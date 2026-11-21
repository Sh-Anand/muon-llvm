//===-- RISCVRegBudget.cpp - Vortex: the registers a function may use -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Muon gives a warp its registers for the whole kernel: a thread is charged for every register the kernel,
// _start and what it calls write, and a core holds min(8, 512 / registers) warps.  Registers below the
// next such boundary cost nothing; one that crosses it costs a warp.
//
// So a kernel may use: the registers it must have (x0, those _start writes, those its code names, x0-x31
// if it calls anything), then as many more as its peak pressure and a margin need, rounded up to the most
// the same number of warps allows.  Everything else is reserved, which also tells the scheduler how far
// it may raise pressure.  -riscv-max-regs caps the budget lower, for more warps at the cost of spills.
// Functions that are not kernels stay within x0-x31, as the runtime libraries do, so a call costs a
// kernel at most x0-x31.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-reg-budget"
#define RISCV_REG_BUDGET_NAME "RISC-V Vortex register budget"

static cl::opt<unsigned> MaxRegs(
    "riscv-max-regs", cl::Hidden, cl::init(0),
    cl::desc("Vortex: give a kernel at most this many registers a thread, rounded up to the most the "
             "same number of warps a core allows; it spills if it needs more"));

static cl::opt<unsigned> Margin(
    "riscv-reg-budget-margin", cl::Hidden, cl::init(2),
    cl::desc("Vortex: registers added to a kernel's peak pressure before its budget is chosen"));

bool llvm::isVortexKernel(const Function &F) {
  return F.hasSection() && F.getSection() == "rad_kernels";
}

// the most registers a thread can have and still give the warps a core holds at `Regs`
unsigned llvm::RISCVRegBudgetStepTop(unsigned Regs) {
  constexpr unsigned PerLane = 512, Slots = 8, Max = 128;
  unsigned Warps = std::min(Slots, PerLane / std::clamp(Regs, 1u, Max));
  return std::min(Max, PerLane / Warps);
}

namespace {

class RISCVRegBudget : public MachineFunctionPass {
public:
  static char ID;
  RISCVRegBudget() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervals>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return RISCV_REG_BUDGET_NAME; }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // end anonymous namespace

char RISCVRegBudget::ID = 0;
INITIALIZE_PASS_BEGIN(RISCVRegBudget, DEBUG_TYPE, RISCV_REG_BUDGET_NAME, false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_END(RISCVRegBudget, DEBUG_TYPE, RISCV_REG_BUDGET_NAME, false, false)
char &llvm::RISCVRegBudgetID = RISCVRegBudget::ID;

bool RISCVRegBudget::runOnMachineFunction(MachineFunction &MF) {
  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
  if (!ST.hasVendorXVortex())
    return false;
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetRegisterInfo &TRI = *ST.getRegisterInfo();
  const Function &F = MF.getFunction();
  bool Kernel = isVortexKernel(F);
  auto Index = [](Register R) -> int {
    return R >= RISCV::X0 && R <= RISCV::X127 ? R - RISCV::X0 : -1;
  };

  // the registers it must have: x0, those _start writes (ra, sp, gp, tp, t0-t2, a0-a7), those its code
  // names (inline asm, call arguments), and x0-x31 if it calls anything
  BitVector Fixed(128);
  for (unsigned R : {0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17})
    Fixed.set(R);
  bool Calls = false;
  // a value copied to or from a register the code names (an inline asm operand, a call argument) will
  // be allocated to that register, so the two share it
  DenseMap<Register, int> Shares;
  for (const MachineBasicBlock &MBB : MF)
    for (const MachineInstr &MI : MBB) {
      Calls |= MI.isCall();
      for (const MachineOperand &MO : MI.operands())
        if (MO.isReg() && MO.getReg().isPhysical() && Index(MO.getReg()) >= 0)
          Fixed.set(Index(MO.getReg()));
      if (MI.isCopy()) {
        Register D = MI.getOperand(0).getReg(), S = MI.getOperand(1).getReg();
        if (D.isPhysical() && S.isVirtual() && Index(D) >= 0 && !MRI.isReserved(D))
          Shares.try_emplace(S, Index(D));
        if (S.isPhysical() && D.isVirtual() && Index(S) >= 0 && !MRI.isReserved(S))
          Shares.try_emplace(D, Index(S));
      }
    }
  if (Calls || !Kernel)
    Fixed.set(0, 32);
  auto *FI = MF.getInfo<RISCVMachineFunctionInfo>();
  if (!Kernel) {
    FI->setRegBudget(Fixed, Fixed.count(), 0);
    MRI.freezeReservedRegs(MF);
    return false;
  }

  // peak pressure: the registers occupied at once by its GPR values and by the registers its code names.
  // A named register and the values that share it take one register, or as many as those values live
  struct Event {
    SlotIndex At;
    int Delta;
    int Reg;   // the named register it shares, or -1
    bool Phys; // the named register itself
  };
  LiveIntervals &LIS = getAnalysis<LiveIntervals>();
  SmallVector<Event, 256> Events;
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    Register R = Register::index2VirtReg(I);
    if (MRI.reg_nodbg_empty(R) || !LIS.hasInterval(R))
      continue;
    const TargetRegisterClass *RC = MRI.getRegClassOrNull(R);
    if (!RC || !RC->getNumRegs() || Index(*RC->begin()) < 0)
      continue;
    auto Shared = Shares.find(R);
    int Reg = Shared == Shares.end() ? -1 : Shared->second;
    for (const LiveRange::Segment &S : LIS.getInterval(R))
      Events.push_back({S.start, 1, Reg, false}), Events.push_back({S.end, -1, Reg, false});
  }
  for (unsigned I : Fixed.set_bits()) {
    MCRegister P = RISCV::X0 + I;
    if (MRI.isReserved(P))
      continue;
    for (MCRegUnit U : TRI.regunits(P))
      for (const LiveRange::Segment &S : LIS.getRegUnit(U))
        Events.push_back({S.start, 1, (int)I, true}), Events.push_back({S.end, -1, (int)I, true});
  }
  llvm::sort(Events, [](const Event &A, const Event &B) {
    return A.At < B.At || (A.At == B.At && A.Delta < B.Delta);
  });
  int Values[128] = {}, Named[128] = {}, Live = 0, Peak = 0;
  auto Takes = [&](int R) { return std::max(Values[R], Named[R] > 0 ? 1 : 0); };
  for (const Event &E : Events) {
    if (E.Reg < 0) {
      Live += E.Delta;
    } else {
      Live -= Takes(E.Reg);
      (E.Phys ? Named : Values)[E.Reg] += E.Delta;
      Live += Takes(E.Reg);
    }
    Peak = std::max(Peak, Live);
  }

  // x0, sp, gp and tp are charged too; the values can also use the fixed registers they do not occupy
  unsigned Need = std::max<unsigned>(Fixed.count(), 4 + Peak) + Margin;
  unsigned Budget = RISCVRegBudgetStepTop(Need);
  if (MaxRegs)
    Budget = std::min(Budget, RISCVRegBudgetStepTop(MaxRegs));

  // the fixed registers, then the allocation order's first others, up to the budget
  BitVector Allowed = Fixed;
  for (MCPhysReg R : RISCV::GPRRegClass) {
    if (Allowed.count() >= Budget)
      break;
    if (!MRI.isReserved(R))
      Allowed.set(Index(R));
  }
  FI->setRegBudget(Allowed, Budget, Need);
  MRI.freezeReservedRegs(MF);
  return false;
}

FunctionPass *llvm::createRISCVRegBudgetPass() { return new RISCVRegBudget(); }
