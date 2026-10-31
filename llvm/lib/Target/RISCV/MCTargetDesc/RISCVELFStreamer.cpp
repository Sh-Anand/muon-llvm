//===-- RISCVELFStreamer.cpp - RISC-V ELF Target Streamer Methods ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides RISC-V specific target streamer methods.
//
//===----------------------------------------------------------------------===//

#include "RISCVELFStreamer.h"
#include "RISCVAsmBackend.h"
#include "RISCVBaseInfo.h"
#include "RISCVMCExpr.h"
#include "RISCVMCTargetDesc.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/RISCVAttributes.h"

using namespace llvm;

// This part is for ELF object output.
RISCVTargetELFStreamer::RISCVTargetELFStreamer(MCStreamer &S,
                                               const MCSubtargetInfo &STI)
    : RISCVTargetStreamer(S), CurrentVendor("riscv") {
  MCAssembler &MCA = getStreamer().getAssembler();
  const FeatureBitset &Features = STI.getFeatureBits();
  auto &MAB = static_cast<RISCVAsmBackend &>(MCA.getBackend());
  setTargetABI(RISCVABI::computeTargetABI(STI.getTargetTriple(), Features,
                                          MAB.getTargetOptions().getABIName()));
  setFlagsFromFeatures(STI);
  // `j label` in `.option norelax; j label; .option relax; ...; label:` needs a
  // relocation to ensure the jump target is correct after linking. This is due
  // to a limitation that shouldForceRelocation has to make the decision upfront
  // without knowing a possibly future .option relax. When RISCVAsmParser is used,
  // its ParseInstruction may call setForceRelocs as well.
  if (STI.hasFeature(RISCV::FeatureRelax))
    static_cast<RISCVAsmBackend &>(MAB).setForceRelocs();
}

RISCVELFStreamer &RISCVTargetELFStreamer::getStreamer() {
  return static_cast<RISCVELFStreamer &>(Streamer);
}

void RISCVTargetELFStreamer::emitDirectiveOptionPush() {}
void RISCVTargetELFStreamer::emitDirectiveOptionPop() {}
void RISCVTargetELFStreamer::emitDirectiveOptionPIC() {}
void RISCVTargetELFStreamer::emitDirectiveOptionNoPIC() {}
void RISCVTargetELFStreamer::emitDirectiveOptionRVC() {}
void RISCVTargetELFStreamer::emitDirectiveOptionNoRVC() {}
void RISCVTargetELFStreamer::emitDirectiveOptionRelax() {}
void RISCVTargetELFStreamer::emitDirectiveOptionNoRelax() {}

void RISCVTargetELFStreamer::emitAttribute(unsigned Attribute, unsigned Value) {
  getStreamer().setAttributeItem(Attribute, Value, /*OverwriteExisting=*/true);
}

void RISCVTargetELFStreamer::emitTextAttribute(unsigned Attribute,
                                               StringRef String) {
  getStreamer().setAttributeItem(Attribute, String, /*OverwriteExisting=*/true);
}

void RISCVTargetELFStreamer::emitIntTextAttribute(unsigned Attribute,
                                                  unsigned IntValue,
                                                  StringRef StringValue) {
  getStreamer().setAttributeItems(Attribute, IntValue, StringValue,
                                  /*OverwriteExisting=*/true);
}

void RISCVTargetELFStreamer::finishAttributeSection() {
  RISCVELFStreamer &S = getStreamer();
  if (S.Contents.empty())
    return;

  S.emitAttributesSection(CurrentVendor, ".riscv.attributes",
                          ELF::SHT_RISCV_ATTRIBUTES, AttributeSection);
}

void RISCVTargetELFStreamer::finish() {
  RISCVTargetStreamer::finish();
  MCAssembler &MCA = getStreamer().getAssembler();
  RISCVABI::ABI ABI = getTargetABI();

  unsigned EFlags = MCA.getELFHeaderEFlags();

  if (hasRVC())
    EFlags |= ELF::EF_RISCV_RVC;
  if (hasTSO())
    EFlags |= ELF::EF_RISCV_TSO;

  switch (ABI) {
  case RISCVABI::ABI_ILP32:
  case RISCVABI::ABI_LP64:
    break;
  case RISCVABI::ABI_ILP32F:
  case RISCVABI::ABI_LP64F:
    EFlags |= ELF::EF_RISCV_FLOAT_ABI_SINGLE;
    break;
  case RISCVABI::ABI_ILP32D:
  case RISCVABI::ABI_LP64D:
    EFlags |= ELF::EF_RISCV_FLOAT_ABI_DOUBLE;
    break;
  case RISCVABI::ABI_ILP32E:
  case RISCVABI::ABI_LP64E:
    EFlags |= ELF::EF_RISCV_RVE;
    break;
  case RISCVABI::ABI_Unknown:
    llvm_unreachable("Improperly initialised target ABI");
  }

  MCA.setELFHeaderEFlags(EFlags);
}

void RISCVTargetELFStreamer::reset() {
  AttributeSection = nullptr;
}

void RISCVTargetELFStreamer::emitDirectiveVariantCC(MCSymbol &Symbol) {
  getStreamer().getAssembler().registerSymbol(Symbol);
  cast<MCSymbolELF>(Symbol).setOther(ELF::STO_RISCV_VARIANT_CC);
}

void RISCVELFStreamer::reset() {
  static_cast<RISCVTargetStreamer *>(getTargetStreamer())->reset();
  MCELFStreamer::reset();
  MappingSymbolCounter = 0;
  LastMappingSymbols.clear();
  LastEMS = EMS_None;
  CurFn = nullptr;
  FnRegs.clear();
  FnRefs.clear();
}

void RISCVELFStreamer::emitDataMappingSymbol() {
  if (LastEMS == EMS_Data)
    return;
  emitMappingSymbol("$d");
  LastEMS = EMS_Data;
}

void RISCVELFStreamer::emitInstructionsMappingSymbol() {
  if (LastEMS == EMS_Instructions)
    return;
  emitMappingSymbol("$x");
  LastEMS = EMS_Instructions;
}

void RISCVELFStreamer::emitMappingSymbol(StringRef Name) {
  auto *Symbol = cast<MCSymbolELF>(getContext().getOrCreateSymbol(
      Name + "." + Twine(MappingSymbolCounter++)));
  emitLabel(Symbol);
  Symbol->setType(ELF::STT_NOTYPE);
  Symbol->setBinding(ELF::STB_LOCAL);
}

void RISCVELFStreamer::changeSection(MCSection *Section,
                                     const MCExpr *Subsection) {
  // We have to keep track of the mapping symbol state of any sections we
  // use. Each one should start off as EMS_None, which is provided as the
  // default constructor by DenseMap::lookup.
  LastMappingSymbols[getPreviousSection().first] = LastEMS;
  LastEMS = LastMappingSymbols.lookup(Section);

  MCELFStreamer::changeSection(Section, Subsection);
}

void RISCVELFStreamer::emitLabel(MCSymbol *Symbol, SMLoc Loc) {
  // .type sym,@function precedes the label in both compiled code and asm
  if (cast<MCSymbolELF>(Symbol)->getType() == ELF::STT_FUNC)
    CurFn = Symbol;
  MCELFStreamer::emitLabel(Symbol, Loc);
}

void RISCVELFStreamer::addRefs(const MCExpr *E) {
  switch (E->getKind()) {
  case MCExpr::SymbolRef: {
    auto &Sym = const_cast<MCSymbol &>(cast<MCSymbolRefExpr>(E)->getSymbol());
    // temporaries are basic blocks and constants, never functions
    if (!Sym.isTemporary())
      FnRefs.insert({CurFn, &Sym});
    break;
  }
  case MCExpr::Binary:
    addRefs(cast<MCBinaryExpr>(E)->getLHS());
    addRefs(cast<MCBinaryExpr>(E)->getRHS());
    break;
  case MCExpr::Unary:
    addRefs(cast<MCUnaryExpr>(E)->getSubExpr());
    break;
  case MCExpr::Target:
    addRefs(cast<RISCVMCExpr>(E)->getSubExpr());
    break;
  case MCExpr::Constant:
    break;
  }
}

void RISCVELFStreamer::emitInstruction(const MCInst &Inst,
                                       const MCSubtargetInfo &STI) {
  if (CurFn) {
    const MCRegisterInfo &MRI = *getContext().getRegisterInfo();
    const MCRegisterClass &GPR = MRI.getRegClass(RISCV::GPRRegClassID);
    const MCInstrDesc &Desc = MCII->get(Inst.getOpcode());
    auto &Mask = FnRegs[CurFn];
    auto AddDef = [&](MCRegister Reg) {
      if (GPR.contains(Reg)) {
        unsigned Idx = MRI.getEncodingValue(Reg) & 127;
        Mask[Idx / 32] |= 1u << (Idx % 32);
      }
    };
    for (unsigned I = 0; I < Desc.getNumDefs(); ++I)
      if (Inst.getOperand(I).isReg())
        AddDef(Inst.getOperand(I).getReg());
    for (MCPhysReg Reg : Desc.implicit_defs())
      AddDef(Reg);
    for (const MCOperand &Op : Inst)
      if (Op.isExpr())
        addRefs(Op.getExpr());
  }
  emitInstructionsMappingSymbol();
  MCELFStreamer::emitInstruction(Inst, STI);
}

void RISCVELFStreamer::finishImpl() {
  if (!FnRegs.empty()) {
    switchSection(getContext().getELFSection(".rad.regs", ELF::SHT_PROGBITS, 0));
    for (auto &[Fn, Mask] : FnRegs) {
      emitSymbolValue(Fn, 4);
      for (uint32_t Word : Mask)
        emitInt32(Word);
    }
    switchSection(getContext().getELFSection(".rad.refs", ELF::SHT_PROGBITS, 0));
    for (auto &[From, To] : FnRefs) {
      emitSymbolValue(From, 4);
      emitSymbolValue(To, 4);
    }
  }
  MCELFStreamer::finishImpl();
}

void RISCVELFStreamer::emitBytes(StringRef Data) {
  emitDataMappingSymbol();
  MCELFStreamer::emitBytes(Data);
}

void RISCVELFStreamer::emitFill(const MCExpr &NumBytes, uint64_t FillValue,
                                SMLoc Loc) {
  emitDataMappingSymbol();
  MCELFStreamer::emitFill(NumBytes, FillValue, Loc);
}

void RISCVELFStreamer::emitValueImpl(const MCExpr *Value, unsigned Size,
                                     SMLoc Loc) {
  emitDataMappingSymbol();
  MCELFStreamer::emitValueImpl(Value, Size, Loc);
}

namespace llvm {
MCELFStreamer *createRISCVELFStreamer(MCContext &C,
                                      std::unique_ptr<MCAsmBackend> MAB,
                                      std::unique_ptr<MCObjectWriter> MOW,
                                      std::unique_ptr<MCCodeEmitter> MCE,
                                      bool RelaxAll) {
  RISCVELFStreamer *S =
      new RISCVELFStreamer(C, std::move(MAB), std::move(MOW), std::move(MCE));
  S->getAssembler().setRelaxAll(RelaxAll);
  return S;
}
} // namespace llvm
