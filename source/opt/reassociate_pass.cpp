// Copyright (c) 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "reassociate_pass.h"

namespace spvtools {
namespace opt {


Pass::Status ReassociatePass::Process() {

  bool modified = false;
  for (Function& function : *get_module()) {
    cfg()->ForEachBlockInPostOrder(function.entry().get(),
      [this, &modified](BasicBlock* bb) {
        if (ProcessInstructionsInBB(bb)) {
          modified = true;
        }
      });
  }
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}


static bool CanReassociate(spv::Op op) {
  switch (op) {
  case spv::Op::OpIAdd:
  case spv::Op::OpFAdd:
  case spv::Op::OpIMul:
  case spv::Op::OpFMul:
  case spv::Op::OpBitwiseXor:
  case spv::Op::OpBitwiseOr:
  case spv::Op::OpBitwiseAnd:
    return true;
  }
  return false;
}

bool ReassociatePass::ProcessInstructionsInBB(BasicBlock* bb) {

  std::unordered_map<spv::Op, std::vector<Instruction*>> inst_by_type;
  
  for (Instruction& inst : *bb) {
    spv::Op op = inst.opcode();
    if (CanReassociate(op)) {
      inst_by_type[op].push_back(&inst);
    }
  }

  bool modified = false;
  for (auto& op_insts : inst_by_type) {
    if (ProcessInstructionsByType(op_insts.first, op_insts.second)) {
      modified = true;
    }
  }

  return modified;
}


bool ReassociatePass::ProcessInstructionsByType(spv::Op op,
                                                std::vector<Instruction*>& insts) {

  bool modified = false;
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  analysis::ConstantManager* const_mgr = context()->get_constant_mgr();

  // 1. Order operators so their id's are sorted.
  //    This prevents emitting duplicate instructions.
  //    e.g:
  //      %28 = OpIAdd %uint %24 %27
  //      %29 = OpIAdd %uint %27 %24
  //    =>
  //      %28 = OpIAdd %uint %27 %24
  //      %29 = OpIAdd %uint %27 %24 <- kill
  {
    std::unordered_map<uint64_t, Instruction*> args_to_inst;
    auto iter = insts.begin();
    while (iter < insts.end()) {
      Instruction* inst = *iter;
      Instruction* lhs = def_use_mgr->GetDef(inst->GetSingleWordInOperand(0));
      Instruction* rhs = def_use_mgr->GetDef(inst->GetSingleWordInOperand(1));

      bool lhs_const = lhs->IsConstant();
      bool rhs_const = rhs->IsConstant();

      bool flip = false;
      // Keep constants on the left
      if (rhs_const == lhs_const) {
        flip = lhs->result_id() < rhs->result_id();
      }
      else if (lhs_const) {
        flip = true;
      }

      if (flip) {
        std::swap(lhs, rhs);
        modified = true;
        inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {lhs->result_id()}},
                             {SPV_OPERAND_TYPE_ID, {rhs->result_id()}}});
      }

      uint64_t hash = (uint64_t(lhs->result_id()) << 32) | rhs->result_id();
      auto found = args_to_inst.find(hash);
      if (found != args_to_inst.end()) {
        modified = true;
        context()->ReplaceAllUsesWith(inst->result_id(), found->second->result_id());
        context()->KillInst(inst);
        iter = insts.erase(iter);
      }
      else {
        if (flip) {
          context()->AnalyzeUses(inst);
        }
        args_to_inst[hash] = inst;
        ++iter;
      }
    }
  }

  //std::unordered_set<Instruction*> known_constant_exprs;

  //for (Instruction* inst : insts) {
  //  Instruction* lhs = def_use_mgr->GetDef(inst->GetSingleWordInOperand(0));
  //  Instruction* rhs = def_use_mgr->GetDef(inst->GetSingleWordInOperand(1));

  //  bool lhs_const = lhs->IsConstant() || (known_constant_exprs.find(lhs) != known_constant_exprs.end());
  //  bool rhs_const = rhs->IsConstant() || (known_constant_exprs.find(rhs) != known_constant_exprs.end());
  //  if (lhs_const && rhs_const) {
  //    known_constant_exprs.insert(inst);
  //    continue;
  //  }



  //}

  return modified;

}

}  // namespace opt
}  // namespace spvtools
