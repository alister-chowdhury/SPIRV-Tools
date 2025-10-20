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
#include "ir_builder.h"

#include <algorithm>

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
                                                std::vector<Instruction*>& in_insts) {

  bool modified = false;
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  analysis::ConstantManager* const_mgr = context()->get_constant_mgr();

  struct Subgraph
  {
    std::vector<Instruction*> inputs;
    std::vector<Instruction*> working_space;
    Instruction* output;
  };

  std::unordered_set<Instruction*> intermediates;

  {
    for (Instruction* inst : in_insts) {
      Instruction* lhs = def_use_mgr->GetDef(inst->GetSingleWordInOperand(0));
      Instruction* rhs = def_use_mgr->GetDef(inst->GetSingleWordInOperand(1));

      // Anything with more than one dep, or is fed into
      // a different type of instruction, is considered
      // an output.
      bool is_output = false;
      def_use_mgr->WhileEachUse(inst, [op, &is_output, num_children=0](Instruction* child, uint32_t) mutable {
        is_output = (num_children++ != 0) || is_output;
        is_output = is_output || (op != child->opcode());
        return !is_output;
      });

      if (is_output) {
        
        Subgraph graph;
        graph.output = inst;
        auto Evaluate = [&](Instruction* node) {
          std::vector<Instruction*> stack;
          while (true) {
            if (intermediates.find(node) != intermediates.end()) {
              intermediates.erase(node);
              graph.working_space.push_back(node);
              stack.push_back(def_use_mgr->GetDef(node->GetSingleWordInOperand(0)));
              node = def_use_mgr->GetDef(node->GetSingleWordInOperand(1));
              continue;
            }
            graph.inputs.push_back(node);
            if (!stack.empty()) {
              node = stack.back();
              stack.pop_back();
              continue;
            }
            break;
          }
        };
        Evaluate(lhs);
        Evaluate(rhs);

        // Singular node, not interesting.
        if (graph.working_space.empty()) {
          continue;
        }

        std::sort(graph.inputs.begin(),
          graph.inputs.end(),
          [&](const Instruction* a, const Instruction* b) {
          bool a_const = a->IsConstant();
          bool b_const = b->IsConstant();
          if (a_const != b_const) {
            return a_const;
          }
          return a->result_id() < b->result_id();
        });
        
        modified = true;
        //continue;

        // And we'll create new, better children
        // TODOOOO:
        // REUSE RESULT_ID!!!
        InstructionBuilder ir_builder(context(), inst);
        uint32_t type = inst->type_id();
        auto AddInto = [op, type, &ir_builder, context=context()](Instruction* a, Instruction* b) {
          uint32_t result_id = context->TakeNextId();
          std::unique_ptr<Instruction> inst(new Instruction(
            context, op, type, result_id,
            {{SPV_OPERAND_TYPE_ID, {a->result_id()}}, {SPV_OPERAND_TYPE_ID, {b->result_id()}}}));
          Instruction* new_inst = ir_builder.AddInstruction(std::move(inst));
          context->UpdateDefUse(new_inst);
          return new_inst;
        };

        Instruction* last = AddInto(graph.inputs[0], graph.inputs[1]);
        for (size_t i = 2; i < graph.inputs.size(); ++i) {
          last = AddInto(last, graph.inputs[i]);
        }
        
        context()->ReplaceAllUsesWith(inst->result_id(),
                                      last->result_id());

        // Destroy all previous children
        for (Instruction* prev_inst : graph.working_space) {
          context()->KillInst(prev_inst);
        }
        context()->KillInst(inst);
      }
      else {
        intermediates.insert(inst);
      }
    }

  }


  return modified;

}

}  // namespace opt
}  // namespace spvtools
