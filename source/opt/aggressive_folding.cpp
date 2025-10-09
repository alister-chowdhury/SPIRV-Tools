// Copyright (c) 2019 Google LLC
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

#include "aggressive_folding.h"

#include <vector>

#include "source/opt/instruction.h"
#include "source/opt/ir_context.h"
#include "source/util/bit_vector.h"

namespace spvtools {
namespace opt {

Pass::Status AggressiveFoldingPass::Process() {
  bool modified = false;

  printf("--CCCC--\n");

  MetadataSet metadata;
  ProcessConstants(metadata);
   for (Function& function : *get_module()) {
     printf("---\n");
     cfg()->ForEachBlockInPostOrder(function.entry().get(),
                                    [&modified, &metadata, this](BasicBlock* bb) {
                                      if (ProcessInstructionsInBB(bb, metadata)) {
                                        modified = true;
                                      }
                                    });
   }

   for (const auto& kv : metadata) {
     printf("%u => %s\n", kv.first, kv.second.PrettyPrint().c_str());
   }

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

void AggressiveFoldingPass::ProcessConstants(MetadataSet& metadata) {
  analysis::ConstantManager* const_mgr = context()->get_constant_mgr();
    for(const Instruction* inst : context()->GetConstants()) {
      switch (inst->opcode()) {
        //case spv::Op::OpConstantTrue:  // TODO
        //case spv::Op::OpConstantFalse: // TODO
        //case spv::Op::OpConstantComposite: // TODO
        case spv::Op::OpConstant: {
          InstructionMeta inst_meta;
          if (auto const_value = const_mgr->GetConstantFromInst(inst)) {
            auto const_type = const_value->type();
            if (const_type->AsInteger() &&
                const_type->AsInteger()->width() == 32) {
              inst_meta.SetConstant(const_value->GetU32());
            } else if (const_type->AsFloat() &&
                       const_type->AsFloat()->width() == 32) {
              inst_meta.SetConstant(const_value->GetFloat());
            }
          }
          metadata.emplace(inst->unique_id(), inst_meta);
          break;
        }
      }
      printf("HI %u %s\n", inst->unique_id(), inst->PrettyPrint().c_str());
    }
}

bool AggressiveFoldingPass::ProcessInstructionsInBB(BasicBlock* bb,
                                                    MetadataSet& metadata) {
  bool modified = false;
  printf("BB\n");
  for (auto inst = bb->begin(); inst != bb->end(); ++inst) {
    if (ProcessInstruction(&*inst, metadata)) {
      modified = true;
    }
  }
  return modified;
}

bool AggressiveFoldingPass::ProcessInstruction(Instruction* inst,
                                               MetadataSet& metadata) {
  // context()->get_def_use_mgr()->ForEachUse(
  printf("HELLO %u %s\n", inst->unique_id(), inst->PrettyPrint().c_str());
  return false;
}

}  // namespace opt
}  // namespace spvtools
