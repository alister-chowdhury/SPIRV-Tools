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
namespace {


// x + y - y = x
// x - y + y = x
// x + -(y + y) = x

//// D = OpSub [OpAdd A, B], A => B
//// D = OpSub [OpAdd A, B], B => A
//// D = OpAdd [OpSub A, B], B | OpAdd B, [OpSub A, B] => A
//// D = OpAdd [OpSub B, A], A | OpAdd A, [OpSub B, A] => B
//void HandleRedundantSubAdd(IRContext* ctx,
//                           BasicBlock* bb,
//                           Instruction* inst) {
//  spv::Op op = inst->opcode();
//  assert(op == spv::Op::OpISub ||
//         op == spv::Op::OpIAdd ||
//         op == spv::Op::OpFSub ||
//         op == spv::Op::OpFAdd);
//  
//  ctx->get_instr_block()
//}

}  // namespace

Pass::Status AggressiveFoldingPass::Process() {
  bool modified = false;

   for (Function& function : *get_module()) {
     cfg()->ForEachBlockInPostOrder(function.entry().get(),
                                    [&modified, this](BasicBlock* bb) {
                                    });
   }

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}


}  // namespace opt
}  // namespace spvtools
