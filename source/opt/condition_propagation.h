// Copyright (c) 2025 Google Inc.
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

#ifndef SOURCE_OPT_CONDITION_PROPAGATION_PASS_H_
#define SOURCE_OPT_CONDITION_PROPAGATION_PASS_H_

#include <unordered_map>
#include <utility>

#include "source/opt/function.h"
#include "source/opt/ir_context.h"
#include "source/opt/pass.h"

namespace spvtools {
namespace opt {
namespace condprop {

struct OpHash {
  size_t operator()(spv::Op op) const {
    return std::hash<uint32_t>{}(uint32_t(op));
  }
};

class InstReplacements;
class ExprQueue;

using UnaryPropagationRule = void (*)(ExprQueue&, bool, uint32_t);
using BinaryPropagationRule = void (*)(ExprQueue&, bool, uint32_t, uint32_t);
using InstPropagationRule = void (*)(ExprQueue&, bool, Instruction*);

using UnaryPropagationRuleMap =
    std::unordered_map<spv::Op, UnaryPropagationRule, OpHash>;
using BinaryPropagationRuleMap =
    std::unordered_map<spv::Op, BinaryPropagationRule, OpHash>;
using InstPropagationRuleMap =
std::unordered_map<spv::Op, InstPropagationRule, OpHash>;

struct Rules {
  condprop::UnaryPropagationRuleMap unary_rules;
  condprop::BinaryPropagationRuleMap binary_rules;
  condprop::InstPropagationRuleMap inst_rules;
};

}  // namespace condprop

// TODO: Desc
class ConditionPropagationPass : public Pass {
 public:
  ConditionPropagationPass();

  const char* name() const override { return "condition-propagation"; }
  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisInstrToBlockMapping |
           IRContext::kAnalysisDecorations | IRContext::kAnalysisCombinators |
           IRContext::kAnalysisCFG | IRContext::kAnalysisDominatorAnalysis |
           IRContext::kAnalysisNameMap | IRContext::kAnalysisConstants |
           IRContext::kAnalysisTypes;
  }

 private:
  bool ProcessFunction(Function* function);

  bool ProcessSelect(Instruction* inst);
  bool ProcessSelectionMerge(Instruction* inst);

  bool ProcessConditional(Instruction* inst);
  bool ApplyReplacements(BasicBlock* root_bb,
                         const std::unordered_set<BasicBlock*>& filtered_bb,
                         condprop::InstReplacements& inst_repl);
  bool ApplyReplacementsToBB(BasicBlock* bb,
                             condprop::InstReplacements& inst_repl);

  bool ProcessSwitch(Instruction* inst);

  condprop::Rules rules;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_CONDITION_PROPAGATION_PASS_H_
