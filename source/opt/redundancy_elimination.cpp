// Copyright (c) 2017 Google Inc.
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

#include "source/opt/redundancy_elimination.h"

#include "source/opt/value_number_table.h"

namespace spvtools {
namespace opt {

Pass::Status RedundancyEliminationPass::Process() {
  bool modified = false;
  ValueNumberTable vnTable(context());

  for (auto& func : *get_module()) {
    if (func.IsDeclaration()) {
      continue;
    }

    // Build the dominator tree for this function. It is how the code is
    // traversed.
    DominatorTree& dom_tree =
        context()->GetDominatorAnalysis(&func)->GetDomTree();

    if (HoistSharedInstrutions(dom_tree.GetRoot(), vnTable)) {
      modified = true;
      context()->InvalidateAnalyses(IRContext::kAnalysisDefUse);
      context()->BuildInvalidAnalyses(IRContext::kAnalysisDefUse);
    }
    if (EliminateRedundanciesFrom(dom_tree.GetRoot(), vnTable)) {
      modified = true;
    }
  }
  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

bool RedundancyEliminationPass::EliminateRedundanciesFrom(
    DominatorTreeNode* bb, const ValueNumberTable& vnTable) {
  struct State {
    DominatorTreeNode* node;
    std::map<uint32_t, uint32_t> value_to_id_map;
  };
  std::vector<State> todo;
  todo.push_back({bb, std::map<uint32_t, uint32_t>()});
  bool modified = false;
  for (size_t next_node = 0; next_node < todo.size(); next_node++) {
    modified |= EliminateRedundanciesInBB(todo[next_node].node->bb_, vnTable,
                                          &todo[next_node].value_to_id_map);
    for (DominatorTreeNode* child : todo[next_node].node->children_) {
      todo.push_back({child, todo[next_node].value_to_id_map});
    }
  }
  return modified;
}

bool RedundancyEliminationPass::HoistSharedInstrutions(
  DominatorTreeNode* bb, const ValueNumberTable& vnTable) {
  bool modified = false;
  for (DominatorTreeNode* child : *bb) {
    if (HoistSharedInstrutions(child, vnTable)) {
      modified = true;
    }
  }
  if (bb->children_.size() < 2) {
    return modified;
  }

  std::vector<std::unordered_set<Instruction*>> child_instructions;
  while(true)
  {
    std::unordered_map<uint32_t, uint32_t> vn_counts;
    uint32_t relevant_child_count = 0;
    for (DominatorTreeNode* child : *bb) {
      bool has_instructions = false;
      child->bb_->ForEachInst([&](Instruction* inst) {
        if (inst->result_id() == 0) {
          return;
        }
        uint32_t vn = vnTable.GetValueNumber(inst);
        if (vn != 0) {
          vn_counts[vn] += 1;
          has_instructions = true;
        }
      });
      if (has_instructions) {
        ++relevant_child_count;
      }
    }
    
    // Stop hoisting if there's only one child
    if (relevant_child_count == 1) {
      break;
    }

    std::unordered_set<uint32_t> dedup_vns;
    for (const auto& vnc : vn_counts) {
      if (vnc.second >= relevant_child_count) {
        dedup_vns.insert(vnc.first);
      }
    }
    // Nothing was duplicated, bail.
    if (dedup_vns.empty()) {
      break;
    }
    std::unordered_map<uint32_t, std::vector<Instruction*>> matching;
    for (DominatorTreeNode* child : *bb) {
      child->bb_->ForEachInst([&](Instruction* inst) {
        if (inst->result_id() == 0) {
          return;
        }
        uint32_t vn = vnTable.GetValueNumber(inst);
        if (dedup_vns.find(vn) != dedup_vns.end()) {
          matching[vn].push_back(inst);
        }
      });
    }

    for (auto& match : matching) {
      std::vector<Instruction*>& instructions = match.second;
      Instruction* promoted = instructions.back();
      instructions.pop_back();
      uint32_t promoted_result = promoted->result_id();

      // Insert the node just before a OpBranch or OpBranchConditional
      promoted->RemoveFromList();
      bb->bb_->rbegin()->PreviousNode()->InsertBefore(std::unique_ptr<Instruction>(promoted));
      context()->set_instr_block(promoted, bb->bb_);

      for (Instruction* demoted : instructions) {
        context()->KillNamesAndDecorates(demoted);
        context()->ReplaceAllUsesWith(demoted->result_id(), promoted_result);
        context()->KillInst(demoted);
      }
    }

    modified = true;
  }

  return modified;
}

}  // namespace opt
}  // namespace spvtools
