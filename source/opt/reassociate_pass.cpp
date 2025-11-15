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

#include <algorithm>
#include <unordered_map>

#include "ir_builder.h"
#include "reassociation_graph.h"
#include "scalar_analysis.h"

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

bool ReassociatePass::ProcessInstructionsInBB(BasicBlock* bb) {
  bool modified = false;

  if (ReassociateFP(bb)) {
    modified = true;
  }
  return modified;
}

// Reassociates chains of:
// OpFAdd, OpFSub, OpFNegate
// OpFMul, OpFDiv, OpVectorTimesScalar
//
// Constants are moved closer together e.g:
//   (A + 5 + B - C - 7) => (5 - 7 + A + B - C) => (-2 + A + B - C)
//   (A * 5 * B * C * 7) => (5 * 7 * A * B * C) => (35 * A * B * C)
//
// Variables far apart can cancel each other out e.g:
//  (A + B + C - A - B)   => C
//  (A * B * C / (A * B)) => C
//
// Variables can be factored e.g:
//   (A * x * x + B * x + C) => x * (A * x + B) + C
//   A * x + B * x + C * x   => x * (A + B + C)
bool ReassociatePass::ReassociateFP(BasicBlock* bb) {
  bool modified = false;
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  analysis::TypeManager* type_mgr = context()->get_type_mgr();

  // Independant graphs, with the key being the root node.
  //
  // For an instruction to be a root it must either:
  //  * Be used in a non-chainable instruction
  //  * Be promoted, because it feeds into multiple roots
  std::unordered_map<Instruction*, std::vector<Instruction*>> graphs;

  // Calculate graphs
  {
    std::unordered_map<Instruction*, Instruction*> inst_to_root;

    auto ShouldHandleInstruction = [def_use_mgr, type_mgr](Instruction* inst) {
      switch (inst->opcode()) {
        case spv::Op::OpFDiv:
        case spv::Op::OpFMul:
        case spv::Op::OpVectorTimesScalar:
        case spv::Op::OpFAdd:
        case spv::Op::OpFSub:
        case spv::Op::OpFNegate:
          break;
        default:
          return false;
      }
      if (inst->IsFloatingPointFoldingAllowed()) {
        analysis::Type* type = type_mgr->GetType(inst->type_id());
        if (analysis::Vector* vec_type = type->AsVector()) {
          if (const analysis::Float* float_type =
                  vec_type->element_type()->AsFloat()) {
            uint32_t width = float_type->width();
            return ((width == 32) || (width == 64));
          }
        }
        if (const analysis::Float* float_type = type->AsFloat()) {
          uint32_t width = float_type->width();
          return ((width == 32) || (width == 64));
        }
      }
      return false;
    };

    for (auto it = bb->rbegin(); it != bb->rend(); ++it) {
      Instruction* inst = &*it;
      if (!ShouldHandleInstruction(inst)) {
        continue;
      }

      Instruction* instr_root = nullptr;
      def_use_mgr->WhileEachUse(
          inst, [&](Instruction* child, uint32_t) mutable {
            // Used in a non-chainable instruction
            auto found_parent = inst_to_root.find(child);
            if (found_parent == inst_to_root.end()) {
              instr_root = nullptr;
              return false;
            }
            // Promote to root, because it feeds into multiple roots
            Instruction* instr_root_local = found_parent->second;
            if (instr_root && (instr_root_local != instr_root)) {
              instr_root = nullptr;
              return false;
            }
            instr_root = instr_root_local;
            return true;
          });

      // New root found!
      if (!instr_root) {
        graphs[inst] = {};
        instr_root = inst;
      }

      inst_to_root[inst] = instr_root;
      graphs[instr_root].push_back(inst);
    }

    // Remove trivial graphs and reverse the instruction order, since we
    // iterated backwards.
    for (auto it = graphs.begin(); it != graphs.end();) {
      if (it->second.size() < 3) {
        it = graphs.erase(it);
        continue;
      }
      std::reverse(it->second.begin(), it->second.end());
      ++it;
    }
  }

  if (graphs.empty()) {
    return modified;
  }

  for (auto& entry : graphs) {
    if (ReassociateFPGraph(entry.first, std::move(entry.second))) {
      modified = true;
    }
  }

  return modified;
}

bool ReassociatePass::ReassociateFPGraph(Instruction* root,
                                         std::vector<Instruction*>&& graph) {
  using namespace reassociate;

  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::ConstantManager* const_mgr = context()->get_constant_mgr();

  analysis::Type* type = type_mgr->GetType(root->type_id());
  FPReassocGraph fpgraph(type, def_use_mgr, const_mgr);

  for (Instruction* inst : graph) {
    fpgraph.AddInstruction(inst);
  }

  const FPNode* root_fp = fpgraph.FindInstruction(root);

  std::cout << "BEFORE OPT:\n\n";
  root_fp->DotGraph(std::cout);
  std::cout << "\n\n";
  root_fp->PrintNode(std::cout);
  std::cout << "\n\n";
  root_fp->PrintEquation(std::cout);
  std::cout.flush();

  root_fp = fpgraph.SimplifyNode(root_fp);

  std::cout << "\n\n\nAFTER OPT:\n\n";
  root_fp->DotGraph(std::cout);
  std::cout << "\n\n";
  root_fp->PrintNode(std::cout);
  std::cout << "\n\n";
  root_fp->PrintEquation(std::cout);
  std::cout.flush();

  std::vector<const FPNode*> nodes = fpgraph.ResolveNodes(root_fp);

  // It's not very likely that we're going to reduce the number
  // of instructions if our node count isn't already less.
  if (nodes.size() >= graph.size()) {
    return false;
  }

  // Special case, if we folded to either a constant or external
  // probably shouldn't need to do this specifically for this tbh
  if (nodes.size() == 1) {
    const FPNode* single = nodes[0];
    bool is_constant = single->node_type == FPNode::kConstant;
    bool is_external = single->node_type == FPNode::kExternal;
    
    if (is_constant || is_external) {
      
      bool can_kill_root = true;
      
      // We may need to convert this to a OpCompositeConstruct
      // if the input was scalar but the output is vector.
      if (is_external && fpgraph.IsVector()) {
        Instruction* ext = def_use_mgr->GetDef(single->result_id);
        can_kill_root = type_mgr->GetType(ext->type_id())->AsVector() != nullptr;
        if (!can_kill_root) {
          root->SetOpcode(spv::Op::OpCompositeConstruct);
          std::vector<Operand> operands;
          for (uint32_t i = 0; i < fpgraph.ElementSize(); ++i) {
            operands.push_back({ SPV_OPERAND_TYPE_LITERAL_INTEGER, {single->result_id} });
          }
          root->SetInOperands(std::move(operands));
        }
      }

      // TODO: If constant, redirect result_id to new constant
      if (is_constant) {

      }

      // TODO: Replace all uses of root with either the external
      // or constant
      if (can_kill_root) {

      }
      
      for (Instruction* inst : graph) {
        if (!can_kill_root && inst == root) {
          // Mark kill
        }
      }

      return true;
    }
  }

  fpgraph.TopologicallySort(nodes);

  // ReassocGraphFP::FPReassocNode* root_node = fpgraph.GetUserExternal(root);
  // root_node->ConvertAddsToMuls(fpgraph);
  //// Run factorisation pass here

  bool modified = false;
  // if ((root_node->flags & ReassocGraphFP::FPReassocNode::kBeenOptimised)) {
  //   modified = true;

  //}
  return modified;
}

}  // namespace opt
}  // namespace spvtools
