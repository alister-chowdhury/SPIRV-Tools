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
#include "scalar_analysis.h"
#include "ir_builder.h"

#include <algorithm>
#include <unordered_map>

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

  // Independant graphs, with the key being the root node.
  //
  // For an instruction to be a root it must either:
  //  * Be used in a non-chainable instruction
  //  * Be promoted, because it feeds into multiple roots
  std::unordered_map<Instruction*, std::vector<Instruction*>> graphs;

  // #1 Calculate graphs
  {
    std::unordered_map<Instruction*, Instruction*> inst_to_root;

    auto ShouldHandleInstruction = [def_use_mgr](Instruction* inst) {
      switch (inst->opcode()) {
      case spv::Op::OpFDiv:
      case spv::Op::OpFMul:
      //case spv::Op::OpVectorTimesScalar: // TODO
      case spv::Op::OpFAdd:
      case spv::Op::OpFSub:
      case spv::Op::OpFNegate:
        return inst->IsFloatingPointFoldingAllowed();
      default:
        break;
      }
      return false;
      };

    for (auto it = bb->rbegin(); it != bb->rend(); ++it) {
      Instruction* inst = &*it;
      if (!ShouldHandleInstruction(inst)) {
        continue;
      }

      Instruction* instr_root = nullptr;
      def_use_mgr->WhileEachUse(inst, [&](Instruction* child, uint32_t) mutable {
        // Used in a non-chainable instruction
        auto found_parent = inst_to_root.find(child);
        if (found_parent == inst_to_root.end()) {
          instr_root = nullptr;
          return false;
        }
        // Promote to root, because it feeds into multiple roots
        Instruction* instr_root_local = found_parent->second;
        if (instr_root && (instr_root_local == instr_root)) {
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


/////

struct ReassocGraphFP {

  struct FPReassocNode {

    enum class NodeType {
      kExternal,
      kAdd,     // add / sub
      kMul      // mul / div
    };

    using FlagsType = uint32_t;
    enum Flags : FlagsType {
      kNone = 0,
      kConstant = (1u << 0),
      kUserExternal = (1u << 1),
      kBeenOptimised = (1u << 2),
      kNeedsCombining = (1u << 3),
      kNeedsHashing = (1u << 4)
    };

    NodeType node_type = NodeType::kExternal;
    FlagsType flags = kNone;
    uint32_t result_id = UINT32_MAX;
    uint64_t hash = 0;
    
    // Comparison for sorting, if it's a constant it'll be
    // prioritised, otherwise use a hash.
    bool operator<(const FPReassocNode& other) const {
      assert((flags & kNeedsHashing) == 0);
      assert((other.flags & kNeedsHashing) == 0);
      if ((flags & kConstant) == (other.flags & kConstant)) {
        return hash < other.hash;
      }
      return (flags & kConstant) != 0;
    }

    bool HasBeenOptimised() const {
      return (flags & kBeenOptimised) == kBeenOptimised;
    }

    void RemoveInput(FPReassocNode* node) {
      assert(node_type != NodeType::kExternal);
      auto found = inputs.find(node);
      if (found != inputs.end()) {
        inputs.erase(found);
        flags |= kNeedsHashing;
        if (inputs.size() == 0) {
          flags |= kConstant;
        }
      }
    }

    void AddInput(FPReassocNode* node, int32_t count) {
      assert(node_type != NodeType::kExternal);
      if (count == 0) {
        return;
      }
      auto found = inputs.find(node);
      if (found == inputs.end()) {
        if ((node->flags & kConstant) != kConstant) {
          flags &= ~kConstant;
        }
        flags |= (node->flags & kBeenOptimised);
        inputs[node] = count;
      }
      else {
        found->second += count;
        // Cancellation took place
        if (found->second == 0) {
          flags |= kBeenOptimised;
          inputs.erase(found);
          if (inputs.size() == 0) {
            flags |= kConstant;
          }
        }
      }
      flags |= kNeedsHashing;
    }

    // Combine inputs that are of the same type
    void CombineInputs() {
      if ((flags & kNeedsCombining) == 0) {
        return;
      }
      assert(node_type != NodeType::kExternal);

      // Clear our current state and re-evaluate it.
      std::unordered_map<FPReassocNode*, int32_t> inputs_local = std::move(inputs);
      flags |= kNeedsHashing;
      flags |= kConstant;

      for (auto it : inputs_local) {
        it.first->CombineInputs();
        flags |= (it.first->flags & kBeenOptimised);
        if (it.first->node_type == node_type) {
          for (auto it2 : it.first->inputs) {
            AddInput(it2.first, it2.second * it.second);
          }
        }
        else {
          AddInput(it.first, it.second);
        }
      }
      flags &= ~kNeedsCombining;
    }

    FPReassocNode* PropagateConstants(ReassocGraphFP& graph, InstructionBuilder& ir_builder, analysis::Type* type);

    // Inputs to this node and their count
    std::unordered_map<FPReassocNode*, int32_t> inputs;
  };

  ReassocGraphFP(analysis::DefUseManager* def_use_mgr_, analysis::ConstantManager* const_mgr_, const ValueNumberTable* vn_table_)
    : def_use_mgr(def_use_mgr_), const_mgr(const_mgr_), vn_table(vn_table_)
  { }

  FPReassocNode* CreateNode(Instruction* inst);
  FPReassocNode* FindOrCreateUserExternal(Instruction* inst);
  FPReassocNode* FindOrCreateUserExternal(uint32_t inst_id) {
    return FindOrCreateUserExternal(def_use_mgr->GetDef(inst_id));
  }
  
  FPReassocNode* GetUserExternal(Instruction* inst) {
    return user_external.at(inst);
  }

  FPReassocNode* CreateConst(double value, analysis::Type* type) {
    uint32_t width = type->AsFloat()->width();
    assert(width == 32 || width == 64);
    uint32_t const_id;
    if (width == 32) {
      const_id = const_mgr->GetFloatConstId((float)value);
    }
    else {
      const_id = const_mgr->GetDoubleConstId(value);
    }
    if (analysis::Vector* vec = type->AsVector()) {
      uint32_t num = vec->element_count();
      std::vector<uint32_t> v;
      v.resize(num);
      for (uint32_t i = 0; i < num; ++i) {
        v[i] = const_id;
      }
      Instruction* inst = const_mgr->GetDefiningInstruction(const_mgr->GetConstant(type, v));
      return FindOrCreateUserExternal(inst);
    }
    return FindOrCreateUserExternal(const_id);
  }

  analysis::DefUseManager* def_use_mgr;
  const ValueNumberTable* vn_table;
  analysis::ConstantManager* const_mgr;
  std::unordered_map<Instruction*, FPReassocNode*> user_external;
  std::vector<std::unique_ptr<FPReassocNode>> storage;
};

ReassocGraphFP::FPReassocNode* ReassocGraphFP::CreateNode(Instruction* inst) {
  assert(user_external.find(inst) == user_external.end());
  FPReassocNode* new_node = new FPReassocNode{};
  user_external[inst] = new_node;
  storage.emplace_back(new_node);
  new_node->result_id = inst->result_id();
  new_node->flags = FPReassocNode::kUserExternal
                  | FPReassocNode::kConstant
                  | FPReassocNode::kNeedsCombining
                  | FPReassocNode::kNeedsHashing
                  ;
  switch (inst->opcode()) {
  case spv::Op::OpFDiv: {
    new_node->node_type = FPReassocNode::NodeType::kMul;
    FPReassocNode* lhs = FindOrCreateUserExternal(inst->GetSingleWordInOperand(0));
    FPReassocNode* rhs = FindOrCreateUserExternal(inst->GetSingleWordInOperand(1));
    new_node->AddInput(lhs, 1);
    new_node->AddInput(rhs, -1);
    break;
  }
  case spv::Op::OpFMul: {
    new_node->node_type = FPReassocNode::NodeType::kMul;
    FPReassocNode* lhs = FindOrCreateUserExternal(inst->GetSingleWordInOperand(0));
    FPReassocNode* rhs = FindOrCreateUserExternal(inst->GetSingleWordInOperand(1));
    new_node->AddInput(lhs, 1);
    new_node->AddInput(rhs, 1);
    break;
  }
  case spv::Op::OpFSub: {
    new_node->node_type = FPReassocNode::NodeType::kAdd;
    FPReassocNode* lhs = FindOrCreateUserExternal(inst->GetSingleWordInOperand(0));
    FPReassocNode* rhs = FindOrCreateUserExternal(inst->GetSingleWordInOperand(1));
    new_node->AddInput(lhs, 1);
    new_node->AddInput(rhs, -1);
    break;
  }
  case spv::Op::OpFAdd: {
    new_node->node_type = FPReassocNode::NodeType::kAdd;
    FPReassocNode* lhs = FindOrCreateUserExternal(inst->GetSingleWordInOperand(0));
    FPReassocNode* rhs = FindOrCreateUserExternal(inst->GetSingleWordInOperand(1));
    new_node->AddInput(lhs, 1);
    new_node->AddInput(rhs, 1);
    break;
  }
  case spv::Op::OpFNegate: {
    new_node->node_type = FPReassocNode::NodeType::kAdd;
    FPReassocNode* lhs = FindOrCreateUserExternal(inst->GetSingleWordInOperand(0));
    new_node->AddInput(lhs, -1);
    break;
  }
  default:
    assert(false);
    break;
  }
  return new_node;
}

ReassocGraphFP::FPReassocNode* ReassocGraphFP::FindOrCreateUserExternal(Instruction* inst) {
  auto found = user_external.find(inst);
  if (found != user_external.end()) {
    return found->second;
  }
  FPReassocNode* new_node = new FPReassocNode{};
  storage.emplace_back(new_node);
  new_node->result_id = inst->result_id();
  new_node->hash = uint64_t(vn_table->GetValueNumber(inst))
                 | (uint64_t(new_node->result_id) << 32)
                 ;
  new_node->flags = FPReassocNode::kUserExternal;
  if (inst->IsConstant()) {
    new_node->flags |= FPReassocNode::kConstant;
  }
  user_external[inst] = new_node;
  return new_node;
}

ReassocGraphFP::FPReassocNode* ReassocGraphFP::FPReassocNode::PropagateConstants( ReassocGraphFP& graph,
                                                                                  InstructionBuilder& ir_builder,
                                                                                  analysis::Type* type) {
  if (node_type == NodeType::kExternal) {
    return this;
  }

  // Clear our current state and re-evaluate it.
  std::unordered_map<FPReassocNode*, int32_t> inputs_local = std::move(inputs);
  flags |= kNeedsHashing;
  flags |= kConstant;

  for (auto it : inputs_local) {
    AddInput(it.first->PropagateConstants(graph, ir_builder, type),
             it.second);
  }

  if ((flags & kConstant) == kConstant) {
    analysis::DefUseManager* def_use_mgr = graph.def_use_mgr;
    analysis::ConstantManager* const_mgr = graph.const_mgr;
    assert(std::all_of(inputs.begin(), inputs.end(), [](auto it) {
      return (it.first->node_type == NodeType::kExternal);
    }));

    // No inputs, due to cancellation.
    // kAdd => 0
    // kMul => 1
    if (inputs.empty()) {
      if (node_type == NodeType::kAdd) {
        return graph.CreateConst(0.0, type);
      }
      else if(node_type == NodeType::kMul) {
        return graph.CreateConst(1.0, type);
      }
      assert(false);
    }

    // For add, start with 0.0
    // result += A * num
    //
    // For mul, start with 1.0
    // for _ in num:
    //    result *= A

  }

  return this;
}


bool ReassociatePass::ReassociateFPGraph(Instruction* root, std::vector<Instruction*>&& graph) {

  ReassocGraphFP fpgraph(context()->get_def_use_mgr(),
                         context()->get_constant_mgr(),
                         context()->GetValueNumberTable());
  for (Instruction* inst : graph) {
    fpgraph.CreateNode(inst);
  }

  bool modified = false;
  ReassocGraphFP::FPReassocNode* root_node = fpgraph.GetUserExternal(root);
  root_node->CombineInputs();

  InstructionBuilder ir_builder(
    context(), root,
    IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);

  analysis::Type* type = context()->get_type_mgr()->GetType(root->type_id());
  root_node = root_node->PropagateConstants(fpgraph,
                                            ir_builder,
                                            type);

  if (root_node->HasBeenOptimised()) {
    modified = true;
  }

  return modified;
}

}  // namespace opt
}  // namespace spvtools
