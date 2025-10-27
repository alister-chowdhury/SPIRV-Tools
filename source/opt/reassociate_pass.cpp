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

#include <algorithm>
#include <unordered_map>

#include "ir_builder.h"
#include "reassociate_pass.h"
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
          // case spv::Op::OpVectorTimesScalar: // TODO
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
          if (const analysis::Float* float_type = vec_type->element_type()->AsFloat()) {
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

  struct FPConstAccum {
    FPConstAccum& operator+=(const FPConstAccum& other) {
      assert(other.vals.size() == vals.size());
      size_t n = vals.size();
      for (size_t i = 0; i < n; ++i) {
        vals[i] += other.vals[i];
      }
      return *this;
    }
    FPConstAccum& operator-=(const FPConstAccum& other) {
      assert(other.vals.size() == vals.size());
      size_t n = vals.size();
      for (size_t i = 0; i < n; ++i) {
        vals[i] -= other.vals[i];
      }
      return *this;
    }
    FPConstAccum& operator*=(const FPConstAccum& other) {
      assert(other.vals.size() == vals.size());
      size_t n = vals.size();
      for (size_t i = 0; i < n; ++i) {
        vals[i] *= other.vals[i];
      }
      return *this;
    }
    FPConstAccum& operator*=(double x) {
      size_t n = vals.size();
      for (size_t i = 0; i < n; ++i) {
        vals[i] *= x;
      }
      return *this;
    }
    FPConstAccum& operator/=(const FPConstAccum& other) {
      assert(other.vals.size() == vals.size());
      size_t n = vals.size();
      for (size_t i = 0; i < n; ++i) {
        vals[i] /= other.vals[i];
      }
      return *this;
    }

    bool AllZero() const {
      for (const double v : vals) {
        if (v != 0.0) {
          return false;
        }
      }
      return true;
    }

    bool AllOne() const {
      for (const double v : vals) {
        if (v != 1.0) {
          return false;
        }
      }
      return true;
    }

    std::vector<double> vals;
  };

  struct FPReassocNode {
    enum class NodeType {
      kInvalid,
      kExternal,
      kAdd,  // add / sub
      kMul   // mul / div
    };

    using FlagsType = uint32_t;
    enum Flags : FlagsType {
      kNone = 0,
      kConstant = (1u << 0),
      kMulZero = (1u << 1),
      kDefaultConstantAccum = (1u << 2),
      kUserExternal = (1u << 3),
      kBeenOptimised = (1u << 4)
    };

    NodeType node_type = NodeType::kInvalid;
    FlagsType flags = kNone;
    uint32_t result_id = UINT32_MAX;
    FPConstAccum const_accum{};

    void AddInput(FPReassocNode* inp, int32_t num);
    void AccumConstant(FPReassocNode* inp, int32_t num);
    void ConvertAddsToMuls(ReassocGraphFP& parent);

    // Inputs to this node and their count
    std::unordered_map<FPReassocNode*, int32_t> inputs;
  };

  ReassocGraphFP(analysis::Type* type_, analysis::TypeManager* type_mgr_,
                 analysis::DefUseManager* def_use_mgr_,
                 analysis::ConstantManager* const_mgr_,
                 const ValueNumberTable* vn_table_)
      : type(type_),
        type_mgr(type_mgr_),
        def_use_mgr(def_use_mgr_),
        const_mgr(const_mgr_),
        vn_table(vn_table_) {

    uint32_t default_const_size = 1;
    if (analysis::Vector* vec_type = type->AsVector()) {
      is_vector = true;
      default_const_size = vec_type->element_count();
      if (const analysis::Float* float_type = vec_type->element_type()->AsFloat()) {
        width = float_type->width();
      }
    }
    else if (const analysis::Float* float_type = type->AsFloat()) {
      width = float_type->width();
    }
    assert((width == 32) || (width == 64));
    default_add_accum.vals.resize(default_const_size, 0.0);
    default_mul_accum.vals.resize(default_const_size, 1.0);
  }

  FPReassocNode* AllocateNode(FPReassocNode::NodeType node_type);
  FPReassocNode* CreateNode(Instruction* inst);
  FPReassocNode* FindOrCreateUserExternal(Instruction* inst);
  FPReassocNode* FindOrCreateUserExternal(uint32_t inst_id) {
    return FindOrCreateUserExternal(def_use_mgr->GetDef(inst_id));
  }
  FPReassocNode* GetUserExternal(Instruction* inst) {
    return user_external.at(inst);
  }

  analysis::Type* type;
  analysis::TypeManager* type_mgr;
  analysis::DefUseManager* def_use_mgr;
  analysis::ConstantManager* const_mgr;
  const ValueNumberTable* vn_table;

  bool is_vector = false;
  uint32_t width = 0;
  FPConstAccum default_add_accum;
  FPConstAccum default_mul_accum;

  std::unordered_map<Instruction*, FPReassocNode*> user_external;
  std::vector<std::unique_ptr<FPReassocNode>> storage;
};

void ReassocGraphFP::FPReassocNode::AccumConstant(FPReassocNode* inp,
                                                  int32_t num) {
  assert((node_type == NodeType::kAdd) || (node_type == NodeType::kMul));

  // Don't bother doing a * 1, or + 0
  if ((inp->node_type == node_type) && (inp->flags & kDefaultConstantAccum)) {
    return;
  }

  // If this is the first constant we've consumed,
  // we've probably not done any extra optimisations.
  // However, if it's the second, then we've folded
  // something.
  if (flags & kDefaultConstantAccum) {
    flags &= ~kDefaultConstantAccum;
  } else {
    flags |= kBeenOptimised;
  }

  if (node_type == NodeType::kMul) {
    if (num > 0) {
      for (int32_t i = 0; i < num; ++i) {
        const_accum *= inp->const_accum;
      }
    } else {
      for (int32_t i = 0; i > num; --i) {
        const_accum /= inp->const_accum;
      }
    }
    // Mul by zero, clear everything!
    if (const_accum.AllZero()) {
      flags |= kMulZero;
      if (!inputs.empty()) {
        flags |= kConstant | kBeenOptimised;
        inputs.clear();
      }
    }
  } else if (node_type == NodeType::kAdd) {
    if (num > 0) {
      for (int32_t i = 0; i < num; ++i) {
        const_accum += inp->const_accum;
      }
    } else {
      for (int32_t i = 0; i > num; --i) {
        const_accum -= inp->const_accum;
      }
    }
  }
}

void ReassocGraphFP::FPReassocNode::AddInput(FPReassocNode* inp, int32_t num) {
  assert((node_type == NodeType::kAdd) || (node_type == NodeType::kMul));
  flags |= (inp->flags & kBeenOptimised);

  if (num == 0) {
    return;
  }
  if (flags & kMulZero) {
    flags |= kBeenOptimised;
    return;
  }
  if ((inp->flags & kConstant)) {
    AccumConstant(inp, num);
    return;
  }
  if (inp->node_type == node_type) {
    AccumConstant(inp, num);
    for (auto kv : inp->inputs) {
      AddInput(kv.first, kv.second * num);
    }
  } else {
    auto found = inputs.find(inp);
    if (found != inputs.end()) {
      int32_t old_v = found->second;
      found->second += num;
      // Some amount of folding has taken place
      if ((old_v > 0) && (num < 0)) {
        flags |= kBeenOptimised;
      }
      // Totally cancelled out
      if (found->second == 0) {
        inputs.erase(found);
        if (inputs.empty()) {
          flags |= kConstant;
        }
      }
    } else {
      flags &= ~kConstant;
      inputs[inp] = num;
    }
  }
}

void ReassocGraphFP::FPReassocNode::ConvertAddsToMuls(ReassocGraphFP& parent) {
  assert(node_type != NodeType::kInvalid);
  if (node_type == NodeType::kExternal) {
    return;
  }

  bool could_use_folding = false;
  for (auto& inp : inputs) {
    inp.first->ConvertAddsToMuls(parent);
    flags |= (inp.first->flags & kBeenOptimised);
    if (node_type == NodeType::kAdd) {
      if ((inp.second <= -3) || (inp.second >= 3)) {
        could_use_folding = true;
      }
    }
  }

  if (could_use_folding) {
    flags |= kBeenOptimised;
    std::vector<FPReassocNode*> new_nodes;
    for (auto it = inputs.begin(); it != inputs.end();) {
      if ((it->second <= -3) || (it->second >= 3)) {
        FPReassocNode* new_node = parent.AllocateNode(NodeType::kMul);
        new_node->flags &= ~kDefaultConstantAccum;
        new_node->const_accum *= double(it->second);
        new_node->AddInput(it->first, 1);
        new_nodes.push_back(new_node);
        it = inputs.erase(it);
      }
      else {
        ++it;
      }
    }
    for (FPReassocNode* new_node : new_nodes) {
      AddInput(new_node, 1);
    }
  }
}

ReassocGraphFP::FPReassocNode* ReassocGraphFP::AllocateNode(
    FPReassocNode::NodeType node_type) {
  FPReassocNode* new_node = new FPReassocNode{};
  new_node->node_type = node_type;
  storage.emplace_back(new_node);
  assert(node_type != FPReassocNode::NodeType::kInvalid);
  if (node_type == FPReassocNode::NodeType::kAdd) {
    new_node->flags =
        FPReassocNode::kDefaultConstantAccum | FPReassocNode::kConstant;
    new_node->const_accum = default_add_accum;
  }
  if (node_type == FPReassocNode::NodeType::kMul) {
    new_node->flags =
        FPReassocNode::kDefaultConstantAccum | FPReassocNode::kConstant;
    new_node->const_accum = default_mul_accum;
  }
  return new_node;
}

ReassocGraphFP::FPReassocNode* ReassocGraphFP::CreateNode(Instruction* inst) {
  assert(user_external.find(inst) == user_external.end());

  FPReassocNode::NodeType node_type = FPReassocNode::NodeType::kInvalid;
  utils::SmallVector<std::pair<FPReassocNode*, int32_t>, 2> inputs;

  switch (inst->opcode()) {
    case spv::Op::OpFDiv: {
      node_type = FPReassocNode::NodeType::kMul;
      FPReassocNode* lhs =
          FindOrCreateUserExternal(inst->GetSingleWordInOperand(0));
      FPReassocNode* rhs =
          FindOrCreateUserExternal(inst->GetSingleWordInOperand(1));
      inputs.push_back({lhs, 1});
      inputs.push_back({rhs, -1});
      break;
    }
    case spv::Op::OpFMul: {
      node_type = FPReassocNode::NodeType::kMul;
      FPReassocNode* lhs =
          FindOrCreateUserExternal(inst->GetSingleWordInOperand(0));
      FPReassocNode* rhs =
          FindOrCreateUserExternal(inst->GetSingleWordInOperand(1));
      inputs.push_back({lhs, 1});
      inputs.push_back({rhs, 1});
      break;
    }
    case spv::Op::OpFSub: {
      node_type = FPReassocNode::NodeType::kAdd;
      FPReassocNode* lhs =
          FindOrCreateUserExternal(inst->GetSingleWordInOperand(0));
      FPReassocNode* rhs =
          FindOrCreateUserExternal(inst->GetSingleWordInOperand(1));
      inputs.push_back({lhs, 1});
      inputs.push_back({rhs, -1});
      break;
    }
    case spv::Op::OpFAdd: {
      node_type = FPReassocNode::NodeType::kAdd;
      FPReassocNode* lhs =
          FindOrCreateUserExternal(inst->GetSingleWordInOperand(0));
      FPReassocNode* rhs =
          FindOrCreateUserExternal(inst->GetSingleWordInOperand(1));
      inputs.push_back({lhs, 1});
      inputs.push_back({rhs, 1});
      break;
    }
    case spv::Op::OpFNegate: {
      node_type = FPReassocNode::NodeType::kAdd;
      FPReassocNode* lhs =
          FindOrCreateUserExternal(inst->GetSingleWordInOperand(0));
      inputs.push_back({lhs, -1});
      break;
    }
    default:
      assert(false);
      break;
  }

  FPReassocNode* new_node = AllocateNode(node_type);
  user_external[inst] = new_node;
  new_node->result_id = inst->result_id();
  new_node->flags |= FPReassocNode::kUserExternal;

  for (const auto& p : inputs) {
    new_node->AddInput(p.first, p.second);
  }

  return new_node;
}

ReassocGraphFP::FPReassocNode* ReassocGraphFP::FindOrCreateUserExternal(
    Instruction* inst) {
  auto found = user_external.find(inst);
  if (found != user_external.end()) {
    return found->second;
  }
  FPReassocNode* new_node = AllocateNode(FPReassocNode::NodeType::kExternal);
  user_external[inst] = new_node;
  new_node->result_id = inst->result_id();
  new_node->flags = FPReassocNode::kUserExternal;
  if (inst->IsConstant()) {
    // If anything seems off, ignore that this instruction was
    // reported as constant and pretend it isn't.
    if (type_mgr->GetType(inst->type_id()) == type) {
      new_node->const_accum.vals.resize(default_add_accum.vals.size(), 0.0);
      if (const analysis::Constant* c =
              const_mgr->FindDeclaredConstant(inst->result_id())) {
        if (c->AsNullConstant()) {
          new_node->flags |= FPReassocNode::kConstant;
        } else if (!is_vector) {
          if (const analysis::FloatConstant* fc = c->AsFloatConstant()) {
            if (width == 32) {
              new_node->flags |= FPReassocNode::kConstant;
              new_node->const_accum.vals[0] = (double)fc->GetFloat();
            } else {
              new_node->flags |= FPReassocNode::kConstant;
              new_node->const_accum.vals[0] = fc->GetDouble();
            }
          }
        }
      }
      if (is_vector) {
        bool ok = true;
        int32_t write_back = 0;
        inst->ForEachInId([&](uint32_t* id) {
          if (write_back >= default_add_accum.vals.size()) {
            return;
          }
          const analysis::Constant* const_op =
              const_mgr->FindDeclaredConstant(*id);
          if (!const_op) {
            ok = false;
            return;
          }
          if (const_op->AsNullConstant()) {
            ++write_back;
            return;
          }
          const analysis::FloatConstant* const_fp = const_op->AsFloatConstant();
          if (!const_fp) {
            ok = false;
            return;
          }
          if (width == 32) {
            new_node->const_accum.vals[write_back++] =
                (double)const_fp->GetFloat();
          } else {
            new_node->const_accum.vals[write_back++] = const_fp->GetDouble();
          }
        });
        if (ok) {
          new_node->flags |= FPReassocNode::kConstant;
        }
      }
    }
  }
  return new_node;
}

bool ReassociatePass::ReassociateFPGraph(Instruction* root,
                                         std::vector<Instruction*>&& graph) {
  analysis::Type* type = context()->get_type_mgr()->GetType(root->type_id());
  ReassocGraphFP fpgraph(
      type, context()->get_type_mgr(), context()->get_def_use_mgr(),
      context()->get_constant_mgr(), context()->GetValueNumberTable());

  for (Instruction* inst : graph) {
    fpgraph.CreateNode(inst);
  }

  ReassocGraphFP::FPReassocNode* root_node = fpgraph.GetUserExternal(root);
  root_node->ConvertAddsToMuls(fpgraph);
  // Run factorisation pass here

  bool modified = false;
  if ((root_node->flags & ReassocGraphFP::FPReassocNode::kBeenOptimised)) {
    modified = true;

  }
  return modified;
}

}  // namespace opt
}  // namespace spvtools
