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

#include "source/opt/condition_propagation.h"

#include <algorithm>
#include <unordered_set>
#include <optional>
#include <vector>

namespace spvtools {
namespace opt {
namespace condprop {

// Adjust a comparison op so the lhs and rhs can be swapped.
// e.g:
//  %a = OpUGreaterThan %x %y
//  %b = OpULessThan %y %x
spv::Op SwapCmpSides(spv::Op a) {
  switch (a) {
  case spv::Op::OpUGreaterThan: return spv::Op::OpULessThan;
  case spv::Op::OpSGreaterThan: return spv::Op::OpSLessThan;
  case spv::Op::OpUGreaterThanEqual: return spv::Op::OpULessThanEqual;
  case spv::Op::OpSGreaterThanEqual: return spv::Op::OpSLessThanEqual;
  case spv::Op::OpULessThan: return spv::Op::OpUGreaterThan;
  case spv::Op::OpSLessThan: return spv::Op::OpSGreaterThan;
  case spv::Op::OpULessThanEqual: return spv::Op::OpUGreaterThanEqual;
  case spv::Op::OpSLessThanEqual: return spv::Op::OpSGreaterThanEqual;
  case spv::Op::OpFOrdLessThan: return spv::Op::OpFOrdGreaterThan;
  case spv::Op::OpFUnordLessThan: return spv::Op::OpFUnordGreaterThan;
  case spv::Op::OpFOrdGreaterThan: return spv::Op::OpFOrdLessThan;
  case spv::Op::OpFUnordGreaterThan: return spv::Op::OpFUnordLessThan;
  case spv::Op::OpFOrdLessThanEqual: return spv::Op::OpFOrdGreaterThanEqual;
  case spv::Op::OpFUnordLessThanEqual: return spv::Op::OpFUnordGreaterThanEqual;
  case spv::Op::OpFOrdGreaterThanEqual: return spv::Op::OpFOrdLessThanEqual;
  case spv::Op::OpFUnordGreaterThanEqual: return spv::Op::OpFOrdGreaterThanEqual;
  default:
    return a;
  }
}

// Merge integer compare operations which have the same inputs.
// e.g:
//   %a = OpIEqual %bool %x %y
//   %b = OpSLessThanEqual %bool %x %y
//  %12 = OpLogicalAnd %bool %a %b
constexpr uint32_t MergeICmpKey(spv::Op a, spv::Op b) {
  uint32_t a_shl = (uint32_t)a - (uint32_t)spv::Op::OpIEqual;
  uint32_t b_shl = (uint32_t)b - (uint32_t)spv::Op::OpIEqual;
  return (1u << a_shl) | (1u << b_shl);
}

std::optional<spv::Op> TryMergeICmp(spv::Op a, spv::Op b, bool& has_paradox) {
  assert((uint32_t)a >= (uint32_t)spv::Op::OpIEqual && (uint32_t)a <= (uint32_t)spv::Op::OpSLessThanEqual);
  assert((uint32_t)b >= (uint32_t)spv::Op::OpIEqual && (uint32_t)b <= (uint32_t)spv::Op::OpSLessThanEqual);

  if (a == b) { return a; }

  switch (MergeICmpKey(a, b)) {

  // (x <= y) && (x >= y) = (x == y)
  case MergeICmpKey(spv::Op::OpSGreaterThanEqual, spv::Op::OpSLessThanEqual):
  case MergeICmpKey(spv::Op::OpUGreaterThanEqual, spv::Op::OpULessThanEqual):
    return spv::Op::OpIEqual;

  // (x == y) && (x <= y) = (x == y)
  // (x == y) && (x >= y) = (x == y)
  case MergeICmpKey(spv::Op::OpIEqual, spv::Op::OpSLessThanEqual):
  case MergeICmpKey(spv::Op::OpIEqual, spv::Op::OpULessThanEqual):
  case MergeICmpKey(spv::Op::OpIEqual, spv::Op::OpSGreaterThanEqual):
  case MergeICmpKey(spv::Op::OpIEqual, spv::Op::OpUGreaterThanEqual):
    return spv::Op::OpIEqual;

  // (x < y)  && (x != y) = (x < y)
  // (x <= y) && (x != y) = (x < y)
  // (x < y)  && (x <= y) = (x < y)
  case MergeICmpKey(spv::Op::OpSLessThan, spv::Op::OpINotEqual):
  case MergeICmpKey(spv::Op::OpSLessThanEqual, spv::Op::OpINotEqual):
  case MergeICmpKey(spv::Op::OpSLessThan, spv::Op::OpSLessThanEqual):
    return spv::Op::OpSLessThan;
  case MergeICmpKey(spv::Op::OpULessThan, spv::Op::OpINotEqual):
  case MergeICmpKey(spv::Op::OpULessThanEqual, spv::Op::OpINotEqual):
  case MergeICmpKey(spv::Op::OpULessThan, spv::Op::OpULessThanEqual):
    return spv::Op::OpULessThan;

  // (x > y)  && (x != y) = (x > y)
  // (x >= y) && (x != y) = (x > y)
  // (x > y)  && (x >= y) = (x > y)
  case MergeICmpKey(spv::Op::OpSGreaterThan, spv::Op::OpINotEqual):
  case MergeICmpKey(spv::Op::OpSGreaterThanEqual, spv::Op::OpINotEqual):
  case MergeICmpKey(spv::Op::OpSGreaterThan, spv::Op::OpSGreaterThanEqual):
    return spv::Op::OpSGreaterThan;
  case MergeICmpKey(spv::Op::OpUGreaterThan, spv::Op::OpINotEqual):
  case MergeICmpKey(spv::Op::OpUGreaterThanEqual, spv::Op::OpINotEqual):
  case MergeICmpKey(spv::Op::OpUGreaterThan, spv::Op::OpUGreaterThanEqual):
    return spv::Op::OpUGreaterThan;

  // Conflicting compares create an paradox
  case MergeICmpKey(spv::Op::OpIEqual, spv::Op::OpINotEqual):
  case MergeICmpKey(spv::Op::OpIEqual, spv::Op::OpSGreaterThan):
  case MergeICmpKey(spv::Op::OpIEqual, spv::Op::OpUGreaterThan):
  case MergeICmpKey(spv::Op::OpIEqual, spv::Op::OpSLessThan):
  case MergeICmpKey(spv::Op::OpIEqual, spv::Op::OpULessThan):
  case MergeICmpKey(spv::Op::OpSGreaterThan, spv::Op::OpSLessThan):
  case MergeICmpKey(spv::Op::OpUGreaterThan, spv::Op::OpULessThan):
  case MergeICmpKey(spv::Op::OpSGreaterThan, spv::Op::OpSLessThanEqual):
  case MergeICmpKey(spv::Op::OpUGreaterThan, spv::Op::OpULessThanEqual):
  case MergeICmpKey(spv::Op::OpSGreaterThanEqual, spv::Op::OpSLessThan):
  case MergeICmpKey(spv::Op::OpUGreaterThanEqual, spv::Op::OpULessThan):
    has_paradox = true;
    break;

  default: break;
  }
  return {};
}


// Merge floating-point compare operations which have the same inputs.
// e.g:
//   %a = OpFOrdEqual %bool %x %y
//   %b = OpFOrdLessEqual %bool %x %y
//  %12 = OpLogicalAnd %bool %a %b
constexpr uint32_t MergeFCmpKey(spv::Op a, spv::Op b) {
  uint32_t a_shl = (uint32_t)a - (uint32_t)spv::Op::OpFOrdEqual;
  uint32_t b_shl = (uint32_t)b - (uint32_t)spv::Op::OpFOrdEqual;
  return (1u << a_shl) | (1u << b_shl);
}

std::optional<spv::Op> TryMergeFCmp(spv::Op a, spv::Op b, bool& has_paradox, bool& has_nan) {
  assert((uint32_t)a >= (uint32_t)spv::Op::OpFOrdEqual || (uint32_t)a <= (uint32_t)spv::Op::OpFUnordGreaterThanEqual);
  assert((uint32_t)b >= (uint32_t)spv::Op::OpFOrdEqual || (uint32_t)b <= (uint32_t)spv::Op::OpFUnordGreaterThanEqual);

  if (a == b) { return a; }

  switch (MergeFCmpKey(a, b)) {
  
  // (x ord_op y) && (x unord_op y) = (x ord_op y)
  case MergeFCmpKey(spv::Op::OpFOrdEqual, spv::Op::OpFUnordEqual): return spv::Op::OpFOrdEqual;
  case MergeFCmpKey(spv::Op::OpFOrdNotEqual, spv::Op::OpFUnordNotEqual): return spv::Op::OpFOrdNotEqual;
  case MergeFCmpKey(spv::Op::OpFOrdLessThan, spv::Op::OpFUnordLessThan): return spv::Op::OpFOrdLessThan;
  case MergeFCmpKey(spv::Op::OpFOrdGreaterThan, spv::Op::OpFUnordGreaterThan): return spv::Op::OpFOrdGreaterThan;
  case MergeFCmpKey(spv::Op::OpFOrdLessThanEqual, spv::Op::OpFUnordLessThanEqual): return spv::Op::OpFOrdLessThanEqual;
  case MergeFCmpKey(spv::Op::OpFOrdGreaterThanEqual, spv::Op::OpFUnordGreaterThanEqual): return spv::Op::OpFOrdLessThanEqual;
  
  // (x <= y) && (x >= y) = (x == y)
  case MergeFCmpKey(spv::Op::OpFOrdLessThanEqual, spv::Op::OpFOrdGreaterThanEqual):
  case MergeFCmpKey(spv::Op::OpFUnordLessThanEqual, spv::Op::OpFOrdGreaterThanEqual):
  case MergeFCmpKey(spv::Op::OpFOrdLessThanEqual, spv::Op::OpFUnordGreaterThanEqual): return spv::Op::OpFOrdEqual;
  case MergeFCmpKey(spv::Op::OpFUnordLessThanEqual, spv::Op::OpFUnordGreaterThanEqual): return spv::Op::OpFUnordEqual;

  // (x == y) && (x <= y) = (x == y)
  // (x == y) && (x >= y) = (x == y)
  case MergeFCmpKey(spv::Op::OpFOrdEqual, spv::Op::OpFOrdLessThanEqual):
  case MergeFCmpKey(spv::Op::OpFUnordEqual, spv::Op::OpFOrdLessThanEqual):
  case MergeFCmpKey(spv::Op::OpFOrdEqual, spv::Op::OpFUnordLessThanEqual):
  case MergeFCmpKey(spv::Op::OpFOrdEqual, spv::Op::OpFOrdGreaterThanEqual):
  case MergeFCmpKey(spv::Op::OpFUnordEqual, spv::Op::OpFOrdGreaterThanEqual):
  case MergeFCmpKey(spv::Op::OpFOrdEqual, spv::Op::OpFUnordGreaterThanEqual): return spv::Op::OpFOrdEqual;
  case MergeFCmpKey(spv::Op::OpFUnordEqual, spv::Op::OpFUnordLessThanEqual):
  case MergeFCmpKey(spv::Op::OpFUnordEqual, spv::Op::OpFUnordGreaterThanEqual): return spv::Op::OpFUnordEqual;
  
  // (x < y)  && (x != y) = (x < y)
  // (x <= y) && (x != y) = (x < y)
  // (x < y)  && (x <= y) = (x < y)
  case MergeFCmpKey(spv::Op::OpFOrdLessThan, spv::Op::OpFOrdNotEqual):
  case MergeFCmpKey(spv::Op::OpFUnordLessThan, spv::Op::OpFOrdNotEqual):
  case MergeFCmpKey(spv::Op::OpFOrdLessThan, spv::Op::OpFUnordNotEqual):
  case MergeFCmpKey(spv::Op::OpFOrdLessThanEqual, spv::Op::OpFOrdNotEqual):
  case MergeFCmpKey(spv::Op::OpFUnordLessThanEqual, spv::Op::OpFOrdNotEqual):
  case MergeFCmpKey(spv::Op::OpFOrdLessThanEqual, spv::Op::OpFUnordNotEqual):
  case MergeFCmpKey(spv::Op::OpFOrdLessThan, spv::Op::OpFOrdLessThanEqual):
  case MergeFCmpKey(spv::Op::OpFUnordLessThan, spv::Op::OpFOrdLessThanEqual):
  case MergeFCmpKey(spv::Op::OpFOrdLessThan, spv::Op::OpFUnordLessThanEqual): return spv::Op::OpFOrdLessThan;
  case MergeFCmpKey(spv::Op::OpFUnordLessThan, spv::Op::OpFUnordNotEqual):
  case MergeFCmpKey(spv::Op::OpFUnordLessThanEqual, spv::Op::OpFUnordNotEqual):
  case MergeFCmpKey(spv::Op::OpFUnordLessThan, spv::Op::OpFUnordLessThanEqual): return spv::Op::OpFUnordLessThan;

  // (x > y)  && (x != y) = (x > y)
  // (x >= y) && (x != y) = (x > y)
  // (x > y)  && (x >= y) = (x > y)
  case MergeFCmpKey(spv::Op::OpFOrdGreaterThan, spv::Op::OpFOrdNotEqual):
  case MergeFCmpKey(spv::Op::OpFUnordGreaterThan, spv::Op::OpFOrdNotEqual):
  case MergeFCmpKey(spv::Op::OpFOrdGreaterThan, spv::Op::OpFUnordNotEqual):
  case MergeFCmpKey(spv::Op::OpFOrdGreaterThanEqual, spv::Op::OpFOrdNotEqual):
  case MergeFCmpKey(spv::Op::OpFUnordGreaterThanEqual, spv::Op::OpFOrdNotEqual):
  case MergeFCmpKey(spv::Op::OpFOrdGreaterThanEqual, spv::Op::OpFUnordNotEqual):
  case MergeFCmpKey(spv::Op::OpFOrdGreaterThan, spv::Op::OpFOrdGreaterThanEqual):
  case MergeFCmpKey(spv::Op::OpFUnordGreaterThan, spv::Op::OpFOrdGreaterThanEqual):
  case MergeFCmpKey(spv::Op::OpFOrdGreaterThan, spv::Op::OpFUnordGreaterThanEqual): return spv::Op::OpFOrdGreaterThan;
  case MergeFCmpKey(spv::Op::OpFUnordGreaterThan, spv::Op::OpFUnordNotEqual):
  case MergeFCmpKey(spv::Op::OpFUnordGreaterThanEqual, spv::Op::OpFUnordNotEqual):
  case MergeFCmpKey(spv::Op::OpFUnordGreaterThan, spv::Op::OpFUnordGreaterThanEqual): return spv::Op::OpFUnordGreaterThan;
  
  // Ordered conflicting compares create an paradox
  case MergeFCmpKey(spv::Op::OpFOrdEqual, spv::Op::OpFOrdNotEqual):
  case MergeFCmpKey(spv::Op::OpFOrdEqual, spv::Op::OpFUnordNotEqual):
  case MergeFCmpKey(spv::Op::OpFUnordEqual, spv::Op::OpFOrdNotEqual):
  case MergeFCmpKey(spv::Op::OpFOrdEqual, spv::Op::OpFOrdGreaterThan):
  case MergeFCmpKey(spv::Op::OpFUnordEqual, spv::Op::OpFOrdGreaterThan):
  case MergeFCmpKey(spv::Op::OpFOrdEqual, spv::Op::OpFUnordGreaterThan):
  case MergeFCmpKey(spv::Op::OpFOrdEqual, spv::Op::OpFOrdLessThan):
  case MergeFCmpKey(spv::Op::OpFUnordEqual, spv::Op::OpFOrdLessThan):
  case MergeFCmpKey(spv::Op::OpFOrdEqual, spv::Op::OpFUnordLessThan):
  case MergeFCmpKey(spv::Op::OpFOrdLessThan, spv::Op::OpFOrdGreaterThan):
  case MergeFCmpKey(spv::Op::OpFOrdLessThan, spv::Op::OpFUnordGreaterThan):
  case MergeFCmpKey(spv::Op::OpFUnordLessThan, spv::Op::OpFOrdGreaterThan):
  case MergeFCmpKey(spv::Op::OpFOrdLessThan, spv::Op::OpFOrdGreaterThanEqual):
  case MergeFCmpKey(spv::Op::OpFOrdLessThan, spv::Op::OpFUnordGreaterThanEqual):
  case MergeFCmpKey(spv::Op::OpFUnordLessThan, spv::Op::OpFOrdGreaterThanEqual):
  case MergeFCmpKey(spv::Op::OpFOrdGreaterThan, spv::Op::OpFOrdLessThanEqual):
  case MergeFCmpKey(spv::Op::OpFOrdGreaterThan, spv::Op::OpFUnordLessThanEqual):
  case MergeFCmpKey(spv::Op::OpFUnordGreaterThan, spv::Op::OpFOrdLessThanEqual):
    has_paradox = true;
    break;

  // Purely unordered conflicting compares mean there is a nan
  case MergeFCmpKey(spv::Op::OpFUnordEqual, spv::Op::OpFUnordNotEqual):
  case MergeFCmpKey(spv::Op::OpFUnordEqual, spv::Op::OpFUnordLessThan):
  case MergeFCmpKey(spv::Op::OpFUnordEqual, spv::Op::OpFUnordGreaterThan):
  case MergeFCmpKey(spv::Op::OpFUnordLessThan, spv::Op::OpFUnordGreaterThan):
  case MergeFCmpKey(spv::Op::OpFUnordLessThan, spv::Op::OpFUnordGreaterThanEqual):
  case MergeFCmpKey(spv::Op::OpFUnordGreaterThan, spv::Op::OpFUnordLessThanEqual):
    has_nan = true;
    break;

  default: break;
  }
  return {};
}

struct InstInfo {

  using FlagBits = uint32_t;
  enum Flags : FlagBits {
    kNoFlags = 0,
    // Type info
    kIsBool = 1 << 0,
    kIsInt = 1 << 1,
    kIsFloat = 1 << 2,
    // Bool
    kIsTrue = 1 << 3,
    kIsFalse = 1 << 4,
    // FP
    kIsNan = 1 << 5,
    kIsNonNan = 1 << 6,
    kIsZero = kIsFalse
  };

  bool IsBool() const {
    return flags & kIsBool;
  }
  bool IsFalse() const {
    return flags & kIsFalse;
  }
  bool IsTrue() const {
    return flags & kIsTrue;
  }

  bool IsInt() const {
    return flags & kIsInt;
  }

  bool IsFloat() const {
    return flags & kIsFloat;
  }
  bool IsNan() const {
    return flags & kIsNan;
  }
  bool IsNonNan() const {
    return flags & kIsNonNan;
  }
  bool IsZero() const {
    return flags & kIsZero;
  }

  bool HasBoolParadox() const {
    return (flags & (kIsBool | kIsTrue | kIsFalse)) == (kIsBool | kIsTrue | kIsFalse);
  }

  void AddICmp(spv::Op op, uint32_t rhs, bool& has_paradox);
  void AddFCmp(spv::Op op, uint32_t rhs, bool& has_paradox, bool& has_nan);

  FlagBits flags = kNoFlags;

  // Relationship to other instructions.
  // These are stored with lhs = this, rhs = other.
  // e.g:
  //  %true = OpFLessThan %bool %this %a
  //  %true = OpFGreaterEqualThan %bool %b %this
  //    => { %a : [OpFLessThan], %b: [OpFLessEqualThan] }
  std::unordered_map<uint32_t, utils::SmallVector<spv::Op, 1>> relations;
};

class ConditionedState {
public:
  ConditionedState() = default;
  ~ConditionedState() = default;

  void AddKnownBool(uint32_t result_id, bool result);

private:
  bool has_paradox = false;
  std::unordered_map<uint32_t, InstInfo> inst_infos;
};


void ConditionedState::AddKnownBool(uint32_t result_id, bool result) {
  InstInfo& info = inst_infos[result_id];
  info.flags |= InstInfo::kIsBool | (result ? InstInfo::kIsTrue : InstInfo::kIsFalse);
  if (info.HasBoolParadox()) {
    has_paradox = true;
  }
}

}  // namespace condprop

ConditionPropagationPass::ConditionPropagationPass() {
}

Pass::Status ConditionPropagationPass::Process() {
  bool modified = false;

  for (Function& function : *get_module()) {
    modified |= ProcessFunction(&function);
  }
  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

bool ConditionPropagationPass::ProcessFunction(Function* function) {
  bool modified = false;

  std::vector<Instruction*> switch_insts;

  function->ForEachInst([&](Instruction* inst) {
    spv::Op op = inst->opcode();
    switch (op) {
      case spv::Op::OpSelect:
        modified |= ProcessSelect(inst);
        break;
      case spv::Op::OpSelectionMerge:
        modified |= ProcessSelectionMerge(inst);
        break;
      default:
        break;
    }
  });
  return modified;
}

// Helper for tracking basic blocks which can't be modified, as
// they can be reached by multiple conflicting conditions.
// e.g:
// if (expr) { ... }  <-- BB #1
// ...                <-- BB #2, but not in an else brace
struct ConditionPropCollisionHelper {
  ConditionPropCollisionHelper(BasicBlock* root, CFG* cfg_) : cfg(cfg_) {
    // Preload all incoming bbs to the collision set.
    // This stops us from incorrectly stomping things when there
    // is a recursive pattern.
    //
    // e.g, (%l1 would end up incorrectly stomping %a):
    //    ...
    //    %l1 = OpLabel
    //    %a = OpUGreaterThan %V %uint_100
    //    OpBranch %l2
    //
    //    %l2 = OpLabel
    //    ...
    //    OpBranchConditional %K %l1 %l99
    if (root) {
      std::unordered_set<uint32_t> seen{root->id()};
      std::vector<uint32_t> queue{root->id()};
      while (!queue.empty()) {
        uint32_t id = queue.back();
        queue.pop_back();
        for (uint32_t v : cfg->preds(id)) {
          if (seen.find(v) == seen.end()) {
            seen.insert(v);
            queue.push_back(v);
          }
        }
      }
      for (uint32_t id : seen) {
        collisions.insert(cfg->block(id));
      }
    }
  }

  std::unordered_set<BasicBlock*> GetReachable(BasicBlock* root_bb) {
    std::unordered_set<BasicBlock*> result{root_bb};
    cfg->ForEachBlockInPostOrder(
        root_bb, [&result](BasicBlock* bb) { result.insert(bb); });
    auto seen_end = seen_blocks.end();
    for (BasicBlock* bb : result) {
      if (seen_blocks.find(bb) != seen_end) {
        collisions.insert(bb);
      }
    }
    seen_blocks.insert(result.begin(), result.end());
    return result;
  }

  void RemoveCollisions(std::unordered_set<BasicBlock*>& reachable) const {
    const auto collisions_end = collisions.end();
    for (auto it = reachable.begin(); it != reachable.end();) {
      if (collisions.find(*it) != collisions_end) {
        it = reachable.erase(it);
      } else {
        ++it;
      }
    }
  }

  CFG* cfg;
  std::unordered_set<BasicBlock*> seen_blocks;
  std::unordered_set<BasicBlock*> collisions;
};

bool ConditionPropagationPass::ProcessSelect(Instruction* inst) {
  assert(inst->opcode() == spv::Op::OpSelect);
  bool modified = false;
  return modified;
}

bool ConditionPropagationPass::ProcessSelectionMerge(Instruction* inst) {
  inst = inst->NextNode();
  switch (inst->opcode()) {
    case spv::Op::OpBranchConditional:
      return ProcessConditional(inst);
    case spv::Op::OpSwitch:
      return ProcessSwitch(inst);
    default:
      return false;
  }
}

bool ConditionPropagationPass::ProcessConditional(Instruction* inst) {
  assert(inst->opcode() == spv::Op::OpBranchConditional);

  bool modified = false;
  return modified;
}


bool ConditionPropagationPass::ProcessSwitch(Instruction* inst) {
  assert(inst->opcode() == spv::Op::OpSwitch);

  // This is just going to be a direct replacement, but because it's a literal,
  // we sort of need to only allocate the constant instruction equivilant, when
  // we know it's actually going to be used.
  // No condition logic etc.
  return false;
}

}  // namespace opt
}  // namespace spvtools
