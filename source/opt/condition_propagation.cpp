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
#include <vector>

namespace spvtools {
namespace opt {
namespace condprop {

class InstReplacements {
 public:
  InstReplacements() {}

  // If two rules directly contradict each other, that creates a paradox.
  // When this is the case, the branch we're acting on cannot be taken.
  bool HasParadox() const { return has_paradox; }
  void SetParadox() { has_paradox = true; }

  // Direct result_id replacement.
  // Any time |src_result| is used by an instruction, replace it with
  // |dst_result|.
  void AddDirectReplacement(uint32_t src_result, uint32_t dst_result) {
    // Currently not checking for if this is could create a paradox, since we
    // could end up with two things that are functionally the same, but differ
    // in result_id. e.g:
    //  OpIEquals(x, 0) && OpIEquals(x, OpNullConstant)
    if (src_result != dst_result) {
      direct.try_emplace(src_result, dst_result);
    }
  }

  // Direct result_id replacement.
  // Any time |src_result| is used by an instruction, it can be replaced by
  // the resulting bool, which may not have a result_id yet.
  void AddDirectBoolReplacement(uint32_t src_result, bool dst_result) {
    auto res = direct_bool.try_emplace(src_result, dst_result).first;
    if (res->second != dst_result) {
      SetParadox();
    }
  }

  // Replacement for unary operators.
  // Any time |op|(|arg|) is encountered, it can replaced with
  // the resulting bool, which may not have a result_id yet.
  void AddInstReplacement(spv::Op op, uint32_t arg, bool dst_result) {
    auto res = unary_bool_replacements[op].try_emplace(arg, dst_result).first;
    if (res->second != dst_result) {
      SetParadox();
    }
  }

  // Replacement for binary operators.
  // Any time |op|(|lhs|, |rhs|) is encountered, it can replaced with
  // the resulting bool, which may not have a result_id yet.
  // Returns *this, so they can be chained.
  void AddInstReplacement(spv::Op op, uint32_t lhs, uint32_t rhs,
                          bool dst_result) {
    uint64_t key = uint64_t(lhs) | (uint64_t(rhs) << 32);
    auto res = binary_bool_replacements[op].try_emplace(key, dst_result).first;
    if (res->second != dst_result) {
      SetParadox();
    }
  }

  // Attempt to get a replacement result_id for a given instruction.
  // If no replacement could be resolved, it will return 0.
  uint32_t GetReplacement(IRContext* context, uint32_t inst_id);

  uint32_t GetConstTrue(IRContext* context);
  uint32_t GetConstFalse(IRContext* context);

 private:
  bool has_paradox = false;

  std::unordered_map<uint32_t, uint32_t> direct;
  std::unordered_map<uint32_t, bool> direct_bool;
  std::unordered_map<spv::Op, std::unordered_map<uint32_t, bool>, OpHash>
      unary_bool_replacements;
  // Stored as |lhs| | (|rhs| << 32)
  std::unordered_map<spv::Op, std::unordered_map<uint64_t, bool>, OpHash>
      binary_bool_replacements;

  uint32_t true_const_id_ = 0u;
  uint32_t false_const_id_ = 0u;
};

uint32_t InstReplacements::GetConstTrue(IRContext* context) {
  if (true_const_id_) {
    return true_const_id_;
  }
  const analysis::Type* bool_type = context->get_type_mgr()->GetBoolType();
  analysis::ConstantManager* const_mgr = context->get_constant_mgr();
  true_const_id_ = const_mgr
                       ->GetDefiningInstruction(
                           const_mgr->GetConstant(bool_type, {uint32_t(true)}))
                       ->result_id();
  return true_const_id_;
}

uint32_t InstReplacements::GetConstFalse(IRContext* context) {
  if (false_const_id_) {
    return false_const_id_;
  }
  const analysis::Type* bool_type = context->get_type_mgr()->GetBoolType();
  analysis::ConstantManager* const_mgr = context->get_constant_mgr();
  false_const_id_ = const_mgr
                        ->GetDefiningInstruction(const_mgr->GetConstant(
                            bool_type, {uint32_t(false)}))
                        ->result_id();
  return false_const_id_;
}

uint32_t InstReplacements::GetReplacement(IRContext* context,
                                          uint32_t inst_id) {
  // result_id => result_id
  {
    auto found_direct = direct.find(inst_id);
    if (found_direct != direct.end()) {
      return found_direct->second;
    }
  }

  // result_id => constant bool
  {
    auto found_direct_bool = direct_bool.find(inst_id);
    if (found_direct_bool != direct_bool.end()) {
      uint32_t result = found_direct_bool->second ? GetConstTrue(context)
                                                  : GetConstFalse(context);
      direct[inst_id] = result;
      return result;
    }
  }

  Instruction* inst = context->get_def_use_mgr()->GetDef(inst_id);

  // binary => constant bool
  {
    auto binary_op_map_it = binary_bool_replacements.find(inst->opcode());
    if (binary_op_map_it != binary_bool_replacements.end()) {
      const auto& binary_map = binary_op_map_it->second;
      uint64_t key = uint64_t(inst->GetSingleWordInOperand(0)) |
                     (uint64_t(inst->GetSingleWordInOperand(1)) << 32);
      auto found_binary = binary_map.find(key);
      if (found_binary != binary_map.end()) {
        uint32_t result = found_binary->second ? GetConstTrue(context)
                                               : GetConstFalse(context);
        direct[inst_id] = result;
        return result;
      }
    }
  }

  // unary => constant bool
  {
    auto unary_op_map_it = unary_bool_replacements.find(inst->opcode());
    if (unary_op_map_it != unary_bool_replacements.end()) {
      const auto& unary_map = unary_op_map_it->second;
      auto found_unary = unary_map.find(inst->GetSingleWordInOperand(0));
      if (found_unary != unary_map.end()) {
        uint32_t result = found_unary->second ? GetConstTrue(context)
                                              : GetConstFalse(context);
        direct[inst_id] = result;
        return result;
      }
    }
  }

  direct[inst_id] = 0;
  return 0;
}

struct ExprQueueItem {
  spv::Op op{};
  uint32_t arg0{};
  uint32_t arg1{};
  bool result{};

  bool operator==(const ExprQueueItem& other) const {
    return (op == other.op) && (arg0 == other.arg0) && (arg1 == other.arg1) &&
           (result == other.result);
  }
  bool operator!=(const ExprQueueItem& other) const {
    return !(*this == other);
  }

  struct Hash {
    size_t operator()(const ExprQueueItem& item) const {
      const uint64_t p = 0x9e3779b97f4a7c15LLU;
      uint64_t h = 0xcbf29ce484222325LLU;
      if (item.result) {
        h = ~h;
      }
      h = (h ^ uint64_t(item.op)) * p;
      h = ((h >> 1) ^ uint64_t(item.arg0)) * p;
      h = ((h >> 1) ^ uint64_t(item.arg1)) * p;
      return size_t(h);
    }
  };
};

class ExprQueue {
 public:
  ExprQueue(IRContext* ctx, const Rules& rules_, InstReplacements& inst_repl_)
      : context_(ctx), rules(&rules_), inst_repl(&inst_repl_) {}

  const UnaryPropagationRuleMap& UnaryRules() const {
    return rules->unary_rules;
  }
  const BinaryPropagationRuleMap& BinaryRules() const {
    return rules->binary_rules;
  }
  const InstPropagationRuleMap& InstRules() const { return rules->inst_rules; }

  InstReplacements& Replacements() const { return *inst_repl; }

  // Process the next inference rule.
  // Returns false if there are nothing left queued,
  // or we encountered a paradox.
  bool ProcessNextRule();

  IRContext* context() const { return context_; }

  Instruction* GetInst(uint32_t result_id) const {
    return context()->get_def_use_mgr()->GetDef(result_id);
  }

  // Mark that all uses of |src|, should be replaced by |dst|,
  // which does not cause extra rules to be evaluated.
  ExprQueue& R(uint32_t src, uint32_t dst) {
    Replacements().AddDirectReplacement(src, dst);
    return *this;
  }

  // Mark that all uses of |src|, should be replaced by |dst|,
  // if |src| has an inference rule, it will queued for evaluation.
  ExprQueue& B(uint32_t src, bool dst);

  // Mark that expressions that match this signature should be
  // replaced by |dst|, it will also be queued for evaluation.
  ExprQueue& Q(spv::Op op, uint32_t arg, bool dst);
  ExprQueue& Q(spv::Op op, uint32_t lhs, uint32_t rhs, bool dst);

 private:
  IRContext* context_;
  const Rules* rules;
  InstReplacements* inst_repl;

  std::unordered_set<ExprQueueItem, ExprQueueItem::Hash> seen_expr;
  std::vector<ExprQueueItem> expr_queue;
};

bool ExprQueue::ProcessNextRule() {
  if (expr_queue.empty() || Replacements().HasParadox()) {
    return false;
  }
  ExprQueueItem item = expr_queue.back();
  expr_queue.pop_back();

  auto binary = BinaryRules().find(item.op);
  if (binary != BinaryRules().end()) {
    (binary->second)(*this, item.result, item.arg0, item.arg1);
    return true;
  }

  auto unary = UnaryRules().find(item.op);
  if (unary != UnaryRules().end()) {
    (unary->second)(*this, item.result, item.arg0);
    return true;
  }

  auto inst = InstRules().find(item.op);
  if (inst != InstRules().end()) {
    (inst->second)(*this, item.result, GetInst(item.arg0));
    return true;
  }

  return true;
}

ExprQueue& ExprQueue::B(uint32_t src, bool dst) {
  Replacements().AddDirectBoolReplacement(src, dst);
  Instruction* inst = context()->get_def_use_mgr()->GetDef(src);
  spv::Op op = inst->opcode();
  if (BinaryRules().find(op) != BinaryRules().end()) {
    return Q(op, inst->GetSingleWordInOperand(0),
             inst->GetSingleWordInOperand(1), dst);
  }
  if (UnaryRules().find(op) != UnaryRules().end()) {
    return Q(op, inst->GetSingleWordInOperand(0), dst);
  }
  if (InstRules().find(op) != InstRules().end()) {
    return Q(op, src, dst);
  }
  return *this;
}

ExprQueue& ExprQueue::Q(spv::Op op, uint32_t arg, bool dst) {
  Replacements().AddInstReplacement(op, arg, dst);
  ExprQueueItem expr_item{op, arg, 0, dst};
  if (seen_expr.emplace(expr_item).second) {
    expr_queue.push_back(expr_item);
  }
  return *this;
}

ExprQueue& ExprQueue::Q(spv::Op op, uint32_t lhs, uint32_t rhs, bool dst) {
  // Keep commutive ops ordered, but add two replacement rules.
  // This prevents us from having to do something similar when actually
  // performing the replacement.
  if (spvOpcodeIsCommutativeBinaryOperator(op)) {
    if (lhs > rhs) {
      std::swap(lhs, rhs);
    }
    Replacements().AddInstReplacement(op, rhs, lhs, dst);
  }
  Replacements().AddInstReplacement(op, lhs, rhs, dst);
  ExprQueueItem expr_item{op, lhs, rhs, dst};
  if (seen_expr.emplace(expr_item).second) {
    expr_queue.push_back(expr_item);
  }
  return *this;
}

void LogicalNot(ExprQueue& q, bool result, uint32_t arg) {
  q.B(arg, !result);  // OpLogicalNot(x) = true  | x = false
                      // OpLogicalNot(x) = false | x = true
}

void CopyObject(ExprQueue& q, bool result, uint32_t arg) { q.B(arg, result); }

void LogicalAnd(ExprQueue& q, bool result, uint32_t lhs, uint32_t rhs) {
  // OpLogicalAnd(x, y) = true | x = true, y = true
  if (result) {
    q.B(lhs, true);
    q.B(rhs, true);
  }
}

void LogicalOr(ExprQueue& q, bool result, uint32_t lhs, uint32_t rhs) {
  // OpLogicalOr(x, y) = false | x = false, y = false
  if (!result) {
    q.B(lhs, false);
    q.B(rhs, false);
  }
}

void LogicalEqual(ExprQueue& q, bool result, uint32_t lhs, uint32_t rhs) {
  // OpLogicalEqual(x, y) = true  | OpLogicalNotEqual(x, y) = false
  // OpLogicalEqual(x, y) = false | OpLogicalNotEqual(x, y) = true
  q.Q(spv::Op::OpLogicalNotEqual, lhs, rhs, !result);

  if (result) {
    if (q.GetInst(rhs)->IsConstant()) {
      q.R(lhs, rhs);
    }
    // TODO: (a == b) && (a == true), we should infer b == true
    else if (q.GetInst(lhs)->IsConstant()) {
      q.R(rhs, lhs);
    }
  }
}

void LogicalNotEqual(ExprQueue& q, bool result, uint32_t lhs, uint32_t rhs) {
  // OpLogicalNotEqual(x, y) = true  | OpLogicalEqual(x, y) = false
  // OpLogicalNotEqual(x, y) = false | OpLogicalEqual(x, y) = true
  q.Q(spv::Op::OpLogicalEqual, lhs, rhs, !result);
}

void IEqual(ExprQueue& q, bool result, uint32_t lhs, uint32_t rhs) {
  // OpIEqual(x, y) = true  | OpINotEqual(x, y) = false
  // OpIEqual(x, y) = false | OpINotEqual(x, y) = true
  q.Q(spv::Op::OpINotEqual, lhs, rhs, !result);

  // OpIEqual(x, y) = true  | OpUGreaterThan(x, y)      = false
  //                          OpSGreaterThan(x, y)      = false
  //                          OpUGreaterThanEqual(x, y) = true
  //                          OpSGreaterThanEqual(x, y) = true
  //                          OpULessThan(x, y)         = false
  //                          OpSLessThan(x, y)         = false
  //                          OpULessThanEqual(x, y)    = true
  //                          OpSLessThanEqual(x, y)    = true
  if (result) {
    static const constexpr spv::Op true_ops[] = {
        spv::Op::OpUGreaterThanEqual, spv::Op::OpSGreaterThanEqual,
        spv::Op::OpULessThanEqual, spv::Op::OpSLessThanEqual};
    for (const spv::Op op : true_ops) {
      q.Q(op, lhs, rhs, true).Q(op, rhs, lhs, true);
    }

    static const constexpr spv::Op false_ops[] = {
        spv::Op::OpUGreaterThan, spv::Op::OpSGreaterThan, spv::Op::OpULessThan,
        spv::Op::OpSLessThan};
    for (const spv::Op op : false_ops) {
      q.Q(op, lhs, rhs, false).Q(op, rhs, lhs, false);
    }
  }

  // TODO: (a == b) && (a == 3), we should infer b == 3
  if (result) {
    if (q.GetInst(rhs)->IsConstant()) {
      q.R(lhs, rhs);
    } else if (q.GetInst(lhs)->IsConstant()) {
      q.R(rhs, lhs);
    }
  }
}

void INotEqual(ExprQueue& q, bool result, uint32_t lhs, uint32_t rhs) {
  // OpINotEqual(x, y) = true  | OpIEqual(x, y) = false
  // OpINotEqual(x, y) = false | OpIEqual(x, y) = true
  q.Q(spv::Op::OpIEqual, lhs, rhs, !result);
}

void UGreaterThan(ExprQueue& q, bool result, uint32_t lhs, uint32_t rhs) {
  // OpUGreaterThan(x, y) = true  | OpULessThanEqual(x, y) = false
  // OpUGreaterThan(x, y) = false | OpULessThanEqual(x, y) = true
  q.Q(spv::Op::OpULessThanEqual, lhs, rhs, !result);

  // OpUGreaterThan(x, y) = true  | OpIEqual(x, y) = false
  //                              | OpULessThan(x, y) = false
  //                              | OpULessThanEqual(x, y) = false
  if (result) {
    q.Q(spv::Op::OpIEqual, lhs, rhs, false);
    q.Q(spv::Op::OpULessThan, lhs, rhs, false);
    q.Q(spv::Op::OpULessThanEqual, lhs, rhs, false);
  }
}

void SGreaterThan(ExprQueue& q, bool result, uint32_t lhs, uint32_t rhs) {
  // OpSGreaterThan(x, y) = true  | OpSLessThanEqual(x, y) = false
  // OpSGreaterThan(x, y) = false | OpSLessThanEqual(x, y) = true
  q.Q(spv::Op::OpSLessThanEqual, lhs, rhs, !result);

  // OpSGreaterThan(x, y) = true  | OpIEqual(x, y) = false
  //                              | OpSLessThan(x, y) = false
  //                              | OpSLessThanEqual(x, y) = false
  if (result) {
    q.Q(spv::Op::OpIEqual, lhs, rhs, false);
    q.Q(spv::Op::OpSLessThan, lhs, rhs, false);
    q.Q(spv::Op::OpSLessThanEqual, lhs, rhs, false);
  }
}

void UGreaterThanEqual(ExprQueue& q, bool result, uint32_t lhs, uint32_t rhs) {
  // OpUGreaterThanEqual(x, y) = true  | OpULessThan(x, y) = false
  // OpUGreaterThanEqual(x, y) = false | OpULessThan(x, y) = true
  q.Q(spv::Op::OpULessThan, lhs, rhs, !result);
}

void SGreaterThanEqual(ExprQueue& q, bool result, uint32_t lhs, uint32_t rhs) {
  // OpSGreaterThanEqual(x, y) = true  | OpSLessThan(x, y) = false
  // OpSGreaterThanEqual(x, y) = false | OpSLessThan(x, y) = true
  q.Q(spv::Op::OpSLessThan, lhs, rhs, !result);
}

void ULessThan(ExprQueue& q, bool result, uint32_t lhs, uint32_t rhs) {
  // OpULessThan(x, y) = true  | OpUGreaterThanEqual(x, y) = false
  // OpULessThan(x, y) = false | OpUGreaterThanEqual(x, y) = true
  q.Q(spv::Op::OpUGreaterThanEqual, lhs, rhs, !result);

  // OpULessThan(x, y) = true  | OpIEqual(x, y) = false
  //                           | OpUGreaterThan(x, y) = false
  //                           | OpUGreaterThanEqual(x, y) = false
  if (result) {
    q.Q(spv::Op::OpIEqual, lhs, rhs, false);
    q.Q(spv::Op::OpUGreaterThan, lhs, rhs, false);
    q.Q(spv::Op::OpUGreaterThanEqual, lhs, rhs, false);
  }
}

void SLessThan(ExprQueue& q, bool result, uint32_t lhs, uint32_t rhs) {
  // OpSLessThan(x, y) = true  | OpSGreaterThanEqual(x, y) = false
  // OpSLessThan(x, y) = false | OpSGreaterThanEqual(x, y) = true
  q.Q(spv::Op::OpSGreaterThanEqual, lhs, rhs, !result);

  // OpSLessThan(x, y) = true  | OpIEqual(x, y) = false
  //                           | OpSGreaterThan(x, y) = false
  //                           | OpSGreaterThanEqual(x, y) = false
  if (result) {
    q.Q(spv::Op::OpIEqual, lhs, rhs, false);
    q.Q(spv::Op::OpSGreaterThan, lhs, rhs, false);
    q.Q(spv::Op::OpSGreaterThanEqual, lhs, rhs, false);
  }
}

void ULessThanEqual(ExprQueue& q, bool result, uint32_t lhs, uint32_t rhs) {
  // OpULessThanEqual(x, y) = true  | OpUGreaterThan(x, y) = false
  // OpULessThanEqual(x, y) = false | OpUGreaterThan(x, y) = true
  q.Q(spv::Op::OpUGreaterThan, lhs, rhs, !result);
}

void SLessThanEqual(ExprQueue& q, bool result, uint32_t lhs, uint32_t rhs) {
  // OpSLessThanEqual(x, y) = true  | OpSGreaterThan(x, y) = false
  // OpSLessThanEqual(x, y) = false | OpSGreaterThan(x, y) = true
  q.Q(spv::Op::OpSGreaterThan, lhs, rhs, !result);
}

void FOrdEqual(ExprQueue& q, bool result, uint32_t lhs, uint32_t rhs) {
  // OpFOrdEqual(x, y) = true  | OpFUnordNotEqual(x, y) = false
  // OpFOrdEqual(x, y) = false | OpFUnordNotEqual(x, y) = true
  q.Q(spv::Op::OpFUnordNotEqual, lhs, rhs, !result);

  // OpFOrdEqual(x, y) = true  | OpFOrdNotEqual(x, y)           = false
  //                             OpFOrdLessThan(x, y)           = false
  //                             OpFUnordLessThan(x, y)         = false
  //                             OpFOrdGreaterThan(x, y)        = false
  //                             OpFUnordGreaterThan(x, y)      = false
  //                             OpFOrdLessThanEqual(x, y)      = true
  //                             OpFUnordLessThanEqual(x, y)    = true
  //                             OpFOrdGreaterThanEqual(x, y)   = true
  //                             OpFUnordGreaterThanEqual(x, y) = true
  if (result) {
    static const constexpr spv::Op true_ops[] = {
        spv::Op::OpFOrdLessThanEqual, spv::Op::OpFUnordLessThanEqual,
        spv::Op::OpFOrdGreaterThanEqual, spv::Op::OpFUnordGreaterThanEqual};
    for (const spv::Op op : true_ops) {
      q.Q(op, lhs, rhs, true).Q(op, rhs, lhs, true);
    }

    static const constexpr spv::Op false_ops[] = {
        spv::Op::OpFOrdNotEqual, spv::Op::OpFOrdLessThan,
        spv::Op::OpFUnordLessThan, spv::Op::OpFOrdGreaterThan,
        spv::Op::OpFUnordGreaterThan};
    for (const spv::Op op : false_ops) {
      q.Q(op, lhs, rhs, false).Q(op, rhs, lhs, false);
    }

    // We need to be careful if we want to propagate a constant.
    // For cases where there is a zero, we could be handling (0 == -0),
    // which would evaluate as true, but if replaced it could change the
    // result of instructions like OpBitcast.
    // So we're only propagating non-zero constants.
    Instruction* lhs_inst = q.GetInst(lhs);
    Instruction* rhs_inst = q.GetInst(rhs);
    if (lhs_inst->IsConstant()) {
      const analysis::Constant* c =
          q.context()->get_constant_mgr()->GetConstantFromInst(lhs_inst);
      if (!c->IsZero()) {
        q.R(rhs, lhs);
      }
    } else if (rhs_inst->IsConstant()) {
      const analysis::Constant* c =
          q.context()->get_constant_mgr()->GetConstantFromInst(rhs_inst);
      if (!c->IsZero()) {
        q.R(lhs, rhs);
      }
    }
  }
}

void FUnordEqual(ExprQueue& q, bool result, uint32_t lhs, uint32_t rhs) {
  // OpFUnordEqual(x, y) = true  | OpFOrdNotEqual(x, y) = false
  // OpFUnordEqual(x, y) = false | OpFOrdNotEqual(x, y) = true
  q.Q(spv::Op::OpFOrdNotEqual, lhs, rhs, !result);

  // OpFUnordEqual(x, y) = true  | OpFUnordNotEqual(x, y)         = false
  //                               OpFUnordLessThan(x, y)         = false
  //                               OpFUnordGreaterThan(x, y)      = false
  //                               OpFUnordLessThanEqual(x, y)    = true
  //                               OpFUnordGreaterThanEqual(x, y) = true
  if (result) {
    static const constexpr spv::Op true_ops[] = {
        spv::Op::OpFUnordLessThanEqual, spv::Op::OpFUnordGreaterThanEqual};
    for (const spv::Op op : true_ops) {
      q.Q(op, lhs, rhs, true).Q(op, rhs, lhs, true);
    }

    static const constexpr spv::Op false_ops[] = {
        spv::Op::OpFUnordNotEqual, spv::Op::OpFUnordLessThan,
        spv::Op::OpFUnordGreaterThan, spv::Op::OpFUnordGreaterThan};
    for (const spv::Op op : false_ops) {
      q.Q(op, lhs, rhs, false).Q(op, rhs, lhs, false);
    }

    // We can't really guarantee either side isn't going to be a NaN,
    // so we can't propagate the constant.
  }
}

void FUnordNotEqual(ExprQueue& q, bool result, uint32_t lhs, uint32_t rhs) {
  // OpFUnordNotEqual(x, y) = true  | OpFOrdEqual(x, y) = false
  // OpFUnordNotEqual(x, y) = false | OpFOrdEqual(x, y) = true
  q.Q(spv::Op::OpFOrdEqual, lhs, rhs, !result);
}

void FOrdNotEqual(ExprQueue& q, bool result, uint32_t lhs, uint32_t rhs) {
  // OpFOrdNotEqual(x, y) = true  | OpFUnordEqual(x, y) = false
  // OpFOrdNotEqual(x, y) = false | OpFUnordEqual(x, y) = true
  q.Q(spv::Op::OpFUnordEqual, lhs, rhs, !result);

  // OpFOrdNotEqual(x, y) = true  | OpFOrdEqual(x, y) = false
  if (result) {
    q.Q(spv::Op::OpFOrdEqual, lhs, rhs, false);
  }
}

void FOrdLessThan(ExprQueue& q, bool result, uint32_t lhs, uint32_t rhs) {
  // OpFOrdLessThan(x, y) = true  | OpFUnordGreaterThanEqual(x, y) = false
  // OpFOrdLessThan(x, y) = false | OpFUnordGreaterThanEqual(x, y) = true
  q.Q(spv::Op::OpFUnordGreaterThanEqual, lhs, rhs, !result);

  // OpFOrdLessThan(x, y) = true  | OpFOrdNotEqual(x, y) = true
  //                              | OpFOrdGreaterThan(x, y) = false
  //                              | OpFOrdGreaterThanEqual(x, y) = false
  if (result) {
    q.Q(spv::Op::OpFOrdNotEqual, lhs, rhs, true);
    q.Q(spv::Op::OpFOrdGreaterThan, lhs, rhs, false);
    q.Q(spv::Op::OpFOrdGreaterThanEqual, lhs, rhs, false);
  }
}

void FUnordLessThan(ExprQueue& q, bool result, uint32_t lhs, uint32_t rhs) {
  // OpFUnordLessThan(x, y) = true  | OpFOrdGreaterThanEqual(x, y) = false
  // OpFUnordLessThan(x, y) = false | OpFOrdGreaterThanEqual(x, y) = true
  q.Q(spv::Op::OpFOrdGreaterThanEqual, lhs, rhs, !result);
}

void FOrdGreaterThan(ExprQueue& q, bool result, uint32_t lhs, uint32_t rhs) {
  // OpFOrdGreaterThan(x, y) = true  | OpFUnordLessThanEqual(x, y) = false
  // OpFOrdGreaterThan(x, y) = false | OpFUnordLessThanEqual(x, y) = true
  q.Q(spv::Op::OpFUnordLessThanEqual, lhs, rhs, !result);

  // OpFOrdGreaterThan(x, y) = true  | OpFOrdNotEqual(x, y) = true
  //                                 | OpFOrdLessThan(x, y) = false
  //                                 | OpFOrdLessThanEqual(x, y) = false
  if (result) {
    q.Q(spv::Op::OpFOrdNotEqual, lhs, rhs, true);
    q.Q(spv::Op::OpFOrdLessThan, lhs, rhs, false);
    q.Q(spv::Op::OpFOrdLessThanEqual, lhs, rhs, false);
  }
}

void FUnordGreaterThan(ExprQueue& q, bool result, uint32_t lhs, uint32_t rhs) {
  // OpFUnordGreaterThan(x, y) = true  | OpFOrdLessThanEqual(x, y) = false
  // OpFUnordGreaterThan(x, y) = false | OpFOrdLessThanEqual(x, y) = true
  q.Q(spv::Op::OpFOrdLessThanEqual, lhs, rhs, !result);
}

void FOrdLessThanEqual(ExprQueue& q, bool result, uint32_t lhs, uint32_t rhs) {
  // OpFOrdLessThanEqual(x, y) = true  | OpFUnordGreaterThan(x, y) = false
  // OpFOrdLessThanEqual(x, y) = false | OpFUnordGreaterThan(x, y) = true
  q.Q(spv::Op::OpFUnordGreaterThan, lhs, rhs, !result);

  // OpFOrdLessThanEqual(x, y) = true  | OpFOrdGreaterThan(x, y) = false
  if (result) {
    q.Q(spv::Op::OpFOrdGreaterThan, lhs, rhs, false);
  }
}

void FUnordLessThanEqual(ExprQueue& q, bool result, uint32_t lhs,
                         uint32_t rhs) {
  // OpFUnordLessThanEqual(x, y) = true  | OpFOrdGreaterThan(x, y) = false
  // OpFUnordLessThanEqual(x, y) = false | OpFOrdGreaterThan(x, y) = true
  q.Q(spv::Op::OpFOrdGreaterThan, lhs, rhs, !result);
}

void FOrdGreaterThanEqual(ExprQueue& q, bool result, uint32_t lhs,
                          uint32_t rhs) {
  // OpFOrdGreaterThanEqual(x, y) = true  | OpFUnordLessThan(x, y) = false
  // OpFOrdGreaterThanEqual(x, y) = false | OpFUnordLessThan(x, y) = true
  q.Q(spv::Op::OpFUnordLessThan, lhs, rhs, !result);

  // OpFOrdGreaterThanEqual(x, y) = true  | OpFOrdLessThan(x, y) = false
  if (result) {
    q.Q(spv::Op::OpFOrdLessThan, lhs, rhs, false);
  }
}

void FUnordGreaterThanEqual(ExprQueue& q, bool result, uint32_t lhs,
                            uint32_t rhs) {
  // OpFUnordGreaterThanEqual(x, y) = true  | OpFOrdLessThan(x, y) = false
  // OpFUnordGreaterThanEqual(x, y) = false | OpFOrdLessThan(x, y) = true
  q.Q(spv::Op::OpFOrdLessThan, lhs, rhs, !result);
}

void Phi(ExprQueue& q, bool result, Instruction* inst) {
  assert(inst->opcode() == spv::Op::OpPhi);

  // If there is a single branch which doesn't directly result in a
  // paradox, then it must be the branch taken.
  // e.g:
  //    result = true | OpPhi %bool %false %branch_1 %value %branch_2
  //    %branch_2 is the only valid branch

  bool has_valid_branch = false;
  bool has_multiple_valid_branches = false;
  uint32_t singular_branch = 0;
  uint32_t singular_value = 0;

  uint32_t num_exprs = inst->NumInOperandWords() / 2;
  for (uint32_t i = 0; i < num_exprs; ++i) {
    uint32_t branch = inst->GetSingleWordInOperand(i * 2 + 1);
    uint32_t value = inst->GetSingleWordInOperand(i * 2);

    Instruction* val_inst = q.GetInst(value);
    while (val_inst->opcode() == spv::Op::OpCopyObject) {
      value = val_inst->GetSingleWordInOperand(0);
      val_inst = q.GetInst(value);
    }
    if (val_inst->IsConstant()) {
      const analysis::Constant* c =
          q.context()->get_constant_mgr()->GetConstantFromInst(val_inst);
      bool is_true = !c->IsZero();
      if (is_true != result) {
        continue;
      }
    }

    if (has_valid_branch) {
      has_multiple_valid_branches = true;
      break;
    }

    has_valid_branch = true;
    singular_branch = branch;
    singular_value = value;
  }

  if (!has_valid_branch) {
    q.Replacements().SetParadox();
    return;
  }

  if (has_multiple_valid_branches) {
    return;
  }

  const uint32_t this_branch =
      q.context()->get_instr_block(inst->result_id())->id();
  CFG* cfg = q.context()->cfg();

  // Walk up the chain for singular dependency branches,
  // if the branch is reached via a selection merge + conditional branch,
  // add the condition to the queue.
  auto ConsumeDependencyChain = [&q, cfg, this_branch](uint32_t branch,
                                                       bool result) {
    uint32_t child_branch = this_branch;
    while (true) {
      BasicBlock* blk = cfg->block(branch);
      if (!blk) {
        break;
      }
      Instruction* found_cnd = nullptr;
      Instruction* merge = blk->GetMergeInst();
      if (merge && merge->opcode() == spv::Op::OpSelectionMerge) {
        Instruction* cnd = merge->NextNode();
        if (cnd->opcode() == spv::Op::OpBranchConditional) {
          found_cnd = cnd;
        }
      }
      if (found_cnd) {
        if (found_cnd->GetSingleWordInOperand(1) == child_branch) {
          q.B(found_cnd->GetSingleWordInOperand(0), result);
        } else {
          assert(found_cnd->GetSingleWordInOperand(2) == child_branch);
          q.B(found_cnd->GetSingleWordInOperand(0), !result);
        }
      }
      const std::vector<uint32_t>& preds = cfg->preds(branch);
      if (preds.size() != 1) {
        break;
      }
      child_branch = branch;
      branch = preds[0];
    }
  };

  q.B(singular_value, result);
  ConsumeDependencyChain(singular_branch, result);
}

void Select(ExprQueue& q, bool result, Instruction* inst) {
  assert(inst->opcode() == spv::Op::OpSelect);

  // If one side of a select can't match the result, then
  // the other side must be taken.

  // OpSelect(expr, a, true)  == false | expr = true, a = false
  // OpSelect(expr, a, false) == true  | expr = true, a = true
  Instruction* false_side = q.GetInst(inst->GetSingleWordInOperand(2));
  if (false_side->IsConstant()) {
    const analysis::Constant* c =
        q.context()->get_constant_mgr()->GetConstantFromInst(false_side);
    bool matches = result == !c->IsZero();
    if (!matches) {
      q.B(inst->GetSingleWordInOperand(0), true);
      q.B(inst->GetSingleWordInOperand(1), result);
    }
  }

  // OpSelect(expr, true, b)  == false | expr = false, b = false
  // OpSelect(expr, false, b) == true  | expr = false, b = true
  Instruction* true_side = q.GetInst(inst->GetSingleWordInOperand(1));
  if (true_side->IsConstant()) {
    const analysis::Constant* c =
        q.context()->get_constant_mgr()->GetConstantFromInst(true_side);
    bool matches = result == !c->IsZero();
    if (!matches) {
      q.B(inst->GetSingleWordInOperand(0), false);
      q.B(inst->GetSingleWordInOperand(2), result);
    }
  }
}

}  // namespace condprop

ConditionPropagationPass::ConditionPropagationPass() {
  auto& unary_rules = rules.unary_rules;
  auto& binary_rules = rules.binary_rules;
  auto& inst_rules = rules.inst_rules;

  unary_rules[spv::Op::OpLogicalNot] = condprop::LogicalNot;
  unary_rules[spv::Op::OpCopyObject] = condprop::CopyObject;

  binary_rules[spv::Op::OpLogicalAnd] = condprop::LogicalAnd;
  binary_rules[spv::Op::OpLogicalOr] = condprop::LogicalOr;
  binary_rules[spv::Op::OpLogicalEqual] = condprop::LogicalEqual;
  binary_rules[spv::Op::OpLogicalNotEqual] = condprop::LogicalNotEqual;
  binary_rules[spv::Op::OpIEqual] = condprop::IEqual;
  binary_rules[spv::Op::OpINotEqual] = condprop::INotEqual;
  binary_rules[spv::Op::OpUGreaterThan] = condprop::UGreaterThan;
  binary_rules[spv::Op::OpSGreaterThan] = condprop::SGreaterThan;
  binary_rules[spv::Op::OpUGreaterThanEqual] = condprop::UGreaterThanEqual;
  binary_rules[spv::Op::OpSGreaterThanEqual] = condprop::SGreaterThanEqual;
  binary_rules[spv::Op::OpULessThan] = condprop::ULessThan;
  binary_rules[spv::Op::OpSLessThan] = condprop::SLessThan;
  binary_rules[spv::Op::OpULessThanEqual] = condprop::ULessThanEqual;
  binary_rules[spv::Op::OpSLessThanEqual] = condprop::SLessThanEqual;
  binary_rules[spv::Op::OpFOrdEqual] = condprop::FOrdEqual;
  binary_rules[spv::Op::OpFUnordEqual] = condprop::FUnordEqual;
  binary_rules[spv::Op::OpFUnordNotEqual] = condprop::FUnordNotEqual;
  binary_rules[spv::Op::OpFOrdNotEqual] = condprop::FOrdNotEqual;
  binary_rules[spv::Op::OpFOrdLessThan] = condprop::FOrdLessThan;
  binary_rules[spv::Op::OpFUnordLessThan] = condprop::FUnordLessThan;
  binary_rules[spv::Op::OpFOrdGreaterThan] = condprop::FOrdGreaterThan;
  binary_rules[spv::Op::OpFUnordGreaterThan] = condprop::FUnordGreaterThan;
  binary_rules[spv::Op::OpFOrdLessThanEqual] = condprop::FOrdLessThanEqual;
  binary_rules[spv::Op::OpFUnordLessThanEqual] = condprop::FUnordLessThanEqual;
  binary_rules[spv::Op::OpFOrdGreaterThanEqual] =
      condprop::FOrdGreaterThanEqual;
  binary_rules[spv::Op::OpFUnordGreaterThanEqual] =
      condprop::FUnordGreaterThanEqual;

  inst_rules[spv::Op::OpPhi] = condprop::Phi;
  inst_rules[spv::Op::OpSelect] = condprop::Select;

  auto CalcTotalUniqueRules = [&] {
    std::set<spv::Op> unique;
    for (const auto& r : unary_rules) {
      unique.insert(r.first);
    }
    for (const auto& r : binary_rules) {
      unique.insert(r.first);
    }
    for (const auto& r : inst_rules) {
      unique.insert(r.first);
    }
    return unique.size();
  };

  assert((unary_rules.size() + binary_rules.size() + inst_rules.size()) ==
             CalcTotalUniqueRules() &&
         "Op has multiple rules!");
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

  // True branch
  {
    condprop::InstReplacements inst_repl;
    condprop::ExprQueue expr_queue(context(), rules, inst_repl);
    expr_queue.B(inst->GetSingleWordInOperand(0), true);
    while (expr_queue.ProcessNextRule());
    if (inst_repl.HasParadox()) {
      uint32_t res = inst_repl.GetConstFalse(context());
      if (res) {
        inst->SetInOperand(0, {res});
      }
      return true;
    }

    if (uint32_t repl_id = inst_repl.GetReplacement(
            context(), inst->GetSingleWordInOperand(1))) {
      inst->SetInOperand(1, {repl_id});
      modified = true;
    }
  }

  // False branch
  {
    condprop::InstReplacements inst_repl;
    condprop::ExprQueue expr_queue(context(), rules, inst_repl);
    expr_queue.B(inst->GetSingleWordInOperand(0), false);
    while (expr_queue.ProcessNextRule());
    if (inst_repl.HasParadox()) {
      uint32_t res = inst_repl.GetConstTrue(context());
      if (res) {
        inst->SetInOperand(0, {res});
      }
      return true;
    }
    if (uint32_t repl_id = inst_repl.GetReplacement(
            context(), inst->GetSingleWordInOperand(2))) {
      inst->SetInOperand(2, {repl_id});
      modified = true;
    }
  }

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

  BasicBlock* true_bb =
      context()->get_instr_block(inst->GetSingleWordInOperand(1));
  BasicBlock* false_bb =
      context()->get_instr_block(inst->GetSingleWordInOperand(2));

  ConditionPropCollisionHelper collision_helper(
      context()->get_instr_block(inst), cfg());
  std::unordered_set<BasicBlock*> true_reachable =
      collision_helper.GetReachable(true_bb);
  std::unordered_set<BasicBlock*> false_reachable =
      collision_helper.GetReachable(false_bb);
  collision_helper.RemoveCollisions(true_reachable);
  collision_helper.RemoveCollisions(false_reachable);

  bool modified = false;

  if (!true_reachable.empty()) {
    condprop::InstReplacements inst_repl;
    condprop::ExprQueue expr_queue(context(), rules, inst_repl);
    expr_queue.B(inst->GetSingleWordInOperand(0), true);
    while (expr_queue.ProcessNextRule());
    if (inst_repl.HasParadox()) {
      uint32_t res = inst_repl.GetConstFalse(context());
      if (res) {
        inst->SetInOperand(0, {res});
      }
      return true;
    }

    if (ApplyReplacements(true_bb, true_reachable, inst_repl)) {
      modified = true;
    }
  }

  if (!false_reachable.empty()) {
    condprop::InstReplacements inst_repl;
    condprop::ExprQueue expr_queue(context(), rules, inst_repl);
    expr_queue.B(inst->GetSingleWordInOperand(0), false);
    while (expr_queue.ProcessNextRule());
    if (inst_repl.HasParadox()) {
      uint32_t res = inst_repl.GetConstTrue(context());
      if (res) {
        inst->SetInOperand(0, {res});
      }
      return true;
    }

    if (ApplyReplacements(false_bb, false_reachable, inst_repl)) {
      modified = true;
    }
  }

  return modified;
}

bool ConditionPropagationPass::ApplyReplacements(
    BasicBlock* root_bb, const std::unordered_set<BasicBlock*>& filtered_bb,
    condprop::InstReplacements& inst_repl) {
  bool modified = false;

  // Apply replacement rules in order, this allows us to build direct
  // replacements that can be used by successors.
  if (ApplyReplacementsToBB(root_bb, inst_repl)) {
    modified = true;
  }
  cfg()->ForEachBlockInPostOrder(root_bb, [&](BasicBlock* bb) {
    if (filtered_bb.find(bb) == filtered_bb.end()) {
      return;
    }
    if (ApplyReplacementsToBB(bb, inst_repl)) {
      modified = true;
    }
  });

  // Apply replacements to any OpPhi which is incoming from any
  // of the filtered bbs.
  // e.g:
  //    float b;
  //    if (a == 1.0) {
  //      b = a;
  //    }
  //    else {
  //      b = 100.0
  //    }
  //
  //    %b = OpPhi %float %a %true_bb %float_100 %false_bb
  //    =>
  //    %b = OpPhi %float %float_1 %true_bb %float_100 %false_bb
  for (BasicBlock* bb : filtered_bb) {
    Instruction* label = bb->GetLabelInst();
    if (!label) {
      continue;
    }
    get_def_use_mgr()->ForEachUse(
        label, [&](Instruction* phi_inst, uint32_t index) {
          if (phi_inst->opcode() != spv::Op::OpPhi) {
            return;
          }
          if ((index & 1) != 1) {
            return;
          }
          uint32_t var_op = phi_inst->GetOperand(index - 1).words[0];
          var_op = inst_repl.GetReplacement(context(), var_op);
          if (var_op) {
            phi_inst->GetOperand(index - 1).words[0] = var_op;
            modified = true;
          }
        });
  }

  return modified;
}

bool ConditionPropagationPass::ApplyReplacementsToBB(
    BasicBlock* bb, condprop::InstReplacements& inst_repl) {
  bool modified = false;

  for (Instruction& inst : *bb) {
    if (!inst.HasResultId()) {
      continue;
    }

    // First try to replace the whole instruction
    if (uint32_t top_id =
            inst_repl.GetReplacement(context(), inst.result_id())) {
      inst.SetOpcode(spv::Op::OpCopyObject);
      inst.SetInOperands({{SPV_OPERAND_TYPE_ID, {top_id}}});
      modified = true;
      continue;
    }

    // Try to replace their incoming ops.
    inst.ForEachInId([&](uint32_t* word) {
      if (uint32_t inner_id = inst_repl.GetReplacement(context(), *word)) {
        *word = inner_id;
        modified = true;
      }
    });
  }

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
