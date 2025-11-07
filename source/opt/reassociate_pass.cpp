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
#include "scalar_analysis.h"
#include "source/util/hash_combine.h"

namespace spvtools {
namespace opt {
namespace reassociate {

template <class NodePtr>
struct IdLessThan {
  bool operator()(const NodePtr first, const NodePtr second) const {
    return first->id < second->id;
  }
};

struct FPConstAccum {
#define CONST_ACCUM_OP_EQ_VEC(op)                         \
  FPConstAccum& operator op(const FPConstAccum & other) { \
    assert(other.vals.size() == vals.size());             \
    size_t n = vals.size();                               \
    for (size_t i = 0; i < n; ++i) {                      \
      vals[i] op other.vals[i];                           \
    }                                                     \
    return *this;                                         \
  }

#define CONST_ACCUM_OP_EQ_SCALAR(op)        \
  FPConstAccum& operator op(double value) { \
    for (double& v : vals) {                \
      v op value;                           \
    }                                       \
    return *this;                           \
  }

  CONST_ACCUM_OP_EQ_VEC(+=)
  CONST_ACCUM_OP_EQ_VEC(-=)
  CONST_ACCUM_OP_EQ_VEC(*=)
  CONST_ACCUM_OP_EQ_VEC(/=)

  CONST_ACCUM_OP_EQ_SCALAR(+=)
  CONST_ACCUM_OP_EQ_SCALAR(-=)
  CONST_ACCUM_OP_EQ_SCALAR(*=)
  CONST_ACCUM_OP_EQ_SCALAR(/=)
  CONST_ACCUM_OP_EQ_SCALAR(=)

#undef CONST_ACCUM_OP_EQ_VEC
#undef CONST_ACCUM_OP_EQ_SCALAR

  FPConstAccum& operator=(const FPConstAccum& other) {
    vals = other.vals;
    return *this;
  }

  bool operator==(const FPConstAccum& other) const {
    assert(other.vals.size() == vals.size());
    size_t n = vals.size();
    for (size_t i = 0; i < n; ++i) {
      if (vals[i] != other.vals[i]) {
        return false;
      }
    }
    return true;
  }

  bool operator!=(const FPConstAccum& other) const { return !(*this == other); }

  bool operator==(double value) const {
    for (const double& v : vals) {
      if (v != value) return false;
    }
    return true;
  }
  bool operator!=(double value) const { return !(*this == value); }

  bool IsZero() const { return *this == 0.0; }
  bool IsOne() const { return *this == 1.0; }
  void SetToDefaultMul() { *this = 1.0; }
  void SetToDefaultAdd() { *this = 0.0; }
  bool IsDefaultMul() const { return IsOne(); }
  bool IsDefaultAdd() const { return IsZero(); }
  double& operator[](int32_t index) { return vals[index]; }
  const double& operator[](int32_t index) const { return vals[index]; }
  size_t size() const { return vals.size(); }

  struct Hash {
    size_t operator()(const FPConstAccum& accum) const {
      size_t hash = std::hash<size_t>{}(accum.vals.size());
      hash = utils::hash_combine(hash, accum.vals);
      return hash;
    }
  };

  std::vector<double> vals;
};

struct FPNode {
  using InputsType =
      std::map<const FPNode*, int32_t, IdLessThan<const FPNode*>>;

  enum NodeType {
    kInvalid,
    kExternal,
    kConstant,
    kAdd,  // add / sub
    kMul   // mul / div
  };

  struct InputsHasher {
    size_t operator()(const InputsType& inputs) const {
      size_t hash = std::hash<size_t>{}(inputs.size());
      for (const auto& inp : inputs) {
        hash = utils::hash_combine(hash, inp.first->id, inp.second);
      }
      return hash;
    }
  };

  struct Hash {
    size_t operator()(const FPNode& node) const {
      size_t consts = FPConstAccum::Hash{}(node.const_accum);
      size_t inputs = InputsHasher{}(node.inputs);
      size_t hash = utils::hash_combine(consts, inputs,
                                        static_cast<uint32_t>(node.node_type));
      // Purposefully ignoring the `result_id`, except for externals
      if (node.node_type == FPNode::kExternal) {
        hash = utils::hash_combine(hash, node.result_id);
      }
      return hash;
    }
  };

  bool operator==(const FPNode& other) const {
    if (node_type != other.node_type) {
      return false;
    }
    // Purposefully ignoring the `result_id`, except for externals
    if (node_type == FPNode::kExternal) {
      if (result_id != other.result_id) {
        return false;
      }
    }
    if (other.const_accum != const_accum) {
      return false;
    }
    return inputs == other.inputs;
  }

  // Add an input to either a kAdd or kMul.
  // If the input is a constant, it'll be consumed.
  // If the input is the same node type, it's inputs
  // and constant accumulation will be absorbed.
  void AddInput(const FPNode* inp, int32_t num);

  // Consume a constant accumulation.
  // Only valid for kAdd (+=), or kMul (*=);
  void ConsumeConstant(FPConstAccum other, int32_t num);

  // After inputs have finished being added:
  // * If the node_type is kMul and it's associated constant is 0,
  //   remove all inputs.
  // * Remove any inputs with a count of 0.
  // * If there are no nodes, convert the node_type to constant.
  // * If there is only one input, with a count of 1 and no extra
  //   constant, propagate its value.
  void SimplifyInputs();

  NodeType node_type = kInvalid;
  // External result id (kExternal)
  uint32_t result_id = UINT32_MAX;
  // Constants that have been accumulated (kAdd,kMul,kConstant)
  FPConstAccum const_accum{};
  // Id of this node wrt the ReassocGraphFP that stores it
  uint32_t id = UINT32_MAX;
  // Inputs to this node and their counts
  InputsType inputs;
};

void FPNode::AddInput(const FPNode* parent, int32_t num) {
  assert(node_type == kAdd || node_type == kMul);
  if (num == 0) {
    return;
  }

  if (parent->node_type == node_type || parent->node_type == kConstant) {
    ConsumeConstant(parent->const_accum, num);
  }

  if (parent->node_type == node_type) {
    for (const auto& parent_input : parent->inputs) {
      inputs.try_emplace(parent_input.first, 0).first->second +=
          parent_input.second * num;
    }
  } else if (parent->node_type == kConstant) {
    return;
  } else {
    inputs.emplace(parent, 0).first->second += num;
  }
}

void FPNode::ConsumeConstant(FPConstAccum other, int32_t num) {
  other *= num;
  switch (node_type) {
    case kAdd:
      const_accum += other;
      break;
    case kMul:
      const_accum *= other;
      break;
    default:
      assert(false);
      break;
  }
}

void FPNode::SimplifyInputs() {
  if (node_type != kAdd && node_type != kMul) {
    return;
  }
  if (node_type == kMul && const_accum.IsZero()) {
    inputs.clear();
  }
  for (auto it = inputs.begin(); it != inputs.end();) {
    if (it->second == 0) {
      it = inputs.erase(it);
      continue;
    }
    ++it;
  }
  if (inputs.empty()) {
    node_type = kConstant;
  } else if ((inputs.size() == 1) &&
             ((node_type == kAdd && const_accum.IsDefaultAdd()) ||
              (node_type == kMul && const_accum.IsDefaultMul()))) {
    const auto& single_child = *inputs.begin();
    if (single_child.second == 1) {
      *this = *single_child.first;
    }
  }
}

struct ReassocGraphFP {
  ReassocGraphFP(analysis::Type* type_,
                 analysis::DefUseManager* def_use_mgr_,
                 analysis::ConstantManager* const_mgr_);

  const FPNode* AddInstruction(Instruction* inst);
  const FPNode* FindInstruction(Instruction* inst);
  const FPNode* FindInstructionOrCreateExternal(Instruction* inst);
  const FPNode* AddNode(FPNode&& node);
  void PrintNode(std::ostream& output, const FPNode* node, int32_t indent = 0);

  // Helpers for constucting nodes
  FPNode Mul(const FPConstAccum& const_accum, const FPNode::InputsType& inputs) const;
  FPNode Mul(const FPNode::InputsType& inputs) const;
  FPNode Mul() const {
    FPNode new_desc;
    new_desc.node_type = FPNode::kMul;
    new_desc.const_accum = default_mul_accum();
    return new_desc;
  }
  FPNode Add(const FPConstAccum& const_accum, const FPNode::InputsType& inputs) const;
  FPNode Add(const FPNode::InputsType& inputs) const;
  FPNode Add() const {
    FPNode new_desc;
    new_desc.node_type = FPNode::kAdd;
    new_desc.const_accum = default_add_accum();
    return new_desc;
  }

  analysis::Type* type;
  analysis::DefUseManager* def_use_mgr;
  analysis::ConstantManager* const_mgr;

  bool is_vector = false;
  uint32_t width = 0;
  const FPConstAccum& default_add_accum() const { return default_zero_accum;}
  const FPConstAccum& default_mul_accum() const { return default_one_accum;}

  FPConstAccum default_zero_accum;
  FPConstAccum default_one_accum;
  
  std::unordered_map<Instruction*, const FPNode*> instr_to_node;
  std::unordered_set<FPNode, FPNode::Hash> storage;
};

ReassocGraphFP::ReassocGraphFP(analysis::Type* type_,
                               analysis::DefUseManager* def_use_mgr_,
                               analysis::ConstantManager* const_mgr_)
    : type(type_),
      def_use_mgr(def_use_mgr_),
      const_mgr(const_mgr_) {
  uint32_t default_const_size = 1;
  if (analysis::Vector* vec_type = type->AsVector()) {
    is_vector = true;
    default_const_size = vec_type->element_count();
    if (const analysis::Float* float_type =
            vec_type->element_type()->AsFloat()) {
      width = float_type->width();
    }
  } else if (const analysis::Float* float_type = type->AsFloat()) {
    width = float_type->width();
  }
  assert((width == 32) || (width == 64));
  default_zero_accum.vals.resize(default_const_size, 0.0);
  default_one_accum.vals.resize(default_const_size, 1.0);
}

const FPNode* ReassocGraphFP::AddNode(FPNode&& node) {
  assert(node.node_type != FPNode::kInvalid);
  node.id = storage.size();
  return &*storage.emplace(std::forward<FPNode>(node)).first;
}

const FPNode* ReassocGraphFP::FindInstruction(Instruction* inst) {
  auto found = instr_to_node.find(inst);
  if (found != instr_to_node.end()) {
    return found->second;
  }
  return nullptr;
}

const FPNode* ReassocGraphFP::FindInstructionOrCreateExternal(
    Instruction* inst) {
  if (const FPNode* found = FindInstruction(inst)) {
    return found;
  }

  FPNode new_node_desc;
  new_node_desc.node_type = FPNode::kExternal;
  new_node_desc.result_id = inst->result_id();

  if (inst->IsConstant()) {
    bool fetched_ok = false;
    FPConstAccum const_values = default_add_accum();

    const analysis::Constant* c =
        const_mgr->FindDeclaredConstant(inst->result_id());
    const analysis::Type* inst_type = c->type();

    if (is_vector && inst_type->AsVector()) {
      if (const analysis::VectorConstant* vc = c->AsVectorConstant()) {
        const std::vector<const analysis::Constant*>& components =
            vc->GetComponents();
        if (components.size() >= const_values.size()) {
          fetched_ok = true;
          for (size_t i = 0; i < const_values.size(); ++i) {
            const analysis::Constant* c2 = components[i];
            const analysis::Float* f = c2->type()->AsFloat();
            if (!f) {
              fetched_ok = false;
              break;
            }
            if (f->width() != 32 && f->width() != 64) {
              fetched_ok = false;
              break;
            }
            const_values[i] =
                (f->width() == 32) ? c->GetFloat() : c->GetDouble();
          }
        }
      }
    } else if (const analysis::Float* f = inst_type->AsFloat()) {
      if (f->width() == 32 || f->width() == 64) {
        fetched_ok = true;
        const_values = (f->width() == 32) ? c->GetFloat() : c->GetDouble();
      }
    }

    if (fetched_ok) {
      new_node_desc.node_type = FPNode::kConstant;
      new_node_desc.const_accum = std::move(const_values);
    }
  }

  const FPNode* new_node = AddNode(std::move(new_node_desc));
  instr_to_node[inst] = new_node;
  return new_node;
}

const FPNode* ReassocGraphFP::AddInstruction(Instruction* inst) {
  FPNode new_node_desc;
  new_node_desc.result_id = inst->result_id();

  auto ResolveInstArg = [&](uint32_t index) {
    return FindInstructionOrCreateExternal(
        def_use_mgr->GetDef(inst->GetSingleWordInOperand(index)));
  };
  switch (inst->opcode()) {
    case spv::Op::OpFDiv:
      new_node_desc.node_type = FPNode::kMul;
      new_node_desc.const_accum = default_mul_accum();
      new_node_desc.AddInput(ResolveInstArg(0), 1);
      new_node_desc.AddInput(ResolveInstArg(1), -1);
      break;
    case spv::Op::OpFMul:
      new_node_desc.node_type = FPNode::kMul;
      new_node_desc.const_accum = default_mul_accum();
      new_node_desc.AddInput(ResolveInstArg(0), 1);
      new_node_desc.AddInput(ResolveInstArg(1), 1);
      break;
    case spv::Op::OpFSub:
      new_node_desc.node_type = FPNode::kAdd;
      new_node_desc.const_accum = default_add_accum();
      new_node_desc.AddInput(ResolveInstArg(0), 1);
      new_node_desc.AddInput(ResolveInstArg(1), -1);
      break;
    case spv::Op::OpFAdd:
      new_node_desc.node_type = FPNode::kAdd;
      new_node_desc.const_accum = default_add_accum();
      new_node_desc.AddInput(ResolveInstArg(0), 1);
      new_node_desc.AddInput(ResolveInstArg(1), 1);
      break;
    case spv::Op::OpFNegate:
      new_node_desc.node_type = FPNode::kAdd;
      new_node_desc.const_accum = default_add_accum();
      new_node_desc.AddInput(ResolveInstArg(0), -1);
      break;
    default:
      assert(false);
      break;
  }
  new_node_desc.SimplifyInputs();
  const FPNode* new_node = AddNode(std::move(new_node_desc));
  instr_to_node[inst] = new_node;
  return new_node;
}

FPNode ReassocGraphFP::Mul(const FPConstAccum& const_accum, const FPNode::InputsType& inputs) const {
  FPNode new_desc;
  new_desc.node_type = FPNode::kMul;
  new_desc.const_accum = const_accum;
  new_desc.inputs = inputs;
  return new_desc;
}

FPNode ReassocGraphFP::Mul(const FPNode::InputsType& inputs) const {
  FPNode new_desc;
  new_desc.node_type = FPNode::kMul;
  new_desc.const_accum = default_mul_accum();
  new_desc.inputs = inputs;
  return new_desc;
}

FPNode ReassocGraphFP::Add(const FPConstAccum& const_accum, const FPNode::InputsType& inputs) const {
  FPNode new_desc;
  new_desc.node_type = FPNode::kAdd;
  new_desc.const_accum = const_accum;
  new_desc.inputs = inputs;
  return new_desc;
}

FPNode ReassocGraphFP::Add(const FPNode::InputsType& inputs) const {
  FPNode new_desc;
  new_desc.node_type = FPNode::kAdd;
  new_desc.const_accum = default_add_accum();
  new_desc.inputs = inputs;
  return new_desc;
}

void ReassocGraphFP::PrintNode(std::ostream& output, const FPNode* node,
                               int32_t indent) {
  std::string indentation(indent * 4, ' ');

  output << indentation;
  switch (node->node_type) {
    case FPNode::kInvalid:
      output << "[ invalid] {\n";
      break;
    case FPNode::kExternal:
      output << "[external] {\n";
      break;
    case FPNode::kConstant:
      output << "[constant] {\n";
      break;
    case FPNode::kAdd:
      output << "[   add  ] {\n";
      break;
    case FPNode::kMul:
      output << "[   mul  ] {\n";
      break;
    default:
      assert(false);
      break;
  }

  bool print_const_accum = false;
  switch (node->node_type) {
    case FPNode::kConstant:
      print_const_accum = true;
      break;
    case FPNode::kAdd:
      print_const_accum = !node->const_accum.IsDefaultAdd();
      break;
    case FPNode::kMul:
      print_const_accum = !node->const_accum.IsDefaultMul();
      break;
    default:
      break;
  }

  if (print_const_accum) {
    output << indentation << "    .const_accum {\n";
    for (double v : node->const_accum.vals) {
      output << indentation << "        " << v << ",\n";
    }
    output << indentation << "    }\n";
  }

  if (node->node_type == FPNode::kExternal) {
    output << indentation << "    .result_id = " << node->result_id << "\n";
  }

  if (!node->inputs.empty()) {
    output << indentation << "    .inputs {\n";
    int32_t indent_child = indent + 2;
    for (const auto& c : node->inputs) {
      if (c.second != 1) {
        output << indentation << "        x" << c.second << "x\n";
      }
      PrintNode(output, c.first, indent_child);
      output << ",\n";
    }
    output << indentation << "    }\n";
  }

  output << indentation << "}";
}

// Merge muls where are inputted to adds.
//
// This includes, wrapping inputs that have multiple counts with a mul.
//  ({ X, 2 }) => mul(.constant = 2, .inputs = ( {X, 1} ))
// 
// Allowing the following rules to take place:
//  (3 * a) + a                 => 4 * a
//  (3 * a) + (2 * a)           => 5 * a
//  (3 * a) + (3 * b) + (3 * c) => 3 * (a + b + c)
//  (3 * a) + (-3 * a)          => 0
//
//  TODO on the mul side:
//  (3 * (10 + 3 * (a + b)))  => 30 + 9 * (a + b)
void MergeAddMulInputs2(ReassocGraphFP& graph, FPNode& new_desc) {

  // Once again, something funky is going on here.

  if (new_desc.node_type != FPNode::kAdd) {
    return;
  }

  if (new_desc.inputs.size() < 2) {
    return;
  }

  const FPConstAccum& zero_accum = graph.default_zero_accum;

  using SharedInputs = std::unordered_map<FPNode::InputsType, FPConstAccum,
                                          FPNode::InputsHasher>;
  using SharedConstants =
      std::unordered_map<FPConstAccum, std::vector<FPNode::InputsType>,
                         FPConstAccum::Hash>;

  SharedInputs shared_inputs;
  SharedConstants shared_constants;

  // First pass, merge shared inputs
  for (const auto& input : std::move(new_desc.inputs)) {
    const FPNode* parent = input.first;
    const int32_t count = input.second;

    FPNode::InputsType parent_inputs;
    FPConstAccum addon = zero_accum;

    // Emulate a kMul node to catch things like:
    // 3*a + a => 4*a
    if (parent->node_type != FPNode::kMul) {
      parent_inputs[parent] = 1;
      addon += count;
    } else {
      parent_inputs = parent->inputs;
      addon = parent->const_accum;
      addon *= count;
    }
    shared_inputs.try_emplace(std::move(parent_inputs), zero_accum).first->second += addon;
  }

  // TODO: Try to merge things like:
  // add { mul( N, {add(A, B)} ), A, B } => mul(N+1, {Add(A, B)}

  // Second pass, merge shared constants
  for (const auto& input : shared_inputs) {
    auto& found = shared_constants.try_emplace(
        input.second, std::vector<FPNode::InputsType>{});
    found.first->second.push_back(input.first);
  }

  // Emit merged muls
  new_desc.inputs.clear();
  for (const auto& input : shared_constants) {
    FPNode new_mul;
    new_mul.node_type = FPNode::kMul;
    new_mul.const_accum = input.first;

    // Don't bother making an add chain
    if (input.second.size() == 1) {
      new_mul.inputs = input.second[0];
    }
    // Create add chain for nested muls
    // e.g: 3 * (a + b + c)
    else {
      FPNode new_add;
      new_add.node_type = FPNode::kAdd;
      new_add.const_accum = graph.default_add_accum();
      for (const auto& child_mul_inputs : input.second) {
        FPNode new_child_mul;
        new_child_mul.node_type = FPNode::kMul;
        new_child_mul.const_accum = graph.default_mul_accum();
        new_child_mul.inputs = child_mul_inputs;
        new_child_mul.SimplifyInputs();
        new_add.AddInput(graph.AddNode(std::move(new_child_mul)), 1);
      }
      new_add.SimplifyInputs();
      new_mul.AddInput(graph.AddNode(std::move(new_add)), 1);
    }
    new_mul.SimplifyInputs();
    new_desc.AddInput(graph.AddNode(std::move(new_mul)), 1);
  }
  new_desc.SimplifyInputs();
}


// Merge muls where are inputted to adds.
//
// This includes, wrapping inputs that have multiple counts with a mul.
//  ({ X, 2 }) => mul(.constant = 2, .inputs = ( {X, 1} ))
// 
// Allowing the following rules to take place:
//  (3 * a) + a                 => 4 * a
//  (3 * a) + (2 * a)           => 5 * a
//  (3 * a) + (-3 * a)          => 0
bool MergeAddMulInputs(ReassocGraphFP& graph, FPNode& new_desc) {

  if (new_desc.node_type != FPNode::kAdd) {
    return false;
  }

  if (new_desc.inputs.size() < 2) {
    return false;
  }

  using MergedInputs = std::unordered_map<FPNode::InputsType, FPConstAccum,
    FPNode::InputsHasher>;
  
  bool has_mergeable = false;
  MergedInputs merged_inputs;

  for(const auto& input : new_desc.inputs) {
    FPNode::InputsType mul_inputs;
    FPConstAccum accum = graph.default_zero_accum;

    // { X, 3 } => Mul(.C=3, { X, 1 })
    if (input.second != 1) {
      has_mergeable = true;
    }

    if (input.first->node_type == FPNode::kMul) {
      mul_inputs = input.first->inputs;
      accum = input.first->const_accum;
      accum *= input.second;
    }
    // Emulate mul, to catch things like 3*a + a => 4*a
    else {
      mul_inputs.emplace(input.first, input.second);
      accum += input.second;
    }
    auto found = merged_inputs.find(mul_inputs);
    if (found != merged_inputs.end()) {
      has_mergeable = true;
      found->second += accum;
    }
    else {
      merged_inputs.emplace(std::move(mul_inputs), std::move(accum));
    }
  }

  if (!has_mergeable) {
    return false;
  }

  new_desc.inputs.clear();
  for (auto& merged_input : merged_inputs) {
    FPNode new_mul = graph.Mul(merged_input.second, merged_input.first);
    new_mul.SimplifyInputs();
    new_desc.AddInput(graph.AddNode(std::move(new_mul)), 1);
  }
  return true;
}

// Merge muls with the same constant that are added together.
//
// Allowing the following rules to take place:
//  (3 * a) + (3 * b) + (3 * c) => 3 * (a + b + c)
bool MergeAddConstMulInputs(ReassocGraphFP& graph, FPNode& new_desc) {

  if (new_desc.node_type != FPNode::kAdd) {
    return false;
  }

  if (new_desc.inputs.size() < 2) {
    return false;
  }

  using MergedConstants =
    std::unordered_map<FPConstAccum, std::vector<FPNode::InputsType>,
    FPConstAccum::Hash>;

  bool has_mergeable = false;
  MergedConstants merged_constants;
  FPNode::InputsType extras;

  for (const auto& input : new_desc.inputs) {
    if (input.first->node_type == FPNode::kMul) {
      FPConstAccum accum = input.first->const_accum;
      accum *= input.second;
      auto found = merged_constants.find(accum);
      if (found != merged_constants.end()) {
        found->second.push_back(input.first->inputs);
        has_mergeable = true;
      }
      else {
        merged_constants.emplace(
          accum, std::vector<FPNode::InputsType>{input.first->inputs});
      }
    }
    else {
      extras.emplace(input.first, input.second);
    }
  }

  if (!has_mergeable) {
    return false;
  }

  new_desc.inputs.clear();
  for (const auto& extra : extras) {
    new_desc.AddInput(extra.first, extra.second);
  }

  for (const auto& merged : merged_constants) {
    FPNode new_mul = graph.Mul();
    new_mul.const_accum = merged.first; // MAKE HELPER

    for (const FPNode::InputsType& new_add_inputs : merged.second) {
      FPNode new_add = graph.Add(new_add_inputs);
      new_add.SimplifyInputs();
      new_mul.AddInput(graph.AddNode(std::move(new_add)), 1);
    }

    new_mul.SimplifyInputs();
    new_desc.AddInput(graph.AddNode(std::move(new_mul)), 1);
  }
  new_desc.SimplifyInputs();

  return true;
}


bool ApplyFPFoldingRules(ReassocGraphFP& graph, FPNode& new_desc) {
  if (new_desc.node_type == FPNode::kConstant || new_desc.node_type == FPNode::kExternal) {
    return false;
  }
  uint64_t prev_hash = FPNode::Hash{}(new_desc);
  MergeAddMulInputs(graph, new_desc);
  MergeAddConstMulInputs(graph, new_desc);
  //  TODO on the mul side:
  //  (3 * (10 + 3 * (a + b)))  => 30 + 9 * (a + b)

  return prev_hash != FPNode::Hash{}(new_desc);
}

static const FPNode* SimplifyFPNode(ReassocGraphFP& graph, const FPNode* node) {

  if (node->node_type == FPNode::kConstant ||
    node->node_type == FPNode::kExternal) {
    return node;
  }

  constexpr uint32_t max_iterations = 20;

  FPNode new_desc;
  new_desc.node_type = node->node_type;
  new_desc.result_id = node->result_id;
  new_desc.const_accum = node->const_accum;
  new_desc.inputs = node->inputs;

  // Simplify top level
  while (ApplyFPFoldingRules(graph, new_desc));

  // Simplify children
  FPNode::InputsType children = std::move(new_desc.inputs);
  new_desc.inputs.clear();
  for (const auto& input : children) {
    new_desc.AddInput(SimplifyFPNode(graph, input.first), input.second);
  }
  new_desc.SimplifyInputs();

  // Simplify top level
  while (ApplyFPFoldingRules(graph, new_desc));

  return graph.AddNode(std::move(new_desc));
}

}  // namespace reassociate

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

  analysis::Type* type = context()->get_type_mgr()->GetType(root->type_id());
  ReassocGraphFP fpgraph(type,
                         context()->get_def_use_mgr(),
                         context()->get_constant_mgr());

  for (Instruction* inst : graph) {
    fpgraph.AddInstruction(inst);
  }

  const FPNode* root_fp = fpgraph.FindInstruction(root);

  std::cout << "BEFORE OPT:\n";
  fpgraph.PrintNode(std::cout, root_fp);
  std::cout.flush();

   root_fp = SimplifyFPNode(fpgraph, root_fp);

   std::cout << "\n\n\nAFTER OPT:\n";
   fpgraph.PrintNode(std::cout, root_fp);
   std::cout.flush();

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
