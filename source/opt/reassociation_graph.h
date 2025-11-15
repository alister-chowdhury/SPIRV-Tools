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

#ifndef SOURCE_OPT_REASSOCIATION_GRAPH_H_
#define SOURCE_OPT_REASSOCIATION_GRAPH_H_

#include <iostream>
#include <map>
#include <variant>
#include <vector>

#include "source/util/hash_combine.h"

namespace spvtools {
namespace opt {

class Instruction;

namespace analysis {
class Type;
class DefUseManager;
class ConstantManager;
}  // namespace analysis

namespace reassociate {

template <class NodePtr>
struct IdLessThan {
  bool operator()(const NodePtr first, const NodePtr second) const {
    return first->id < second->id;
  }
};

// Helper which acts as a container of constant values for
// floating point formats.
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

  bool operator==(double value) const {
    for (const double& v : vals) {
      if (v != value) return false;
    }
    return true;
  }

  bool operator!=(const FPConstAccum& other) const { return !(*this == other); }
  bool operator!=(double value) const { return !(*this == value); }

  bool IsZero() const { return *this == 0.0; }
  bool IsOne() const { return *this == 1.0; }
  bool IsMinusOne() const { return *this == -1.0; }

  bool IsDefaultMul() const { return IsOne(); }
  void SetToDefaultMul() { *this = 1.0; }

  bool IsDefaultAdd() const { return IsZero(); }
  void SetToDefaultAdd() { *this = 0.0; }

  void Negate() {
    for (double& v : vals) {
      v = -v;
    }
  }

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

// Node for floating-point operations.
//
// Can represent:
// * A external instruction
// * A constant value
// * A series of add / sub operations
// * A series of mul / div operations
struct FPNode {
  enum NodeType {
    kInvalid,
    kExternal,
    kConstant,
    kAdd,  // add / sub
    kMul   // mul / div
  };

  // The inputs are stored as a map using the internal id assigned to it
  // by the graph, so they are consistent between compilers and operating
  // systems.
  // (Compared to using an unordered_map and directly using the pointer)
  using InputsType =
      std::map<const FPNode*, int32_t, IdLessThan<const FPNode*>>;

  struct InputsHash {
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
      size_t inputs = InputsHash{}(node.inputs);
      size_t hash = utils::hash_combine(consts, inputs,
                                        static_cast<uint32_t>(node.node_type));
      // Purposefully ignoring the `result_id`, except for externals
      if (node.node_type == FPNode::kExternal) {
        hash = utils::hash_combine(hash, node.result_id);
      }
      return hash;
    }
  };

  bool operator==(const FPNode& other) const;
  bool operator!=(const FPNode& other) const { return !(*this == other); }

  // Add an input to either a kAdd or kMul.
  // * If the input is a constant, it'll be consumed.
  // * If the input is the same node type, it's inputs
  //   and constant accumulation will be absorbed.
  void AddInput(const FPNode* inp, int32_t num);

  // Consume a constant accumulation.
  // Only valid for kAdd (+=), or kMul (*=);
  void ConsumeConstant(FPConstAccum other, int32_t num);

  // Simplify inputs, generally should be called after all
  // inputs have been added.
  // Applying the following operations:
  // * If the node_type is kMul and it's associated constant is 0,
  //   remove all inputs.
  // * Remove any inputs with a count of 0.
  // * If there are no nodes, convert the node_type to constant.
  // * If there is only one input, with a count of 1 and no extra
  //   constant, propagate its value.
  bool SimplifyInputs();

  // Create a dotgraph representation of this node and its inputs.
  void DotGraph(std::ostream& output) const;

  // Print a descriptive representation of this node and its inputs.
  void PrintNode(std::ostream& output, int32_t indent = 0) const;

  // Print a equation representation of this node and its inputs.
  void PrintEquation(std::ostream& output) const;

  NodeType node_type = kInvalid;

  // External result id (kExternal)
  uint32_t result_id = UINT32_MAX;

  // Constants that have been accumulated (kAdd, kMul, kConstant)
  FPConstAccum const_accum{};

  // Id of this node wrt the ReassocGraphFP that stores it
  uint32_t id = UINT32_MAX;

  // Inputs to this node and their counts.
  // This should only be set for kAdd and kMul.
  // For kAdd, this is effectively a multiplier.
  //   { { X, 2 }, { Y, 3 }} => X*2 + Y*2
  // For kMul, this is effectively an exponent.
  //   { {X , 2 }, { Y, 3 }} => X*X * Y*Y*Y
  InputsType inputs;
};

// Reassociation graph for floating-point formats.
//
class FPReassocGraph {
 public:
  FPReassocGraph(analysis::Type* type, analysis::DefUseManager* def_use_mgr,
                 analysis::ConstantManager* const_mgr);

  // Helper constructor for running tests.
  // Adding instruction won't work, but doing things like adding nodes
  // and folding will.
  FPReassocGraph(uint32_t num_components)
      : type(nullptr),
        def_use_mgr(nullptr),
        const_mgr(nullptr),
        is_vector(num_components > 1) {
    default_zero_accum.vals.resize(num_components, 0.0);
    default_one_accum.vals.resize(num_components, 1.0);
  }

  bool IsVector() const {
    return is_vector;
  }

  uint32_t ElementSize() const {
    return default_zero_accum.size();
  }

  // Add a node to the graphs internal storage, which will de-duplicate
  // nodes with the same representation.
  const FPNode* AddNode(FPNode&& node);

  // Add an instruction to the graph.
  // Supported types are:
  // * OpFDiv
  // * OpFMul
  // * OpFSub
  // * OpFAdd
  // * OpFNegate
  const FPNode* AddInstruction(Instruction* inst);

  // Find an instruction that was previous added.
  // Will return a nullptr if not found.
  const FPNode* FindInstruction(Instruction* inst) const;

  // Find an instruction that was previous added, if it wasn't found,
  // add it as an external node and return that.
  const FPNode* FindInstructionOrCreateExternal(Instruction* inst);

  // Helpers for constucting nodes
  FPNode MakeExternal(uint32_t result_id) const {
    FPNode desc;
    desc.node_type = FPNode::kExternal;
    desc.result_id = result_id;
    return desc;
  }
  FPNode MakeConst(const FPConstAccum& const_accum) const {
    FPNode desc;
    desc.node_type = FPNode::kConstant;
    desc.const_accum = const_accum;
    return desc;
  }
  FPNode MakeMul(const FPConstAccum& const_accum,
                 const FPNode::InputsType& inputs) const {
    FPNode desc;
    desc.node_type = FPNode::kMul;
    desc.const_accum = const_accum;
    for (const auto& input : inputs) {
      desc.AddInput(input.first, input.second);
    }
    return desc;
  }
  FPNode MakeMul(const FPNode::InputsType& inputs) const {
    FPNode desc;
    desc.node_type = FPNode::kMul;
    desc.const_accum = DefaultMulAccum();
    for (const auto& input : inputs) {
      desc.AddInput(input.first, input.second);
    }
    return desc;
  }
  FPNode MakeMul() const {
    FPNode desc;
    desc.node_type = FPNode::kMul;
    desc.const_accum = DefaultMulAccum();
    return desc;
  }
  FPNode MakeAdd(const FPConstAccum& const_accum,
                 const FPNode::InputsType& inputs) const {
    FPNode desc;
    desc.node_type = FPNode::kAdd;
    desc.const_accum = const_accum;
    for (const auto& input : inputs) {
      desc.AddInput(input.first, input.second);
    }
    return desc;
  }
  FPNode MakeAdd(const FPNode::InputsType& inputs) const {
    FPNode desc;
    desc.node_type = FPNode::kAdd;
    desc.const_accum = DefaultAddAccum();
    for (const auto& input : inputs) {
      desc.AddInput(input.first, input.second);
    }
    return desc;
  }
  FPNode MakeAdd() const {
    FPNode desc;
    desc.node_type = FPNode::kAdd;
    desc.const_accum = DefaultAddAccum();
    return desc;
  }

  // Expand coefficients of add chains.
  // This is mainly to help factor more constants together.
  //
  //  (3 * (a + b)) + (2 * a) + (2 * b) => (5 * a) + (5 * b)
  bool ExpandCoefficients(FPNode& desc);

  // Convert add chains into muls and merge muls with the same inputs.
  //
  // Allowing the following rules to take place:
  //  a + a + a                   => 3 * a
  //  (3 * a) + a                 => 4 * a
  //  (3 * a) + (2 * a)           => 5 * a
  //  (3 * a) + (-3 * a)          => 0
  bool MergeAddMulInputs(FPNode& desc);

  // Merge muls with the same constant that are added together.
  //
  // Allowing the following rules to take place:
  //  (3 * a) + (3 * b) + (3 * c) => 3 * (a + b + c)
  //  (5 * a) + (-5 * b)          => 5 * (a - b)
  bool FactorAddConstMulInputs(FPNode& desc);

  // Attempt to propagate a muls constant value to a
  // single add that only contains mul inputs, which
  // already have a mul by constant taking place.
  //
  // Allowing the following rules to take place:
  //  (3 * (10 + 3 * (a + b)))  => 30 + 9 * (a + b)
  bool PropagateConstMulAddInputs(FPNode& desc);

  // Merge muls with a shared input that are added together.
  //
  // Allowing the following rules to take place:
  // (x * y) + (x * z)        => x * (y + z)
  // (5 * x * x) + (3 * x)    => x * (3 + (x * 5))
  bool FactorAddMulInputs(FPNode& desc);

  // Convert muls in add chains, which have a constant
  // of -1 to be 1.
  //
  // Allowing the following rules to take place:
  //  (-1 * a) + b = b - a
  //  b - (-1 * a) = b + a
  bool HoistMulByNegOne(FPNode& desc);

  // Applies folding rules sequentially once.
  bool ApplyFoldingRules(FPNode& desc);

  // Apply folding rules to a node and all its children.
  const FPNode* SimplifyNode(const FPNode* node);

  // Resolve all nodes used in the graph.
  std::vector<const FPNode*> ResolveNodes(const FPNode* node) const;

  // Resolve all nodes used in the graph.
  static void TopologicallySort(std::vector<const FPNode*>& nodes);

  const FPConstAccum& DefaultZeroAccum() const { return default_zero_accum; }
  const FPConstAccum& DefaultOneAccum() const { return default_one_accum; }
  const FPConstAccum& DefaultAddAccum() const { return default_zero_accum; }
  const FPConstAccum& DefaultMulAccum() const { return default_one_accum; }

 private:
  analysis::Type* type;
  analysis::DefUseManager* def_use_mgr;
  analysis::ConstantManager* const_mgr;

  FPConstAccum default_zero_accum;
  FPConstAccum default_one_accum;

  bool is_vector = false;
  uint32_t width = 0;

  std::unordered_map<Instruction*, const FPNode*> instr_to_node;
  std::unordered_set<FPNode, FPNode::Hash> storage;
};

}  // namespace reassociate
}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_REASSOCIATE_H_
