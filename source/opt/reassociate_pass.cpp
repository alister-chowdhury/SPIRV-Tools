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
#include "source/util/hash_combine.h"

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
#define CONST_ACCUM_OP_EQ_VEC(op)                          \
    FPConstAccum& operator op(const FPConstAccum& other) { \
      assert(other.vals.size() == vals.size());            \
      size_t n = vals.size();                              \
      for (size_t i = 0; i < n; ++i) {                     \
        vals[i] op other.vals[i];                          \
      }                                                    \
      return *this;                                        \
    }

#define CONST_ACCUM_OP_CMP_VEC(op)                      \
    bool operator op(const FPConstAccum& other) const { \
      assert(other.vals.size() == vals.size());         \
      size_t n = vals.size();                           \
      for (size_t i = 0; i < n; ++i) {                  \
        if (!(vals[i] op other.vals[i])) {              \
          return false;                                 \
        }                                               \
      }                                                 \
      return true;                                      \
    }

#define CONST_ACCUM_OP_EQ_SCALAR(op)          \
    FPConstAccum& operator op(double value) { \
      for (double& v : vals) {                \
        v op value;                           \
      }                                       \
      return *this;                           \
    }

#define CONST_ACCUM_OP_CMP_SCALAR(op)      \
    bool operator op(double value) const { \
      for (const double& v : vals) {       \
        if (!(v op value)) return false;   \
      }                                    \
      return true;                         \
    }

    CONST_ACCUM_OP_EQ_VEC(+=)
    CONST_ACCUM_OP_EQ_VEC(-=)
    CONST_ACCUM_OP_EQ_VEC(*=)
    CONST_ACCUM_OP_EQ_VEC(/=)
    CONST_ACCUM_OP_CMP_VEC(==)
    CONST_ACCUM_OP_CMP_VEC(!=)

    CONST_ACCUM_OP_EQ_SCALAR(+=)
    CONST_ACCUM_OP_EQ_SCALAR(-=)
    CONST_ACCUM_OP_EQ_SCALAR(*=)
    CONST_ACCUM_OP_EQ_SCALAR(/=)
    CONST_ACCUM_OP_EQ_SCALAR(=)
    CONST_ACCUM_OP_CMP_SCALAR(==)
    CONST_ACCUM_OP_CMP_SCALAR(!=)

#undef CONST_ACCUM_OP_EQ_VEC
#undef CONST_ACCUM_OP_EQ_SCALAR

    FPConstAccum& operator=(const FPConstAccum& other) {
      vals = other.vals;
      return *this;
    }

    bool IsZero() const {
      return *this == 0.0;
    }

    bool IsOne() const {
      return *this == 1.0;
    }

    void SetToDefaultMul() {
      *this = 1.0;
    }

    void SetToDefaultAdd() {
      *this = 0.0;
    }

    bool IsDefaultMul() const {
      return IsOne();
    }

    bool IsDefaultAdd() const {
      return IsZero();
    }

    double& operator[](int32_t index) {
      return vals[index];
    }

    const double& operator[](int32_t index) const {
      return vals[index];
    }

    size_t size() const {
      return vals.size();
    }

    struct Hash {
      size_t operator()(const FPConstAccum& accum) const {
        size_t hash = std::hash<size_t>{}(accum.vals.size());
        hash = utils::hash_combine(hash, accum.vals);
        return hash;
      }
    };

    std::vector<double> vals;
  };

  enum class NodeType {
    kInvalid,
    kExternal,
    kConstant,
    kAdd,  // add / sub
    kMul   // mul / div
  };

  struct FPNode {

    using InputsType = std::map<const FPNode*, int32_t>;

    struct InputsHasher {
      size_t operator()(const InputsType& inputs) const {
        size_t hash = std::hash<size_t>{}(inputs.size());
        for (const auto& inp : inputs) {
          hash = utils::hash_combine(hash, inp.first, inp.second);
        }
        return hash;
      }
    };

    struct Hash {
      size_t operator()(const FPNode& node) const {
        size_t consts = FPConstAccum::Hash{}(node.const_accum);
        size_t inputs = InputsHasher{}(node.inputs);
        size_t hash = utils::hash_combine(
            consts, inputs, static_cast<uint32_t>(node.node_type));
        // Purposefully ignoring the `result_id`, except for externals
        if (node.node_type == NodeType::kExternal) {
          hash = utils::hash_combine(hash, node.result_id);
        }
        return hash;
      }
    };

    bool operator==(const FPNode& other) const {
      if (node_type != other.node_type) {
        return false;
      }
      if (node_type == NodeType::kExternal) {
        if (result_id != other.result_id) {
          return false;
        }
      }
      if (other.const_accum != const_accum) {
        return false;
      }
      return inputs == other.inputs;
    }

    NodeType node_type = NodeType::kInvalid;
    uint32_t result_id = UINT32_MAX;
    FPConstAccum const_accum{};

    // Forward point to an already factored
    // version of this node.
    mutable const FPNode* factored = nullptr;

    void AddInput(const FPNode* inp, int32_t num);
    void ConsumeConstant(FPConstAccum other, int32_t num);

    // After inputs have finished being added:
    // * If the node_type is kMul and it's associated constant is 0,
    //   remove all inputs.
    // * Remove any inputs with a count of 0.
    // * If there are no nodes, convert the node_type to constant.
    // * If there is only one input, with a count of 1 and no extra
    //   constant, propagate its value.
    void SimplifyInputs();
    
    // Inputs to this node and their counts
    InputsType inputs;
  };

  ReassocGraphFP(analysis::Type* type_, analysis::TypeManager* type_mgr_,
                 analysis::DefUseManager* def_use_mgr_,
                 analysis::ConstantManager* const_mgr_)
      : type(type_),
        type_mgr(type_mgr_),
        def_use_mgr(def_use_mgr_),
        const_mgr(const_mgr_) {

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

  analysis::Type* type;
  analysis::TypeManager* type_mgr;
  analysis::DefUseManager* def_use_mgr;
  analysis::ConstantManager* const_mgr;

  bool is_vector = false;
  uint32_t width = 0;
  FPConstAccum default_add_accum;
  FPConstAccum default_mul_accum;

  const FPNode* AddInstruction(Instruction* inst);
  const FPNode* FindInstruction(Instruction* inst);
  const FPNode* FindInstructionOrCreateExternal(Instruction* inst);
  const FPNode* AddNode(FPNode&& node);


  // NEED TO CHANGE THIS SO:
  // (3 * a) + a => 4 * a
  // 1. Inputs with a count >= 2 or count < 0 are wrapped with a kMul. << HACKING FOR ABOVE
  //     TO BE ALL THINGS
  // 2. kMul inputs that only differ by their constant values
  //    are merged together.
  // 3. Simplify the above.
  // 4. kMuls with the same constant, but different inputs are
  //    merged, such that:
  //    (3 * a) + (3 * b) + (3 * c) => 3 * (a + b + c) //// TODO
  // 5. Factor: /// (Maybe????)
  //    (a * b) + (4 * a) + (3 * b) + 12 => (3 + a) * (4 + b)
  void ApplyAddFactorizationRules(FPNode& new_desc);


  const FPNode* ApplyFactorizationRules(const FPNode* root);

  // REMOVE THIS
  void PrintNode(std::ostream& output, const FPNode* node, int32_t indent = 0);

  std::unordered_map<Instruction*, const FPNode*> instr_to_node;
  std::unordered_set<FPNode, FPNode::Hash> storage;
};

const ReassocGraphFP::FPNode* ReassocGraphFP::FindInstruction(Instruction* inst) {
  auto found = instr_to_node.find(inst);
  if (found != instr_to_node.end()) {
    return found->second;
  }
  return nullptr;
}

const ReassocGraphFP::FPNode* ReassocGraphFP::FindInstructionOrCreateExternal(
    Instruction* inst) {
  if (const FPNode* found = FindInstruction(inst)) {
    return found;
  }

  FPNode new_node_desc;
  new_node_desc.node_type = NodeType::kExternal;
  new_node_desc.result_id = inst->result_id();

  if (inst->IsConstant()) {
    bool fetched_ok = false;
    FPConstAccum const_values = default_add_accum;

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
      new_node_desc.node_type = NodeType::kConstant;
      new_node_desc.const_accum = std::move(const_values);
    }
  }

  const ReassocGraphFP::FPNode* new_node = AddNode(std::move(new_node_desc));
  instr_to_node[inst] = new_node;
  return new_node;
}

const ReassocGraphFP::FPNode* ReassocGraphFP::AddNode(FPNode&& node) {
  assert(node.node_type != NodeType::kInvalid);
  return &*storage.emplace(std::forward<FPNode>(node)).first;
}

const ReassocGraphFP::FPNode* ReassocGraphFP::AddInstruction(Instruction* inst) {

  FPNode new_node_desc;
  new_node_desc.result_id = inst->result_id();

  auto ResolveInstArg = [&](uint32_t index) {
    return FindInstructionOrCreateExternal(
        def_use_mgr->GetDef(inst->GetSingleWordInOperand(index)));
  };
  switch (inst->opcode()) {
  case spv::Op::OpFDiv:
    new_node_desc.node_type = NodeType::kMul;
    new_node_desc.const_accum = default_mul_accum;
    new_node_desc.AddInput(ResolveInstArg(0),  1);
    new_node_desc.AddInput(ResolveInstArg(1), -1);
    break;
  case spv::Op::OpFMul:
    new_node_desc.node_type = NodeType::kMul;
    new_node_desc.const_accum = default_mul_accum;
    new_node_desc.AddInput(ResolveInstArg(0),  1);
    new_node_desc.AddInput(ResolveInstArg(1),  1);
    break;
  case spv::Op::OpFSub:
    new_node_desc.node_type = NodeType::kAdd;
    new_node_desc.const_accum = default_add_accum;
    new_node_desc.AddInput(ResolveInstArg(0),  1);
    new_node_desc.AddInput(ResolveInstArg(1), -1);
    break;
  case spv::Op::OpFAdd:
    new_node_desc.node_type = NodeType::kAdd;
    new_node_desc.const_accum = default_add_accum;
    new_node_desc.AddInput(ResolveInstArg(0),  1);
    new_node_desc.AddInput(ResolveInstArg(1),  1);
    break;
  case spv::Op::OpFNegate:
    new_node_desc.node_type = NodeType::kAdd;
    new_node_desc.const_accum = default_add_accum;
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

void ReassocGraphFP::FPNode::AddInput(const FPNode* parent, int32_t num) {
  assert(node_type == NodeType::kAdd || node_type == NodeType::kMul);
  if (num == 0) {
    return;
  }

  if (parent->node_type == node_type || parent->node_type == NodeType::kConstant) {
    ConsumeConstant(parent->const_accum, num);
  }

  if (parent->node_type == node_type) {
    for (const auto& parent_input : parent->inputs) {
      inputs.try_emplace(parent_input.first, 0).first->second += parent_input.second * num;
    }
  }
  else if (parent->node_type == NodeType::kConstant) {
    return;
  }
  else {
    inputs.emplace(parent, 0).first->second += num;
  }
}

void ReassocGraphFP::FPNode::ConsumeConstant(FPConstAccum other, int32_t num) {
  other *= num;
  switch (node_type) {
    case NodeType::kAdd:
      const_accum += other;
      break;
    case NodeType::kMul:
      const_accum *= other;
      break;
    default:
      assert(false);
      break;
  }
}

void ReassocGraphFP::FPNode::SimplifyInputs() {
  if (node_type != NodeType::kAdd && node_type != NodeType::kMul) {
    return;
  }
  if (node_type == NodeType::kMul && const_accum.IsZero()) {
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
    node_type = NodeType::kConstant;
  } else if ((inputs.size() == 1) &&
             ((node_type == NodeType::kAdd && const_accum.IsDefaultAdd()) ||
              (node_type == NodeType::kMul && const_accum.IsDefaultMul()))) {
    const auto& single_child = *inputs.begin();
    if (single_child.second == 1) {
      *this = *single_child.first;
    }
  }
}

void ReassocGraphFP::ApplyAddFactorizationRules(FPNode& new_desc) {
  
  assert(new_desc.node_type == NodeType::kAdd);

  // Split this stuff into different functions, its getting a bit hard to follow.

  std::unordered_map<FPNode::InputsType, FPConstAccum, FPNode::InputsHasher>
    merged_mul_inputs;

  // Wrap inputs with a count >= 2 || count < 0 with a kMul
  // Accumulate kMuls with the same inputs
  {
    FPNode::InputsType local_inputs = std::move(new_desc.inputs);
    for (const auto& input : local_inputs) {
      const FPNode* parent = input.first;
      int32_t count = input.second;

      if (parent->node_type != NodeType::kMul && ((count > 2) || (count < 0))) {
        FPNode new_mul;
        new_mul.node_type = NodeType::kMul;
        new_mul.const_accum = default_mul_accum;
        new_mul.const_accum *= count;
        new_mul.AddInput(parent, 1);
        new_mul.SimplifyInputs();
        parent = AddNode(std::move(new_mul));
        count = 1;
      }

      // Hack, to force wrap everything in a mul
      // so (a * 3) + a => a * 4
      if (parent->node_type != NodeType::kMul) {
        FPNode::InputsType tmp;
        tmp[parent] = 1;
        merged_mul_inputs.try_emplace(tmp, default_add_accum)
          .first->second += 1;
      }

      if (parent->node_type != NodeType::kMul) {
        // hack, see above
        //new_desc.AddInput(parent, count);
      } else {
        const FPConstAccum& zero =
            default_add_accum;  // default_add_accum is [0,...,0]
        FPConstAccum& const_accum =
            merged_mul_inputs.try_emplace(parent->inputs, default_add_accum)
                .first->second;
        FPConstAccum addon = parent->const_accum;
        addon *= count;
        const_accum += addon;
      }
    }
  }

  // Accumulate kMuls with the same constant
  std::unordered_map<FPConstAccum, std::vector<FPNode::InputsType>,
                     FPConstAccum::Hash>
      merged_muls;

  for (const auto& input : merged_mul_inputs) {
    std::vector<FPNode::InputsType>& merged_inputs = merged_muls.try_emplace(
      input.second,
      std::vector<FPNode::InputsType>{}
    ).first->second;
    merged_inputs.push_back(input.first);
  }

  // Emit merged muls
  for (const auto& input : merged_muls) {
    FPNode new_mul;
    new_mul.node_type = NodeType::kMul;
    new_mul.const_accum = input.first;
    if (input.second.size() == 1) {
      new_mul.inputs = input.second[0];
    }
    else {
      FPNode new_add;
      new_add.node_type = NodeType::kAdd;
      new_add.const_accum = default_add_accum;
      for (const auto& child_mul_inputs : input.second) {
        FPNode new_child_mul;
        new_child_mul.node_type = NodeType::kMul;
        new_child_mul.const_accum = default_mul_accum;
        new_child_mul.inputs = child_mul_inputs;
        new_child_mul.SimplifyInputs();
        new_add.AddInput(ApplyFactorizationRules(AddNode(std::move(new_child_mul))), 1);
      }
      new_add.SimplifyInputs();
      new_mul.AddInput(ApplyFactorizationRules(AddNode(std::move(new_add))), 1);
    }
    new_mul.SimplifyInputs();
    new_desc.AddInput(ApplyFactorizationRules(AddNode(std::move(new_mul))), 1);
  }

  new_desc.SimplifyInputs();
}

const ReassocGraphFP::FPNode* ReassocGraphFP::ApplyFactorizationRules(const FPNode* root) {
  assert(root->node_type != NodeType::kInvalid);
  if ((root->node_type == NodeType::kConstant) || (root->node_type == NodeType::kExternal)) {
    return root;
  }

  if (root->factored) {
    return root->factored;
  }

  FPNode new_desc;
  new_desc.node_type = root->node_type;
  new_desc.result_id = root->result_id;
  new_desc.const_accum = root->const_accum;
  for (const auto& input : root->inputs) {
    new_desc.AddInput(ApplyFactorizationRules(input.first), input.second);
  }
  new_desc.SimplifyInputs();

  if (new_desc.node_type == NodeType::kAdd) {
    ApplyAddFactorizationRules(new_desc);
  }

  root->factored = AddNode(std::move(new_desc));
  return root->factored;
}


void ReassocGraphFP::PrintNode(std::ostream& output, const FPNode* node, int32_t indent) {
  std::string indentation(indent * 4, ' ');

  output << indentation;
  switch (node->node_type) {
  case NodeType::kInvalid : output << "[ invalid] {\n"; break;
  case NodeType::kExternal: output << "[external] {\n"; break;
  case NodeType::kConstant: output << "[constant] {\n"; break;
  case NodeType::kAdd:      output << "[   add  ] {\n"; break;
  case NodeType::kMul:      output << "[   mul  ] {\n"; break;
  default:
    assert(false);
    break;
  }

  bool print_const_accum = false;
  switch (node->node_type) {
  case NodeType::kConstant: print_const_accum = true; break;
  case NodeType::kAdd:      print_const_accum = !node->const_accum.IsDefaultAdd(); break;
  case NodeType::kMul:      print_const_accum = !node->const_accum.IsDefaultMul(); break;
  default:
    break;
  }

  if (print_const_accum) {
    output << indentation   << "    .const_accum {\n";
    for (double v : node->const_accum.vals) {
      output << indentation << "        " << v << ",\n";
    }
    output << indentation   << "    }\n";
  }

  if (node->node_type == NodeType::kExternal) {
    output << indentation   << "    .result_id = " << node->result_id << "\n";
  }

  if (!node->inputs.empty()) {
    output << indentation   << "    .inputs {\n";
    int32_t indent_child = indent + 2;
    for (const auto& c : node->inputs) {
      if (c.second != 1) {
        output << indentation << "        " << c.second << "x\n";
      }
      PrintNode(output, c.first, indent_child);
      output << ",\n";
    }
    output << indentation   << "    }\n";
  }

  output << indentation << "}";
}

bool ReassociatePass::ReassociateFPGraph(Instruction* root,
                                         std::vector<Instruction*>&& graph) {
  analysis::Type* type = context()->get_type_mgr()->GetType(root->type_id());
  ReassocGraphFP fpgraph(
      type, context()->get_type_mgr(), context()->get_def_use_mgr(),
      context()->get_constant_mgr());

  for (Instruction* inst : graph) {
    fpgraph.AddInstruction(inst);
  }

  const ReassocGraphFP::FPNode* root_fp = fpgraph.FindInstruction(root);

  std::cout << "BEFORE OPT:\n";
  fpgraph.PrintNode(std::cout, root_fp);
  std::cout.flush();

  root_fp = fpgraph.ApplyFactorizationRules(root_fp);

  std::cout << "\n\n\nAFTER OPT:\n";
  fpgraph.PrintNode(std::cout, root_fp);
  std::cout.flush();

  //ReassocGraphFP::FPReassocNode* root_node = fpgraph.GetUserExternal(root);
  //root_node->ConvertAddsToMuls(fpgraph);
  //// Run factorisation pass here

  bool modified = false;
  //if ((root_node->flags & ReassocGraphFP::FPReassocNode::kBeenOptimised)) {
  //  modified = true;

  //}
  return modified;
}

}  // namespace opt
}  // namespace spvtools
