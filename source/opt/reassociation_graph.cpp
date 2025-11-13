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

#include "reassociation_graph.h"

#include <unordered_map>

#include "def_use_manager.h"
#include "instruction.h"
#include "type_manager.h"
#include "types.h"

namespace spvtools {
namespace opt {
namespace reassociate {

bool FPNode::operator==(const FPNode& other) const {
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
  switch (node_type) {
    case kAdd:
      other *= num;
      const_accum += other;
      break;
    case kMul: {
      FPConstAccum other_accum = other;
      other_accum.SetToDefaultMul();
      for (int32_t i = 0; i < num; ++i) {
        other_accum *= other;
      }
      for (int32_t i = 0; i > num; --i) {
        other_accum /= other;
      }
      const_accum *= other_accum;
      break;
    }
    default:
      assert(false);
      break;
  }
}

bool FPNode::SimplifyInputs() {
  if (node_type != kAdd && node_type != kMul) {
    return false;
  }
  bool has_changes = false;
  if (node_type == kMul && const_accum.IsZero()) {
    if (!inputs.empty()) {
      has_changes = true;
      inputs.clear();
    }
  }
  for (auto it = inputs.begin(); it != inputs.end();) {
    if (it->second == 0) {
      has_changes = true;
      it = inputs.erase(it);
      continue;
    }
    ++it;
  }
  if (inputs.empty()) {
    has_changes = true;
    node_type = kConstant;
  } else if ((inputs.size() == 1) &&
             ((node_type == kAdd && const_accum.IsDefaultAdd()) ||
              (node_type == kMul && const_accum.IsDefaultMul()))) {
    const auto& single_child = *inputs.begin();
    if (single_child.second == 1) {
      has_changes = true;
      *this = *single_child.first;
    }
  }
  return has_changes;
}

struct FPDotGraphBuilder {
  FPDotGraphBuilder(std::ostream& output_) : output(output_) {
    output << "digraph {\n";
  }

  ~FPDotGraphBuilder() { output << "}\n"; }

  std::string TakeOp(const char op, const std::string& group) {
    std::string name = "a" + std::to_string(op_counter++);
    output << name << " [shape=square label=\"" << op << "\" group=" << group
           << "]\n";
    return name;
  }

  std::string TakeConst(const FPConstAccum& accum) {
    std::string name = "c" + std::to_string(const_counter++);
    output << name << " [shape=underline label=\"";
    size_t n = accum.size();
    for (size_t i = 0; i < n; ++i) {
      output << accum[i];
      if (i < (n - 1)) {
        output << ", ";
      }
    }
    output << "\"]\n";
    return name;
  }

  const std::string& VisitChildren(const FPNode* node) {
    auto found = seen_nodes.find(node);
    if (found != seen_nodes.end()) {
      return found->second;
    }

    std::string last;
    if ((node->node_type == FPNode::kConstant) ||
        (node->node_type == FPNode::kAdd &&
         !node->const_accum.IsDefaultAdd()) ||
        (node->node_type == FPNode::kMul &&
         !node->const_accum.IsDefaultMul())) {
      last = TakeConst(node->const_accum);
    }

    if (node->node_type == FPNode::kExternal) {
      last = "e_" + std::to_string(node->result_id);
      output << last << " [shape=cds label=\"id=" << node->result_id << "\"]\n";
    }

    if (node->node_type == FPNode::kAdd || node->node_type == FPNode::kMul) {
      std::string group = "g" + std::to_string(group_counter++);

      const char additive_op = node->node_type == FPNode::kAdd ? '+' : '*';
      const char reductive_op = node->node_type == FPNode::kAdd ? '-' : '/';

      for (const auto& input : node->inputs) {
        const std::string& child = VisitChildren(input.first);
        for (int32_t i = 0; i < input.second; ++i) {
          if (last.empty()) {
            last = child;
            continue;
          }
          std::string next = TakeOp(additive_op, group);
          output << '{' << last << ' ' << child << "}->" << next << "\n";
          last = next;
        }
        for (int32_t i = 0; i > input.second; --i) {
          if (last.empty()) {
            // Add '-' prefix
            if (node->node_type == FPNode::kAdd) {
              last = TakeOp(reductive_op, group);
              output << child << "->" << last << "\n";
            } else {
              last = child;
            }
            continue;
          }
          std::string next = TakeOp(reductive_op, group);
          output << '{' << last << ' ' << child << "}->" << next << "\n";
          last = next;
        }
      }
    }
    if (last.empty()) {
      last = "_error_";
    }
    return seen_nodes.emplace(node, last).first->second;
  }

  std::ostream& output;
  uint32_t group_counter = 0;
  uint32_t op_counter = 0;
  uint32_t const_counter = 0;
  std::unordered_map<const FPNode*, std::string> seen_nodes;
};

void FPNode::DotGraph(std::ostream& output) const {
  FPDotGraphBuilder dot_builder(output);
  dot_builder.VisitChildren(this);
}

void FPNode::PrintNode(std::ostream& output, int32_t indent) const {
  std::string indentation(indent * 4, ' ');
  output << indentation;
  switch (node_type) {
    case kInvalid:
      output << "[ invalid] {\n";
      break;
    case kExternal:
      output << "[external] {\n";
      break;
    case kConstant:
      output << "[constant] {\n";
      break;
    case kAdd:
      output << "[   add  ] {\n";
      break;
    case kMul:
      output << "[   mul  ] {\n";
      break;
    default:
      assert(false);
      break;
  }
  bool print_const_accum = false;
  switch (node_type) {
    case FPNode::kConstant:
      print_const_accum = true;
      break;
    case FPNode::kAdd:
      print_const_accum = !const_accum.IsDefaultAdd();
      break;
    case FPNode::kMul:
      print_const_accum = !const_accum.IsDefaultMul();
      break;
    default:
      break;
  }
  if (print_const_accum) {
    output << indentation << "    .const_accum {\n";
    for (double v : const_accum.vals) {
      output << indentation << "        " << v << ",\n";
    }
    output << indentation << "    }\n";
  }
  if (node_type == FPNode::kExternal) {
    output << indentation << "    .result_id = " << result_id << "\n";
  }
  if (!inputs.empty()) {
    output << indentation << "    .inputs {\n";
    int32_t indent_child = indent + 2;
    for (const auto& c : inputs) {
      if (c.second != 1) {
        output << indentation << "        " << c.second << "x\n";
      }
      c.first->PrintNode(output, indent_child);
      output << ",\n";
    }
    output << indentation << "    }\n";
  }
  output << indentation << "}";
}

void FPNode::PrintEquation(std::ostream& output) const {
  if (node_type == FPNode::kInvalid) {
    output << "INVALID";
    return;
  }
  if (node_type == FPNode::kExternal) {
    output << "{" << result_id << "}";
    return;
  }

  bool print_const_accum = false;
  bool print_braces = false;
  switch (node_type) {
    case FPNode::kConstant:
      print_const_accum = true;
      break;
    case FPNode::kAdd:
      print_const_accum = !const_accum.IsDefaultAdd();
      print_braces = true;
      break;
    case FPNode::kMul:
      print_const_accum = !const_accum.IsDefaultMul();
      print_braces = true;
      break;
    default:
      break;
  }

  if (print_braces) {
    output << '(';
  }
  output << ' ';

  bool first_op = true;
  const char add_op = node_type == FPNode::kAdd ? '+' : '*';
  const char sub_op = node_type == FPNode::kAdd ? '-' : '/';

  if (print_const_accum) {
    size_t n = const_accum.size();
    if (n != 1) {
      output << '[';
    }
    for (size_t i = 0; i < n; ++i) {
      output << const_accum[i];
      if (i < (n - 1)) {
        output << ", ";
      }
    }
    if (n != 1) {
      output << ']';
    }

    first_op = false;
    output << ' ';
  }

  for (const auto& input : inputs) {
    for (int32_t i = 0; i < input.second; ++i) {
      if (!first_op) {
        output << add_op << ' ';
      }
      input.first->PrintEquation(output);
      output << ' ';
      first_op = false;
    }
    for (int32_t i = 0; i > input.second; --i) {
      if (!first_op) {
        output << sub_op << ' ';
      }
      // '-' prefix
      else if (node_type == FPNode::kAdd) {
        output << '-';
      }
      input.first->PrintEquation(output);
      output << ' ';
      first_op = false;
    }
  }

  if (print_braces) {
    output << ')';
  }
}

FPReassocGraph::FPReassocGraph(analysis::Type* type_,
                               analysis::DefUseManager* def_use_mgr_,
                               analysis::ConstantManager* const_mgr_)
    : type(type_), def_use_mgr(def_use_mgr_), const_mgr(const_mgr_) {
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

const FPNode* FPReassocGraph::AddNode(FPNode&& node) {
  assert(node.node_type != FPNode::kInvalid);
  node.id = storage.size();
  return &*storage.emplace(std::forward<FPNode>(node)).first;
}

const FPNode* FPReassocGraph::FindInstruction(Instruction* inst) const {
  auto found = instr_to_node.find(inst);
  if (found != instr_to_node.end()) {
    return found->second;
  }
  return nullptr;
}

const FPNode* FPReassocGraph::FindInstructionOrCreateExternal(
    Instruction* inst) {
  if (const FPNode* found = FindInstruction(inst)) {
    return found;
  }

  FPNode new_node_desc;
  new_node_desc.node_type = FPNode::kExternal;
  new_node_desc.result_id = inst->result_id();

  if (inst->IsConstant()) {
    bool fetched_ok = false;
    FPConstAccum const_values = DefaultZeroAccum();

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

const FPNode* FPReassocGraph::AddInstruction(Instruction* inst) {
  FPNode new_node_desc;
  new_node_desc.result_id = inst->result_id();

  auto ResolveInstArg = [&](uint32_t index) {
    return FindInstructionOrCreateExternal(
        def_use_mgr->GetDef(inst->GetSingleWordInOperand(index)));
  };
  switch (inst->opcode()) {
    case spv::Op::OpFDiv:
      new_node_desc.node_type = FPNode::kMul;
      new_node_desc.const_accum = DefaultMulAccum();
      new_node_desc.AddInput(ResolveInstArg(0), 1);
      new_node_desc.AddInput(ResolveInstArg(1), -1);
      break;
    case spv::Op::OpFMul:
      new_node_desc.node_type = FPNode::kMul;
      new_node_desc.const_accum = DefaultMulAccum();
      new_node_desc.AddInput(ResolveInstArg(0), 1);
      new_node_desc.AddInput(ResolveInstArg(1), 1);
      break;
    case spv::Op::OpFSub:
      new_node_desc.node_type = FPNode::kAdd;
      new_node_desc.const_accum = DefaultAddAccum();
      new_node_desc.AddInput(ResolveInstArg(0), 1);
      new_node_desc.AddInput(ResolveInstArg(1), -1);
      break;
    case spv::Op::OpFAdd:
      new_node_desc.node_type = FPNode::kAdd;
      new_node_desc.const_accum = DefaultAddAccum();
      new_node_desc.AddInput(ResolveInstArg(0), 1);
      new_node_desc.AddInput(ResolveInstArg(1), 1);
      break;
    case spv::Op::OpFNegate:
      new_node_desc.node_type = FPNode::kAdd;
      new_node_desc.const_accum = DefaultAddAccum();
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

bool FPReassocGraph::ExpandCoefficients(FPNode& desc) {
  if (desc.node_type != FPNode::kAdd) {
    return false;
  }

  // Extract composites and if told to, add them as inputs to desc if told too.
  auto TryToExtractComposite = [&](const auto& input, bool apply) {
    if (input.first->node_type != FPNode::kMul ||
        (input.first->inputs.size() != 1)) {
      return false;
    }
    auto first_child_entry = input.first->inputs.begin();
    // Not handling any powers here.
    // (Although that might be worthwhile for factorisation)
    if (first_child_entry->second != 1) {
      return false;
    }
    const FPNode* first_child = first_child_entry->first;
    if ((first_child->node_type == FPNode::kAdd) &&
            (first_child->inputs.size() > 1) ||
        !first_child->const_accum.IsDefaultAdd()) {
      if (apply) {
        if (!first_child->const_accum.IsDefaultAdd()) {
          FPConstAccum addon = first_child->const_accum;
          addon *= input.first->const_accum;
          desc.const_accum += addon;
        }
        for (const auto& child_input : first_child->inputs) {
          FPNode new_mul = MakeMul({{child_input.first, 1}});
          new_mul.const_accum = input.first->const_accum;
          new_mul.const_accum *= child_input.second;
          new_mul.SimplifyInputs();
          desc.AddInput(AddNode(std::move(new_mul)), 1);
        }
      }
      return true;
    }
    return false;
  };

  bool has_any_composites = false;
  for (const auto& input : desc.inputs) {
    if (TryToExtractComposite(input, false)) {
      has_any_composites = true;
      break;
    }
  }
  if (!has_any_composites) {
    return false;
  }
  FPNode::InputsType old_inputs = std::move(desc.inputs);
  desc.inputs.clear();
  for (const auto& input : old_inputs) {
    if (!TryToExtractComposite(input, true)) {
      desc.AddInput(input.first, input.second);
    }
  }
  desc.SimplifyInputs();
  return true;
}

bool FPReassocGraph::MergeAddMulInputs(FPNode& desc) {
  if (desc.node_type != FPNode::kAdd) {
    return false;
  }

  using MergedInputs =
      std::unordered_map<FPNode::InputsType, FPConstAccum, FPNode::InputsHash>;

  bool has_mergeable = false;
  MergedInputs merged_inputs;

  for (const auto& input : desc.inputs) {
    FPNode::InputsType mul_inputs;
    FPConstAccum accum = DefaultAddAccum();

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
      mul_inputs.emplace(input.first, 1);
      accum += input.second;
    }
    auto found = merged_inputs.find(mul_inputs);
    if (found != merged_inputs.end()) {
      has_mergeable = true;
      found->second += accum;
    } else {
      merged_inputs.emplace(std::move(mul_inputs), std::move(accum));
    }
  }

  if (!has_mergeable) {
    return false;
  }

  desc.inputs.clear();
  for (auto& merged_input : merged_inputs) {
    FPNode new_mul = MakeMul(merged_input.second, merged_input.first);
    new_mul.SimplifyInputs();
    desc.AddInput(AddNode(std::move(new_mul)), 1);
  }
  desc.SimplifyInputs();
  return true;
}

bool FPReassocGraph::FactorAddConstMulInputs(FPNode& desc) {
  if (desc.node_type != FPNode::kAdd) {
    return false;
  }

  if (desc.inputs.size() < 2) {
    return false;
  }

  struct AddMulInput {
    int32_t add_count{};
    FPNode::InputsType nested_mul_inputs;
  };

  using MergedConstants =
      std::unordered_map<FPConstAccum, std::vector<AddMulInput>,
                         FPConstAccum::Hash>;

  bool has_mergeable = false;
  MergedConstants merged_constants;
  FPNode::InputsType extras;

  for (const auto& input : desc.inputs) {
    if (input.first->node_type == FPNode::kMul) {
      AddMulInput add_mul_input;
      add_mul_input.add_count = 1;
      add_mul_input.nested_mul_inputs = input.first->inputs;

      FPConstAccum accum = input.first->const_accum;
      accum *= input.second;
      auto found = merged_constants.find(accum);

      // Attempt to merge with an already existing entry
      if (found != merged_constants.end()) {
        found->second.emplace_back(std::move(add_mul_input));
        has_mergeable = true;
      }
      // Try again, but negated, turning this into a subtract.
      FPConstAccum neg_accum = accum;
      neg_accum.Negate();
      found = merged_constants.find(neg_accum);
      if (found != merged_constants.end()) {
        add_mul_input.add_count = -1;
        found->second.emplace_back(std::move(add_mul_input));
        has_mergeable = true;
      }
      // Fallback to just adding a new entry
      else {
        merged_constants.emplace(accum,
                                 std::vector<AddMulInput>{add_mul_input});
      }
    } else {
      extras.emplace(input.first, input.second);
    }
  }

  if (!has_mergeable) {
    return false;
  }

  desc.inputs.clear();
  for (const auto& extra : extras) {
    desc.AddInput(extra.first, extra.second);
  }

  for (const auto& merged : merged_constants) {
    FPNode mul_constant = MakeMul();
    mul_constant.const_accum = merged.first;
    FPNode new_add = MakeAdd();
    for (const AddMulInput& add_mul_inputs : merged.second) {
      FPNode nested_mul = MakeMul(add_mul_inputs.nested_mul_inputs);
      nested_mul.SimplifyInputs();
      new_add.AddInput(AddNode(std::move(nested_mul)),
                       add_mul_inputs.add_count);
    }
    new_add.SimplifyInputs();
    mul_constant.AddInput(AddNode(std::move(new_add)), 1);
    mul_constant.SimplifyInputs();
    desc.AddInput(AddNode(std::move(mul_constant)), 1);
  }
  desc.SimplifyInputs();
  return true;
}

bool FPReassocGraph::PropagateConstMulAddInputs(FPNode& desc) {
  if (desc.node_type != FPNode::kMul) {
    return false;
  }
  if (desc.const_accum.IsDefaultMul()) {
    return false;
  }

  const FPNode* found = nullptr;

  for (const auto& input : desc.inputs) {
    if (input.first->node_type == FPNode::kAdd && input.second == 1) {
      bool relevant_candidate = true;
      for (const auto& add_input : input.first->inputs) {
        if (add_input.first->node_type != FPNode::kMul) {
          relevant_candidate = false;
          break;
        }
        // Don't introduce a mul that wasn't already going
        // to take place.
        if (add_input.first->const_accum.IsDefaultMul() ||
            add_input.first->const_accum.IsMinusOne()) {
          relevant_candidate = false;
          break;
        }
      }
      if (relevant_candidate) {
        found = input.first;
        break;
      }
    }
  }

  if (!found) {
    return false;
  }

  desc.inputs.erase(found);
  FPNode new_add = MakeAdd();
  new_add.const_accum = desc.const_accum;
  new_add.const_accum *= found->const_accum;

  for (const auto& original_add_input : found->inputs) {
    FPNode new_mul = *original_add_input.first;
    new_mul.const_accum *= desc.const_accum;
    new_mul.const_accum *= original_add_input.second;
    new_mul.SimplifyInputs();
    new_add.AddInput(AddNode(std::move(new_mul)), 1);
  }
  new_add.SimplifyInputs();

  desc.const_accum.SetToDefaultMul();
  desc.AddInput(AddNode(std::move(new_add)), 1);
  desc.SimplifyInputs();
  return true;
}

bool FPReassocGraph::FactorAddMulInputs(FPNode& desc) {
  // Factor whichever coefficient which is used the most.
  auto FactorNext = [&] {
    using FactorMap =
        std::unordered_map<const FPNode*, std::vector<const FPNode*>>;

    if (desc.node_type != FPNode::kAdd) {
      return false;
    }

    // Nodes which share a positive coefficient
    FactorMap positive_nodes;
    size_t positive_max = 0;
    const FPNode* positive_node = nullptr;

    // Nodes which share a negative coefficient
    FactorMap negative_nodes;
    size_t negative_max = 0;
    const FPNode* negative_node = nullptr;

    for (const auto& input : desc.inputs) {
      if (input.first->node_type != FPNode::kMul) {
        continue;
      }
      for (const auto& mul_input : input.first->inputs) {
        bool use_positive = mul_input.second >= 0;
        FactorMap& target_map = use_positive ? positive_nodes : negative_nodes;
        size_t& target_max = use_positive ? positive_max : negative_max;
        const FPNode*& target_node =
            use_positive ? positive_node : negative_node;

        auto found = target_map.find(mul_input.first);
        if (found != target_map.end()) {
          found->second.push_back(input.first);
          if (found->second.size() > target_max) {
            target_max = found->second.size();
            target_node = mul_input.first;
          }
        } else {
          target_map.emplace(mul_input.first,
                             std::vector<const FPNode*>{input.first});
        }
      }
    }

    if ((positive_max < 2) && (negative_max < 2)) {
      return false;
    }

    bool use_positive = (positive_max >= negative_max);
    FactorMap& target_map = use_positive ? positive_nodes : negative_nodes;
    const FPNode* target_node = use_positive ? positive_node : negative_node;
    std::vector<const FPNode*> children = target_map.at(target_node);

    FPNode new_mul = MakeMul();
    new_mul.AddInput(target_node, use_positive ? 1 : -1);

    FPNode new_add = MakeAdd();
    for (const FPNode* child : children) {
      desc.inputs.erase(child);
      FPNode fixed_child = *child;
      fixed_child.AddInput(target_node, use_positive ? -1 : 1);
      fixed_child.SimplifyInputs();
      new_add.AddInput(AddNode(std::move(fixed_child)), 1);
    }
    new_add.SimplifyInputs();
    new_mul.AddInput(AddNode(std::move(new_add)), 1);
    desc.AddInput(AddNode(std::move(new_mul)), 1);
    desc.SimplifyInputs();
    return true;
  };

  bool did_any_work = FactorNext();
  if (did_any_work) {
    // Keep factoring until we can't.
    while (FactorNext());
  }
  return did_any_work;
}

bool FPReassocGraph::HoistMulByNegOne(FPNode& desc) {
  if (desc.node_type != FPNode::kAdd) {
    return false;
  }

  bool has_any_mul_neg_one = false;
  for (const auto& input : desc.inputs) {
    if (input.first->node_type == FPNode::kMul &&
        input.first->const_accum.IsMinusOne()) {
      has_any_mul_neg_one = true;
      break;
    }
  }

  if (!has_any_mul_neg_one) {
    return false;
  }

  FPNode::InputsType old_inputs = std::move(desc.inputs);
  desc.inputs.clear();
  for (const auto& input : old_inputs) {
    if (input.first->node_type == FPNode::kMul &&
        input.first->const_accum.IsMinusOne()) {
      FPNode new_mul = MakeMul(input.first->inputs);
      new_mul.SimplifyInputs();
      desc.AddInput(AddNode(std::move(new_mul)), -input.second);
    } else {
      desc.AddInput(input.first, input.second);
    }
  }
  desc.SimplifyInputs();
  return true;
}

bool FPReassocGraph::ApplyFoldingRules(FPNode& desc) {
  FPNode prev = desc;

  bool applied = false;

  if (ExpandCoefficients(desc)) {
    applied = true;
  }
  if (MergeAddMulInputs(desc)) {
    applied = true;
  }
  if (FactorAddConstMulInputs(desc)) {
    applied = true;
  }
  if (PropagateConstMulAddInputs(desc)) {
    applied = true;
  }
  if (FactorAddMulInputs(desc)) {
    applied = true;
  }
  if (HoistMulByNegOne(desc)) {
    applied = true;
  }

  return applied && (prev != desc);
}

const FPNode* FPReassocGraph::SimplifyNode(const FPNode* node) {
  if (node->node_type == FPNode::kConstant ||
      node->node_type == FPNode::kExternal) {
    return node;
  }

  // TODO for instruction emitting:
  //
  // It would be "nice" to try and reduce the number of
  // "duplicate" mul operations which are fed into add-chains
  // which can't be merged:
  //
  // e.g:
  //    b + (50 * x) and c + (-50 * x)
  //    =>
  //    b + (50 * x) and c - (50 * x)
  //
  //
  // Likewise:
  //    OpAdd C = a, b
  //    OpAdd D = b, C
  //  ...
  //    OpAdd E = b, b
  //    OpAdd F = E, c
  //    =>
  //    OpAdd U = b, b
  //    OpAdd D = U, a
  //    ...
  //    OpAdd F = U, c
  //
  // So the total number of operations can be brought down.

  FPNode new_desc;
  new_desc.node_type = node->node_type;
  new_desc.result_id = node->result_id;
  new_desc.const_accum = node->const_accum;
  new_desc.inputs = node->inputs;

  // 1. Apply folding rules to this node.
  // 2. Apply folding rules to children.
  // 3. Apply folding rules to this node again,
  //    if that did anything, go back to 1,
  //    since we likely have new children.
  do {
    // Top level simplify
    while (ApplyFoldingRules(new_desc));

    // Simplify children
    FPNode::InputsType children = std::move(new_desc.inputs);
    new_desc.inputs.clear();
    for (const auto& input : children) {
      new_desc.AddInput(SimplifyNode(input.first), input.second);
    }
    new_desc.SimplifyInputs();
  } while (ApplyFoldingRules(new_desc));

  return AddNode(std::move(new_desc));
}

}  // namespace reassociate
}  // namespace opt
}  // namespace spvtools
