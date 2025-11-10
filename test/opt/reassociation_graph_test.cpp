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

#include "source/opt/reassociation_graph.h"

#include "gtest/gtest.h"
#include "source/opt/instruction.h"

namespace spvtools {
namespace opt {
namespace reassociate {

void PrintTo(const FPNode& fpnode, std::ostream* os) {
  *os << "\nFull:\n";
  fpnode.PrintNode(*os);
  *os << "\n";
  *os << "Equation: ";
  fpnode.PrintEquation(*os);
  *os << "\n";
}

namespace {

class ReassocGraphBuilderTest : public ::testing::Test {
 protected:
  ReassocGraphBuilderTest() : graph(1u) {}

  const FPNode* GetExternal(uint32_t id) {
    return graph.AddNode(std::move(graph.MakeExternal(id)));
  }
  const FPNode* GetConst(double value) {
    FPConstAccum val = graph.DefaultZeroAccum();
    val = value;
    return graph.AddNode(std::move(graph.MakeConst(val)));
  }
  const FPNode* GetAdd(const FPNode::InputsType& inputs) {
    return graph.AddNode(std::move(graph.MakeAdd(inputs)));
  }
  const FPNode* GetMul(const FPNode::InputsType& inputs) {
    return graph.AddNode(std::move(graph.MakeMul(inputs)));
  }

  FPReassocGraph graph;
};

// Make sure nodes are correctly de-duplicating and are not colliding with
// each other.
TEST_F(ReassocGraphBuilderTest, TestFPDeduplication) {
  {
    const FPNode* external_1 = GetExternal(1);
    const FPNode* external_2 = GetExternal(2);
    const FPNode* const_1 = GetConst(1.0);
    const FPNode* const_2 = GetConst(2.0);
    const FPNode* add_ex1_ex2 = GetAdd({{external_1, 1}, {external_2, 1}});
    const FPNode* add_cn1_ex2 = GetAdd({{const_1, 1}, {external_2, 1}});
    const FPNode* add_ex1_cn2 = GetAdd({{external_1, 1}, {const_2, 1}});
    const FPNode* mul_ex1_ex2 = GetMul({{external_1, 1}, {external_2, 1}});
    const FPNode* mul_cn1_ex2 = GetMul({{const_1, 1}, {external_2, 1}});
    const FPNode* mul_ex1_cn2 = GetMul({{external_1, 1}, {const_2, 1}});
    const FPNode* add_chain =
        GetAdd({{mul_ex1_ex2, 1}, {mul_cn1_ex2, 2}, {mul_ex1_cn2, 1}});
    const FPNode* mul_chain =
        GetMul({{add_ex1_ex2, 1}, {add_cn1_ex2, 2}, {add_ex1_cn2, 1}});

    EXPECT_TRUE(external_1->result_id == 1);
    EXPECT_EQ(external_1, GetExternal(1));

    EXPECT_TRUE(external_2->result_id == 2);
    EXPECT_EQ(external_2, GetExternal(2));
    EXPECT_NE(external_1, external_2);

    EXPECT_TRUE(const_1->const_accum == 1.0);
    EXPECT_EQ(const_1, GetConst(1.0));

    EXPECT_TRUE(const_2->const_accum == 2.0);
    EXPECT_EQ(const_2, GetConst(2.0));
    EXPECT_NE(const_1, const_2);

    EXPECT_EQ(add_ex1_ex2->inputs.at(external_1), 1);
    EXPECT_EQ(add_ex1_ex2->inputs.at(external_2), 1);
    EXPECT_EQ(add_ex1_ex2, GetAdd({{external_1, 1}, {external_2, 1}}));
    EXPECT_NE(add_ex1_ex2, GetAdd({{external_1, 2}, {external_2, 1}}));
    EXPECT_NE(add_ex1_ex2, GetAdd({{external_1, 1}, {external_2, 2}}));

    EXPECT_EQ(add_cn1_ex2->inputs.at(external_2), 1);
    EXPECT_TRUE(add_cn1_ex2->const_accum == 1.0);
    EXPECT_EQ(add_cn1_ex2, GetAdd({{const_1, 1}, {external_2, 1}}));
    EXPECT_NE(add_cn1_ex2, GetAdd({{const_1, 2}, {external_2, 1}}));
    EXPECT_NE(add_cn1_ex2, GetAdd({{const_1, 1}, {external_2, 2}}));

    EXPECT_EQ(add_ex1_cn2->inputs.at(external_1), 1);
    EXPECT_TRUE(add_ex1_cn2->const_accum == 2.0);
    EXPECT_EQ(add_ex1_cn2, GetAdd({{external_1, 1}, {const_2, 1}}));
    EXPECT_NE(add_ex1_cn2, GetAdd({{external_1, 2}, {const_2, 1}}));
    EXPECT_NE(add_ex1_cn2, GetAdd({{external_1, 1}, {const_2, 2}}));

    EXPECT_EQ(mul_ex1_ex2->inputs.at(external_1), 1);
    EXPECT_EQ(mul_ex1_ex2->inputs.at(external_2), 1);
    EXPECT_EQ(mul_ex1_ex2, GetMul({{external_1, 1}, {external_2, 1}}));
    EXPECT_NE(mul_ex1_ex2, GetMul({{external_1, 2}, {external_2, 1}}));
    EXPECT_NE(mul_ex1_ex2, GetMul({{external_1, 1}, {external_2, 2}}));

    EXPECT_EQ(mul_cn1_ex2->inputs.at(external_2), 1);
    EXPECT_TRUE(mul_cn1_ex2->const_accum == 1.0);
    EXPECT_EQ(mul_cn1_ex2, GetMul({{const_1, 1}, {external_2, 1}}));
    EXPECT_EQ(mul_cn1_ex2,
              GetMul({{const_1, 2}, {external_2, 1}}));  // Equal due to 1^2 = 1
    EXPECT_NE(mul_cn1_ex2, GetMul({{const_1, 1}, {external_2, 2}}));

    EXPECT_EQ(mul_ex1_cn2->inputs.at(external_1), 1);
    EXPECT_TRUE(mul_ex1_cn2->const_accum == 2.0);
    EXPECT_EQ(mul_ex1_cn2, GetMul({{external_1, 1}, {const_2, 1}}));
    EXPECT_NE(mul_ex1_cn2, GetMul({{external_1, 2}, {const_2, 1}}));
    EXPECT_NE(mul_ex1_cn2, GetMul({{external_1, 1}, {const_2, 2}}));

    EXPECT_EQ(add_chain->inputs.at(mul_ex1_ex2), 1);
    EXPECT_EQ(add_chain->inputs.at(mul_cn1_ex2), 2);
    EXPECT_EQ(add_chain->inputs.at(mul_ex1_cn2), 1);
    EXPECT_EQ(add_chain,
              GetAdd({{mul_ex1_ex2, 1}, {mul_cn1_ex2, 2}, {mul_ex1_cn2, 1}}));
    EXPECT_NE(add_chain,
              GetAdd({{mul_ex1_ex2, 2}, {mul_cn1_ex2, 2}, {mul_ex1_cn2, 1}}));
    EXPECT_NE(add_chain,
              GetAdd({{mul_ex1_ex2, 1}, {mul_cn1_ex2, 1}, {mul_ex1_cn2, 1}}));
    EXPECT_NE(add_chain,
              GetAdd({{mul_ex1_ex2, 1}, {mul_cn1_ex2, 1}, {mul_ex1_cn2, 2}}));
    EXPECT_NE(add_chain, GetAdd({{mul_cn1_ex2, 2}, {mul_ex1_cn2, 1}}));
    EXPECT_NE(add_chain, GetAdd({{mul_ex1_ex2, 1}, {mul_ex1_cn2, 1}}));
    EXPECT_NE(add_chain, GetAdd({{mul_ex1_ex2, 1}, {mul_cn1_ex2, 1}}));

    EXPECT_EQ(mul_chain->inputs.at(add_ex1_ex2), 1);
    EXPECT_EQ(mul_chain->inputs.at(add_cn1_ex2), 2);
    EXPECT_EQ(mul_chain->inputs.at(add_ex1_cn2), 1);
    EXPECT_EQ(mul_chain,
              GetMul({{add_ex1_ex2, 1}, {add_cn1_ex2, 2}, {add_ex1_cn2, 1}}));
    EXPECT_NE(mul_chain,
              GetMul({{add_ex1_ex2, 2}, {add_cn1_ex2, 2}, {add_ex1_cn2, 1}}));
    EXPECT_NE(mul_chain,
              GetMul({{add_ex1_ex2, 1}, {add_cn1_ex2, 1}, {add_ex1_cn2, 1}}));
    EXPECT_NE(mul_chain,
              GetMul({{add_ex1_ex2, 1}, {add_cn1_ex2, 1}, {add_ex1_cn2, 2}}));
    EXPECT_NE(mul_chain, GetMul({{add_cn1_ex2, 2}, {add_ex1_cn2, 1}}));
    EXPECT_NE(mul_chain, GetMul({{add_ex1_ex2, 1}, {add_ex1_cn2, 1}}));
    EXPECT_NE(mul_chain, GetMul({{add_ex1_ex2, 1}, {add_cn1_ex2, 1}}));
  }
}

TEST_F(ReassocGraphBuilderTest, TestFPNodeMerging) {
  const FPNode* external_1 = GetExternal(1);
  const FPNode* external_2 = GetExternal(2);
  const FPNode* external_3 = GetExternal(3);

  // (a + b) + c = a + b + c
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(external_3, 1);
    add.AddInput(GetAdd({{external_1, 1}, {external_2, 1}}), 1);
    EXPECT_EQ(add, graph.MakeAdd({
                       {external_1, 1},
                       {external_2, 1},
                       {external_3, 1},
                   }));
  }

  // (a * b) * c = a * b * c
  {
    FPNode mul = graph.MakeMul();
    mul.AddInput(external_3, 1);
    mul.AddInput(GetMul({{external_1, 1}, {external_2, 1}}), 1);
    EXPECT_EQ(mul, graph.MakeMul({
                       {external_1, 1},
                       {external_2, 1},
                       {external_3, 1},
                   }));
  }

  // (a + 10) + (b + 15) = a + b + 25
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetAdd({{GetConst(10), 1}, {external_1, 1}}), 1);
    add.AddInput(GetAdd({{GetConst(15), 1}, {external_2, 1}}), 1);
    EXPECT_EQ(add, graph.MakeAdd(
                       {{GetConst(25), 1}, {external_1, 1}, {external_2, 1}}));
  }

  // (a * 10) * (b * 15) = a * b * 150
  {
    FPNode mul = graph.MakeMul();
    mul.AddInput(GetMul({{GetConst(10), 1}, {external_1, 1}}), 1);
    mul.AddInput(GetMul({{GetConst(15), 1}, {external_2, 1}}), 1);
    EXPECT_EQ(mul, graph.MakeMul(
                       {{GetConst(150), 1}, {external_1, 1}, {external_2, 1}}));
  }

  // (a + 10) + (a + 2) = a + a + 12
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetAdd({{GetConst(10), 1}, {external_1, 1}}), 1);
    add.AddInput(GetAdd({{GetConst(2), 1}, {external_1, 1}}), 1);
    EXPECT_EQ(add, graph.MakeAdd({{GetConst(12), 1}, {external_1, 2}}));
  }

  // (a * 10) * (a * 2) = a * a * 20
  {
    FPNode mul = graph.MakeMul();
    mul.AddInput(GetMul({{GetConst(10), 1}, {external_1, 1}}), 1);
    mul.AddInput(GetMul({{GetConst(2), 1}, {external_1, 1}}), 1);
    EXPECT_EQ(mul, graph.MakeMul({{GetConst(20), 1}, {external_1, 2}}));
  }
}

TEST_F(ReassocGraphBuilderTest, TestFPSimplifyInputs) {
  const FPNode* external_1 = GetExternal(1);
  const FPNode* external_2 = GetExternal(2);
  const FPNode* external_3 = GetExternal(3);
  const FPNode* const_0 = GetConst(0.0);
  const FPNode* const_1 = GetConst(1.0);
  const FPNode* const_2 = GetConst(2.0);

  // Mul by zero
  {
    FPNode mul = graph.MakeMul();
    mul.AddInput(const_0, 1);
    mul.AddInput(external_1, 1);
    mul.AddInput(external_2, 1);
    mul.SimplifyInputs();
    EXPECT_EQ(mul.node_type, FPNode::kConstant);
    EXPECT_EQ(mul.inputs.size(), 0);
    EXPECT_TRUE(mul.const_accum.IsZero());
  }

  // Remove zero inputs
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(external_1, 1);
    add.AddInput(external_2, 1);
    add.AddInput(external_3, 1);
    add.AddInput(external_1, -1);
    add.SimplifyInputs();
    EXPECT_EQ(add.node_type, FPNode::kAdd);
    EXPECT_EQ(add.inputs.size(), 2);
    EXPECT_EQ(add.inputs.find(external_1), add.inputs.end());
    EXPECT_EQ(add.inputs.at(external_2), 1);
    EXPECT_EQ(add.inputs.at(external_3), 1);

    FPNode mul = graph.MakeMul();
    mul.AddInput(external_1, 1);
    mul.AddInput(external_2, 1);
    mul.AddInput(external_3, 1);
    mul.AddInput(external_1, -1);
    mul.SimplifyInputs();
    EXPECT_EQ(mul.node_type, FPNode::kMul);
    EXPECT_EQ(mul.inputs.size(), 2);
    EXPECT_EQ(mul.inputs.find(external_1), mul.inputs.end());
    EXPECT_EQ(mul.inputs.at(external_2), 1);
    EXPECT_EQ(mul.inputs.at(external_3), 1);
  }

  // Propagate to const
  {
    FPNode add = graph.MakeAdd();
    add.SimplifyInputs();
    EXPECT_EQ(add.node_type, FPNode::kConstant);
    EXPECT_EQ(add.inputs.size(), 0);
    EXPECT_TRUE(add.const_accum.IsDefaultAdd());

    FPNode mul = graph.MakeMul();
    mul.SimplifyInputs();
    EXPECT_EQ(mul.node_type, FPNode::kConstant);
    EXPECT_EQ(mul.inputs.size(), 0);
    EXPECT_TRUE(mul.const_accum.IsDefaultMul());
  }

  // Propagate to single input
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(external_1, 1);
    add.AddInput(external_2, 1);
    add.AddInput(external_1, -1);
    add.SimplifyInputs();
    EXPECT_EQ(add, *external_2);

    add = graph.MakeAdd();
    add.AddInput(external_1, 1);
    add.SimplifyInputs();
    EXPECT_EQ(add, *external_1);

    FPNode mul = graph.MakeMul();
    mul.AddInput(external_1, 1);
    mul.AddInput(external_2, 1);
    mul.AddInput(external_1, -1);
    mul.SimplifyInputs();
    EXPECT_EQ(mul, *external_2);

    mul = graph.MakeMul();
    mul.AddInput(external_1, 1);
    mul.SimplifyInputs();
    EXPECT_EQ(mul, *external_1);
  }

  // Don't propagate if count != 1
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(external_1, 2);
    add.SimplifyInputs();
    EXPECT_NE(add, *external_1);

    FPNode mul = graph.MakeMul();
    mul.AddInput(external_1, 2);
    mul.SimplifyInputs();
    EXPECT_NE(mul, *external_1);
  }
}

TEST_F(ReassocGraphBuilderTest, TestFPExpandCoefficients) {
  const FPNode* external_1 = GetExternal(1);
  const FPNode* external_2 = GetExternal(2);
  const FPNode* external_3 = GetExternal(3);

  // a = no change
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(external_1, 1);
    add.SimplifyInputs();
    FPNode add_2 = add;
    EXPECT_FALSE(graph.ExpandCoefficients(add_2));
    EXPECT_EQ(add, add_2);
  }

  // a + b = no change
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(external_1, 1);
    add.AddInput(external_2, 1);
    add.SimplifyInputs();
    FPNode add_2 = add;
    EXPECT_FALSE(graph.ExpandCoefficients(add_2));
    EXPECT_EQ(add, add_2);
  }

  // (5 * a) + b = no change
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{external_1, 1}, {GetConst(5.0), 1}}), 1);
    add.AddInput(external_2, 1);
    add.SimplifyInputs();
    FPNode add_2 = add;
    EXPECT_FALSE(graph.ExpandCoefficients(add_2));
    EXPECT_EQ(add, add_2);
  }

  // (5 * a) + (3 * b) = no change
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{external_1, 1}, {GetConst(5.0), 1}}), 1);
    add.AddInput(GetMul({{external_2, 1}, {GetConst(3.0), 1}}), 1);
    add.SimplifyInputs();
    FPNode add_2 = add;
    EXPECT_FALSE(graph.ExpandCoefficients(add_2));
    EXPECT_EQ(add, add_2);
  }

  // (5 * a) + (3 * b) + (2 * c) = no change
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{external_1, 1}, {GetConst(5.0), 1}}), 1);
    add.AddInput(GetMul({{external_2, 1}, {GetConst(3.0), 1}}), 1);
    add.AddInput(GetMul({{external_3, 1}, {GetConst(2.0), 1}}), 1);
    add.SimplifyInputs();
    FPNode add_2 = add;
    EXPECT_FALSE(graph.ExpandCoefficients(add_2));
    EXPECT_EQ(add, add_2);
  }

  // (5 * a) + (5 * b) + (5 * c) = no change
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{external_1, 1}, {GetConst(5.0), 1}}), 1);
    add.AddInput(GetMul({{external_2, 1}, {GetConst(5.0), 1}}), 1);
    add.AddInput(GetMul({{external_3, 1}, {GetConst(5.0), 1}}), 1);
    add.SimplifyInputs();
    FPNode add_2 = add;
    EXPECT_FALSE(graph.ExpandCoefficients(add_2));
    EXPECT_EQ(add, add_2);
  }

  // (3 * (a + b))^2 + 5 = no change
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{GetAdd({{external_1, 1}, {external_2, 1}}), 2},
                         {GetConst(3.0), 1}}),
                 1);
    add.AddInput(GetConst(5.0), 1);
    add.SimplifyInputs();
    FPNode add_2 = add;
    EXPECT_FALSE(graph.ExpandCoefficients(add_2));
    EXPECT_EQ(add, add_2);
  }

  // (3 * (a + b)) + 5 = (3 * a) + (3 * b) + 5
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{GetAdd({{external_1, 1}, {external_2, 1}}), 1},
                         {GetConst(3.0), 1}}),
                 1);
    add.AddInput(GetConst(5.0), 1);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.ExpandCoefficients(add));
    EXPECT_EQ(add, graph.MakeAdd({
                       {GetConst(5.0), 1},
                       {GetMul({{GetConst(3.0), 1}, {external_1, 1}}), 1},
                       {GetMul({{GetConst(3.0), 1}, {external_2, 1}}), 1},
                   }));
  }

  // (3 * (a + 10)) + 5 = (3 * a) + 35
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{GetAdd({{external_1, 1}, {GetConst(10.0), 1}}), 1},
                         {GetConst(3.0), 1}}),
                 1);
    add.AddInput(GetConst(5.0), 1);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.ExpandCoefficients(add));
    EXPECT_EQ(add, graph.MakeAdd({
                       {GetConst(35.0), 1},
                       {GetMul({{GetConst(3.0), 1}, {external_1, 1}}), 1},
                   }));
  }

  // (3 * (a + b + 10)) + 5 = (3 * a) + (3 * b) + 35
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(
        GetMul(
            {{GetAdd({{external_1, 1}, {external_2, 1}, {GetConst(10.0), 1}}),
              1},
             {GetConst(3.0), 1}}),
        1);
    add.AddInput(GetConst(5.0), 1);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.ExpandCoefficients(add));
    EXPECT_EQ(add, graph.MakeAdd({
                       {GetConst(35.0), 1},
                       {GetMul({{GetConst(3.0), 1}, {external_1, 1}}), 1},
                       {GetMul({{GetConst(3.0), 1}, {external_2, 1}}), 1},
                   }));
  }

  // (3 * (a + b)) + c = (3 * a) + (3 * b) + c
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{GetConst(3.0), 1},
                         {GetAdd({{external_1, 1}, {external_2, 1}}), 1}}),
                 1);
    add.AddInput(external_3, 1);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.ExpandCoefficients(add));
    EXPECT_EQ(add, graph.MakeAdd({
                       {external_3, 1},
                       {GetMul({{GetConst(3.0), 1}, {external_1, 1}}), 1},
                       {GetMul({{GetConst(3.0), 1}, {external_2, 1}}), 1},
                   }));
  }

  // (3 * (a + b)) + (2 * c) = (3 * a) + (3 * b) + (2 * c)
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{GetConst(3.0), 1},
                         {GetAdd({{external_1, 1}, {external_2, 1}}), 1}}),
                 1);
    add.AddInput(GetMul({{GetConst(2.0), 1}, {external_3, 1}}), 1);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.ExpandCoefficients(add));
    EXPECT_EQ(add, graph.MakeAdd({
                       {GetMul({{GetConst(2.0), 1}, {external_3, 1}}), 1},
                       {GetMul({{GetConst(3.0), 1}, {external_1, 1}}), 1},
                       {GetMul({{GetConst(3.0), 1}, {external_2, 1}}), 1},
                   }));
  }

  // (3 * (a + b)) + (2 * a) + (2 * b) = (3 * a) + (3 * b) + (2 * a) + (2 * b)
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{GetConst(3.0), 1},
                         {GetAdd({{external_1, 1}, {external_2, 1}}), 1}}),
                 1);
    add.AddInput(GetMul({{GetConst(2.0), 1}, {external_1, 1}}), 1);
    add.AddInput(GetMul({{GetConst(2.0), 1}, {external_2, 1}}), 1);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.ExpandCoefficients(add));
    EXPECT_EQ(add, graph.MakeAdd({
                       {GetMul({{GetConst(2.0), 1}, {external_1, 1}}), 1},
                       {GetMul({{GetConst(2.0), 1}, {external_2, 1}}), 1},
                       {GetMul({{GetConst(3.0), 1}, {external_1, 1}}), 1},
                       {GetMul({{GetConst(3.0), 1}, {external_2, 1}}), 1},
                   }));
  }

  // (3 * (a + b)) + (2 * a) + (2 * a) = (3 * a) + (3 * b) + (2 * a) + (2 * a)
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{GetConst(3.0), 1},
                         {GetAdd({{external_1, 1}, {external_2, 1}}), 1}}),
                 1);
    add.AddInput(GetMul({{GetConst(2.0), 1}, {external_1, 1}}), 2);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.ExpandCoefficients(add));
    EXPECT_EQ(add, graph.MakeAdd({
                       {GetMul({{GetConst(2.0), 1}, {external_1, 1}}), 2},
                       {GetMul({{GetConst(3.0), 1}, {external_1, 1}}), 1},
                       {GetMul({{GetConst(3.0), 1}, {external_2, 1}}), 1},
                   }));
  }

  // (5 * (a + b)) + (4 * (b + c)) = (5 * a) + (5 * b) + (4 * b) + (4 * c)
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{GetConst(5.0), 1},
                         {GetAdd({{external_1, 1}, {external_2, 1}}), 1}}),
                 1);
    add.AddInput(GetMul({{GetConst(4.0), 1},
                         {GetAdd({{external_2, 1}, {external_3, 1}}), 1}}),
                 1);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.ExpandCoefficients(add));
    EXPECT_EQ(add, graph.MakeAdd(
                       {{GetMul({{GetConst(5.0), 1}, {external_1, 1}}), 1},
                        {GetMul({{GetConst(5.0), 1}, {external_2, 1}}), 1},
                        {GetMul({{GetConst(4.0), 1}, {external_2, 1}}), 1},
                        {GetMul({{GetConst(4.0), 1}, {external_3, 1}}), 1}}));
  }

  // (5 * (a + b + 10)) + (4 * (b + c + 20))
  // = (5 * a) + (5 * b) + (4 * b) + (4 * c) + 130
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(
        GetMul({{GetConst(5.0), 1},
                {GetAdd({{external_1, 1}, {external_2, 1}, {GetConst(10), 1}}),
                 1}}),
        1);
    add.AddInput(
        GetMul({{GetConst(4.0), 1},
                {GetAdd({{external_2, 1}, {external_3, 1}, {GetConst(20), 1}}),
                 1}}),
        1);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.ExpandCoefficients(add));
    EXPECT_EQ(add, graph.MakeAdd(
                       {{GetConst(130.0), 1},
                        {GetMul({{GetConst(5.0), 1}, {external_1, 1}}), 1},
                        {GetMul({{GetConst(5.0), 1}, {external_2, 1}}), 1},
                        {GetMul({{GetConst(4.0), 1}, {external_2, 1}}), 1},
                        {GetMul({{GetConst(4.0), 1}, {external_3, 1}}), 1}}));
  }
}

TEST_F(ReassocGraphBuilderTest, TestFPMergeAddMulInputs) {
  const FPNode* external_1 = GetExternal(1);
  const FPNode* external_2 = GetExternal(2);
  const FPNode* external_3 = GetExternal(3);

  // a = no change
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(external_1, 1);
    add.SimplifyInputs();
    FPNode add_2 = add;
    EXPECT_FALSE(graph.MergeAddMulInputs(add_2));
    EXPECT_EQ(add, add_2);
  }

  // a + b = no change
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(external_1, 1);
    add.AddInput(external_2, 1);
    add.SimplifyInputs();
    FPNode add_2 = add;
    EXPECT_FALSE(graph.MergeAddMulInputs(add_2));
    EXPECT_EQ(add, add_2);
  }

  // (5 * a) + b = no change
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{external_1, 1}, {GetConst(5.0), 1}}), 1);
    add.AddInput(external_2, 1);
    add.SimplifyInputs();
    FPNode add_2 = add;
    EXPECT_FALSE(graph.MergeAddMulInputs(add_2));
    EXPECT_EQ(add, add_2);
  }

  // (5 * a) + (3 * b) = no change
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{external_1, 1}, {GetConst(5.0), 1}}), 1);
    add.AddInput(GetMul({{external_2, 1}, {GetConst(3.0), 1}}), 1);
    add.SimplifyInputs();
    FPNode add_2 = add;
    EXPECT_FALSE(graph.MergeAddMulInputs(add_2));
    EXPECT_EQ(add, add_2);
  }

  // (5 * a) + (3 * b) + (2 * c) = no change
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{external_1, 1}, {GetConst(5.0), 1}}), 1);
    add.AddInput(GetMul({{external_2, 1}, {GetConst(3.0), 1}}), 1);
    add.AddInput(GetMul({{external_3, 1}, {GetConst(2.0), 1}}), 1);
    add.SimplifyInputs();
    FPNode add_2 = add;
    EXPECT_FALSE(graph.MergeAddMulInputs(add_2));
    EXPECT_EQ(add, add_2);
  }

  // (5 * a) + (5 * b) + (5 * c) = no change
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{external_1, 1}, {GetConst(5.0), 1}}), 1);
    add.AddInput(GetMul({{external_2, 1}, {GetConst(5.0), 1}}), 1);
    add.AddInput(GetMul({{external_3, 1}, {GetConst(5.0), 1}}), 1);
    add.SimplifyInputs();
    FPNode add_2 = add;
    EXPECT_FALSE(graph.MergeAddMulInputs(add_2));
    EXPECT_EQ(add, add_2);
  }

  // a + a => (2 * a)
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(external_1, 2);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.MergeAddMulInputs(add));
    EXPECT_EQ(add, graph.MakeMul({{external_1, 1}, {GetConst(2.0), 1}}));
  }

  // a + a + b => 2 * a + b
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(external_1, 2);
    add.AddInput(external_2, 1);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.MergeAddMulInputs(add));
    EXPECT_EQ(add,
              graph.MakeAdd({{GetMul({{external_1, 1}, {GetConst(2.0), 1}}), 1},
                             {external_2, 1}}));
  }

  // a + a + b + b => (2 * a) + (2 * b)
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(external_1, 2);
    add.AddInput(external_2, 2);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.MergeAddMulInputs(add));
    EXPECT_EQ(add, graph.MakeAdd(
                       {{GetMul({{external_1, 1}, {GetConst(2.0), 1}}), 1},
                        {GetMul({{external_2, 1}, {GetConst(2.0), 1}}), 1}}));
  }

  // a + a + a => (3 * a)
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(external_1, 3);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.MergeAddMulInputs(add));
    EXPECT_EQ(add, graph.MakeMul({{external_1, 1}, {GetConst(3.0), 1}}));
  }

  // a + a + a + b => (3 * a) + b
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(external_1, 3);
    add.AddInput(external_2, 1);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.MergeAddMulInputs(add));
    EXPECT_EQ(add,
              graph.MakeAdd({{GetMul({{external_1, 1}, {GetConst(3.0), 1}}), 1},
                             {external_2, 1}}));
  }

  // a + a + a + b + b => (3 * a) + (2 * b)
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(external_1, 3);
    add.AddInput(external_2, 2);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.MergeAddMulInputs(add));
    EXPECT_EQ(add, graph.MakeAdd(
                       {{GetMul({{external_1, 1}, {GetConst(3.0), 1}}), 1},
                        {GetMul({{external_2, 1}, {GetConst(2.0), 1}}), 1}}));
  }

  // -a => (-1 * a)
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(external_1, -1);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.MergeAddMulInputs(add));
    EXPECT_EQ(add, graph.MakeMul({{external_1, 1}, {GetConst(-1.0), 1}}));
  }

  // -a - a => (-2 * a)
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(external_1, -2);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.MergeAddMulInputs(add));
    EXPECT_EQ(add, graph.MakeMul({{external_1, 1}, {GetConst(-2.0), 1}}));
  }

  // -a - a - a => (-3 * a)
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(external_1, -3);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.MergeAddMulInputs(add));
    EXPECT_EQ(add, graph.MakeMul({{external_1, 1}, {GetConst(-3.0), 1}}));
  }

  // a + a + a + b + b - c => (3 * a) + (2 * b) + (-1 * c)
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(external_1, 3);
    add.AddInput(external_2, 2);
    add.AddInput(external_3, -1);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.MergeAddMulInputs(add));
    EXPECT_EQ(add, graph.MakeAdd(
                       {{GetMul({{external_1, 1}, {GetConst(3.0), 1}}), 1},
                        {GetMul({{external_2, 1}, {GetConst(2.0), 1}}), 1},
                        {GetMul({{external_3, 1}, {GetConst(-1.0), 1}}), 1}}));
  }

  // (2 * a) + (2 * a) = (4 * a)
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{external_1, 1}, {GetConst(2.0), 1}}), 2);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.MergeAddMulInputs(add));
    EXPECT_EQ(add, graph.MakeMul({{external_1, 1}, {GetConst(4.0), 1}}));
  }

  // (2 * a) + (3 * a) = (5 * a)
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{external_1, 1}, {GetConst(3.0), 1}}), 1);
    add.AddInput(GetMul({{external_1, 1}, {GetConst(2.0), 1}}), 1);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.MergeAddMulInputs(add));
    EXPECT_EQ(add, graph.MakeMul({{external_1, 1}, {GetConst(5.0), 1}}));
  }

  // (2 * a) + (3 * a) + (3 * a) = (8 * a)
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{external_1, 1}, {GetConst(3.0), 1}}), 2);
    add.AddInput(GetMul({{external_1, 1}, {GetConst(2.0), 1}}), 1);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.MergeAddMulInputs(add));
    EXPECT_EQ(add, graph.MakeMul({{external_1, 1}, {GetConst(8.0), 1}}));
  }

  // (5 * a) + a = (6 * a)
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{external_1, 1}, {GetConst(5.0), 1}}), 1);
    add.AddInput(external_1, 1);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.MergeAddMulInputs(add));
    EXPECT_EQ(add, graph.MakeMul({{external_1, 1}, {GetConst(6.0), 1}}));
  }

  // (5 * a) + (5 * a) + a = (11 * a)
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{external_1, 1}, {GetConst(5.0), 1}}), 2);
    add.AddInput(external_1, 1);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.MergeAddMulInputs(add));
    EXPECT_EQ(add, graph.MakeMul({{external_1, 1}, {GetConst(11.0), 1}}));
  }

  // (5 * a) + (4 * b) + a = (6 * a) + (4 * b)
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{external_1, 1}, {GetConst(5.0), 1}}), 1);
    add.AddInput(GetMul({{external_2, 1}, {GetConst(4.0), 1}}), 1);
    add.AddInput(external_1, 1);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.MergeAddMulInputs(add));
    EXPECT_EQ(add, graph.MakeAdd(
                       {{GetMul({{external_1, 1}, {GetConst(6.0), 1}}), 1},
                        {GetMul({{external_2, 1}, {GetConst(4.0), 1}}), 1}}));
  }

  // (5 * a) + (3 * a) = (8 * a)
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{external_1, 1}, {GetConst(5.0), 1}}), 1);
    add.AddInput(GetMul({{external_1, 1}, {GetConst(3.0), 1}}), 1);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.MergeAddMulInputs(add));
    EXPECT_EQ(add, graph.MakeMul({{external_1, 1}, {GetConst(8.0), 1}}));
  }

  // (5 * a) + (3 * a) + a = (9 * a)
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{external_1, 1}, {GetConst(5.0), 1}}), 1);
    add.AddInput(GetMul({{external_1, 1}, {GetConst(3.0), 1}}), 1);
    add.AddInput(external_1, 1);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.MergeAddMulInputs(add));
    EXPECT_EQ(add, graph.MakeMul({{external_1, 1}, {GetConst(9.0), 1}}));
  }

  // (5 * a) + (3 * a) + (2 * a) + a = (11 * a)
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{external_1, 1}, {GetConst(5.0), 1}}), 1);
    add.AddInput(GetMul({{external_1, 1}, {GetConst(3.0), 1}}), 1);
    add.AddInput(GetMul({{external_1, 1}, {GetConst(2.0), 1}}), 1);
    add.AddInput(external_1, 1);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.MergeAddMulInputs(add));
    EXPECT_EQ(add, graph.MakeMul({{external_1, 1}, {GetConst(11.0), 1}}));
  }

  // (5 * a) + (-5 * a) = 0
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{external_1, 1}, {GetConst(5.0), 1}}), 1);
    add.AddInput(GetMul({{external_1, 1}, {GetConst(-5.0), 1}}), 1);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.MergeAddMulInputs(add));
    EXPECT_EQ(add, *GetConst(0.0));
  }

  // (5 * a) + (-4 * a) = a
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{external_1, 1}, {GetConst(5.0), 1}}), 1);
    add.AddInput(GetMul({{external_1, 1}, {GetConst(-4.0), 1}}), 1);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.MergeAddMulInputs(add));
    EXPECT_EQ(add, *external_1);
  }
}

TEST_F(ReassocGraphBuilderTest, TestFPFactorAddConstMulInputs) {
  const FPNode* external_1 = GetExternal(1);
  const FPNode* external_2 = GetExternal(2);
  const FPNode* external_3 = GetExternal(3);

  // a = no change
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(external_1, 1);
    add.SimplifyInputs();
    FPNode add_2 = add;
    EXPECT_FALSE(graph.FactorAddConstMulInputs(add_2));
    EXPECT_EQ(add, add_2);
  }

  // a + b = no change
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(external_1, 1);
    add.AddInput(external_2, 1);
    add.SimplifyInputs();
    FPNode add_2 = add;
    EXPECT_FALSE(graph.FactorAddConstMulInputs(add_2));
    EXPECT_EQ(add, add_2);
  }

  // (5 * a) + b = no change
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{external_1, 1}, {GetConst(5.0), 1}}), 1);
    add.AddInput(external_2, 1);
    add.SimplifyInputs();
    FPNode add_2 = add;
    EXPECT_FALSE(graph.FactorAddConstMulInputs(add_2));
    EXPECT_EQ(add, add_2);
  }

  // (5 * a) + (3 * b) = no change
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{external_1, 1}, {GetConst(5.0), 1}}), 1);
    add.AddInput(GetMul({{external_2, 1}, {GetConst(3.0), 1}}), 1);
    add.SimplifyInputs();
    FPNode add_2 = add;
    EXPECT_FALSE(graph.FactorAddConstMulInputs(add_2));
    EXPECT_EQ(add, add_2);
  }

  // (5 * a) + (3 * b) + (2 * c) = no change
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{external_1, 1}, {GetConst(5.0), 1}}), 1);
    add.AddInput(GetMul({{external_2, 1}, {GetConst(3.0), 1}}), 1);
    add.AddInput(GetMul({{external_3, 1}, {GetConst(2.0), 1}}), 1);
    add.SimplifyInputs();
    FPNode add_2 = add;
    EXPECT_FALSE(graph.FactorAddConstMulInputs(add_2));
    EXPECT_EQ(add, add_2);
  }

  // (5 * a) + (5 * b) = 5 * (a + b)
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{external_1, 1}, {GetConst(5.0), 1}}), 1);
    add.AddInput(GetMul({{external_2, 1}, {GetConst(5.0), 1}}), 1);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.FactorAddConstMulInputs(add));
    EXPECT_EQ(add,
              graph.MakeMul({{GetConst(5.0), 1},
                             {GetAdd({{external_1, 1}, {external_2, 1}}), 1}}));
  }

  // (5 * a) + (5 * b) + c = (5 * (a + b)) + c
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{external_1, 1}, {GetConst(5.0), 1}}), 1);
    add.AddInput(GetMul({{external_2, 1}, {GetConst(5.0), 1}}), 1);
    add.AddInput(external_3, 1);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.FactorAddConstMulInputs(add));
    EXPECT_EQ(add,
              graph.MakeAdd(
                  {{external_3, 1},
                   {GetMul({{GetConst(5.0), 1},
                            {GetAdd({{external_1, 1}, {external_2, 1}}), 1}}),
                    1}}));
  }

  // (4 * a) + (4 * b) + (4 * c) = 4 * (a + b + c)
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{external_1, 1}, {GetConst(4.0), 1}}), 1);
    add.AddInput(GetMul({{external_2, 1}, {GetConst(4.0), 1}}), 1);
    add.AddInput(GetMul({{external_3, 1}, {GetConst(4.0), 1}}), 1);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.FactorAddConstMulInputs(add));
    EXPECT_EQ(add,
              graph.MakeMul(
                  {{GetConst(4.0), 1},
                   {GetAdd({{external_1, 1}, {external_2, 1}, {external_3, 1}}),
                    1}}));
  }

  // (4 * a) + (4 * b) + (4 * c) + 50 = 4 * (a + b + c) + 50
  {
    FPNode add = graph.MakeAdd();
    add.AddInput(GetMul({{external_1, 1}, {GetConst(4.0), 1}}), 1);
    add.AddInput(GetMul({{external_2, 1}, {GetConst(4.0), 1}}), 1);
    add.AddInput(GetMul({{external_3, 1}, {GetConst(4.0), 1}}), 1);
    add.AddInput(GetConst(50.0), 1);
    add.SimplifyInputs();
    EXPECT_TRUE(graph.FactorAddConstMulInputs(add));
    EXPECT_EQ(
        add,
        graph.MakeAdd(
            {{GetConst(50.0), 1},
             {GetMul(
                  {{GetConst(4.0), 1},
                   {GetAdd({{external_1, 1}, {external_2, 1}, {external_3, 1}}),
                    1}}),
              1}}));
  }
}

TEST_F(ReassocGraphBuilderTest, TestFPPropagateConstMulAddInputs) {
  const FPNode* external_1 = GetExternal(1);
  const FPNode* external_2 = GetExternal(2);
  const FPNode* external_3 = GetExternal(3);

  // a = no change
  {
    FPNode mul = graph.MakeMul();
    mul.AddInput(external_1, 1);
    mul.SimplifyInputs();
    FPNode mul_2 = mul;
    EXPECT_FALSE(graph.PropagateConstMulAddInputs(mul_2));
    EXPECT_EQ(mul, mul_2);
  }

  // a * b = no change
  {
    FPNode mul = graph.MakeMul();
    mul.AddInput(external_1, 1);
    mul.AddInput(external_2, 1);
    mul.SimplifyInputs();
    FPNode mul_2 = mul;
    EXPECT_FALSE(graph.PropagateConstMulAddInputs(mul_2));
    EXPECT_EQ(mul, mul_2);
  }

  // (5 + a) * b = no change
  {
    FPNode mul = graph.MakeMul();
    mul.AddInput(GetAdd({{external_1, 1}, {GetConst(5.0), 1}}), 1);
    mul.AddInput(external_2, 1);
    mul.SimplifyInputs();
    FPNode mul_2 = mul;
    EXPECT_FALSE(graph.PropagateConstMulAddInputs(mul_2));
    EXPECT_EQ(mul, mul_2);
  }

  // (5 + (a * 2)) * b = no change
  {
    FPNode mul = graph.MakeMul();
    mul.AddInput(GetAdd({{GetMul({{external_1, 1}, {GetConst(2.0), 1}}), 1},
                         {GetConst(5.0), 1}}),
                 1);
    mul.AddInput(external_2, 1);
    mul.SimplifyInputs();
    FPNode mul_2 = mul;
    EXPECT_FALSE(graph.PropagateConstMulAddInputs(mul_2));
    EXPECT_EQ(mul, mul_2);
  }

  // (3 * (10 + (3 * a * b)) = 30 + (9 * a * b)
  {
    FPNode mul = graph.MakeMul();
    mul.AddInput(GetConst(3.0), 1);
    mul.AddInput(
        GetAdd({{GetConst(10.0), 1},
                {GetMul({{GetConst(3.0), 1}, {external_1, 1}, {external_2, 1}}),
                 1}}),
        1);
    mul.SimplifyInputs();
    EXPECT_TRUE(graph.PropagateConstMulAddInputs(mul));
    EXPECT_EQ(
        mul,
        graph.MakeAdd(
            {{GetMul({{GetConst(9.0), 1}, {external_1, 1}, {external_2, 1}}),
              1},
             {GetConst(30.0), 1}}));
  }

  // (3 * (10 + 3 * (a + b))) = 30 + 9 * (a + b)
  {
    FPNode mul = graph.MakeMul();
    mul.AddInput(GetConst(3.0), 1);
    mul.AddInput(
        GetAdd({{GetConst(10.0), 1},
                {GetMul({{GetConst(3.0), 1},
                         {GetAdd({{external_1, 1}, {external_2, 1}}), 1}}),
                 1}}),
        1);
    mul.SimplifyInputs();
    EXPECT_TRUE(graph.PropagateConstMulAddInputs(mul));
    EXPECT_EQ(mul,
              graph.MakeAdd(
                  {{GetMul({{GetConst(9.0), 1},
                            {GetAdd({{external_1, 1}, {external_2, 1}}), 1}}),
                    1},
                   {GetConst(30.0), 1}}));
  }

  // 2 * (10 + 3 * (a + b))) * c = (20 + 6 * (a + b)) * c
  {
    FPNode mul = graph.MakeMul();
    mul.AddInput(GetConst(2.0), 1);
    mul.AddInput(
        GetAdd({{GetConst(10.0), 1},
                {GetMul({{GetConst(3.0), 1},
                         {GetAdd({{external_1, 1}, {external_2, 1}}), 1}}),
                 1}}),
        1);
    mul.AddInput(external_3, 1);
    mul.SimplifyInputs();
    EXPECT_TRUE(graph.PropagateConstMulAddInputs(mul));
    EXPECT_EQ(
        mul,
        graph.MakeMul(
            {{external_3, 1},
             {GetAdd(
                  {{GetMul({{GetConst(6.0), 1},
                            {GetAdd({{external_1, 1}, {external_2, 1}}), 1}}),
                    1},
                   {GetConst(20.0), 1}}),
              1}}));
  }
}

}  // namespace
}  // namespace reassociate
}  // namespace opt
}  // namespace spvtools
