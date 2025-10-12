// Copyright (c) 2019 Google LLC
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

#include "aggressive_folding.h"

#include <vector>

#include "source/opt/instruction.h"
#include "source/opt/ir_context.h"
#include "source/util/bit_vector.h"

namespace spvtools {
namespace opt {
namespace {

struct AFPOpTracking
{
  union Bits {
    uint32_t u32;
    int32_t  i32;
    float    f32;
    bool     b;

    int64_t i64() const { return i32; }
    uint64_t u64() const { return u32; }
    double f64() const { return f32; }
  };

  struct PerComponent {
    
    bool valid = false;
    Bits min_value;
    Bits max_value;
    Bits signed_bits;
    Bits unsigned_bits;

    void Invalidate() {
      valid = false;
      min_value.u32 = 0u;
      max_value.u32 = 0xffffffffu;
      signed_bits.u32 = 0u;
      unsigned_bits.u32 = 0u;
    }

    static PerComponent InitConstant(uint32_t constant) {
      PerComponent pc {};
      pc.valid = true;
      pc.min_value.u32 = constant;
      pc.max_value.u32 = constant;
      pc.signed_bits.u32 = constant;
      pc.unsigned_bits.u32 = ~constant;
      return pc;
    }
    static PerComponent InitConstant(float constant) {
      PerComponent pc {};
      pc.valid = true;
      pc.min_value.f32 = constant;
      pc.max_value.f32 = constant;
      pc.signed_bits.f32 = constant;
      pc.unsigned_bits.u32 = ~pc.signed_bits.u32;
      return pc;
    }
    static PerComponent InitConstant(bool constant) {
      PerComponent pc {};
      pc.valid = true;
      pc.min_value.b = constant;
      pc.max_value.b = constant;
      pc.signed_bits.b = constant;
      pc.unsigned_bits.b = !constant;
      return pc;
    }

    bool IsConstant(spv::Op underlying_type) const {
      switch (underlying_type) {
      case spv::Op::OpTypeBool: return min_value.b == max_value.b;
      case spv::Op::OpTypeInt: return min_value.u32 == max_value.u32;
      case spv::Op::OpTypeFloat: return min_value.f32 == max_value.f32;
      }
      return false;
    }

    void SetI32(int64_t new_min_value, int64_t new_max_value) {
      Invalidate();
      if (new_min_value > new_max_value) {
        std::swap(new_min_value, new_min_value);
      }
      // Only keep tracking if no over/underflows occur
      if ((new_min_value >= INT32_MIN) && (new_max_value <= INT32_MAX)) {
        valid = true;
        min_value.i32 = new_min_value;
        max_value.i32 = new_max_value;
        if (IsConstant(spv::Op::OpTypeInt)) {
          signed_bits.u32 = min_value.u32;
          unsigned_bits.u32 = ~signed_bits.u32;
        }
      }
    }

    void SetU32(uint64_t new_min_value, uint64_t new_max_value) {
      Invalidate();
      if (new_min_value > new_max_value) {
        std::swap(new_min_value, new_min_value);
      }
      // Only keep tracking if no overflows occur
      if (new_max_value <= UINT32_MAX) {
        valid = true;
        min_value.u32 = new_min_value;
        max_value.u32 = new_max_value;
        if (IsConstant(spv::Op::OpTypeInt)) {
          signed_bits.u32 = min_value.u32;
          unsigned_bits.u32 = ~signed_bits.u32;
        }
      }
    }

    void SetF32(double new_min_value, double new_max_value) {
      Invalidate();
      if (new_min_value > new_max_value) {
        std::swap(new_min_value, new_min_value);
      }
      // Only keep tracking if no over/underflows occur
      if ((new_min_value >= FLT_MIN)
          && (new_max_value <= FLT_MAX)
          && !isinf(new_min_value)
          && !isinf(new_max_value)
          && !isnan(new_min_value)
          && !isnan(new_max_value)) {
        valid = true;
        min_value.f32 = new_min_value;
        max_value.f32 = new_max_value;
        if (IsConstant(spv::Op::OpTypeFloat)) {
          signed_bits.u32 = min_value.u32;
          unsigned_bits.u32 = ~signed_bits.u32;
        }
      }
    }

#define I32_OP(name, op)\
    PerComponent name (const PerComponent& B) const {\
    PerComponent pc {};\
    if (valid && B.valid) {\
      pc.SetI32(min_value.i64() op B.min_value.i64(),\
                max_value.i64() op B.max_value.i64());\
    }\
    return pc;\
    }
    I32_OP(iadd, +)
    I32_OP(isub, -)
    I32_OP(imul, *)
#undef I32_OP

#define U32_OP(name, op)\
    PerComponent name (const PerComponent& B) const {\
      PerComponent pc {};\
      if (valid && B.valid) {\
        pc.SetU32(min_value.u64() op B.min_value.u64(),\
                  max_value.u64() op B.max_value.u64());\
      }\
      return pc;\
    }
    U32_OP(uadd, +)
    U32_OP(usub, -)
    U32_OP(umul, *)
#undef U32_OP

    PerComponent fadd(const PerComponent& B) const {
      PerComponent pc {};
      if (valid && B.valid) {
        pc.SetF32(min_value.f64() + B.min_value.f64(),
                  max_value.f64() + B.max_value.f64());
      }
      return pc;
    }
  };

  /*
      uint32_t component_type_id = type_inst->GetSingleWordInOperand(0);
    Instruction* def_component_type =
        context_->get_def_use_mgr()->GetDef(component_type_id);
    return def_component_type != nullptr &&
           IsFoldableScalarType(def_component_type);
  */

  // If num_components != 0, we're assuming we're working with a OpTypeVector.
  // Not currently attempting to do anything fancy with matrices.
  uint32_t num_components = 0u;

  // The underlying data type (e.g spv::Op::OpTypeFloat)
  spv::Op underlying_type = spv::Op::OpNop;


  bool InitialiseFromConstant(const Instruction* inst) {
    switch (inst->opcode()) {
    case spv::Op::OpConstantTrue:
      num_components = 0u;
      underlying_type = spv::Op::OpTypeBool;
      return true;
    case spv::Op::OpConstantFalse:
      num_components = 0u;
      underlying_type = spv::Op::OpTypeBool;
      return true;
    }

    return false;
  }


};

}  // namespace

Pass::Status AggressiveFoldingPass::Process() {
  bool modified = false;

   for (Function& function : *get_module()) {
     cfg()->ForEachBlockInPostOrder(function.entry().get(),
                                    [&modified, this](BasicBlock* bb) {
                                    });
   }

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}


}  // namespace opt
}  // namespace spvtools
