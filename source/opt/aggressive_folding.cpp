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

// Only handle 32bits
// * Floats are mostly simple (?)
//
// * Integers can be treated as unsigned and signed
//   so the valid range needs to be handled carefully.
struct OpTrackingScalarValues
{
  union Bits {
    uint32_t u32;
    int32_t  i32;
    float    f32;
  };

  std::vector<uint32_t>              possible_refs;
  std::vector<std::pair<Bits, Bits>> possible_val_ranges;

  bool HasData() const {
    return !( possible_refs.empty()
      && possible_val_ranges.empty());
  }

  bool HasRefData() const {
    return !possible_refs.empty();
  }

  bool HasValueData() const {
    return !possible_val_ranges.empty();
  }

  // Merge or add an int range
  // Expects A < B in unsigned
  void MergeIntRange(Bits A, Bits B) {
    // Attempt to merge with existing values first
    auto it = possible_val_ranges.begin();
    while (it < possible_val_ranges.end()) {
      Bits C = it->first;
      Bits D = it->second;
      if ((C.u32 >= A.u32) && (C.u32 <= B.u32)) {
        A.u32 = std::min(A.u32, C.u32);
        B.u32 = std::min(B.u32, D.u32);
        it = possible_val_ranges.erase(it);
      }
      else {
        ++it;
      }
    }
    possible_val_ranges.push_back({ A, B });
  }

  // Push an int range to the stack, if B < A, we're assuming an under/overflow
  // has occured and split it into two ranges [0, B] and [A, U32MAX]
  void PushIntRange(Bits A, Bits B) {
    if (A.u32 > B.u32) {
      MergeIntRange(Bits{ 0 }, B);
      MergeIntRange(A, Bits{ UINT32_MAX });
    }
    else {
      MergeIntRange(A, B);
    }
  }

  // If we have an int range which is [0, UINTMAX], then practically
  // speaking, we know nothing and should just mark this as having no
  // data.
  void RemoveDataForUnboundedInt() {
    for (const auto& p : possible_val_ranges) {
      if (p.first.u32 == 0 && p.second.u32 == UINT32_MAX) {
        possible_val_ranges.clear();
        break;
      }
    }
  }

  OpTrackingScalarValues uadd(const OpTrackingScalarValues& b) const {
    OpTrackingScalarValues v;
    if (HasValueData() && b.HasValueData()) {
      for (const std::pair<Bits, Bits>& a_bits : possible_val_ranges) {
        for (const std::pair<Bits, Bits>& b_bits : b.possible_val_ranges) {
          Bits b00 = { a_bits.first.u32 - b_bits.first.u32 };
          Bits b01 = { a_bits.second.u32 - b_bits.first.u32 };
          Bits b10 = { a_bits.first.u32 - b_bits.second.u32 };
          Bits b11 = { a_bits.second.u32 - b_bits.second.u32 };
          v.PushIntRange(b00, b11);
          // Extra data added to account for over/underflow.
          v.PushIntRange(b00, b01);
          v.PushIntRange(b10, b11);
        }
      }
      v.RemoveDataForUnboundedInt();
    }
    return v;
  }

  OpTrackingScalarValues usub(const OpTrackingScalarValues& b) const {
    
    OpTrackingScalarValues v;

    if (possible_refs.size() == 1
      && b.possible_refs.size() == 1
      && possible_refs[0] == b.possible_refs[0]) {
      v.possible_val_ranges.clear();
      v.possible_val_ranges.push_back({ Bits{0}, Bits{0} });
      return v;
    }

    if (HasValueData() && b.HasValueData()) {
      for (const std::pair<Bits, Bits>& a_bits : possible_val_ranges) {
        for (const std::pair<Bits, Bits>& b_bits : b.possible_val_ranges) {
          Bits b00 = { a_bits.first.u32 - b_bits.first.u32 };
          Bits b01 = { a_bits.second.u32 - b_bits.first.u32 };
          Bits b10 = { a_bits.first.u32 - b_bits.second.u32 };
          Bits b11 = { a_bits.second.u32 - b_bits.second.u32 };
          v.PushIntRange(b00, b11);
          // Extra data added to account for over/underflow.
          v.PushIntRange(b00, b01);
          v.PushIntRange(b10, b11);
        }
      }
      v.RemoveDataForUnboundedInt();
    }

    return v;
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
