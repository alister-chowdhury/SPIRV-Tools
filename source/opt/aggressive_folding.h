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

#ifndef SOURCE_OPT_AGGRESSIVE_FOLDING_H_
#define SOURCE_OPT_AGGRESSIVE_FOLDING_H_

#include <unordered_map>
#include <unordered_set>

// TODO Remove
#include <sstream>

#include "source/opt/ir_context.h"
#include "source/opt/module.h"
#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// TODO DESC

class AggressiveFoldingPass : public Pass {
 public:
  const char* name() const override { return "aggressive-folding"; }
  Status Process() override;

  // Return the mask of preserved Analyses.
  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDefUse |
           IRContext::kAnalysisInstrToBlockMapping |
           IRContext::kAnalysisCombinators | IRContext::kAnalysisCFG |
           IRContext::kAnalysisDominatorAnalysis |
           IRContext::kAnalysisLoopAnalysis | IRContext::kAnalysisNameMap |
           IRContext::kAnalysisConstants | IRContext::kAnalysisTypes;
  }

 private:
  struct RefOrValue {
    enum AvailableFlags {
      kAvailableNone = 0 << 0,
      kAvailableRef = 1 << 0,
      kAvailableInt32 = 1 << 1,
      kAvailableFloat32 = 1 << 2,
    };

    AvailableFlags flags = kAvailableNone;
    uint32_t ref = 0;

    union {
      int32_t i32;
      uint32_t u32;
      float f32;
    };

    bool HasConstantValue() const { return (flags & ~kAvailableRef) != 0; }
    bool HasConstantI32() const { return (flags & kAvailableInt32) != 0; }
    bool HasConstantF32() const { return (flags & kAvailableFloat32) != 0; }
    bool HasRef() const { return (flags & kAvailableRef) != 0; }

    // TODO: Remove
    std::string PrettyPrint() const {
      std::stringstream ss;
      ss << "[";
      if (HasConstantI32()) {
        ss << "int s=" << i32 << ", u=" << u32 << ";";
      }
      if (HasConstantF32()) {
        ss << "float =" << f32 << ";";
      }
      if (HasRef()) {
        ss << "ref=" << ref;
      }
      ss << "]";
      return ss.str();
    }
  };

  struct InstructionMeta {
    RefOrValue min_;
    RefOrValue max_;
    RefOrValue signedbits;
    RefOrValue unsignedbits;

    std::string PrettyPrint() const {
      std::stringstream ss;
      ss << "{min=" << min_.PrettyPrint() << ",max=" << max_.PrettyPrint()
         << ",sbt=" << signedbits.PrettyPrint()
         << ",ubt=" << unsignedbits.PrettyPrint() << "}";
      return ss.str();
    }

    void SetConstant(uint32_t value) {
      min_.u32 = value;
      min_.flags = RefOrValue::kAvailableInt32;
      max_ = min_;
      signedbits = min_;
      unsignedbits = signedbits;
      unsignedbits.u32 = ~unsignedbits.u32;
    }

    void SetConstant(float value) {
      min_.f32 = value;
      min_.flags = RefOrValue::kAvailableFloat32;
      max_ = min_;
      signedbits = min_;
      signedbits.flags = RefOrValue::kAvailableInt32;
      unsignedbits = signedbits;
      unsignedbits.u32 = ~unsignedbits.u32;
    }

    // Test if both the min and max converge to a constant known value
    bool CanCollapseToConstant() const {
      if ((min_.flags & max_.flags & ~RefOrValue::kAvailableRef) != 0) {
        return min_.HasConstantF32() ? min_.f32 == max_.f32
                                     : min_.u32 == max_.u32;
      }
      return false;
    }

    // Test if both the min and max converge to a constant known ref
    bool CanCollapseToRef() const {
      return min_.HasRef() && max_.HasRef() && (min_.ref == max_.ref);
    }
  };

  using MetadataSet = std::unordered_map<uint32_t, InstructionMeta>;

  void ProcessConstants(MetadataSet& metadata);
  bool ProcessInstructionsInBB(BasicBlock* bb, MetadataSet& metadata);
  bool ProcessInstruction(Instruction* inst, MetadataSet& metadata);
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_AGGRESSIVE_FOLDING_H_
