// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_TRANSPOSE_CONV_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_TRANSPOSE_CONV_H_

#include "delegate/intel_openvino/operations/operations_base.h"

namespace tflite {
namespace openvinodelegate {

class TransposeConv : public OperationsBase {
 public:
  TransposeConv(int operationIndex) {}
  TfLiteStatus CreateNode() override;
  void SetCustom(bool isCustom) { isConvolution2dTransposeBias = true; }

 private:
  bool isConvolution2dTransposeBias = false;
};

}  // namespace openvinodelegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_TRANSPOSE_CONV_H_
