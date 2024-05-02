// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_SOFTMAX_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_SOFTMAX_H_

#include "delegate/intel_openvino/operations/operations_base.h"

namespace tflite {
namespace openvinodelegate {

class Softmax : public OperationsBase {
 public:
  Softmax(int operationIndex) {}
  TfLiteStatus CreateNode() override;
};

}  // namespace openvinodelegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_SOFTMAX_H_
