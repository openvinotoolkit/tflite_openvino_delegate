// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_RELU6_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_RELU6_H_

#include "delegate/intel_openvino/operations/operations_base.h"

namespace tflite {
namespace openvinodelegate {

class Relu6 : public OperationsBase {
 public:
  Relu6(int operationIndex) {}
  TfLiteStatus CreateNode() override;
};

}  // namespace openvinodelegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_RELU6_H_
