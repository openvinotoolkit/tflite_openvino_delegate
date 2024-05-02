// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "delegate/intel_openvino/operations/include/dequantize.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus Dequantize::CreateNode() {
  auto inputNode = getInputNode(tensor_indices_[0]);
  if (inputNode == nullptr) {
    TFLITE_LOG(INFO) << "input node  is null\n";
    return kTfLiteError;
  }

  output_node =
      std::make_shared<ov::opset8::Convert>(inputNode, ov::element::f32);

  return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
