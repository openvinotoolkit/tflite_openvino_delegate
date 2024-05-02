// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "delegate/intel_openvino/operations/include/add.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus Add::CreateNode() {
  TfLiteAddParams *add_params = (TfLiteAddParams *)GetBuiltinData();
  auto input_node_1 = getInputNode(tensor_indices_[INPUT_NODE_1]);
  if (input_node_1 == nullptr) {
    TFLITE_LOG(INFO) << "input node 1 is null\n";
    return kTfLiteError;
  }
  auto input_node_2 = getInputNode(tensor_indices_[INPUT_NODE_2]);
  if (input_node_2 == nullptr) {
    TFLITE_LOG(INFO) << "input Node 2 is null\n";
    return kTfLiteError;
  }

  auto add_node = std::make_shared<ov::opset8::Add>(
      input_node_1, input_node_2, ov::op::AutoBroadcastType::NUMPY);
  output_node = ApplyActivation(add_node, add_params->activation);
  return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
