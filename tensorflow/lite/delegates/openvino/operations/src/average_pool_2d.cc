// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "delegate/intel_openvino/operations/include/average_pool_2d.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus AveragePool2D::CreateNode() {
  TfLitePoolParams *avg_pool_params = (TfLitePoolParams *)GetBuiltinData();
  auto input_node = getInputNode(tensor_indices_[INPUT_NODE_1]);
  if (input_node == nullptr) {
    TFLITE_LOG(ERROR) << "input node is null\n";
    return kTfLiteError;
  }

  std::vector<size_t> strides;
  std::vector<size_t> kernel;
  std::vector<size_t> padding_begin, padding_end;
  ov::op::PadType auto_pad;
  size_t padding_top, padding_bottom, padding_left, padding_right = 0;

  TfLiteStatus tf_status = CalculatePadding(avg_pool_params->padding, auto_pad);
  if (tf_status == kTfLiteError) {
    TFLITE_LOG(ERROR) << "Invalid Padding\n";
    return kTfLiteError;
  }

  strides = {(size_t)avg_pool_params->stride_height,
             (size_t)avg_pool_params->stride_width};
  kernel = {(size_t)avg_pool_params->filter_height,
            (size_t)avg_pool_params->filter_width};
  padding_begin = {padding_top, padding_left};
  padding_end = {padding_bottom, padding_right};

  auto average_pool_2d_node = std::make_shared<ov::opset8::AvgPool>(
      input_node, ov::Strides(strides), ov::Shape(padding_begin),
      ov::Shape(padding_end), ov::Shape(kernel), true,
      ov::op::RoundingType::FLOOR, auto_pad);
  output_node =
      ApplyActivation(average_pool_2d_node, avg_pool_params->activation);

  return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
