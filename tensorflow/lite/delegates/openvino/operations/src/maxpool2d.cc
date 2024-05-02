// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "delegate/intel_openvino/operations/include/maxpool2d.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus MaxPool2D::CreateNode() {
  const TfLitePoolParams *maxpool2d_params =
      (TfLitePoolParams *)GetBuiltinData();

  auto input_node = getInputNode(tensor_indices_[INPUT_NODE_1]);

  std::vector<size_t> strides{(size_t)maxpool2d_params->stride_height,
                              (size_t)maxpool2d_params->stride_width};

  // will be ignored since auto_pad is specified
  std::vector<std::size_t> padding_begin = {0, 0};
  std::vector<std::size_t> padding_end = {0, 0};
  // ToDo: According to tf release note:
  // `tf.nn.max_pool2d` now supports explicit padding.
  // need to support.

  std::vector<size_t> kernel{(size_t)maxpool2d_params->filter_height,
                             (size_t)maxpool2d_params->filter_width};

  // Set padding scheme to PadType::VALID for valid or unknown
  ov::op::PadType auto_pad = (maxpool2d_params->padding == kTfLitePaddingSame)
                                 ? ov::op::PadType::SAME_UPPER
                                 : ov::op::PadType::VALID;

  auto maxpool2d_node = std::make_shared<ov::opset3::MaxPool>(
      input_node, ov::Strides(strides), ov::Shape(padding_begin),
      ov::Shape(padding_end), ov::Shape(kernel), ov::op::RoundingType::FLOOR,
      auto_pad);

  output_node = ApplyActivation(maxpool2d_node, maxpool2d_params->activation);
  return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
