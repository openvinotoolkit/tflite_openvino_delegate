// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "delegate/intel_openvino/operations/include/conv2d.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus Conv2D::CreateNode() {
  const TfLiteConvParams *conv2d_params = (TfLiteConvParams *)GetBuiltinData();
  std::vector<int> filter_dims = GetDims(tensor_indices_[FILTER_NODE]);
  std::vector<size_t> strides;
  std::vector<std::ptrdiff_t> padding_begin, padding_end;
  std::vector<size_t> dilations;
  ov::op::PadType auto_pad;
  int filter_size = 0;
  int padding_top, padding_bottom, padding_left, padding_right = 0;

  TfLiteStatus status = CalculatePadding(conv2d_params->padding, auto_pad);
  if (status != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Invalid padding type in conv2d\n";
    return kTfLiteError;
  }

  strides = {(size_t)conv2d_params->stride_height,
             (size_t)conv2d_params->stride_width};
  padding_begin = {padding_top, padding_left};
  padding_end = {padding_bottom, padding_right};
  dilations = {(size_t)conv2d_params->dilation_height_factor,
               (size_t)conv2d_params->dilation_width_factor};
  auto input_node = getInputNode(tensor_indices_[INPUT_NODE_1]);
  auto filter_node = getInputNode(tensor_indices_[FILTER_NODE]);
  auto bias_node = getInputNode(tensor_indices_[BIAS_NODE]);

  if (!GetGraphNodeManager()->isIndexAParam(tensor_indices_[FILTER_NODE])) {
    ov::AxisVector order = {0, 3, 1, 2};
    const auto order_node = ov::opset3::Constant::create(
        ov::element::i64, ov::Shape{order.size()}, order);
    filter_node =
        std::make_shared<ov::opset3::Transpose>(filter_node, order_node);
  }

  auto conv_node = std::make_shared<ov::opset8::Convolution>(
      input_node, filter_node, ov::Strides(strides),
      ov::CoordinateDiff(padding_begin), ov::CoordinateDiff(padding_end),
      ov::Strides(dilations), auto_pad);
  auto bias_dims = GetDims(tensor_indices_[BIAS_NODE]);
  std::vector<uint32_t> shape(conv_node->get_shape().size(), 1);
  shape[1] = bias_dims[0];
  auto shape_node =
      CreateConstNode(ov::element::i32, ov::Shape{shape.size()}, shape);

  bias_node =
      std::make_shared<ov::opset3::Reshape>(bias_node, shape_node, true);

  output_node = std::make_shared<ov::opset3::Add>(
      conv_node, bias_node, ov::op::AutoBroadcastType::NUMPY);

  output_node = ApplyActivation(output_node, conv2d_params->activation);
  return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
