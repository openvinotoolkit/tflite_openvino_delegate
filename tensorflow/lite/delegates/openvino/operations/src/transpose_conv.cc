// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "delegate/intel_openvino/operations/include/transpose_conv.h"

namespace tflite {
namespace openvinodelegate {
TfLiteStatus TransposeConv::CreateNode() {
  const TfLiteTransposeConvParams *transpose_conv_params =
      (TfLiteTransposeConvParams *)GetBuiltinData();
  std::shared_ptr<ov::Node> weights_node = nullptr;
  std::shared_ptr<ov::Node> input_node = nullptr;
  if (!isConvolution2dTransposeBias) {
    weights_node = getInputNode(tensor_indices_[TRANSPOSE_CONV_WEIGHTS]);
    input_node = getInputNode(tensor_indices_[TRANSPOSE_CONV_INPUT]);
  } else {
    input_node = getInputNode(tensor_indices_[0]);
    weights_node = getInputNode(tensor_indices_[1]);
  }
  bool has_bias = false;
  std::shared_ptr<ov::Node> bias_node = nullptr;
  if (!isConvolution2dTransposeBias) {
    if (tensor_indices_size_ >= 4) {
      bias_node = getInputNode(tensor_indices_[TRANSPOSE_CONV_BIAS]);
      has_bias = true;
    }
  } else {
    bias_node = getInputNode(tensor_indices_[2]);
    has_bias = true;
  }
  std::vector<size_t> strides = {(size_t)transpose_conv_params->stride_height,
                                 (size_t)transpose_conv_params->stride_width};
  size_t dilation_width_factor = 1.0, dilation_height_factor = 1.0;
  std::vector<size_t> dilations = {dilation_height_factor,
                                   dilation_width_factor};
  ov::op::PadType auto_pad = ov::op::PadType::SAME_UPPER;
  int padding_top = 0, padding_bottom = 0, padding_left = 0, padding_right = 0;

  if (transpose_conv_params->padding == kTfLitePaddingUnknown) {
  } else if (transpose_conv_params->padding == kTfLitePaddingSame) {
    auto_pad = ov::op::PadType::SAME_UPPER;
  } else if (transpose_conv_params->padding == kTfLitePaddingValid) {
    auto_pad = ov::op::PadType::VALID;
  }

  std::vector<std::ptrdiff_t> padding_begin = {padding_top, padding_left};
  std::vector<std::ptrdiff_t> padding_end = {padding_bottom, padding_right};

  // TODO: Possibly add a test compilation flag to change the order for test
  // mode
  //  Comment below transpose lines for test mode
  ov::AxisVector order = {3, 0, 1, 2};
  const auto order_node = ov::opset3::Constant::create(
      ov::element::i64, ov::Shape{order.size()}, order);
  weights_node =
      std::make_shared<ov::opset3::Transpose>(weights_node, order_node);

  std::shared_ptr<ov::Node> transpose_conv_node = nullptr;
  size_t spatial_dimensions_size = 2;
  int32_t output_shape[4];
  std::vector<int32_t> spatial_dimensions(spatial_dimensions_size);
  if (!isConvolution2dTransposeBias) {
    GetTensorData(tensor_indices_[0], &output_shape);
    spatial_dimensions[0] = output_shape[1];
    spatial_dimensions[1] = output_shape[2];
  } else {
    std::vector<int32_t> conv_output_shape = GetDims(tensor_indices_[0]);
    spatial_dimensions[0] = conv_output_shape[1] * 2;
    spatial_dimensions[1] = conv_output_shape[2] * 2;
  }
  auto output_shape_node = std::make_shared<ov::opset8::Constant>(
      ov::element::i32, ov::Shape{spatial_dimensions_size}, spatial_dimensions);

  transpose_conv_node = std::make_shared<ov::opset3::ConvolutionBackpropData>(
      input_node, weights_node, output_shape_node, ov::Strides(strides),
      ov::CoordinateDiff(padding_begin), ov::CoordinateDiff(padding_end),
      ov::Strides(dilations), auto_pad);

  if (has_bias) {
    std::vector<int> bias_dims;
    if (!isConvolution2dTransposeBias) {
      bias_dims = GetDims(tensor_indices_[TRANSPOSE_CONV_BIAS]);
    } else {
      bias_dims = GetDims(tensor_indices_[2]);
    }
    std::vector<uint32_t> shape(transpose_conv_node->get_shape().size(), 1);
    shape[1] = bias_dims[0];
    auto shape_node =
        CreateConstNode(ov::element::i32, ov::Shape{shape.size()}, shape);
    bias_node =
        std::make_shared<ov::opset3::Reshape>(bias_node, shape_node, true);
    output_node = std::make_shared<ov::opset3::Add>(
        transpose_conv_node, bias_node, ov::op::AutoBroadcastType::NUMPY);
  } else {
    output_node = transpose_conv_node;
  }
  if (!isConvolution2dTransposeBias) {
    output_node =
        ApplyActivation(output_node, transpose_conv_params->activation);
  }
  return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
