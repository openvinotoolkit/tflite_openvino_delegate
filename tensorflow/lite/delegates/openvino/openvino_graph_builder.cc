// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino_graph_builder.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus OpenVINOGraphBuilder::CreateNodeFromTfLiteOp(
    int node_id, TfLiteRegistrationExternal *registration,
    TfLiteOpaqueNode *node, TfLiteOpaqueContext *context) {
  if (node_id < 0) return kTfLiteError;
  if (registration == nullptr || node == nullptr || context == nullptr)
    return kTfLiteError;

  std::shared_ptr<OperationsBase> operation_node;
  TfLiteStatus node_status =
      CreateOpClass(node_id, registration, operation_node);
  if (node_status != kTfLiteOk || !operation_node) return kTfLiteError;
  operation_node->SetGraphData(context, node_manager_.get());

  const int *inputs_data;
  int num_inputs;
  TfLiteStatus status = TfLiteOpaqueNodeInputs(node, &inputs_data, &num_inputs);
  if (status != kTfLiteOk) return status;
  if (TfLiteRegistrationExternalGetBuiltInCode(registration) ==
          kTfLiteBuiltinCustom &&
      (strcmp(TfLiteRegistrationExternalGetCustomName(registration),
              "Convolution2DTransposeBias") == 0)) {
    const void *init_data;
    int size;
    if (TfLiteOpaqueNodeGetCustomInitialData(node, &init_data, &size) !=
        kTfLiteOk) {
      return kTfLiteDelegateError;
    }
    operation_node->UpdateNodeInfo((void *)inputs_data, num_inputs,
                                   (void *)init_data);
  } else {
    operation_node->UpdateNodeInfo((void *)inputs_data, num_inputs,
                                   TfLiteOpaqueNodeGetBuiltinData(node));
  }
  if (operation_node->CreateNode() != kTfLiteOk)
    return kTfLiteError;
  else {
    std::shared_ptr<ov::Node> result_node = operation_node->GetOpResultNode();
    if (result_node == nullptr) return kTfLiteError;

    const int *outputs;
    int num_outputs;
    TfLiteStatus tf_status =
        TfLiteOpaqueNodeOutputs(node, &outputs, &num_outputs);
    if (tf_status != kTfLiteOk) return tf_status;
    node_manager_->setOutputAtOperandIndex(outputs[0], result_node);

    return kTfLiteOk;
  }
}

TfLiteStatus OpenVINOGraphBuilder::CreateOpClass(
    int operationIndex, TfLiteRegistrationExternal *registration,
    std::shared_ptr<OperationsBase> &op_base) {
  if (operationIndex < 0) return kTfLiteError;
  if (registration == nullptr) return kTfLiteError;

  switch (TfLiteRegistrationExternalGetBuiltInCode(registration)) {
    case kTfLiteBuiltinAdd: {
      op_base = std::make_shared<Add>(operationIndex);
      return kTfLiteOk;
    }
    case kTfLiteBuiltinAveragePool2d: {
      op_base = std::make_shared<AveragePool2D>(operationIndex);
      return kTfLiteOk;
    }
    case kTfLiteBuiltinConv2d: {
      op_base = std::make_shared<Conv2D>(operationIndex);
      return kTfLiteOk;
    }
    case kTfLiteBuiltinConcatenation: {
      op_base = std::make_shared<Concat>(operationIndex);
      return kTfLiteOk;
    }
    case kTfLiteBuiltinCustom: {
      if (strcmp(TfLiteRegistrationExternalGetCustomName(registration),
                 "Convolution2DTransposeBias") == 0) {
        auto transpose_conv = std::make_shared<TransposeConv>(operationIndex);
        transpose_conv->SetCustom(true);
        op_base = transpose_conv;
        return kTfLiteOk;
      }
      return kTfLiteError;
    }
    case kTfLiteBuiltinDepthwiseConv2d: {
      op_base = std::make_shared<DepthwiseConv2D>(operationIndex);
      return kTfLiteOk;
    }
    case kTfLiteBuiltinDequantize: {
      op_base = std::make_shared<Dequantize>(operationIndex);
      return kTfLiteOk;
    }
    case kTfLiteBuiltinMul: {
      op_base = std::make_shared<Mul>(operationIndex);
      return kTfLiteOk;
    }
    case kTfLiteBuiltinResizeBilinear: {
      op_base = std::make_shared<ResizeBilinear>(operationIndex);
      return kTfLiteOk;
    }
    case kTfLiteBuiltinRelu: {
      op_base = std::make_shared<Relu>(operationIndex);
      return kTfLiteOk;
    }
    case kTfLiteBuiltinRelu6: {
      op_base = std::make_shared<Relu6>(operationIndex);
      return kTfLiteOk;
    }
    case kTfLiteBuiltinLogistic: {
      op_base = std::make_shared<Logistic>(operationIndex);
      return kTfLiteOk;
    }
    case kTfLiteBuiltinHardSwish: {
      op_base = std::make_shared<HardSwish>(operationIndex);
      return kTfLiteOk;
    }
    case kTfLiteBuiltinSoftmax: {
      op_base = std::make_shared<Softmax>(operationIndex);
      return kTfLiteOk;
    }
    case kTfLiteBuiltinTanh: {
      op_base = std::make_shared<Tanh>(operationIndex);
      return kTfLiteOk;
    }
    case kTfLiteBuiltinReshape: {
      op_base = std::make_shared<Reshape>(operationIndex);
      return kTfLiteOk;
    }
    case kTfLiteBuiltinMaxPool2d: {
      op_base = std::make_shared<MaxPool2D>(operationIndex);
      return kTfLiteOk;
    }
    case kTfLiteBuiltinMean: {
      op_base = std::make_shared<Mean>(operationIndex);
      return kTfLiteOk;
    }
    case kTfLiteBuiltinTransposeConv: {
      op_base = std::make_shared<TransposeConv>(operationIndex);
      return kTfLiteOk;
    }
    default:
      op_base = nullptr;
      return kTfLiteError;
  }
}

}  // namespace openvinodelegate
}  // namespace tflite
