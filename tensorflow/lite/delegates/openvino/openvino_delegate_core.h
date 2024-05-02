// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_DELEGATE_CORE_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_DELEGATE_CORE_H_
#include <iostream>
#include <openvino/openvino.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/serialize.hpp>
#include <openvino/runtime/core.hpp>
#include <vector>

#include "openvino_graph_builder.h"
#include "operations/openvino_node_manager.h"

namespace tflite {
namespace openvinodelegate {
class OpenVINODelegateCore {
 public:
  OpenVINODelegateCore(std::string_view plugins_path)
      : openvino_delegate_core_(ov::Core()) {
    plugins_location_ = plugins_path;
  }
  TfLiteStatus OpenVINODelegateInit() {
    std::vector<std::string> ov_devices =
        openvino_delegate_core_.get_available_devices();
    if (std::find(ov_devices.begin(), ov_devices.end(), "CPU") ==
        ov_devices.end()) {
      return kTfLiteDelegateError;
    } else {
      return kTfLiteOk;
    }
  }

  std::vector<int> getComputeInputs() { return compute_inputs_; }

  std::vector<int> getOutputs() { return outputs_; }

  ov::InferRequest getInferRequest() const { return infer_request_; }

  TfLiteStatus CreateGraphfromTfLite(TfLiteOpaqueContext *context,
                                     const TfLiteOpaqueDelegateParams *params);

 private:
  std::unique_ptr<OpenVINOGraphBuilder> openvino_graph_builder_;
  ov::Core openvino_delegate_core_;
  std::string plugins_location_;
  std::shared_ptr<ov::Model> model_;
  ov::CompiledModel compiled_model_;
  std::string ov_device_ = "CPU";
  std::vector<int> compute_inputs_ = {};
  std::vector<int> outputs_ = {};
  ov::InferRequest infer_request_;
};
}  // namespace openvinodelegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_DELEGATE_KERNEL_H_
