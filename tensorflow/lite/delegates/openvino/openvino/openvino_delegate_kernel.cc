/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "openvino_delegate_kernel.h"

#include "openvino_delegate_core.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus OpenVINODelegateKernel::Init(TfLiteOpaqueContext *context,
                                          const TfLiteOpaqueDelegateParams *params) {
    TFLITE_LOG(INFO) << "Openvino delegate version 2.15.2v "
                     << "\n";
    // Should we do some NPU Init here.
    TfLiteStatus init_status = ov_delegate_core_->OpenVINODelegateInit();
    if (init_status != kTfLiteOk) {
        return init_status;
    }

    TfLiteStatus set_status = ov_delegate_core_->CreateGraphfromTfLite(context, params);
    if (set_status != kTfLiteOk) {
        return set_status;
    }
    // TODO: get device string from flags

    return kTfLiteOk;
}

TfLiteStatus OpenVINODelegateKernel::Prepare(TfLiteOpaqueContext *context, TfLiteOpaqueNode *node) {
    TFLITE_LOG(INFO) << "inside Prepare \n";
    return kTfLiteOk;
}

TfLiteStatus OpenVINODelegateKernel::Eval(TfLiteOpaqueContext *context, TfLiteOpaqueNode *node) {
    std::vector<int> compute_inputs = ov_delegate_core_->getComputeInputs();
    size_t i = 0;
    for (int t : compute_inputs) {
        ov::Tensor inputBlob = ov_delegate_core_->getInferRequest().get_input_tensor(i++);
        uint8_t *dest = (uint8_t *)inputBlob.data<float>();

        const TfLiteOpaqueTensor *opaque_input_tensor =
            TfLiteOpaqueContextGetOpaqueTensor(context, t);
        auto len = TfLiteOpaqueTensorByteSize(opaque_input_tensor);
        void *srcPtr = TfLiteOpaqueTensorData(opaque_input_tensor);

        float *src = (float *)srcPtr;
        std::memcpy((uint8_t *)dest, (uint8_t *)srcPtr, len);
    }
    ov_delegate_core_->getInferRequest().start_async();
    ov_delegate_core_->getInferRequest().wait_for(std::chrono::milliseconds(10000));
    std::vector<int> outputs = ov_delegate_core_->getOutputs();
    size_t o = 0;
    for (int t : outputs) {
        ov::Tensor outputBlob = ov_delegate_core_->getInferRequest().get_output_tensor(o);
        const TfLiteOpaqueTensor *opaque_output_tensor =
            TfLiteOpaqueContextGetOpaqueTensor(context, *(outputs.begin()));
        void *srcPtr = TfLiteOpaqueTensorData(opaque_output_tensor);
        uint8_t *dest = (uint8_t *)outputBlob.data<float>();
        auto len = TfLiteOpaqueTensorByteSize(opaque_output_tensor);
        std::memcpy((void *)srcPtr, (void *)dest, len);
        o++;
    }

    return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
