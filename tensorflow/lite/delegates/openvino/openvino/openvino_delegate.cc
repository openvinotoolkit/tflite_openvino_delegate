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
#include "openvino_delegate.h"

#include "openvino/runtime/core.hpp"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/utils/simple_opaque_delegate.h"

namespace tflite {
namespace openvinodelegate {
bool OpenVINODelegate::CheckInputsType(const int tensor_id, const TfLiteOpaqueContext *context,
                                       TfLiteType expected_type) const {
    const TfLiteOpaqueTensor *opaque_tensor =
        TfLiteOpaqueContextGetOpaqueTensor(context, tensor_id);
    TfLiteType type = TfLiteOpaqueTensorType(opaque_tensor);
    return expected_type == type;
}

bool OpenVINODelegate::CheckDataTypeSupported(
    const TfLiteOpaqueContext *context, const TfLiteOpaqueNode *node,
    std::vector<std::vector<TfLiteType>> supported_types) const {
    const int *inputs;
    int num_inputs;
    auto tf_status = TfLiteOpaqueNodeInputs(node, &inputs, &num_inputs);
    for (int i = 0; i < supported_types.size(); i++) {
        int tensor_id = inputs[i];
        bool supported = false;
        for (TfLiteType type : supported_types[i])
            supported = CheckInputsType(tensor_id, context, type);
        if (supported == false) return false;
    }
    return true;
}

bool OpenVINODelegate::CheckDims(const TfLiteOpaqueContext *context, const TfLiteOpaqueNode *node,
                                 std::vector<std::vector<int>> dims_size) const {
    const int *inputs;
    int num_inputs;
    bool supported;
    auto tf_status = TfLiteOpaqueNodeInputs(node, &inputs, &num_inputs);
    if (num_inputs != dims_size.size()) return false;
    for (int i = 0; i < dims_size.size(); i++) {
        supported = false;
        const TfLiteOpaqueTensor *opaque_tensor =
            TfLiteOpaqueContextGetOpaqueTensor(context, inputs[i]);
        for (int j = 0; j < dims_size[i].size(); j++) {
            if (TfLiteOpaqueTensorNumDims(opaque_tensor) == dims_size[i][j]) {
                supported = true;
                int size = 1;
                for (int k = 0; k < dims_size[i][j]; k++)
                    size *= TfLiteOpaqueTensorDim(opaque_tensor, k);
                if (size == 0) return false;
            }
        }
        if (supported == false) return false;
    }
    return supported;
}

bool OpenVINODelegate::CheckNodeSupportByOpenVINO(const TfLiteRegistrationExternal *registration,
                                                  const TfLiteOpaqueNode *node,
                                                  TfLiteOpaqueContext *context) const {
    switch (TfLiteRegistrationExternalGetBuiltInCode(registration)) {
        case kTfLiteBuiltinAdd: {
            return CheckDataTypeSupported(context, node, {{kTfLiteFloat32}, {kTfLiteFloat32}}) &&
                   CheckDims(context, node, {{1, 2, 3, 4}, {1, 2, 3, 4}});
        }
        case kTfLiteBuiltinAveragePool2d: {
            return CheckDataTypeSupported(context, node, {{kTfLiteFloat32}});
        }
        case kTfLiteBuiltinConv2d: {
            const int *inputs;
            int num_inputs;
            auto tf_status = TfLiteOpaqueNodeInputs(node, &inputs, &num_inputs);
            if (num_inputs == 2) {
                return CheckDataTypeSupported(context, node,
                                              {{kTfLiteFloat32}, {kTfLiteFloat32}}) &&
                       CheckDims(context, node, {{4}, {4}});
            } else if (num_inputs == 3) {
                return CheckDataTypeSupported(
                           context, node, {{kTfLiteFloat32}, {kTfLiteFloat32}, {kTfLiteFloat32}}) &&
                       CheckDims(context, node, {{4}, {4}, {1}});
            } else
                return false;
        }
        case kTfLiteBuiltinConcatenation: {
            return CheckDataTypeSupported(context, node, {{kTfLiteFloat32}, {kTfLiteFloat32}}) &&
                   CheckDims(context, node, {{4}, {4}});
        }
        case kTfLiteBuiltinCustom: {
            if (strcmp(TfLiteRegistrationExternalGetCustomName(registration),
                       "Convolution2DTransposeBias") == 0) {
                return CheckDataTypeSupported(
                           context, node, {{kTfLiteFloat32}, {kTfLiteFloat32}, {kTfLiteFloat32}}) &&
                       CheckDims(context, node, {{4}, {4}, {1}});
            }
            return false;
        }
        case kTfLiteBuiltinDepthwiseConv2d: {
            const int *inputs;
            int num_inputs;
            auto tf_status = TfLiteOpaqueNodeInputs(node, &inputs, &num_inputs);
            if (num_inputs == 2) {
                return CheckDataTypeSupported(context, node,
                                              {{kTfLiteFloat32}, {kTfLiteFloat32}}) &&
                       CheckDims(context, node, {{4}, {4}});
            } else if (num_inputs == 3) {
                return CheckDataTypeSupported(
                           context, node, {{kTfLiteFloat32}, {kTfLiteFloat32}, {kTfLiteFloat32}}) &&
                       CheckDims(context, node, {{4}, {4}, {1}});
            } else
                return false;
        }
        case kTfLiteBuiltinDequantize: {
            return CheckDataTypeSupported(context, node, {{kTfLiteFloat16}});
        }
        case kTfLiteBuiltinResizeBilinear: {
            return CheckDataTypeSupported(context, node, {{kTfLiteFloat32}});
        }
        case kTfLiteBuiltinRelu: {
            return CheckDataTypeSupported(context, node, {{kTfLiteFloat32}});
        }
        case kTfLiteBuiltinRelu6: {
            return CheckDataTypeSupported(context, node, {{kTfLiteFloat32}});
        }
        case kTfLiteBuiltinLogistic: {
            return CheckDataTypeSupported(context, node, {{kTfLiteFloat32}});
        }
        case kTfLiteBuiltinHardSwish: {
            return CheckDataTypeSupported(context, node, {{kTfLiteFloat32}});
        }
        case kTfLiteBuiltinMul: {
            return CheckDataTypeSupported(context, node, {{kTfLiteFloat32}, {kTfLiteFloat32}}) &&
                   CheckDims(context, node, {{1, 2, 3, 4}, {1, 2, 3, 4}});
        }
        case kTfLiteBuiltinSoftmax: {
            TfLiteSoftmaxParams *softmax_params =
                (TfLiteSoftmaxParams *)TfLiteOpaqueNodeGetBuiltinData(node);
            if (softmax_params->beta != 1.0f) {
                TFLITE_LOG(INFO) << "Unsupported Softmax op, beta value is not 1.0 \n";
                return false;
            }
            return CheckDataTypeSupported(context, node, {{kTfLiteFloat32}});
        }
        case kTfLiteBuiltinTanh: {
            return CheckDataTypeSupported(context, node, {{kTfLiteFloat32}});
        }
        case kTfLiteBuiltinReshape: {
            return CheckDataTypeSupported(context, node, {{kTfLiteFloat32}, {kTfLiteInt32}}) &&
                   CheckDims(context, node, {{1, 2, 3, 4}, {1}});
        }
        case kTfLiteBuiltinMaxPool2d: {
            return CheckDataTypeSupported(context, node, {{kTfLiteFloat32}});
        }
        case kTfLiteBuiltinMean: {
            return CheckDataTypeSupported(context, node, {{kTfLiteFloat32}}) &&
                   CheckDims(context, node, {{4}, {1}});
        }
        case kTfLiteBuiltinTransposeConv: {
            const int *inputs_data;
            int num_inputs;
            TfLiteOpaqueNodeInputs(node, &inputs_data, &num_inputs);
            if (num_inputs == 3) {
                return CheckDataTypeSupported(
                           context, node, {{kTfLiteInt32}, {kTfLiteFloat32}, {kTfLiteFloat32}}) &&
                       CheckDims(context, node, {{1}, {4}, {4}});
            } else if (num_inputs == 4) {
                return CheckDataTypeSupported(context, node,
                                              {{kTfLiteInt32},
                                               {kTfLiteFloat32},
                                               {kTfLiteFloat32},
                                               {kTfLiteFloat32}}) &&
                       CheckDims(context, node, {{1}, {4}, {4}, {1}});
            } else {
                return false;
            }
        }
        default:
            return false;
    }
}

bool OpenVINODelegate::IsNodeSupportedByDelegate(const TfLiteRegistrationExternal *registration,
                                                 const TfLiteOpaqueNode *node,
                                                 TfLiteOpaqueContext *context) const {
    if (registration == nullptr || node == nullptr || context == nullptr) return false;
    bool check = CheckNodeSupportByOpenVINO(registration, node, context);
    return check;
}

TfLiteStatus OpenVINODelegate::Initialize(TfLiteOpaqueContext *context) { return kTfLiteOk; }

const char *OpenVINODelegate::Name() const { return "OpenVINO SimpleOpaqueDelegate"; }

std::unique_ptr<tflite::SimpleOpaqueDelegateKernelInterface>
OpenVINODelegate::CreateDelegateKernelInterface() {
    return std::unique_ptr<tflite::openvinodelegate::OpenVINODelegateKernel>(
        new tflite::openvinodelegate::OpenVINODelegateKernel());
}
}  // namespace openvinodelegate
}  // namespace tflite

TfLiteDelegate *TFL_CAPI_EXPORT
TfLiteCreateOpenVINODelegate(const TfLiteOpenVINODelegateOptions *options) {
    auto ovdelegate_ = std::make_unique<tflite::openvinodelegate::OpenVINODelegate>(options);
    return tflite::TfLiteOpaqueDelegateFactory::CreateSimpleDelegate(std::move(ovdelegate_));
}

void TFL_CAPI_EXPORT TfLiteDeleteOpenVINODelegate(TfLiteOpaqueDelegate *delegate) { return; }

TfLiteOpenVINODelegateOptions TFL_CAPI_EXPORT TfLiteOpenVINODelegateOptionsDefault() {
    TfLiteOpenVINODelegateOptions result;
    result.debug_level = 0;
    result.plugins_path = "/tmp/plugins.xml";
    return result;
}
