/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>
#include <utility>

#include "openvino_delegate.h"
#include "tensorflow/lite/acceleration/configuration/c/delegate_plugin.h"
#include "tensorflow/lite/acceleration/configuration/c/stable_delegate.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/stable_delegate_interface.h"
#include "tensorflow/lite/delegates/utils/simple_opaque_delegate.h"

namespace {

TfLiteOpaqueDelegate *OpenVINOStableDelegateCreateFunc(const void *tflite_settings) {
    TfLiteOpenVINODelegateOptions options = TfLiteOpenVINODelegateOptionsDefault();
    auto delegate = std::make_unique<tflite::openvinodelegate::OpenVINODelegate>(&options);
    return tflite::TfLiteOpaqueDelegateFactory::CreateSimpleDelegate(std::move(delegate));
}

void OpenVINOStableDelegateDestroyFunc(TfLiteOpaqueDelegate *openvino_stable_delegate) {
    tflite::TfLiteOpaqueDelegateFactory::DeleteSimpleDelegate(openvino_stable_delegate);
}

int OpenVINOStableDelegateErrnoFunc(TfLiteOpaqueDelegate *openvino_stable_delegate) {
    // no-op
    return 0;
}

const TfLiteOpaqueDelegatePlugin openvino_stable_delegate_plugin = {
    OpenVINOStableDelegateCreateFunc, OpenVINOStableDelegateDestroyFunc,
    OpenVINOStableDelegateErrnoFunc};

const TfLiteStableDelegate openvino_stable_delegate = {
    TFL_STABLE_DELEGATE_ABI_VERSION, kOpenVINOStableDelegateName, kOpenVINOStableDelegateVersion,
    &openvino_stable_delegate_plugin};

}  // namespace

extern "C" const TfLiteStableDelegate TFL_TheStableDelegate = openvino_stable_delegate;
