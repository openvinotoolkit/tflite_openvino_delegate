// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "delegate/utils/experimental/stable_delegate/delegate_loader.h"
#include "openvino_delegate.h"
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/core/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"

namespace {

using tflite::TFLiteSettings;
using tflite::TFLiteSettingsBuilder;
using tflite::delegates::utils::LoadDelegateFromSharedLibrary;

class OpenVINODelegateExternalTest : public testing::Test {
 protected:
  const TfLiteStableDelegate *stable_delegate_handle;
  flatbuffers::Offset<TFLiteSettings> tflite_settings;
  TfLiteInterpreter *interpreter;
  TfLiteModel *model;
};

TEST_F(OpenVINODelegateExternalTest, LoadExternalDelegateLibrary) {
  // Load stable opaque_delegate that implements the ADD operation
  // from a shared libary file.
  stable_delegate_handle = LoadDelegateFromSharedLibrary(
      "bazel-bin/delegate/intel_openvino/"
      "libtensorflowlite_intel_openvino_delegate.so");
  ASSERT_NE(stable_delegate_handle, nullptr);
  EXPECT_STREQ(stable_delegate_handle->delegate_abi_version,
               TFL_STABLE_DELEGATE_ABI_VERSION);
  EXPECT_STREQ(stable_delegate_handle->delegate_name,
               kOpenVINOStableDelegateName);
  EXPECT_STREQ(stable_delegate_handle->delegate_version,
               kOpenVINOStableDelegateVersion);
  ASSERT_NE(stable_delegate_handle->delegate_plugin, nullptr);

  // Build TFLiteSettings flatbuffer and pass into opaque_delegate plugin
  // create method.
  flatbuffers::FlatBufferBuilder flatbuffer_builder;
  TFLiteSettingsBuilder tflite_settings_builder(flatbuffer_builder);
  tflite_settings = tflite_settings_builder.Finish();
  flatbuffer_builder.Finish(tflite_settings);
  const TFLiteSettings *settings = flatbuffers::GetRoot<TFLiteSettings>(
      flatbuffer_builder.GetBufferPointer());
  TfLiteOpaqueDelegate *opaque_delegate =
      stable_delegate_handle->delegate_plugin->create(settings);
  ASSERT_NE(opaque_delegate, nullptr);

  //
  // Create the model and the interpreter
  //
  model = TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);
  TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate);
  interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, nullptr);
  // The options can be deleted immediately after interpreter creation.
  TfLiteInterpreterOptionsDelete(options);

  //
  // Allocate the tensors and fill the input tensor.
  //
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
  TfLiteTensor *input_tensor =
      TfLiteInterpreterGetInputTensor(interpreter, /*input_index=*/0);
  ASSERT_NE(input_tensor, nullptr);
  const float kTensorCellValue = 3.f;
  int64_t n = tflite::NumElements(input_tensor);
  std::vector<float> input(n, kTensorCellValue);
  ASSERT_EQ(TfLiteTensorCopyFromBuffer(input_tensor, input.data(),
                                       input.size() * sizeof(float)),
            kTfLiteOk);

  //
  // Run the interpreter and read the output tensor.
  //
  ASSERT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);

  const TfLiteTensor *output_tensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 0);
  ASSERT_NE(output_tensor, nullptr);
  std::vector<float> output(n, 0);
  ASSERT_EQ(TfLiteTensorCopyToBuffer(output_tensor, output.data(),
                                     output.size() * sizeof(float)),
            kTfLiteOk);

  // The 'add.bin' model does the following operation ('t_output' denotes the
  // single output tensor, and 't_input' denotes the single input tensor):
  //
  // t_output = t_input + t_input + t_input = t_input * 3
  for (int i = 0; i < output.size(); ++i) {
    EXPECT_EQ(output[i], kTensorCellValue * 3);
  }

  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
  stable_delegate_handle->delegate_plugin->destroy(opaque_delegate);
}

}  // namespace
