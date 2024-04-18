#include "openvino_delegate_core.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "openvino_graph_builder.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/core/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/kernel_util.h"

class OpenVINODelegateCoreTest : public testing::Test {
protected:
    TfLiteInterpreter* interpreter_ = nullptr;
    TfLiteOpaqueDelegate* opaque_delegate_ = nullptr;
    TfLiteModel* model_ = nullptr;
};

TEST_F(OpenVINODelegateCoreTest, CreateGraphfromTfLite) {
    TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
    opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext* opaque_context,
                                         TfLiteOpaqueDelegate* opaque_delegate_,
                                         void* data) -> TfLiteStatus {
        auto reg_ex = TfLiteRegistrationExternalCreate(
            kTfLiteBuiltinDelegate, "Test driver Openvino delegate", /*version=*/1);
        TfLiteRegistrationExternalSetInit(
            reg_ex,
            [](TfLiteOpaqueContext* opaque_context, const char* buffer, size_t length) -> void* {
                const TfLiteOpaqueDelegateParams* params =
                    reinterpret_cast<const TfLiteOpaqueDelegateParams*>(buffer);
                const std::unordered_set<int> inputs(
                    &params->input_tensors->data[0],
                    &params->input_tensors->data[params->input_tensors->size]);

                auto ov_delegate_core_test =
                    std::make_unique<tflite::openvinodelegate::OpenVINODelegateCore>("");
                EXPECT_EQ(kTfLiteOk, ov_delegate_core_test->OpenVINODelegateInit());
                EXPECT_EQ(kTfLiteOk,
                          ov_delegate_core_test->CreateGraphfromTfLite(opaque_context, params));
                void* void_fake_ptr;
                return void_fake_ptr;
            });
        return kTfLiteOk;
    };

    model_ = TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
    opaque_delegate_ = TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate_);
    interpreter_ = TfLiteInterpreterCreate(model_, options);
    ;

    TfLiteInterpreterOptionsDelete(options);
    TfLiteInterpreterDelete(interpreter_);
    TfLiteModelDelete(model_);
    TfLiteOpaqueDelegateDelete(opaque_delegate_);
}

TEST_F(OpenVINODelegateCoreTest, CreateGraphfromTfLite_InvalidContext) {
    TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
    opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext* opaque_context,
                                         TfLiteOpaqueDelegate* opaque_delegate_,
                                         void* data) -> TfLiteStatus {
        auto reg_ex = TfLiteRegistrationExternalCreate(
            kTfLiteBuiltinDelegate, "Test driver Openvino delegate", /*version=*/1);
        TfLiteRegistrationExternalSetInit(
            reg_ex,
            [](TfLiteOpaqueContext* opaque_context, const char* buffer, size_t length) -> void* {
                const TfLiteOpaqueDelegateParams* params =
                    reinterpret_cast<const TfLiteOpaqueDelegateParams*>(buffer);
                const std::unordered_set<int> inputs(
                    &params->input_tensors->data[0],
                    &params->input_tensors->data[params->input_tensors->size]);

                // test for context with null ptr
                opaque_context == nullptr;
                auto ov_delegate_core_test =
                    std::make_unique<tflite::openvinodelegate::OpenVINODelegateCore>("");
                EXPECT_EQ(kTfLiteOk, ov_delegate_core_test->OpenVINODelegateInit());
                EXPECT_EQ(kTfLiteError,
                          ov_delegate_core_test->CreateGraphfromTfLite(opaque_context, params));
                void* void_fake_ptr;
                return void_fake_ptr;
            });
        return kTfLiteOk;
    };

    model_ = TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
    opaque_delegate_ = TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate_);
    interpreter_ = TfLiteInterpreterCreate(model_, options);
    ;

    TfLiteInterpreterOptionsDelete(options);
    TfLiteInterpreterDelete(interpreter_);
    TfLiteModelDelete(model_);
    TfLiteOpaqueDelegateDelete(opaque_delegate_);
}

TEST_F(OpenVINODelegateCoreTest, CreateGraphfromTfLite_InvalidParams) {
    TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
    opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext* opaque_context,
                                         TfLiteOpaqueDelegate* opaque_delegate_,
                                         void* data) -> TfLiteStatus {
        auto reg_ex = TfLiteRegistrationExternalCreate(
            kTfLiteBuiltinDelegate, "Test driver Openvino delegate", /*version=*/1);
        TfLiteRegistrationExternalSetInit(
            reg_ex,
            [](TfLiteOpaqueContext* opaque_context, const char* buffer, size_t length) -> void* {
                // test for params with null ptr
                const TfLiteOpaqueDelegateParams* params = nullptr;
                auto ov_delegate_core_test =
                    std::make_unique<tflite::openvinodelegate::OpenVINODelegateCore>("");
                EXPECT_EQ(kTfLiteOk, ov_delegate_core_test->OpenVINODelegateInit());
                EXPECT_EQ(kTfLiteError,
                          ov_delegate_core_test->CreateGraphfromTfLite(opaque_context, params));
                void* void_fake_ptr;
                return void_fake_ptr;
            });
        return kTfLiteOk;
    };

    model_ = TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
    opaque_delegate_ = TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate_);
    interpreter_ = TfLiteInterpreterCreate(model_, options);
    ;

    TfLiteInterpreterOptionsDelete(options);
    TfLiteInterpreterDelete(interpreter_);
    TfLiteModelDelete(model_);
    TfLiteOpaqueDelegateDelete(opaque_delegate_);
}