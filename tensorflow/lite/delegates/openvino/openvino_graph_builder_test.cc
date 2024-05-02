// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino_graph_builder.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <openvino/pass/manager.hpp>
#include <openvino/pass/serialize.hpp>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/core/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"

std::function<void(TfLiteOpaqueTensor *opaque_tensor, const int index)>
    test_function_;

class OpenVINOGraphBuilderTest : public testing::Test {
 protected:
  void SetUp() override {
    model = tflite::FlatBufferModel::BuildFromFile(
        "tensorflow/lite/testdata/add.bin");
    ASSERT_NE(model, nullptr);
  }
  void TearDown() override { TfLiteOpaqueDelegateDelete(opaque_delegate_); }

 protected:
  std::unique_ptr<tflite::Interpreter> interpreter_;
  TfLiteOpaqueDelegate *opaque_delegate_ = nullptr;
  TfLiteOpaqueContext *test_graph_context = nullptr;
  std::unique_ptr<tflite::FlatBufferModel> model;
};

TfLiteOpaqueTensor *create_opaque_tensor(TfLiteTensor *t) {
  return (TfLiteOpaqueTensor *)t;
}

TEST_F(OpenVINOGraphBuilderTest, AddInputParamsTest_VALID) {
  TfLiteTensor t;
  const int kNumElements = 32;
  const int kBytes = sizeof(float) * kNumElements;
  memset(&t, 0, sizeof(TfLiteTensor));
  t.bytes = kBytes;
  t.data_is_stale = true;
  t.allocation_type = kTfLiteDynamic;
  t.type = kTfLiteFloat32;
  t.dims = TfLiteIntArrayCreate(2);
  t.dims->data[0] = 4;
  t.dims->data[1] = 8;
  t.dims_signature = TfLiteIntArrayCopy(t.dims);
  t.buffer_handle = 5;

  TfLiteOpaqueTensor *opaque_t = create_opaque_tensor(&t);

  auto openvino_graph_builder_test =
      std::make_unique<tflite::openvinodelegate::OpenVINOGraphBuilder>(
          std::make_unique<NodeManager>());

  EXPECT_EQ(kTfLiteOk,
            openvino_graph_builder_test->AddInputParams(opaque_t, 0));
  EXPECT_EQ(true, openvino_graph_builder_test->getNodeManagerSize() == 1);
}

TEST_F(OpenVINOGraphBuilderTest, AddInputParamsTest_InvalidTensor) {
  auto openvino_graph_builder_test =
      std::make_unique<tflite::openvinodelegate::OpenVINOGraphBuilder>(
          std::make_unique<NodeManager>());

  EXPECT_EQ(kTfLiteError,
            openvino_graph_builder_test->AddInputParams(nullptr, 0));
  EXPECT_EQ(false, openvino_graph_builder_test->getNodeManagerSize() == 1);
}

TEST_F(OpenVINOGraphBuilderTest, AddInputParamsTest_InvalidIndex) {
  TfLiteTensor t;
  const int kNumElements = 32;
  const int kBytes = sizeof(float) * kNumElements;
  memset(&t, 0, sizeof(TfLiteTensor));
  t.bytes = kBytes;
  // t.delegate = &delegate;
  t.data_is_stale = true;
  t.allocation_type = kTfLiteDynamic;
  t.type = kTfLiteFloat32;
  t.dims = TfLiteIntArrayCreate(2);
  t.dims->data[0] = 4;
  t.dims->data[1] = 8;
  t.dims_signature = TfLiteIntArrayCopy(t.dims);
  t.buffer_handle = 5;

  TfLiteOpaqueTensor *opaque_t = create_opaque_tensor(&t);

  auto openvino_graph_builder_test =
      std::make_unique<tflite::openvinodelegate::OpenVINOGraphBuilder>(
          std::make_unique<NodeManager>());

  EXPECT_EQ(kTfLiteError,
            openvino_graph_builder_test->AddInputParams(opaque_t, -1));
  EXPECT_EQ(false, openvino_graph_builder_test->getNodeManagerSize() == 1);
}

TEST_F(OpenVINOGraphBuilderTest, convertNHWCtoNCHW_Valid) {
  std::vector<int> dims = {1, 2, 3, 4};
  auto input = std::make_shared<ov::opset3::Parameter>(
      ov::element::Type_t::f32, ov::Shape(dims.begin(), dims.end()));

  auto openvino_graph_builder_test =
      std::make_unique<tflite::openvinodelegate::OpenVINOGraphBuilder>(
          std::make_unique<NodeManager>());

  std::shared_ptr<ov::Node> interim;
  EXPECT_EQ(kTfLiteOk, openvino_graph_builder_test->convertNHWCtoNCHW(
                           dims, input, interim));
}

TEST_F(OpenVINOGraphBuilderTest, convertNHWCtoNCHW_InvalidNode) {
  std::vector<int> dims = {1, 2, 3, 4};
  auto openvino_graph_builder_test =
      std::make_unique<tflite::openvinodelegate::OpenVINOGraphBuilder>(
          std::make_unique<NodeManager>());

  std::shared_ptr<ov::Node> interim;
  EXPECT_EQ(kTfLiteError, openvino_graph_builder_test->convertNHWCtoNCHW(
                              dims, nullptr, interim));
}

TEST_F(OpenVINOGraphBuilderTest, convertNHWCtoNCHW_InvalidDims) {
  std::vector<int> dims = {1, 2, 3, 4, 5, 6};
  auto input = std::make_shared<ov::opset3::Parameter>(
      ov::element::Type_t::f32, ov::Shape(dims.begin(), dims.end()));

  auto openvino_graph_builder_test =
      std::make_unique<tflite::openvinodelegate::OpenVINOGraphBuilder>(
          std::make_unique<NodeManager>());

  std::shared_ptr<ov::Node> interim;
  EXPECT_EQ(kTfLiteError, openvino_graph_builder_test->convertNHWCtoNCHW(
                              dims, input, interim));
}

TEST_F(OpenVINOGraphBuilderTest, CreateConstNode_Invalid) {
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  bool delegate_prepared = false;

  opaque_delegate_builder.data = &delegate_prepared;
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext *opaque_context,
                                       TfLiteOpaqueDelegate *opaque_delegate_,
                                       void *data) -> TfLiteStatus {
    auto delegate_prepared = static_cast<bool *>(data);
    *delegate_prepared = true;
    TfLiteIntArray *execution_plan;
    TF_LITE_ENSURE_STATUS(
        TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan));
    for (int i = 0; i < execution_plan->size; ++i) {
      TfLiteOpaqueNode *node = nullptr;
      TfLiteRegistrationExternal *registration = nullptr;
      TfLiteOpaqueContextGetNodeAndRegistration(opaque_context, i, &node,
                                                &registration);
      bool is_supported = false;
      const int *inputs_data;
      int num_inputs;
      TfLiteStatus tf_status =
          TfLiteOpaqueNodeInputs(node, &inputs_data, &num_inputs);
      const int t = inputs_data[0];
      auto opaque_tensor =
          TfLiteOpaqueContextGetOpaqueTensor(opaque_context, t);
      auto openvino_graph_builder_test =
          std::make_unique<tflite::openvinodelegate::OpenVINOGraphBuilder>(
              std::make_unique<NodeManager>());

      EXPECT_EQ(kTfLiteError, openvino_graph_builder_test->CreateConstNode(
                                  opaque_context, 0));
    }

    TfLiteRegistrationExternal *registration_external =
        TfLiteRegistrationExternalCreate(kTfLiteBuiltinDelegate,
                                         /*name*/ nullptr,
                                         /*version=*/1);
    return TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        opaque_context, registration_external, execution_plan,
        opaque_delegate_);
  };

  opaque_delegate_ = TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  builder.AddDelegate(opaque_delegate_);

  EXPECT_EQ(kTfLiteOk, builder(&interpreter_));
  EXPECT_TRUE(delegate_prepared);
  ASSERT_NE(interpreter_, nullptr);
}

TEST_F(OpenVINOGraphBuilderTest, CreateConstNode_InvalidContext) {
  auto openvino_graph_builder_test =
      std::make_unique<tflite::openvinodelegate::OpenVINOGraphBuilder>(
          std::make_unique<NodeManager>());
  EXPECT_EQ(kTfLiteError,
            openvino_graph_builder_test->CreateConstNode(nullptr, 0));
}

TEST_F(OpenVINOGraphBuilderTest, CreateConstNode_InvalidIndex) {
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  bool delegate_prepared = false;

  opaque_delegate_builder.data = &delegate_prepared;
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext *opaque_context,
                                       TfLiteOpaqueDelegate *opaque_delegate_,
                                       void *data) -> TfLiteStatus {
    auto delegate_prepared = static_cast<bool *>(data);
    *delegate_prepared = true;
    TfLiteIntArray *execution_plan;
    TF_LITE_ENSURE_STATUS(
        TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan));
    for (int i = 0; i < execution_plan->size; ++i) {
      TfLiteOpaqueNode *node = nullptr;
      TfLiteRegistrationExternal *registration = nullptr;
      TfLiteOpaqueContextGetNodeAndRegistration(opaque_context, i, &node,
                                                &registration);
      bool is_supported = false;
      const int *inputs_data;
      int num_inputs;
      TfLiteStatus tf_status =
          TfLiteOpaqueNodeInputs(node, &inputs_data, &num_inputs);
      const int t = inputs_data[0];
      auto opaque_tensor =
          TfLiteOpaqueContextGetOpaqueTensor(opaque_context, t);
      auto openvino_graph_builder_test =
          std::make_unique<tflite::openvinodelegate::OpenVINOGraphBuilder>(
              std::make_unique<NodeManager>());

      EXPECT_EQ(kTfLiteError, openvino_graph_builder_test->CreateConstNode(
                                  opaque_context, 1));
    }

    TfLiteRegistrationExternal *registration_external =
        TfLiteRegistrationExternalCreate(kTfLiteBuiltinDelegate,
                                         /*name*/ nullptr,
                                         /*version=*/1);
    return TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        opaque_context, registration_external, execution_plan,
        opaque_delegate_);
  };
  opaque_delegate_ = TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  builder.AddDelegate(opaque_delegate_);

  EXPECT_EQ(kTfLiteOk, builder(&interpreter_));
  EXPECT_TRUE(delegate_prepared);
  ASSERT_NE(interpreter_, nullptr);
}

TEST_F(OpenVINOGraphBuilderTest, UpdateResultNodes_Valid) {
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  bool delegate_prepared = false;

  opaque_delegate_builder.data = &delegate_prepared;
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext *opaque_context,
                                       TfLiteOpaqueDelegate *opaque_delegate_,
                                       void *data) -> TfLiteStatus {
    auto delegate_prepared = static_cast<bool *>(data);
    *delegate_prepared = true;
    auto reg_ex = TfLiteRegistrationExternalCreate(
        kTfLiteBuiltinDelegate, "Test driver Openvino delegate", /*version=*/1);
    TfLiteRegistrationExternalSetInit(
        reg_ex,
        [](TfLiteOpaqueContext *opaque_context, const char *buffer,
           size_t length) -> void * {
          auto openvino_graph_builder_test =
              std::make_unique<tflite::openvinodelegate::OpenVINOGraphBuilder>(
                  std::make_unique<NodeManager>());
          std::vector<int> compute_inputs_ = {};
          std::vector<int> outputs_ = {};
          const TfLiteOpaqueDelegateParams *params =
              reinterpret_cast<const TfLiteOpaqueDelegateParams *>(buffer);
          const std::unordered_set<int> inputs(
              &params->input_tensors->data[0],
              &params->input_tensors->data[params->input_tensors->size]);

          for (int o = 0; o < params->output_tensors->size; o++) {
            const int output_tensor_idx = params->output_tensors->data[o];
            outputs_.push_back(output_tensor_idx);
          }
          for (int i = 0; i < params->nodes_to_replace->size; ++i) {
            const int delegate_node_id = params->nodes_to_replace->data[i];
            TfLiteOpaqueNode *node = nullptr;
            TfLiteRegistrationExternal *registration = nullptr;
            TfLiteOpaqueContextGetNodeAndRegistration(
                opaque_context, delegate_node_id, &node, &registration);
            int inputs_size = TfLiteOpaqueNodeNumberOfInputs(node);
            for (int k = 0; k < inputs_size; k++) {
              const int *inputs_data;
              int num_inputs;
              TfLiteStatus tf_status =
                  TfLiteOpaqueNodeInputs(node, &inputs_data, &num_inputs);
              const int t = inputs_data[k];
              const void *data = nullptr;
              auto opaque_tensor =
                  TfLiteOpaqueContextGetOpaqueTensor(opaque_context, t);
              auto allocation_type =
                  TfLiteOpaqueTensorGetAllocationType(opaque_tensor);
              if (allocation_type == kTfLiteMmapRo) {
                data = TfLiteOpaqueTensorData(opaque_tensor);

                if (openvino_graph_builder_test->CreateConstNode(
                        opaque_context, t) != kTfLiteOk)
                  exit(0);
              }
              if (inputs.count(t) != 0) {
                if (data == nullptr) {
                  if (openvino_graph_builder_test->AddInputParams(
                          opaque_tensor, t) != kTfLiteOk)
                    exit(0);
                  compute_inputs_.push_back(t);
                }
              }
            }
            if (openvino_graph_builder_test->CreateNodeFromTfLiteOp(
                    delegate_node_id, registration, node, opaque_context) !=
                kTfLiteOk)
              exit(0);
          }

          EXPECT_EQ(kTfLiteOk, openvino_graph_builder_test->UpdateResultNodes(
                                   opaque_context, outputs_));
          std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(
              openvino_graph_builder_test->getResultNodes(),
              openvino_graph_builder_test->getInputParams());
          ov::Core openvino_delegate_core_;
          ov::CompiledModel compiled_model_;
          std::string deviceStr = "CPU";
          ov::InferRequest infer_request_;
          if (model) {
            compiled_model_ =
                openvino_delegate_core_.compile_model(model, deviceStr);
          }

          infer_request_ = compiled_model_.create_infer_request();
          return &compiled_model_;
        });

    TfLiteRegistrationExternalSetInvoke(
        reg_ex,
        [](TfLiteOpaqueContext *context, TfLiteOpaqueNode *opaque_node)
            -> TfLiteStatus { return kTfLiteOk; });

    TfLiteRegistrationExternalSetFree(
        reg_ex, [](TfLiteOpaqueContext *context, void *data) {});

    TfLiteIntArray *execution_plan;
    TF_LITE_ENSURE_STATUS(
        TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan));
    TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        opaque_context, reg_ex, execution_plan, opaque_delegate_);
    return kTfLiteOk;
  };

  TfLiteModel *model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");

  TfLiteOpaqueDelegate *opaque_delegate =
      TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);
  TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate);
  TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, options);

  EXPECT_TRUE(delegate_prepared);

  TfLiteInterpreterOptionsDelete(options);
  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
  TfLiteOpaqueDelegateDelete(opaque_delegate);
}

TEST_F(OpenVINOGraphBuilderTest, UpdateResultNodes_InvalidContext) {
  auto openvino_graph_builder_test =
      std::make_unique<tflite::openvinodelegate::OpenVINOGraphBuilder>(
          std::make_unique<NodeManager>());
  EXPECT_EQ(kTfLiteError,
            openvino_graph_builder_test->UpdateResultNodes(nullptr, {0}));
}

TEST_F(OpenVINOGraphBuilderTest, UpdateResultNodes_InvalidIndex) {
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  bool delegate_prepared = false;

  opaque_delegate_builder.data = &delegate_prepared;
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext *opaque_context,
                                       TfLiteOpaqueDelegate *opaque_delegate_,
                                       void *data) -> TfLiteStatus {
    auto delegate_prepared = static_cast<bool *>(data);
    *delegate_prepared = true;
    auto reg_ex = TfLiteRegistrationExternalCreate(
        kTfLiteBuiltinDelegate, "Test driver Openvino delegate", /*version=*/1);
    TfLiteRegistrationExternalSetInit(
        reg_ex,
        [](TfLiteOpaqueContext *opaque_context, const char *buffer,
           size_t length) -> void * {
          auto openvino_graph_builder_test =
              std::make_unique<tflite::openvinodelegate::OpenVINOGraphBuilder>(
                  std::make_unique<NodeManager>());
          std::vector<int> compute_inputs_ = {};
          std::vector<int> outputs_ = {};
          const TfLiteOpaqueDelegateParams *params =
              reinterpret_cast<const TfLiteOpaqueDelegateParams *>(buffer);
          const std::unordered_set<int> inputs(
              &params->input_tensors->data[0],
              &params->input_tensors->data[params->input_tensors->size]);

          for (int o = 0; o < params->output_tensors->size; o++) {
            const int output_tensor_idx = params->output_tensors->data[o];
            outputs_.push_back(output_tensor_idx);
          }
          for (int i = 0; i < params->nodes_to_replace->size; ++i) {
            const int delegate_node_id = params->nodes_to_replace->data[i];
            TfLiteOpaqueNode *node = nullptr;
            TfLiteRegistrationExternal *registration = nullptr;
            TfLiteOpaqueContextGetNodeAndRegistration(
                opaque_context, delegate_node_id, &node, &registration);
            int inputs_size = TfLiteOpaqueNodeNumberOfInputs(node);
            for (int k = 0; k < inputs_size; k++) {
              const int *inputs_data;
              int num_inputs;
              TfLiteStatus tf_status =
                  TfLiteOpaqueNodeInputs(node, &inputs_data, &num_inputs);
              const int t = inputs_data[k];
              const void *data = nullptr;
              auto opaque_tensor =
                  TfLiteOpaqueContextGetOpaqueTensor(opaque_context, t);
              auto allocation_type =
                  TfLiteOpaqueTensorGetAllocationType(opaque_tensor);
              if (allocation_type == kTfLiteMmapRo) {
                data = TfLiteOpaqueTensorData(opaque_tensor);

                if (openvino_graph_builder_test->CreateConstNode(
                        opaque_context, t) != kTfLiteOk)
                  exit(0);
              }
              if (inputs.count(t) != 0) {
                if (data == nullptr) {
                  static int count = 0;
                  if (count == 0) {
                    if (openvino_graph_builder_test->AddInputParams(
                            opaque_tensor, t) != kTfLiteOk)
                      exit(0);
                    compute_inputs_.push_back(t);
                    count++;
                  }
                }
              }
            }
            if (openvino_graph_builder_test->CreateNodeFromTfLiteOp(
                    delegate_node_id, registration, node, opaque_context) !=
                kTfLiteOk)
              exit(0);
          }

          EXPECT_EQ(kTfLiteError,
                    openvino_graph_builder_test->UpdateResultNodes(
                        opaque_context, {}));
          std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(
              openvino_graph_builder_test->getResultNodes(),
              openvino_graph_builder_test->getInputParams());
          ov::Core openvino_delegate_core_;
          ov::CompiledModel compiled_model_;
          std::string deviceStr = "CPU";
          ov::InferRequest infer_request_;
          if (model) {
            compiled_model_ =
                openvino_delegate_core_.compile_model(model, deviceStr);
            ov::pass::Manager manager;
            manager.register_pass<ov::pass::Serialize>("/tmp/model.xml",
                                                       "/tmp/model.bin");
            manager.run_passes(model);
          }

          infer_request_ = compiled_model_.create_infer_request();
          return &compiled_model_;
        });

    TfLiteRegistrationExternalSetInvoke(
        reg_ex,
        [](TfLiteOpaqueContext *context, TfLiteOpaqueNode *opaque_node)
            -> TfLiteStatus { return kTfLiteOk; });

    TfLiteRegistrationExternalSetFree(
        reg_ex, [](TfLiteOpaqueContext *context, void *data) {});

    TfLiteIntArray *execution_plan;
    TF_LITE_ENSURE_STATUS(
        TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan));
    TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        opaque_context, reg_ex, execution_plan, opaque_delegate_);
    return kTfLiteOk;
  };

  TfLiteModel *model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");

  TfLiteOpaqueDelegate *opaque_delegate =
      TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);
  TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate);
  TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, options);

  EXPECT_TRUE(delegate_prepared);

  TfLiteInterpreterOptionsDelete(options);
  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
  TfLiteOpaqueDelegateDelete(opaque_delegate);
}

TEST_F(OpenVINOGraphBuilderTest, CreateOpClass_ValidIndex) {
  struct TfLiteRegistrationExternal registration;
  registration.version = 1;
  registration.builtin_code = kTfLiteBuiltinAdd;
  registration.node_index = 1;
  int operationIndex = 1;

  std::shared_ptr<tflite::openvinodelegate::OperationsBase> op_base;
  auto openvino_graph_builder_test =
      std::make_unique<tflite::openvinodelegate::OpenVINOGraphBuilder>(
          std::make_unique<NodeManager>());

  EXPECT_EQ(kTfLiteOk, openvino_graph_builder_test->CreateOpClass(
                           operationIndex, &registration, op_base));
}

TEST_F(OpenVINOGraphBuilderTest, CreateOpClass_InvalidIndex) {
  struct TfLiteRegistrationExternal registration;
  registration.version = 1;
  registration.builtin_code = kTfLiteBuiltinAdd;
  registration.node_index = -1;
  int operationIndex = -1;

  std::shared_ptr<tflite::openvinodelegate::OperationsBase> op_base;
  auto openvino_graph_builder_test =
      std::make_unique<tflite::openvinodelegate::OpenVINOGraphBuilder>(
          std::make_unique<NodeManager>());

  EXPECT_EQ(kTfLiteError, openvino_graph_builder_test->CreateOpClass(
                              operationIndex, &registration, op_base));
}

TEST_F(OpenVINOGraphBuilderTest, CreateOpClass_InvalidRegistration) {
  int operationIndex = 0;

  std::shared_ptr<tflite::openvinodelegate::OperationsBase> op_base;
  auto openvino_graph_builder_test =
      std::make_unique<tflite::openvinodelegate::OpenVINOGraphBuilder>(
          std::make_unique<NodeManager>());

  EXPECT_EQ(kTfLiteError, openvino_graph_builder_test->CreateOpClass(
                              operationIndex, nullptr, op_base));
}

TEST_F(OpenVINOGraphBuilderTest, CreateNodeFromTfLiteOp_Valid) {
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  bool delegate_prepared = false;

  opaque_delegate_builder.data = &delegate_prepared;
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext *opaque_context,
                                       TfLiteOpaqueDelegate *opaque_delegate_,
                                       void *data) -> TfLiteStatus {
    auto delegate_prepared = static_cast<bool *>(data);
    *delegate_prepared = true;
    auto reg_ex = TfLiteRegistrationExternalCreate(
        kTfLiteBuiltinDelegate, "Test driver Openvino delegate", /*version=*/1);
    TfLiteRegistrationExternalSetInit(
        reg_ex,
        [](TfLiteOpaqueContext *opaque_context, const char *buffer,
           size_t length) -> void * {
          auto openvino_graph_builder_test =
              std::make_unique<tflite::openvinodelegate::OpenVINOGraphBuilder>(
                  std::make_unique<NodeManager>());
          std::vector<int> compute_inputs_ = {};
          std::vector<int> outputs_ = {};
          const TfLiteOpaqueDelegateParams *params =
              reinterpret_cast<const TfLiteOpaqueDelegateParams *>(buffer);
          const std::unordered_set<int> inputs(
              &params->input_tensors->data[0],
              &params->input_tensors->data[params->input_tensors->size]);

          for (int o = 0; o < params->output_tensors->size; o++) {
            const int output_tensor_idx = params->output_tensors->data[o];
            outputs_.push_back(output_tensor_idx);
          }
          for (int i = 0; i < params->nodes_to_replace->size; ++i) {
            const int delegate_node_id = params->nodes_to_replace->data[i];
            TfLiteOpaqueNode *node = nullptr;
            TfLiteRegistrationExternal *registration = nullptr;
            TfLiteOpaqueContextGetNodeAndRegistration(
                opaque_context, delegate_node_id, &node, &registration);
            int inputs_size = TfLiteOpaqueNodeNumberOfInputs(node);
            for (int k = 0; k < inputs_size; k++) {
              const int *inputs_data;
              int num_inputs;
              TfLiteStatus tf_status =
                  TfLiteOpaqueNodeInputs(node, &inputs_data, &num_inputs);
              const int t = inputs_data[k];
              const void *data = nullptr;
              auto opaque_tensor =
                  TfLiteOpaqueContextGetOpaqueTensor(opaque_context, t);
              auto allocation_type =
                  TfLiteOpaqueTensorGetAllocationType(opaque_tensor);
              if (allocation_type == kTfLiteMmapRo) {
                data = TfLiteOpaqueTensorData(opaque_tensor);

                if (openvino_graph_builder_test->CreateConstNode(
                        opaque_context, t) != kTfLiteOk)
                  exit(0);
              }
              if (inputs.count(t) != 0) {
                if (data == nullptr) {
                  static int count = 0;
                  if (count == 0) {
                    if (openvino_graph_builder_test->AddInputParams(
                            opaque_tensor, t) != kTfLiteOk)
                      exit(0);
                    compute_inputs_.push_back(t);
                    count++;
                  }
                }
              }
            }
            EXPECT_EQ(
                kTfLiteOk,
                openvino_graph_builder_test->CreateNodeFromTfLiteOp(
                    delegate_node_id, registration, node, opaque_context));
          }

          openvino_graph_builder_test->UpdateResultNodes(opaque_context, {});
          std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(
              openvino_graph_builder_test->getResultNodes(),
              openvino_graph_builder_test->getInputParams());
          ov::Core openvino_delegate_core_;
          ov::CompiledModel compiled_model_;
          std::string deviceStr = "CPU";
          ov::InferRequest infer_request_;
          if (model) {
            compiled_model_ =
                openvino_delegate_core_.compile_model(model, deviceStr);
            ov::pass::Manager manager;
            manager.register_pass<ov::pass::Serialize>("/tmp/model.xml",
                                                       "/tmp/model.bin");
            manager.run_passes(model);
          }

          infer_request_ = compiled_model_.create_infer_request();
          return &compiled_model_;
        });

    TfLiteRegistrationExternalSetInvoke(
        reg_ex,
        [](TfLiteOpaqueContext *context, TfLiteOpaqueNode *opaque_node)
            -> TfLiteStatus { return kTfLiteOk; });

    TfLiteRegistrationExternalSetFree(
        reg_ex, [](TfLiteOpaqueContext *context, void *data) {});

    TfLiteIntArray *execution_plan;
    TF_LITE_ENSURE_STATUS(
        TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan));
    TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        opaque_context, reg_ex, execution_plan, opaque_delegate_);
    return kTfLiteOk;
  };

  TfLiteModel *model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");

  TfLiteOpaqueDelegate *opaque_delegate =
      TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);
  TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate);
  TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, options);

  EXPECT_TRUE(delegate_prepared);

  TfLiteInterpreterOptionsDelete(options);
  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
  TfLiteOpaqueDelegateDelete(opaque_delegate);
}

TEST_F(OpenVINOGraphBuilderTest, CreateNodeFromTfLiteOp_InvalidNodeId) {
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  bool delegate_prepared = false;

  opaque_delegate_builder.data = &delegate_prepared;
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext *opaque_context,
                                       TfLiteOpaqueDelegate *opaque_delegate_,
                                       void *data) -> TfLiteStatus {
    auto delegate_prepared = static_cast<bool *>(data);
    *delegate_prepared = true;
    auto reg_ex = TfLiteRegistrationExternalCreate(
        kTfLiteBuiltinDelegate, "Test driver Openvino delegate", /*version=*/1);

    TfLiteRegistrationExternalSetInit(
        reg_ex,
        [](TfLiteOpaqueContext *opaque_context, const char *buffer,
           size_t length) -> void * {
          auto openvino_graph_builder_test =
              std::make_unique<tflite::openvinodelegate::OpenVINOGraphBuilder>(
                  std::make_unique<NodeManager>());
          std::vector<int> compute_inputs_ = {};
          std::vector<int> outputs_ = {};
          const TfLiteOpaqueDelegateParams *params =
              reinterpret_cast<const TfLiteOpaqueDelegateParams *>(buffer);
          const std::unordered_set<int> inputs(
              &params->input_tensors->data[0],
              &params->input_tensors->data[params->input_tensors->size]);

          for (int o = 0; o < params->output_tensors->size; o++) {
            const int output_tensor_idx = params->output_tensors->data[o];
            outputs_.push_back(output_tensor_idx);
          }
          for (int i = 0; i < params->nodes_to_replace->size; ++i) {
            const int delegate_node_id = params->nodes_to_replace->data[i];
            TfLiteOpaqueNode *node = nullptr;
            TfLiteRegistrationExternal *registration = nullptr;
            TfLiteOpaqueContextGetNodeAndRegistration(
                opaque_context, delegate_node_id, &node, &registration);
            int inputs_size = TfLiteOpaqueNodeNumberOfInputs(node);
            for (int k = 0; k < inputs_size; k++) {
              const int *inputs_data;
              int num_inputs;
              TfLiteStatus tf_status =
                  TfLiteOpaqueNodeInputs(node, &inputs_data, &num_inputs);
              const int t = inputs_data[k];
              const void *data = nullptr;
              auto opaque_tensor =
                  TfLiteOpaqueContextGetOpaqueTensor(opaque_context, t);
              auto allocation_type =
                  TfLiteOpaqueTensorGetAllocationType(opaque_tensor);
              if (allocation_type == kTfLiteMmapRo) {
                data = TfLiteOpaqueTensorData(opaque_tensor);

                if (openvino_graph_builder_test->CreateConstNode(
                        opaque_context, t) != kTfLiteOk)
                  exit(0);
              }
              if (inputs.count(t) != 0) {
                if (data == nullptr) {
                  static int count = 0;
                  if (count == 0) {
                    if (openvino_graph_builder_test->AddInputParams(
                            opaque_tensor, t) != kTfLiteOk)
                      exit(0);
                    compute_inputs_.push_back(t);
                    count++;
                  }
                }
              }
            }
            EXPECT_EQ(kTfLiteError,
                      openvino_graph_builder_test->CreateNodeFromTfLiteOp(
                          -1, registration, node, opaque_context));
          }

          openvino_graph_builder_test->UpdateResultNodes(opaque_context, {});
          std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(
              openvino_graph_builder_test->getResultNodes(),
              openvino_graph_builder_test->getInputParams());
          ov::Core openvino_delegate_core_;
          ov::CompiledModel compiled_model_;
          std::string deviceStr = "CPU";
          ov::InferRequest infer_request_;
          if (model) {
            compiled_model_ =
                openvino_delegate_core_.compile_model(model, deviceStr);
            ov::pass::Manager manager;
            manager.register_pass<ov::pass::Serialize>("/tmp/model.xml",
                                                       "/tmp/model.bin");
            manager.run_passes(model);
          }

          infer_request_ = compiled_model_.create_infer_request();
          return &compiled_model_;
        });

    TfLiteRegistrationExternalSetInvoke(
        reg_ex,
        [](TfLiteOpaqueContext *context, TfLiteOpaqueNode *opaque_node)
            -> TfLiteStatus { return kTfLiteOk; });

    TfLiteRegistrationExternalSetFree(
        reg_ex, [](TfLiteOpaqueContext *context, void *data) {});

    TfLiteIntArray *execution_plan;
    TF_LITE_ENSURE_STATUS(
        TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan));
    TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        opaque_context, reg_ex, execution_plan, opaque_delegate_);
    return kTfLiteOk;
  };

  TfLiteModel *model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");

  TfLiteOpaqueDelegate *opaque_delegate =
      TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);
  TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate);
  TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, options);

  EXPECT_TRUE(delegate_prepared);

  TfLiteInterpreterOptionsDelete(options);
  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
  TfLiteOpaqueDelegateDelete(opaque_delegate);
}

TEST_F(OpenVINOGraphBuilderTest, CreateNodeFromTfLiteOp_InvalidRegistration) {
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  bool delegate_prepared = false;

  opaque_delegate_builder.data = &delegate_prepared;
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext *opaque_context,
                                       TfLiteOpaqueDelegate *opaque_delegate_,
                                       void *data) -> TfLiteStatus {
    auto delegate_prepared = static_cast<bool *>(data);
    *delegate_prepared = true;
    auto reg_ex = TfLiteRegistrationExternalCreate(
        kTfLiteBuiltinDelegate, "Test driver Openvino delegate", /*version=*/1);
    TfLiteRegistrationExternalSetInit(
        reg_ex,
        [](TfLiteOpaqueContext *opaque_context, const char *buffer,
           size_t length) -> void * {
          auto openvino_graph_builder_test =
              std::make_unique<tflite::openvinodelegate::OpenVINOGraphBuilder>(
                  std::make_unique<NodeManager>());
          std::vector<int> compute_inputs_ = {};
          std::vector<int> outputs_ = {};
          const TfLiteOpaqueDelegateParams *params =
              reinterpret_cast<const TfLiteOpaqueDelegateParams *>(buffer);
          const std::unordered_set<int> inputs(
              &params->input_tensors->data[0],
              &params->input_tensors->data[params->input_tensors->size]);

          for (int o = 0; o < params->output_tensors->size; o++) {
            const int output_tensor_idx = params->output_tensors->data[o];
            outputs_.push_back(output_tensor_idx);
          }
          for (int i = 0; i < params->nodes_to_replace->size; ++i) {
            const int delegate_node_id = params->nodes_to_replace->data[i];
            TfLiteOpaqueNode *node = nullptr;
            TfLiteRegistrationExternal *registration = nullptr;
            TfLiteOpaqueContextGetNodeAndRegistration(
                opaque_context, delegate_node_id, &node, &registration);
            int inputs_size = TfLiteOpaqueNodeNumberOfInputs(node);
            for (int k = 0; k < inputs_size; k++) {
              const int *inputs_data;
              int num_inputs;
              TfLiteStatus tf_status =
                  TfLiteOpaqueNodeInputs(node, &inputs_data, &num_inputs);
              const int t = inputs_data[k];
              const void *data = nullptr;
              auto opaque_tensor =
                  TfLiteOpaqueContextGetOpaqueTensor(opaque_context, t);
              auto allocation_type =
                  TfLiteOpaqueTensorGetAllocationType(opaque_tensor);
              if (allocation_type == kTfLiteMmapRo) {
                data = TfLiteOpaqueTensorData(opaque_tensor);

                if (openvino_graph_builder_test->CreateConstNode(
                        opaque_context, t) != kTfLiteOk)
                  exit(0);
              }
              if (inputs.count(t) != 0) {
                if (data == nullptr) {
                  static int count = 0;
                  if (count == 0) {
                    if (openvino_graph_builder_test->AddInputParams(
                            opaque_tensor, t) != kTfLiteOk)
                      exit(0);
                    compute_inputs_.push_back(t);
                    count++;
                  }
                }
              }
            }
            EXPECT_EQ(kTfLiteError,
                      openvino_graph_builder_test->CreateNodeFromTfLiteOp(
                          delegate_node_id, nullptr, node, opaque_context));
          }

          openvino_graph_builder_test->UpdateResultNodes(opaque_context, {});
          std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(
              openvino_graph_builder_test->getResultNodes(),
              openvino_graph_builder_test->getInputParams());
          ov::Core openvino_delegate_core_;
          ov::CompiledModel compiled_model_;
          std::string deviceStr = "CPU";
          ov::InferRequest infer_request_;
          if (model) {
            compiled_model_ =
                openvino_delegate_core_.compile_model(model, deviceStr);
            ov::pass::Manager manager;
            manager.register_pass<ov::pass::Serialize>("/tmp/model.xml",
                                                       "/tmp/model.bin");
            manager.run_passes(model);
          }

          infer_request_ = compiled_model_.create_infer_request();
          return &compiled_model_;
        });

    TfLiteRegistrationExternalSetInvoke(
        reg_ex,
        [](TfLiteOpaqueContext *context, TfLiteOpaqueNode *opaque_node)
            -> TfLiteStatus { return kTfLiteOk; });

    TfLiteRegistrationExternalSetFree(
        reg_ex, [](TfLiteOpaqueContext *context, void *data) {});

    TfLiteIntArray *execution_plan;
    TF_LITE_ENSURE_STATUS(
        TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan));
    TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        opaque_context, reg_ex, execution_plan, opaque_delegate_);
    return kTfLiteOk;
  };

  TfLiteModel *model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");

  TfLiteOpaqueDelegate *opaque_delegate =
      TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);
  TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate);
  TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, options);

  EXPECT_TRUE(delegate_prepared);

  TfLiteInterpreterOptionsDelete(options);
  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
  TfLiteOpaqueDelegateDelete(opaque_delegate);
}

TEST_F(OpenVINOGraphBuilderTest, CreateNodeFromTfLiteOp_InvalidNode) {
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  bool delegate_prepared = false;

  opaque_delegate_builder.data = &delegate_prepared;
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext *opaque_context,
                                       TfLiteOpaqueDelegate *opaque_delegate_,
                                       void *data) -> TfLiteStatus {
    auto delegate_prepared = static_cast<bool *>(data);
    *delegate_prepared = true;
    auto reg_ex = TfLiteRegistrationExternalCreate(
        kTfLiteBuiltinDelegate, "Test driver Openvino delegate", /*version=*/1);
    TfLiteRegistrationExternalSetInit(
        reg_ex,
        [](TfLiteOpaqueContext *opaque_context, const char *buffer,
           size_t length) -> void * {
          auto openvino_graph_builder_test =
              std::make_unique<tflite::openvinodelegate::OpenVINOGraphBuilder>(
                  std::make_unique<NodeManager>());
          std::vector<int> compute_inputs_ = {};
          std::vector<int> outputs_ = {};
          const TfLiteOpaqueDelegateParams *params =
              reinterpret_cast<const TfLiteOpaqueDelegateParams *>(buffer);
          const std::unordered_set<int> inputs(
              &params->input_tensors->data[0],
              &params->input_tensors->data[params->input_tensors->size]);

          for (int o = 0; o < params->output_tensors->size; o++) {
            const int output_tensor_idx = params->output_tensors->data[o];
            outputs_.push_back(output_tensor_idx);
          }
          for (int i = 0; i < params->nodes_to_replace->size; ++i) {
            const int delegate_node_id = params->nodes_to_replace->data[i];
            TfLiteOpaqueNode *node = nullptr;
            TfLiteRegistrationExternal *registration = nullptr;
            TfLiteOpaqueContextGetNodeAndRegistration(
                opaque_context, delegate_node_id, &node, &registration);
            int inputs_size = TfLiteOpaqueNodeNumberOfInputs(node);
            for (int k = 0; k < inputs_size; k++) {
              const int *inputs_data;
              int num_inputs;
              TfLiteStatus tf_status =
                  TfLiteOpaqueNodeInputs(node, &inputs_data, &num_inputs);
              const int t = inputs_data[k];
              const void *data = nullptr;
              auto opaque_tensor =
                  TfLiteOpaqueContextGetOpaqueTensor(opaque_context, t);
              auto allocation_type =
                  TfLiteOpaqueTensorGetAllocationType(opaque_tensor);
              if (allocation_type == kTfLiteMmapRo) {
                data = TfLiteOpaqueTensorData(opaque_tensor);

                if (openvino_graph_builder_test->CreateConstNode(
                        opaque_context, t) != kTfLiteOk)
                  exit(0);
              }
              if (inputs.count(t) != 0) {
                if (data == nullptr) {
                  static int count = 0;
                  if (count == 0) {
                    if (openvino_graph_builder_test->AddInputParams(
                            opaque_tensor, t) != kTfLiteOk)
                      exit(0);
                    compute_inputs_.push_back(t);
                    count++;
                  }
                }
              }
            }
            EXPECT_EQ(
                kTfLiteError,
                openvino_graph_builder_test->CreateNodeFromTfLiteOp(
                    delegate_node_id, registration, nullptr, opaque_context));
          }

          openvino_graph_builder_test->UpdateResultNodes(opaque_context, {});
          std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(
              openvino_graph_builder_test->getResultNodes(),
              openvino_graph_builder_test->getInputParams());
          ov::Core openvino_delegate_core_;
          ov::CompiledModel compiled_model_;
          std::string deviceStr = "CPU";
          ov::InferRequest infer_request_;
          if (model) {
            compiled_model_ =
                openvino_delegate_core_.compile_model(model, deviceStr);
            ov::pass::Manager manager;
            manager.register_pass<ov::pass::Serialize>("/tmp/model.xml",
                                                       "/tmp/model.bin");
            manager.run_passes(model);
          }

          infer_request_ = compiled_model_.create_infer_request();
          return &compiled_model_;
        });

    TfLiteRegistrationExternalSetInvoke(
        reg_ex,
        [](TfLiteOpaqueContext *context, TfLiteOpaqueNode *opaque_node)
            -> TfLiteStatus { return kTfLiteOk; });

    TfLiteRegistrationExternalSetFree(
        reg_ex, [](TfLiteOpaqueContext *context, void *data) {});

    TfLiteIntArray *execution_plan;
    TF_LITE_ENSURE_STATUS(
        TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan));
    TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        opaque_context, reg_ex, execution_plan, opaque_delegate_);
    return kTfLiteOk;
  };

  TfLiteModel *model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");

  TfLiteOpaqueDelegate *opaque_delegate =
      TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);
  TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate);
  TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, options);

  EXPECT_TRUE(delegate_prepared);

  TfLiteInterpreterOptionsDelete(options);
  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
  TfLiteOpaqueDelegateDelete(opaque_delegate);
}

TEST_F(OpenVINOGraphBuilderTest, CreateNodeFromTfLiteOp_InvalidContext) {
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  bool delegate_prepared = false;

  opaque_delegate_builder.data = &delegate_prepared;
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext *opaque_context,
                                       TfLiteOpaqueDelegate *opaque_delegate_,
                                       void *data) -> TfLiteStatus {
    auto delegate_prepared = static_cast<bool *>(data);
    *delegate_prepared = true;
    auto reg_ex = TfLiteRegistrationExternalCreate(
        kTfLiteBuiltinDelegate, "Test driver Openvino delegate", /*version=*/1);
    TfLiteRegistrationExternalSetInit(
        reg_ex,
        [](TfLiteOpaqueContext *opaque_context, const char *buffer,
           size_t length) -> void * {
          auto openvino_graph_builder_test =
              std::make_unique<tflite::openvinodelegate::OpenVINOGraphBuilder>(
                  std::make_unique<NodeManager>());
          std::vector<int> compute_inputs_ = {};
          std::vector<int> outputs_ = {};
          const TfLiteOpaqueDelegateParams *params =
              reinterpret_cast<const TfLiteOpaqueDelegateParams *>(buffer);
          const std::unordered_set<int> inputs(
              &params->input_tensors->data[0],
              &params->input_tensors->data[params->input_tensors->size]);

          for (int o = 0; o < params->output_tensors->size; o++) {
            const int output_tensor_idx = params->output_tensors->data[o];
            outputs_.push_back(output_tensor_idx);
          }
          for (int i = 0; i < params->nodes_to_replace->size; ++i) {
            const int delegate_node_id = params->nodes_to_replace->data[i];
            TfLiteOpaqueNode *node = nullptr;
            TfLiteRegistrationExternal *registration = nullptr;
            TfLiteOpaqueContextGetNodeAndRegistration(
                opaque_context, delegate_node_id, &node, &registration);
            int inputs_size = TfLiteOpaqueNodeNumberOfInputs(node);
            for (int k = 0; k < inputs_size; k++) {
              const int *inputs_data;
              int num_inputs;
              TfLiteStatus tf_status =
                  TfLiteOpaqueNodeInputs(node, &inputs_data, &num_inputs);
              const int t = inputs_data[k];
              const void *data = nullptr;
              auto opaque_tensor =
                  TfLiteOpaqueContextGetOpaqueTensor(opaque_context, t);
              auto allocation_type =
                  TfLiteOpaqueTensorGetAllocationType(opaque_tensor);
              if (allocation_type == kTfLiteMmapRo) {
                data = TfLiteOpaqueTensorData(opaque_tensor);

                if (openvino_graph_builder_test->CreateConstNode(
                        opaque_context, t) != kTfLiteOk)
                  exit(0);
              }
              if (inputs.count(t) != 0) {
                if (data == nullptr) {
                  static int count = 0;
                  if (count == 0) {
                    if (openvino_graph_builder_test->AddInputParams(
                            opaque_tensor, t) != kTfLiteOk)
                      exit(0);
                    compute_inputs_.push_back(t);
                    count++;
                  }
                }
              }
            }
            EXPECT_EQ(kTfLiteError,
                      openvino_graph_builder_test->CreateNodeFromTfLiteOp(
                          delegate_node_id, registration, node, nullptr));
          }

          openvino_graph_builder_test->UpdateResultNodes(opaque_context, {});
          std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(
              openvino_graph_builder_test->getResultNodes(),
              openvino_graph_builder_test->getInputParams());
          ov::Core openvino_delegate_core_;
          ov::CompiledModel compiled_model_;
          std::string deviceStr = "CPU";
          ov::InferRequest infer_request_;
          if (model) {
            compiled_model_ =
                openvino_delegate_core_.compile_model(model, deviceStr);
            ov::pass::Manager manager;
            manager.register_pass<ov::pass::Serialize>("/tmp/model.xml",
                                                       "/tmp/model.bin");
            manager.run_passes(model);
          }

          infer_request_ = compiled_model_.create_infer_request();
          return &compiled_model_;
        });

    TfLiteRegistrationExternalSetInvoke(
        reg_ex,
        [](TfLiteOpaqueContext *context, TfLiteOpaqueNode *opaque_node)
            -> TfLiteStatus { return kTfLiteOk; });

    TfLiteRegistrationExternalSetFree(
        reg_ex, [](TfLiteOpaqueContext *context, void *data) {});

    TfLiteIntArray *execution_plan;
    TF_LITE_ENSURE_STATUS(
        TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan));
    TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        opaque_context, reg_ex, execution_plan, opaque_delegate_);
    return kTfLiteOk;
  };

  TfLiteModel *model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");

  TfLiteOpaqueDelegate *opaque_delegate =
      TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);
  TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate);
  TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, options);

  EXPECT_TRUE(delegate_prepared);

  TfLiteInterpreterOptionsDelete(options);
  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
  TfLiteOpaqueDelegateDelete(opaque_delegate);
}
