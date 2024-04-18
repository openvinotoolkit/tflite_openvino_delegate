#include <gtest/gtest.h>

#include "openvino_graph_builder.h"

namespace tflite {
namespace openvinodelegate {

TEST(AddInputParams, checkInvalidDims) {
    const std::vector<int> x = {};
    TfLiteIntArray *lite = TfLiteIntArrayCreate(x.size());
    for (size_t i = 0; i < x.size(); i++) lite->data[i] = x[i];

    TfLiteTensor t_ptr;
    t_ptr.dims = lite;
    std::unique_ptr<OpenVINOGraphBuilder> graph_builder_test =
        std::make_unique<OpenVINOGraphBuilder>(std::make_unique<NodeManager>());
    EXPECT_EQ(graph_builder_test->AddInputParams(t_ptr, 1), kTfLiteError);
    TfLiteIntArrayFree(lite);
}

TEST(AddInputParams, checkValidDims) {
    TfLiteIntArray t;
    const std::vector<int> x = {1, 3, 3, 2};
    TfLiteIntArray *lite = TfLiteIntArrayCreate(x.size());
    for (size_t i = 0; i < x.size(); i++) lite->data[i] = x[i];

    TfLiteTensor t_ptr;
    t_ptr.dims = lite;
    std::unique_ptr<OpenVINOGraphBuilder> graph_builder_test =
        std::make_unique<OpenVINOGraphBuilder>(std::make_unique<NodeManager>());
    EXPECT_EQ(graph_builder_test->AddInputParams(t_ptr, 1), kTfLiteOk);
    TfLiteIntArrayFree(lite);
}

TEST(AddInputParams, checkNodeState) {
    TfLiteIntArray t;
    const std::vector<int> x = {1, 3, 3, 2};
    TfLiteIntArray *lite = TfLiteIntArrayCreate(x.size());
    for (size_t i = 0; i < x.size(); i++) lite->data[i] = x[i];

    TfLiteTensor t_ptr;
    t_ptr.dims = lite;
    std::unique_ptr<OpenVINOGraphBuilder> graph_builder =
        std::make_unique<OpenVINOGraphBuilder>(std::make_unique<NodeManager>());
    EXPECT_EQ(graph_builder->AddInputParams(t_ptr, 1), kTfLiteOk);
    EXPECT_EQ(graph_builder->getNodeManagerSize() == 1, true);
    TfLiteIntArrayFree(lite);
}

TEST(CreateConstNode, checkInvalidDims) {
    TfLiteIntArray t;
    const std::vector<int> x = {};
    TfLiteIntArray *lite = TfLiteIntArrayCreate(x.size());
    for (size_t i = 0; i < x.size(); i++) lite->data[i] = x[i];

    TfLiteTensor t_ptr;
    t_ptr.dims = lite;
    std::unique_ptr<OpenVINOGraphBuilder> graph_builder_test =
        std::make_unique<OpenVINOGraphBuilder>(std::make_unique<NodeManager>());
    EXPECT_EQ(graph_builder_test->CreateConstNode(t_ptr, 1), kTfLiteError);
    TfLiteIntArrayFree(lite);
}

TEST(CreateConstNode, checkNullConstData) {
    TfLiteIntArray t;
    const std::vector<int> x = {1, 3, 3, 2};
    TfLiteIntArray *lite = TfLiteIntArrayCreate(x.size());
    for (size_t i = 0; i < x.size(); i++) lite->data[i] = x[i];
    TfLitePtrUnion tensor_data;
    tensor_data.raw_const = nullptr;
    TfLiteTensor t_ptr;
    t_ptr.dims = lite;
    t_ptr.data = tensor_data;
    std::unique_ptr<OpenVINOGraphBuilder> graph_builder_test =
        std::make_unique<OpenVINOGraphBuilder>(std::make_unique<NodeManager>());
    EXPECT_EQ(graph_builder_test->CreateConstNode(t_ptr, 1), kTfLiteError);
    TfLiteIntArrayFree(lite);
}

TEST(CreateConstNode, checkNodeState) {
    TfLiteIntArray t;
    const std::vector<int> x = {1, 3, 3, 3};
    TfLiteIntArray *lite = TfLiteIntArrayCreate(x.size());
    for (size_t i = 0; i < x.size(); i++) lite->data[i] = x[i];
    TfLitePtrUnion tensor_data;
    float float_data[3][3][3] = {{{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}},
                                 {{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}},
                                 {{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}}};
    tensor_data.raw_const = (const char *)&float_data;
    TfLiteTensor t_ptr;
    t_ptr.dims = lite;
    t_ptr.data = tensor_data;
    std::unique_ptr<OpenVINOGraphBuilder> graph_builder =
        std::make_unique<OpenVINOGraphBuilder>(std::make_unique<NodeManager>());
    EXPECT_EQ(graph_builder->CreateConstNode(t_ptr, 1), kTfLiteOk);
    EXPECT_EQ(graph_builder->getNodeManagerSize() == 1, true);
    TfLiteIntArrayFree(lite);
}

}  // namespace openvinodelegate
}  // namespace tflite
