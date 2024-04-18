#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_GRAPH_BUILDER_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_GRAPH_BUILDER_H_
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset8.hpp>
#include <vector>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/openvino/operations/include/add.h"
#include "tensorflow/lite/delegates/openvino/operations/include/average_pool_2d.h"
#include "tensorflow/lite/delegates/openvino/operations/include/concat.h"
#include "tensorflow/lite/delegates/openvino/operations/include/conv2d.h"
#include "tensorflow/lite/delegates/openvino/operations/include/depthwise_conv2d.h"
#include "tensorflow/lite/delegates/openvino/operations/include/dequantize.h"
#include "tensorflow/lite/delegates/openvino/operations/include/hardswish.h"
#include "tensorflow/lite/delegates/openvino/operations/include/logistic.h"
#include "tensorflow/lite/delegates/openvino/operations/include/maxpool2d.h"
#include "tensorflow/lite/delegates/openvino/operations/include/mean.h"
#include "tensorflow/lite/delegates/openvino/operations/include/mul.h"
#include "tensorflow/lite/delegates/openvino/operations/include/relu.h"
#include "tensorflow/lite/delegates/openvino/operations/include/relu6.h"
#include "tensorflow/lite/delegates/openvino/operations/include/reshape.h"
#include "tensorflow/lite/delegates/openvino/operations/include/resize_bilinear.h"
#include "tensorflow/lite/delegates/openvino/operations/include/softmax.h"
#include "tensorflow/lite/delegates/openvino/operations/include/tanh.h"
#include "tensorflow/lite/delegates/openvino/operations/include/transpose_conv.h"

#include "tensorflow/lite/delegates/openvino/operations/openvino_node_manager.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace openvinodelegate {

class OpenVINOGraphBuilder {
public:
    OpenVINOGraphBuilder(std::unique_ptr<NodeManager> node_manager) {
        node_manager_ = std::move(node_manager);
    }

    TfLiteStatus convertNHWCtoNCHW(std::vector<int> node_dims, std::shared_ptr<ov::Node> input,
                                   std::shared_ptr<ov::Node> &transposed_node) {
        if (input == nullptr) return kTfLiteError;
        if (node_dims.size() <= 0) return kTfLiteError;

        ov::AxisVector order = {0, 3, 1, 2};
        const auto order_node = std::make_shared<ov::opset8::Constant>(
            ov::element::i32, ov::Shape{order.size()}, order);

        if (node_dims.size() < 4 && node_dims.size() > 0) {
            auto size = node_dims.size();
            for (int i = 0; i < 4 - size; i++) {
                node_dims.insert(node_dims.begin(), 1);
            }
            auto new_size =
                std::make_shared<ov::opset8::Constant>(ov::element::i32, ov::Shape{4}, node_dims);
            input = std::make_shared<ov::opset8::Reshape>(input, new_size, false);
        }

        if (node_dims.size() >= 5) {
            TFLITE_LOG(ERROR) << "5D or greater than 5D tensors are not supported\n";
            return kTfLiteError;
        }
        transposed_node = std::make_shared<ov::opset3::Transpose>(input, order_node);

        return kTfLiteOk;
    }

    TfLiteStatus AddInputParams(const TfLiteOpaqueTensor *t, const int index) {
        if (t == nullptr) return kTfLiteError;
        if (index < 0) return kTfLiteError;

        int32_t num_dims = TfLiteOpaqueTensorNumDims(t);
        std::vector<int> dims(num_dims);
        for (int i = 0; i < num_dims; i++) {
            dims[i] = TfLiteOpaqueTensorDim(t, i);
        }

        if (dims.size() <= 0) return kTfLiteError;

        auto input = std::make_shared<ov::opset3::Parameter>(ov::element::f32,
                                                             ov::Shape(dims.begin(), dims.end()));
        if (input == NULL) {
            return kTfLiteError;
        }
        input_params_.push_back(input);

        std::shared_ptr<ov::Node> interim;
        if (convertNHWCtoNCHW(dims, input, interim) != kTfLiteOk) return kTfLiteError;
        if (interim == nullptr) return kTfLiteError;
        node_manager_->setOutputAtOperandIndex(index, interim);
        node_manager_->insertIndexParameters(index);

        return kTfLiteOk;
    }

    TfLiteStatus CreateConstNode(const TfLiteOpaqueContext *context, const int index) {
        if (context == nullptr) return kTfLiteError;
        const TfLiteOpaqueTensor *t = TfLiteOpaqueContextGetOpaqueTensor(context, index);
        int32_t num_dims;
        ov::element::Type ov_element_type;
        num_dims = TfLiteOpaqueTensorNumDims(t);
        std::vector<int> dims(num_dims);
        for (int i = 0; i < num_dims; i++) {
            dims[i] = TfLiteOpaqueTensorDim(t, i);
        }

        if (dims.size() <= 0) return kTfLiteError;

        const void *data = TfLiteOpaqueTensorData(t);
        if (data == NULL) {
            return kTfLiteError;
        }

        TfLiteType tensor_type = TfLiteOpaqueTensorType(t);
        switch (tensor_type) {
            case kTfLiteFloat32:
                ov_element_type = ov::element::f32;
                break;
            case kTfLiteInt32:
                ov_element_type = ov::element::i32;
                break;
            case kTfLiteUInt8:
                ov_element_type = ov::element::u8;
                break;
            case kTfLiteInt64:
                ov_element_type = ov::element::i64;
                break;
            case kTfLiteBool:
                ov_element_type = ov::element::boolean;
                break;
            case kTfLiteInt16:
                ov_element_type = ov::element::i16;
                break;
            case kTfLiteInt8:
                ov_element_type = ov::element::i8;
                break;
            case kTfLiteFloat16:
                ov_element_type = ov::element::f16;
                break;
            case kTfLiteFloat64:
                ov_element_type = ov::element::f64;
                break;
            case kTfLiteUInt64:
                ov_element_type = ov::element::u64;
                break;
            case kTfLiteUInt32:
                ov_element_type = ov::element::u32;
                break;
            case kTfLiteUInt16:
                ov_element_type = ov::element::u16;
                break;
            case kTfLiteInt4:
                ov_element_type = ov::element::i4;
                break;
            default:
                TFLITE_LOG(ERROR) << "Element type " << tensor_type << " not supported\n";
                return kTfLiteError;
        }

        auto const_node = std::make_shared<ov::opset8::Constant>(
            ov_element_type, ov::Shape(dims.begin(), dims.end()), data);
        if (const_node == NULL) {
            TFLITE_LOG(INFO) << "Error in creating const node\n";
            return kTfLiteError;
        }
        node_manager_->setOutputAtOperandIndex(index, const_node);

        return kTfLiteOk;
    }

    TfLiteStatus UpdateResultNodes(const TfLiteOpaqueContext *context, std::vector<int> outputs) {
        if (context == nullptr) return kTfLiteError;
        if (outputs.size() < 1) return kTfLiteError;

        for (auto o : outputs) {
            auto out_node = node_manager_->getInterimNodeOutput(o);
            auto dims = out_node->get_shape();
            if (dims.size() == 4) {
                ov::AxisVector order;
                order = {0, 2, 3, 1};
                const auto order_node = std::make_shared<ov::opset8::Constant>(
                    ov::element::i64, ov::Shape{order.size()}, order);
                out_node = std::make_shared<ov::opset3::Transpose>(out_node, order_node);
                if (out_node == NULL) {
                    TFLITE_LOG(INFO) << "Error in creating transpose for result node\n";
                    return kTfLiteError;
                }
            }
            result_nodes_.push_back(out_node);
        }

        return kTfLiteOk;
    }

    std::vector<std::shared_ptr<ov::Node>> getResultNodes() { return result_nodes_; }

    std::vector<std::shared_ptr<ov::opset3::Parameter>> getInputParams() { return input_params_; }

    size_t getNodeManagerSize() const { return node_manager_->getNodeCount(); }

    TfLiteStatus CreateNodeFromTfLiteOp(int node_id, TfLiteRegistrationExternal *registration,
                                        TfLiteOpaqueNode *node, TfLiteOpaqueContext *context);
    TfLiteStatus CreateOpClass(int operationIndex, TfLiteRegistrationExternal *registration,
                               std::shared_ptr<OperationsBase> &op_base);

private:
    std::shared_ptr<NodeManager> node_manager_;
    std::vector<std::shared_ptr<ov::opset3::Parameter>> input_params_;
    std::vector<std::shared_ptr<ov::Node>> result_nodes_;
};
}  // namespace openvinodelegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_GRAPH_BUILDER_H_
