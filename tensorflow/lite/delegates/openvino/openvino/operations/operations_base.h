#ifndef TENSORFLOW_LITE_DELEGATES_OPERATIONS_BASE_H_
#define TENSORFLOW_LITE_DELEGATES_OPERATIONS_BASE_H_

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset8.hpp>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/openvino/operations/openvino_node_manager.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/tools/logging.h"

#define TFLITE_INPUT_NODE_1 0
#define TFLITE_INPUT_NODE_2 1
#define TFLITE_FILTER_NODE 1
#define TFLITE_BIAS_NODE 2
#define TFLITE_SHAPE_NODE 1
#define TFLITE_TRANSPOSE_CONV_OUTPUT_SHAPE 0
#define TFLITE_TRANSPOSE_CONV_WEIGHTS 1
#define TFLITE_TRANSPOSE_CONV_INPUT 2
#define TFLITE_TRANSPOSE_CONV_BIAS 3

namespace tflite {
namespace openvinodelegate {

class OperationsBase {
public:
    void UpdateNodeInfo(void *data, int size, void *builtin_data) {
        tensor_indices_ = (int *)data;
        tensor_indices_size_ = size;
        SetBuiltinData(builtin_data);
    }
    void SetGraphData(const TfLiteOpaqueContext *context, NodeManager *node_manager) {
        context_ = context;
        node_manager_ = node_manager;
    }

    std::shared_ptr<ov::Node> GetOpResultNode() { return output_node; }
    virtual TfLiteStatus CreateNode() = 0;

protected:
    // tflite runtime related info to be added in Model BUilder
    int operation_index_;
    std::shared_ptr<ov::Node> output_node;
    void *GetBuiltinData() { return builtin_data_; }
    void SetBuiltinData(void *builtin_data) { builtin_data_ = builtin_data; }
    std::shared_ptr<ov::Node> getInputNode(int index) {
        return node_manager_->getInterimNodeOutput(index);
    }
    NodeManager *GetGraphNodeManager() { return node_manager_; }

    template <typename T>
    std::shared_ptr<ov::Node> CreateConstNode(ov::element::Type elementType, ov::Shape shape,
                                              std::vector<T> data) {
        return std::make_shared<ov::opset8::Constant>(elementType, shape, data);
    }

    TfLiteStatus CalculatePadding(TfLitePadding padding, ov::op::PadType &auto_pad) {
        switch (padding) {
            case kTfLitePaddingSame: {
                auto_pad = ov::op::PadType::SAME_UPPER;
                return kTfLiteOk;
            }
            case kTfLitePaddingValid: {
                auto_pad = ov::op::PadType::VALID;
                return kTfLiteOk;
            }
            default:
                return kTfLiteError;
        }
    }

    std::shared_ptr<ov::Node> ApplyActivation(std::shared_ptr<ov::Node> input,
                                              TfLiteFusedActivation activation) {
        // TODO: change activation type from Tflite to OV runtime
        switch (activation) {
            case kTfLiteActNone:
                return input;
            case kTfLiteActRelu:
                return std::make_shared<ov::opset8::Relu>(input);
            case kTfLiteActReluN1To1:
                return std::make_shared<ov::opset8::Clamp>(input, -1, 1);
            case kTfLiteActRelu6:
                return std::make_shared<ov::opset8::Clamp>(input, 0, 6);
            case kTfLiteActTanh:
                return std::make_shared<ov::opset8::Tanh>(input);
            case kTfLiteActSignBit:
                return nullptr;
            case kTfLiteActSigmoid:
                return std::make_shared<ov::opset8::Sigmoid>(input);
            default:
                return nullptr;
        }
    }

    std::vector<int> GetDims(int index) {
        auto t = TfLiteOpaqueContextGetOpaqueTensor(context_, index);
        int32_t num_dims;
        num_dims = TfLiteOpaqueTensorNumDims(t);
        std::vector<int> dims(num_dims);
        for (int i = 0; i < num_dims; i++) {
            dims[i] = TfLiteOpaqueTensorDim(t, i);
        }
        return dims;
    }

    void GetTensorData(int index, void *data) {
        auto opaque_tensor = TfLiteOpaqueContextGetOpaqueTensor(context_, index);
        void *tensor_data = TfLiteOpaqueTensorData(opaque_tensor);
        auto size = TfLiteOpaqueTensorByteSize(opaque_tensor);
        std::memcpy(data, tensor_data, size);
    }

    TfLiteStatus GetTensorType(TfLiteOpaqueTensor *t, ov::element::Type *ov_element_type) {
        TfLiteType tensor_type = TfLiteOpaqueTensorType(t);
        switch (tensor_type) {
            case kTfLiteFloat32:
                *ov_element_type = ov::element::f32;
                break;
            case kTfLiteInt32:
                *ov_element_type = ov::element::i32;
                break;
            case kTfLiteUInt8:
                *ov_element_type = ov::element::u8;
                break;
            case kTfLiteInt64:
                *ov_element_type = ov::element::i64;
                break;
            case kTfLiteBool:
                *ov_element_type = ov::element::boolean;
                break;
            case kTfLiteInt16:
                *ov_element_type = ov::element::i16;
                break;
            case kTfLiteInt8:
                *ov_element_type = ov::element::i8;
                break;
            case kTfLiteFloat16:
                *ov_element_type = ov::element::f16;
                break;
            case kTfLiteFloat64:
                *ov_element_type = ov::element::f64;
                break;
            case kTfLiteUInt64:
                *ov_element_type = ov::element::u64;
                break;
            case kTfLiteUInt32:
                *ov_element_type = ov::element::u32;
                break;
            case kTfLiteUInt16:
                *ov_element_type = ov::element::u16;
                break;
            case kTfLiteInt4:
                *ov_element_type = ov::element::i4;
                break;
            default:
                TFLITE_LOG(ERROR) << "Element type not supported\n";
                return kTfLiteError;
        }
        return kTfLiteOk;
    }

    void *GetTensorDataPtrAndSize(int index, unsigned int *size) {
        auto opaque_tensor = TfLiteOpaqueContextGetOpaqueTensor(context_, index);
        void *tensor_data = TfLiteOpaqueTensorData(opaque_tensor);
        ov::element::Type ov_element_type;

        if (GetTensorType(opaque_tensor, &ov_element_type) != kTfLiteOk) {
            *size = 0;
            return nullptr;
        }
        *size = TfLiteOpaqueTensorByteSize(opaque_tensor) / sizeof(ov_element_type);
        return tensor_data;
    }

    int *tensor_indices_;
    int tensor_indices_size_;

private:
    void *builtin_data_ = nullptr;
    int op_type_ = 0;
    NodeManager *node_manager_;
    const TfLiteOpaqueContext *context_;
};

}  // namespace openvinodelegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_OPERATIOSN_BASE_H_
