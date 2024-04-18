#include "tensorflow/lite/delegates/openvino/operations/include/mean.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus Mean::CreateNode() {
    TfLiteReducerParams *reduce_params = (TfLiteReducerParams *)GetBuiltinData();

    auto input_node = getInputNode(tensor_indices_[TFLITE_INPUT_NODE_1]);
    if (input_node == nullptr) {
        TFLITE_LOG(INFO) << "input node is null\n";
        return kTfLiteError;
    }

    std::shared_ptr<ov::Node> reduction_axes = getInputNode(tensor_indices_[TFLITE_INPUT_NODE_2]);
    if (reduction_axes == nullptr) {
        TFLITE_LOG(INFO) << "reduction_axes is null\n";
        return kTfLiteError;
    }

    std::vector<int32_t> axes_vec = {};
    unsigned int size;
    void *data = GetTensorDataPtrAndSize(tensor_indices_[1], &size);
    if (size == 0 or data == nullptr) {
        TFLITE_LOG(INFO) << "Failed to get reduction_axes data\n";
        return kTfLiteError;
    }

    for (auto i = 0; i < size; i++) {
        int ax = *((int *)(data) + i);
        if (ax == 1)
            axes_vec.push_back(2);
        else if (ax == 2)
            axes_vec.push_back(3);
        else if (ax == 3)
            axes_vec.push_back(1);
        else
            axes_vec.push_back(ax);
    }

    auto axes_node = CreateConstNode(ov::element::i32, {size}, axes_vec);
    if (axes_node == nullptr) {
        TFLITE_LOG(INFO) << "Failed to create const node for axes\n";
        return kTfLiteError;
    }

    bool keep_dims = (reduce_params->keep_dims > 0) ? true : false;
    output_node = std::make_shared<ov::op::v1::ReduceMean>(input_node, axes_node, keep_dims);

    return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
