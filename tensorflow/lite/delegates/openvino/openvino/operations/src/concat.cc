#include "tensorflow/lite/delegates/openvino/operations/include/concat.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus Concat::CreateNode() {
    TfLiteConcatenationParams *concat_params = (TfLiteConcatenationParams *)GetBuiltinData();
    auto inputNode1 = getInputNode(tensor_indices_[0]);
    if (inputNode1 == nullptr) {
        TFLITE_LOG(INFO) << "input node 1 is null\n";
        return kTfLiteError;
    }

    auto inputNode2 = getInputNode(tensor_indices_[1]);
    if (inputNode2 == nullptr) {
        TFLITE_LOG(INFO) << "input Node 2 is null\n";
        return kTfLiteError;
    }

    // TODO: Replace the hard coded value with logic.
    int axis = 1;
    size_t n = tensor_indices_size_;
    std::vector<ov::Output<ov::Node>> inputs;
    for (size_t i = 0; i < n; i++) {
        auto inputOp = getInputNode(tensor_indices_[i]);
        inputs.push_back(inputOp);
    }

    auto concatNode = std::make_shared<ov::opset8::Concat>(inputs, axis);
    output_node = ApplyActivation(concatNode, concat_params->activation);

    return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
