#include "tensorflow/lite/delegates/openvino/operations/include/tanh.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus Tanh::CreateNode() {
    auto input_node = getInputNode(tensor_indices_[TFLITE_INPUT_NODE_1]);
    if (input_node == nullptr) {
        TFLITE_LOG(ERROR) << "input node is null\n";
        return kTfLiteError;
    }

    output_node = ApplyActivation(input_node, kTfLiteActTanh);
    return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
