#include "tensorflow/lite/delegates/openvino/operations/include/relu.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus Relu::CreateNode() {
    auto input_node = getInputNode(tensor_indices_[TFLITE_INPUT_NODE_1]);
    if (input_node == nullptr) {
        return kTfLiteError;
    }
    output_node = ApplyActivation(input_node, kTfLiteActRelu);
    return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
