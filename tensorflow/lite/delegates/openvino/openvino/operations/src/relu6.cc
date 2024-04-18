#include "tensorflow/lite/delegates/openvino/operations/include/relu6.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus Relu6::CreateNode() {
    auto input_node = getInputNode(tensor_indices_[TFLITE_INPUT_NODE_1]);
    if (input_node == nullptr) {
        return kTfLiteError;
    }
    output_node = ApplyActivation(input_node, kTfLiteActRelu6);
    return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
