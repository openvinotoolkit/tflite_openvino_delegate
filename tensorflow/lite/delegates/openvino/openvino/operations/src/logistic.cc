#include "tensorflow/lite/delegates/openvino/operations/include/logistic.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus Logistic::CreateNode() {
    // Creating input nodes
    auto input_node = getInputNode(tensor_indices_[TFLITE_INPUT_NODE_1]);
    if (input_node == nullptr) {
        return kTfLiteError;
    }
    output_node = ApplyActivation(input_node, kTfLiteActSigmoid);
    return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
