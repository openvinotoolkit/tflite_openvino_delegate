#include "tensorflow/lite/delegates/openvino/operations/include/reshape.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus Reshape::CreateNode() {
    // arg - input node
    auto input_node = getInputNode(tensor_indices_[TFLITE_INPUT_NODE_1]);
    if (input_node == nullptr) {
        TFLITE_LOG(ERROR) << "input node is null\n";
        return kTfLiteError;
    }

    // shape_pattern - shape node
    ov::Output<ov::Node> shape_node = getInputNode(tensor_indices_[TFLITE_SHAPE_NODE]);

    // special_zero
    // Set false since Keras doesn't have special_zero argument

    output_node = std::make_shared<ov::opset3::Reshape>(input_node, shape_node, false);
    if (output_node == nullptr) {
        TFLITE_LOG(ERROR) << "output node is null\n";
        return kTfLiteError;
    }

    return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
