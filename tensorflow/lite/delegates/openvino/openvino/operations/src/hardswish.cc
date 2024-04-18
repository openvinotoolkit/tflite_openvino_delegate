#include "tensorflow/lite/delegates/openvino/operations/include/hardswish.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus HardSwish::CreateNode() {
    auto input_node = getInputNode(tensor_indices_[TFLITE_INPUT_NODE_1]);
    if (input_node == nullptr) {
        return kTfLiteError;
    }
    output_node = std::make_shared<ov::op::v4::HSwish>(input_node);
    return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
