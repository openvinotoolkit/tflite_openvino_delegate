#include "tensorflow/lite/delegates/openvino/operations/include/softmax.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus Softmax::CreateNode() {
    TfLiteSoftmaxParams *softmax_params = (TfLiteSoftmaxParams *)GetBuiltinData();
    auto input_node_1 = getInputNode(tensor_indices_[TFLITE_INPUT_NODE_1]);
    if (input_node_1 == nullptr) {
        TFLITE_LOG(INFO) << "input node 1 is null\n";
        return kTfLiteError;
    }

    // NOTE: assumption here is: Tensorflow always computes softmax along
    // channel(last) dimesnsion. After transpose, our channel shifts to dim 1,
    // which is default axis attribute for Softmax.
    output_node = std::make_shared<ov::opset8::Softmax>(input_node_1);
    return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
