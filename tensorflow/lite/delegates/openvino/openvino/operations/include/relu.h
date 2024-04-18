#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_RELU_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_RELU_H_

#include "tensorflow/lite/delegates/openvino/operations/operations_base.h"

namespace tflite {
namespace openvinodelegate {

class Relu : public OperationsBase {
public:
    Relu(int operationIndex) {}
    TfLiteStatus CreateNode() override;
};

}  // namespace openvinodelegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_RELU_H_
