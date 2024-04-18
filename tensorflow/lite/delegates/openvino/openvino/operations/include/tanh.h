#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_TANH_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_TANH_H_

#include "tensorflow/lite/delegates/openvino/operations/operations_base.h"

namespace tflite {
namespace openvinodelegate {

class Tanh : public OperationsBase {
public:
    Tanh(int operationIndex) {}
    TfLiteStatus CreateNode() override;
};

}  // namespace openvinodelegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_TANH_H_
