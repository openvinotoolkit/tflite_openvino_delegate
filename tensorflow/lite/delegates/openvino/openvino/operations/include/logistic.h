#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_LOGISTIC_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_LOGISTIC_H_

#include "tensorflow/lite/delegates/openvino/operations/operations_base.h"

namespace tflite {
namespace openvinodelegate {

class Logistic : public OperationsBase {
public:
    Logistic(int operationIndex) {}
    TfLiteStatus CreateNode() override;
};

}  // namespace openvinodelegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_LOGISTIC_H_
