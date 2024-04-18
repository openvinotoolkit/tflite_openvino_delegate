#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_RESHAPE_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_RESHAPE_H_

#include "tensorflow/lite/delegates/openvino/operations/operations_base.h"

namespace tflite {
namespace openvinodelegate {

class Reshape : public OperationsBase {
public:
    Reshape(int operationIndex) {}
    TfLiteStatus CreateNode() override;
};

}  // namespace openvinodelegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_RESHAPE_H_
