#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_MAXPOOL2D_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_MAXPOOL2D_H_

#include "tensorflow/lite/delegates/openvino/operations/operations_base.h"

namespace tflite {
namespace openvinodelegate {

class MaxPool2D : public OperationsBase {
public:
    MaxPool2D(int operationIndex) {}
    TfLiteStatus CreateNode() override;
};

}  // namespace openvinodelegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_MAXPOOL2D_H_
