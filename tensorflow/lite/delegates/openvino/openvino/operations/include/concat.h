#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_CONCAT_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_CONCAT_H_

#include "tensorflow/lite/delegates/openvino/operations/operations_base.h"

namespace tflite {
namespace openvinodelegate {

class Concat : public OperationsBase {
public:
    Concat(int operationIndex) {}
    TfLiteStatus CreateNode() override;
};

}  // namespace openvinodelegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_CONCAT_H_
