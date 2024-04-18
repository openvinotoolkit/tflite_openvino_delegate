#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_DEQUANTIZE_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_DEQUANTIZE_H_

#include "tensorflow/lite/delegates/openvino/operations/operations_base.h"

namespace tflite {
namespace openvinodelegate {

class Dequantize : public OperationsBase {
public:
    Dequantize(int operationIndex) {}
    TfLiteStatus CreateNode() override;
};

}  // namespace openvinodelegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_DEQUANTIZE_H_
