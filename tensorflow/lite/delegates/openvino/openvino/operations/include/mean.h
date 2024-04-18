#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_MEAN_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_MEAN_H_

#include "tensorflow/lite/delegates/openvino/operations/operations_base.h"

namespace tflite {
namespace openvinodelegate {

class Mean : public OperationsBase {
public:
    Mean(int operationIndex) {}
    TfLiteStatus CreateNode() override;
};

}  // namespace openvinodelegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_MEAN_H_
