#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_DEPTHWISE_CONV2D_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_DEPTHWISE_CONV2D_H_

#include "tensorflow/lite/delegates/openvino/operations/operations_base.h"

namespace tflite {
namespace openvinodelegate {

class DepthwiseConv2D : public OperationsBase {
public:
    DepthwiseConv2D(int operationIndex) {}
    TfLiteStatus CreateNode() override;
};

}  // namespace openvinodelegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_DEPTHWISE_CONV2D_H_
