#include "tensorflow/lite/delegates/openvino/operations/include/resize_bilinear.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus ResizeBilinear::CreateNode() {
    const TfLiteResizeBilinearParams *resize_bilinearParams =
        (TfLiteResizeBilinearParams *)GetBuiltinData();
    auto input_node = getInputNode(tensor_indices_[TFLITE_INPUT_NODE_1]);
    auto shape_node = getInputNode(tensor_indices_[TFLITE_INPUT_NODE_2]);
    struct ov::op::v11::Interpolate::InterpolateAttrs attrs;

    attrs.mode = ov::op::v11::Interpolate::InterpolateMode::LINEAR_ONNX;
    attrs.shape_calculation_mode = ov::op::v11::Interpolate::ShapeCalcMode::SIZES;

    if (resize_bilinearParams->align_corners == true) {
        attrs.coordinate_transformation_mode =
            ov::op::v11::Interpolate::CoordinateTransformMode::ALIGN_CORNERS;
    } else if (resize_bilinearParams->half_pixel_centers == true) {
        attrs.coordinate_transformation_mode =
            ov::op::v11::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    } else {
        attrs.coordinate_transformation_mode =
            ov::op::v11::Interpolate::CoordinateTransformMode::ASYMMETRIC;
    }

    std::vector<int32_t> axes_vec = {2, 3};
    auto axes_node = CreateConstNode(ov::element::i32, {2}, axes_vec);
    if (axes_node == nullptr) {
        TFLITE_LOG(INFO) << "axes node is null \n";
        return kTfLiteError;
    }

    output_node =
        std::make_shared<ov::op::v11::Interpolate>(input_node, shape_node, axes_node, attrs);

    return kTfLiteOk;
}
}  // namespace openvinodelegate
}  // namespace tflite
