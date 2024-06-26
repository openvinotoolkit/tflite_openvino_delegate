// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_DELEGATE_H_

#include "openvino_delegate_kernel.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/utils/simple_opaque_delegate.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

static const char kOpenVINOStableDelegateName[] = "intel_openvino_delegate";
static const char kOpenVINOStableDelegateVersion[] = "1.0.0";

struct TFL_CAPI_EXPORT TfLiteOpenVINODelegateOptions {
  /* debug_level for the OpenVINO delegate*/
  int debug_level;

  /* path for the OpenVINO plugins
  char *plugins_path; */

  /* Device for OpenVINO to select
      Currently we support CPU and NPU
  char* device_type*/
  ;
};

TfLiteOpenVINODelegateOptions TFL_CAPI_EXPORT
TfLiteOpenVINODelegateOptionsDefault();

TfLiteOpaqueDelegate *TFL_CAPI_EXPORT
TfLiteCreateOpenVINODelegate(const TfLiteOpenVINODelegateOptions *options);

void TFL_CAPI_EXPORT
TfLiteDeleteOpenVINODelegate(TfLiteOpaqueDelegate *delegate);

namespace tflite {
namespace openvinodelegate {

// forward declaration
class OpenVINODelegateTestPeer;

class OpenVINODelegate : public SimpleOpaqueDelegateInterface {
 public:
  explicit OpenVINODelegate(const TfLiteOpenVINODelegateOptions *options)
      : options_(*options) {
    if (options == nullptr) options_ = TfLiteOpenVINODelegateOptionsDefault();
  }

  bool IsNodeSupportedByDelegate(const TfLiteRegistrationExternal *registration,
                                 const TfLiteOpaqueNode *node,
                                 TfLiteOpaqueContext *context) const override;

  TfLiteStatus Initialize(TfLiteOpaqueContext *context) override;

  const char *Name() const override;

  std::unique_ptr<SimpleOpaqueDelegateKernelInterface>
  CreateDelegateKernelInterface() override;

 private:
  TfLiteOpenVINODelegateOptions options_;
  friend class OpenVINODelegateTestPeer;
  bool CheckInputsType(const int tensor_id, const TfLiteOpaqueContext *context,
                       TfLiteType expected_type) const;
  bool CheckDataTypeSupported(
      const TfLiteOpaqueContext *context, const TfLiteOpaqueNode *node,
      std::vector<std::vector<TfLiteType>> supported_types) const;
  bool CheckDims(const TfLiteOpaqueContext *context,
                 const TfLiteOpaqueNode *node,
                 std::vector<std::vector<int>> dims_size) const;
  bool CheckNodeSupportByOpenVINO(
      const TfLiteRegistrationExternal *registration,
      const TfLiteOpaqueNode *node, TfLiteOpaqueContext *context) const;
};
}  // namespace openvinodelegate
}  // namespace tflite

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_DELEGATE_H_
