// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts_no_undefined","tflite_cc_shared_object")
load("//tensorflow/lite:special_rules.bzl", "internal_visibility_allowlist", "tflite_portable_test_suite_combined")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files([
    "openvino_delegate.h",
])

cc_library(
    name ="openvino_graph_builder",
    srcs = ["openvino_graph_builder.cc"],
    hdrs = ["openvino_graph_builder.h"],
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/c:c_api_experimental",
        "//tensorflow/lite/delegates/openvino/operations:operations_base",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/tools:logging",
        "//tensorflow/lite/kernels:builtin_ops",
        "@intel_openvino//:openvino",
    ],
)

cc_library(
    name ="openvino_delegate_core",
    srcs = ["openvino_delegate_core.cc"],
    hdrs = ["openvino_delegate_core.h"],
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        ":openvino_graph_builder",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/tools:logging",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:c_api_experimental",
        "//tensorflow/lite/kernels:kernel_util",
        "@intel_openvino//:openvino",
    ],
)

cc_library(
    name ="openvino_delegate_kernel",
    srcs = ["openvino_delegate_kernel.cc"],
    hdrs = ["openvino_delegate_kernel.h"],
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        ":openvino_delegate_core",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/tools:logging",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:c_api_experimental",
        "//tensorflow/lite/delegates/utils:simple_opaque_delegate",
        "//tensorflow/lite/delegates/utils:simple_delegate",
        "//tensorflow/lite/kernels:kernel_util",
        "@intel_openvino//:openvino",
    ],
)

cc_library(
    name = "openvino_delegate",
    srcs = ["openvino_delegate.cc"],
    hdrs = ["openvino_delegate.h"],
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        ":openvino_delegate_kernel",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/kernels:padding",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/c:c_api_experimental",
        "//tensorflow/lite/delegates/utils:simple_opaque_delegate",
        "//tensorflow/lite/kernels:builtin_ops",
        "@intel_openvino//:openvino",
    ],
)
cc_test(
    name = "openvino_graph_builder_test",
    srcs = ["openvino_graph_builder_test.cc"],
    linkopts = select({
        "//conditions:default": [],
    }),
    deps = [
        ":openvino_graph_builder",
        "@com_google_googletest//:gtest_main",
    ],
)

'''cc_library(
    name ="openvino_delegate_kernel",
    srcs = ["openvino_delegate_kernel.cc"],
    hdrs = ["openvino_delegate_kernel.h"],
    tags = [
        "manual",
        "nobuilder",
    ],
    linkstatic = True,
    deps = [
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/tools:logging",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/delegates/utils:simple_opaque_delegate",
        "//tensorflow/lite/kernels:kernel_util",
        "@intel_openvino//:openvino",
    ],
)
'''

cc_library(
    name = "openvino_external_delegate",
    srcs = [
        "openvino_delegate_external.cc",
    ],
    copts = tflite_copts() + ["-fexceptions"],
    deps = [
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/c:c_api_experimental",
        "//tensorflow/lite/c:c_api_types",
        ":openvino_delegate",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools:logging",
        "//tensorflow/lite/delegates/utils:simple_opaque_delegate",
        "//tensorflow/lite/delegates/utils/experimental/stable_delegate:stable_delegate_interface",
        "//tensorflow/lite/acceleration/configuration/c:delegate_plugin",
        "//tensorflow/lite/acceleration/configuration/c:stable_delegate",
        "//tensorflow/lite/delegates/external:external_delegate_interface",
        "//tensorflow/lite/delegates/utils/experimental/stable_delegate:delegate_loader",
    ],
)

tflite_cc_shared_object(
    name = "tensorflowlite_openvino_stable_delegate",
    testonly = True,
    linkopts = tflite_linkopts_no_undefined() + select({
        "//tensorflow:windows": [],
        "//conditions:default": [
            # Expose necessary symbols only.
            "-Wl,--version-script,$(location //tensorflow/lite/delegates/utils/experimental/stable_delegate:version_script.lds)",
        ],
    }),
    per_os_targets = True,
    deps = [
        ":openvino_external_delegate",
        "//tensorflow/lite/delegates/utils/experimental/stable_delegate:version_script.lds",
    ],
)

cc_test(
    name = "openvino_delegate_external_test",
    srcs = ["openvino_delegate_external_test.cc"],
    linkopts = select({
        "//conditions:default": [],
    }),
    deps = [
        ":openvino_external_delegate",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "openvino_delegate_core_test",
    srcs = ["openvino_delegate_core_test.cc"],
    linkopts = select({
        "//conditions:default": [],
    }),
    deps = [
        ":openvino_delegate_core",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "openvino_delegate_provider",
    srcs = ["//tensorflow/lite/tools/delegates/openvino_delegate_provider.cc"],
    copts = tflite_copts(),
    deps = [
        ":openvino_delegate",
        "//tensorflow/lite/tools/delegates:delegate_provider_hdr",
     ],
    alwayslink = 1,
)

cc_library(
    name = "openvino_delegate_hdrs_only",
    hdrs = ["openvino_delegate.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility_allowlist(),
    deps = [
        "//tensorflow/lite/c:common",
    ],
)

cc_library(
    name = "openvino_delegate_test_mode",
    srcs = ["openvino_delegate.cc"],
    hdrs = ["openvino_delegate.h"],
    copts = tflite_copts() + ["-fexceptions"] + ["-DOPENVINO_DELEGATE_TEST_MODE=1"],
    linkstatic = True,
    deps = [
        ":openvino_delegate_kernel",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:padding",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal/utils:sparsity_format_converter",
        "@intel_openvino//:openvino",
    ],
)

cc_test(
    name = "openvino_delegate_builder_test",
    srcs = ["openvino_delegate_builder_test.cc"],
    linkopts = select({
        "//conditions:default": [],
    }),
    deps = [
        ":openvino_delegate_test_mode",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "openvino_delegate_test",
    srcs = ["openvino_delegate_test.cc"],
    linkopts = select({
        "//conditions:default": [],
    }),
    deps = [
        ":openvino_delegate",
        "@com_google_googletest//:gtest_main",
    ],
)

filegroup(
    name = "openvino_delegate_tests",
    testonly = True,
    srcs = [
        "openvino_graph_builder_test", 
        "openvino_delegate_core_test",
        "openvino_delegate_external_test",
        "openvino_delegate_test",
    ]
)
