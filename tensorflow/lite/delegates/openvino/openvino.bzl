def _openvino_native_impl(repository_ctx):
    openvino_native_dir = repository_ctx.os.environ["OPENVINO_NATIVE_DIR"]
    repository_ctx.symlink(openvino_native_dir, "openvino")
    repository_ctx.file("BUILD", """
load("@org_tensorflow//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")
cc_library_with_tflite(
    name = "openvino",
    hdrs = glob(["openvino/runtime/include", "openvino/runtime/include/ie/cpp", "openvino/runtime/include/ie"]),
    srcs = ["openvino/lib64/libopenvino.so.2023.2.0"],
    includes = ["openvino/runtime/include/ie/cpp",
                "openvino/runtime/include/ie",
                "openvino/runtime/include"],
    visibility = ["//visibility:public"],
)
    """)

openvino_configure = repository_rule(
    implementation = _openvino_native_impl,
    local = True,
    environ = ["OPENVINO_NATIVE_DIR"],
)
