#include <string>
#include <vector>

#include "tensorflow/lite/delegates/external/external_delegate_interface.h"
#include "tensorflow/lite/delegates/openvino/openvino_delegate.h"
#include "tensorflow/lite/tools/command_line_flags.h"

TfLiteDelegate *CreateOVDelegateFromOptions(const char *const *options_keys,
                                            const char *const *options_values, size_t num_options) {
    TfLiteOpenVINODelegateOptions options = TfLiteOpenVINODelegateOptionsDefault();
    std::vector<const char *> argv;
    int argc = num_options + 1;
    argv.reserve(num_options + 1);
    std::vector<std::string> option_args;
    option_args.reserve(num_options);
    for (int i = 0; i < num_options; i++) {
        option_args.emplace_back("--");
        option_args.rbegin()->append(options_keys[i]);
        option_args.rbegin()->append("=");
        option_args.rbegin()->append(options_values[i]);
        argv.push_back(option_args.rbegin()->c_str());
    }

    constexpr char kDebugLevel[] = "debug_level";
    constexpr char kPluginsPath[] = "plugins_path";
    constexpr char kDeviceType[] = "device_type";

    std::vector<tflite::Flag> flag_list = {
        tflite::Flag::CreateFlag(kDebugLevel, &options.debug_level,
                                 "Debug Level for OpenVINO delegate."),
        /*tflite::Flag::CreateFlag(kPluginsPath,
                                 &options.plugins_path,
                                 "Plugins.xml path.")
        /*tflite::Flag::CreateFlag(kDeviceType,
                                 &options.device_type,
                                 "Device Type."), */
    };

    if (!tflite::Flags::Parse(&argc, argv.data(), flag_list)) {
        return nullptr;
    }

    TFLITE_LOG(INFO) << "OpenVINO delegate: debug_level set to " << options.debug_level << ".";
    /* TFLITE_LOG(INFO) << "OpenVINO delegate: plugins_path set to "
                     << options.plugins_path << ".";
    TFLITE_LOG(INFO) << "OpenVINO delegate: device_type set to "
                     << options.device_type << "."; */

    return TfLiteCreateOpenVINODelegate(&options);
}

extern "C" {

// Defines two symbols that need to be exported to use the TFLite external
// delegate. See tensorflow/lite/delegates/external for details.
extern TFL_EXTERNAL_DELEGATE_EXPORT TfLiteDelegate *tflite_plugin_create_delegate(
    const char *const *options_keys, const char *const *options_values, size_t num_options,
    void (*report_error)(const char *)) {
    return CreateOVDelegateFromOptions(options_keys, options_values, num_options);
}

TFL_EXTERNAL_DELEGATE_EXPORT void tflite_plugin_destroy_delegate(TfLiteDelegate *delegate) {}

}  // extern "C"
