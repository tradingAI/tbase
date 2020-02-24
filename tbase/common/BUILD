package(default_visibility = ["//visibility:public"])

py_library(
    name = "py_default_lib",
    srcs = [
        "cmd_util.py",
        "eval.py",
        "logger.py",
        "optimizers.py",
        "random_process.py",
        "replay_buffer.py",
        "replay_memory.py",
        "torch_utils.py",
    ],
    deps = [
        "@com_github_iminders_tgym//tgym:py_default_lib",
        "@com_github_iminders_tgym//tgym/envs:py_default_lib",
    ],
)

py_test(
    name = "cmd_util_test",
    srcs = [
        "cmd_util_test.py",
    ],
    deps = [
        ":py_default_lib",
    ],
)

py_test(
    name = "eval_test",
    srcs = [
        "eval_test.py",
    ],
    deps = [
        ":py_default_lib",
    ],
)

py_test(
    name = "torch_utils_test",
    srcs = [
        "torch_utils_test.py",
    ],
    deps = [
        ":py_default_lib",
    ],
)

py_test(
    name = "random_process_test",
    srcs = [
        "random_process_test.py",
    ],
    deps = [
        ":py_default_lib",
    ],
)
