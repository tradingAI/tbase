package(default_visibility = ["//visibility:public"])

py_library(
    name = "py_default_lib",
    srcs = [
        "cmd_util.py",
    ],
    deps = [
        "@com_github_iminders_tgym//tgym:py_default_lib",
        "@com_github_iminders_tgym//tgym/envs:py_default_lib",
    ],
)

py_test(
    name = "cmd_util",
    srcs = [
        "cmd_util.py",
    ],
    deps = [
        ":py_default_lib",
    ],
)
