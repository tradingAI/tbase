package(default_visibility = ["//visibility:public"])

py_library(
    name = "py_default_lib",
    srcs = [
        "base.py",
        "polices.py",
        "values.py",
    ],
    deps = [
        "//tbase/common:py_default_lib",
    ],
)

py_test(
    name = "polices_test",
    srcs = [
        "polices_test.py",
    ],
    deps = [
        ":py_default_lib",
    ],
)
