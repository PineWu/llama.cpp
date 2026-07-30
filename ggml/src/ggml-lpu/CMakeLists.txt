# LPU backend for llama.cpp / ggml
#
# Required: LPU_INSTALL_DIR (cmake cache) or LPU_HOME (environment variable)
# pointing to the LPU SDK root directory.
#
# The LPU SDK is expected to provide:
#   ${LPU_INSTALL_DIR}/include/   -- header files (lpu_runtime.h, lpu_nn.h, ...)
#   ${LPU_INSTALL_DIR}/lib/       -- shared / static libraries
#
# Library names are placeholders; adjust to match the actual LPU SDK layout.

if (NOT LPU_INSTALL_DIR)
    if (DEFINED ENV{LPU_HOME})
        set(LPU_INSTALL_DIR $ENV{LPU_HOME})
        message(STATUS "LPU: LPU_INSTALL_DIR set from LPU_HOME=$ENV{LPU_HOME}")
    else()
        message(FATAL_ERROR
            "LPU: LPU_INSTALL_DIR is not set. "
            "Please pass -DLPU_INSTALL_DIR=/path/to/lpu_sdk or "
            "set the LPU_HOME environment variable.")
    endif()
endif()

# Verify the SDK directory exists
if (NOT EXISTS "${LPU_INSTALL_DIR}")
    message(FATAL_ERROR "LPU: LPU_INSTALL_DIR='${LPU_INSTALL_DIR}' does not exist.")
endif()

message(STATUS "LPU: Using LPU SDK at ${LPU_INSTALL_DIR}")

# Locate LPU runtime and NN libraries
# Adjust library names to match your actual LPU SDK
find_library(LPU_RUNTIME_LIB
    NAMES lpu_runtime liblpu_runtime
    PATHS "${LPU_INSTALL_DIR}/lib" "${LPU_INSTALL_DIR}/lib64"
    NO_DEFAULT_PATH
)
find_library(LPU_NN_LIB
    NAMES lpu_nn liblpu_nn
    PATHS "${LPU_INSTALL_DIR}/lib" "${LPU_INSTALL_DIR}/lib64"
    NO_DEFAULT_PATH
)

if (NOT LPU_RUNTIME_LIB)
    message(FATAL_ERROR "LPU: Cannot find lpu_runtime library in ${LPU_INSTALL_DIR}/lib")
endif()
if (NOT LPU_NN_LIB)
    message(FATAL_ERROR "LPU: Cannot find lpu_nn library in ${LPU_INSTALL_DIR}/lib")
endif()

message(STATUS "LPU: lpu_runtime -> ${LPU_RUNTIME_LIB}")
message(STATUS "LPU: lpu_nn      -> ${LPU_NN_LIB}")

# Build the backend library using the ggml helper macro
# (same pattern as ggml-cann, ggml-zendnn, etc.)
ggml_add_backend_library(ggml-lpu
    ggml-lpu.cpp
    lpu_ops.cpp
    lpu_sdk.cpp
)

target_include_directories(ggml-lpu PRIVATE
    "${LPU_INSTALL_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(ggml-lpu PRIVATE
    ${LPU_RUNTIME_LIB}
    ${LPU_NN_LIB}
)

target_link_directories(ggml-lpu PRIVATE
    "${LPU_INSTALL_DIR}/lib"
    "${LPU_INSTALL_DIR}/lib64"
)
