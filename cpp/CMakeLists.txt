cmake_minimum_required(VERSION 3.16)
project(ic_knockoff_poly_reg
    VERSION 0.1.0
    DESCRIPTION "IC-Knock-Poly: Iterative Conditional Knockoffs for Sparse Rational Polynomial Regression"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type defaults
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------------------------
# Library target
# ---------------------------------------------------------------------------
add_library(ic_knockoff
    src/matrix_ops.cpp
    src/polynomial.cpp
    src/knockoffs.cpp
    src/posi.cpp
)

target_include_directories(ic_knockoff
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Required for linking this static library into a shared library on Linux/ELF
set_target_properties(ic_knockoff PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Link math library on Unix
if(UNIX)
    target_link_libraries(ic_knockoff PUBLIC m)
endif()

# ---------------------------------------------------------------------------
# Test executable
# ---------------------------------------------------------------------------
enable_testing()

add_executable(test_ic_knockoff tests/test_main.cpp)
target_link_libraries(test_ic_knockoff PRIVATE ic_knockoff)

add_test(NAME unit_tests COMMAND test_ic_knockoff)

# ---------------------------------------------------------------------------
# Python ctypes shared library
# ---------------------------------------------------------------------------
# Produces libic_knockoff_py.so (Linux/macOS) or ic_knockoff_py.dll (Windows).
# Python code loads this via ctypes to use the C++/Rust kernels.
add_library(ic_knockoff_py SHARED
    bindings/py_bindings.cpp
)

target_include_directories(ic_knockoff_py
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/bindings
)

target_link_libraries(ic_knockoff_py PRIVATE ic_knockoff)

# Strip the 'lib' prefix on Windows so Python can find it as 'ic_knockoff_py.dll'
set_target_properties(ic_knockoff_py PROPERTIES
    OUTPUT_NAME "ic_knockoff_py"
)

# ---------------------------------------------------------------------------
# Baseline comparison runner
# ---------------------------------------------------------------------------
add_executable(baseline_runner
    baselines/baseline_runner.cpp
    baselines/baselines.cpp
)
target_include_directories(baseline_runner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/baselines
)
target_link_libraries(baseline_runner PRIVATE ic_knockoff)

