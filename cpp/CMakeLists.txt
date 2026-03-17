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
