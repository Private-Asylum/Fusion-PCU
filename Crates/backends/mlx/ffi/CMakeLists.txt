cmake_minimum_required(VERSION 3.25)
project(pcu_mlx_bridge VERSION 1.0.0 LANGUAGES CXX)

if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    message(FATAL_ERROR "Fusion PCU MLX requires Apple silicon macOS")
endif()

# An explicitly built private bridge keeps SDK-absent Cargo hosts buildable. There is no
# FetchContent, installation rule, or mutation of the caller's MLX environment.
find_package(MLX 0.32.3 EXACT CONFIG REQUIRED)
add_library(pcu_mlx_bridge SHARED bridge.cpp)
target_compile_features(pcu_mlx_bridge PRIVATE cxx_std_20)
target_link_libraries(pcu_mlx_bridge PRIVATE mlx)
set_target_properties(pcu_mlx_bridge PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    BUILD_RPATH "${MLX_LIBRARY_DIRS}")
target_compile_options(pcu_mlx_bridge PRIVATE -Wall -Wextra -Wpedantic -Werror)
