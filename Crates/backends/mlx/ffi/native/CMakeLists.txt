cmake_minimum_required(VERSION 3.25)
project(pcu_mlx_c_direct VERSION 1.0.0 LANGUAGES CXX)

if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    message(FATAL_ERROR "Fusion PCU MLX requires Apple silicon macOS")
endif()
set(CMAKE_OSX_DEPLOYMENT_TARGET "26.2" CACHE STRING "Exact qualified SDK minimum" FORCE)
find_package(MLX 0.32.3 EXACT CONFIG REQUIRED)

set(PCU_MLX_C_PREPARED_SOURCE "" CACHE PATH "Exact full upstream mlx-c tree prepared by Cargo's ffi/rust/build")
if(NOT EXISTS "${PCU_MLX_C_PREPARED_SOURCE}/provenance.json"
   OR NOT EXISTS "${PCU_MLX_C_PREPARED_SOURCE}/source-hashes.txt")
    message(FATAL_ERROR "Cargo's ffi/rust/build must prepare the exact upstream C source before configuring native")
endif()
file(READ "${PCU_MLX_C_PREPARED_SOURCE}/provenance.json" provenance)
string(JSON source_revision GET "${provenance}" source_revision)
if(NOT source_revision STREQUAL "a341b4925024b88b2c593468f16e12f5e17315da")
    message(FATAL_ERROR "Fusion PCU MLX requires the qualified upstream C0.7.0 source revision")
endif()

enable_language(C)
add_library(pcu_mlx_c_abi_contract OBJECT "${CMAKE_CURRENT_LIST_DIR}/c/tests/abi/abi.c")
target_include_directories(pcu_mlx_c_abi_contract PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/c/include"
    "${PCU_MLX_C_PREPARED_SOURCE}")
target_compile_features(pcu_mlx_c_abi_contract PRIVATE c_std_11)
target_compile_options(pcu_mlx_c_abi_contract PRIVATE -Wall -Wextra -Wpedantic -Werror)

set(MLX_C_USE_SYSTEM_MLX ON CACHE BOOL "" FORCE)
set(MLX_C_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
add_subdirectory("${PCU_MLX_C_PREPARED_SOURCE}" "${CMAKE_CURRENT_BINARY_DIR}/upstream")

# Keep the complete upstream source inventory and public operators. Only the
# centralized reporter is replaced; the replay extension retains a primitive.
get_target_property(upstream_sources mlxc SOURCES)
list(FILTER upstream_sources EXCLUDE REGEX "/error\\.cpp$")
set_property(TARGET mlxc PROPERTY SOURCES "${upstream_sources}")
target_sources(mlxc PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/cpp/reporting/reporting.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/cpp/manifest/manifest.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/cpp/replay/replay.cpp")
target_include_directories(mlxc PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/c/include"
    "${CMAKE_CURRENT_LIST_DIR}/cpp")
target_compile_definitions(mlxc PRIVATE PCU_MLX_C_SOURCE_REVISION="${source_revision}")
target_compile_features(mlxc PRIVATE cxx_std_20)
set_target_properties(mlxc PROPERTIES OUTPUT_NAME pcu_mlx_c_direct
    INSTALL_NAME_DIR "@rpath" BUILD_RPATH "${MLX_LIBRARY_DIRS}")
target_compile_options(mlxc PRIVATE -Wall -Wextra -Wpedantic)
