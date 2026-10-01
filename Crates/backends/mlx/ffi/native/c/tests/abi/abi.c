/* Compile as C11 against the actual pinned upstream C headers. The extension
 * interface exposes no C++ types or private implementation headers. */
#include "pcu_mlx/error.h"
#include "pcu_mlx/manifest.h"
#include "pcu_mlx/replay.h"
#include "pcu_mlx/safety.h"
#include "mlx/c/closure.h"
#include "mlx/c/device.h"
#include "mlx/c/string.h"
#include "mlx/c/vector.h"
#include <stddef.h>

_Static_assert(sizeof(void*) == 8, "qualified Apple arm64 pointer width");
_Static_assert(sizeof(mlx_array) == 8, "array opaque handle width");
_Static_assert(sizeof(mlx_device) == 8, "device opaque handle width");
_Static_assert(sizeof(mlx_stream) == 8, "stream opaque handle width");
_Static_assert(sizeof(mlx_closure) == 8, "closure opaque handle width");
_Static_assert(sizeof(mlx_vector_array) == 8, "vector opaque handle width");
_Static_assert(sizeof(mlx_string) == 8, "string opaque handle width");
_Static_assert(MLX_FLOAT32 == 10, "selected F32 dtype vocabulary");
_Static_assert(sizeof(pcu_mlx_c_error_scope) == 32, "safety ABI1 scope size");
_Static_assert(offsetof(pcu_mlx_c_error_scope, previous) == 0, "scope previous");
_Static_assert(offsetof(pcu_mlx_c_error_scope, message) == 8, "scope message");
_Static_assert(offsetof(pcu_mlx_c_error_scope, capacity) == 16, "scope capacity");
_Static_assert(offsetof(pcu_mlx_c_error_scope, failed) == 24, "scope failed");
_Static_assert(offsetof(pcu_mlx_c_error_scope, active) == 28, "scope active");
_Static_assert(sizeof(pcu_mlx_c_manifest) == 136, "safety ABI1 manifest size");
_Static_assert(offsetof(pcu_mlx_c_manifest, float32_dtype) == 32, "manifest dtype");
_Static_assert(offsetof(pcu_mlx_c_manifest, flags) == 36, "manifest flags");
_Static_assert(offsetof(pcu_mlx_c_manifest, native_header_version) == 40, "header version");
_Static_assert(offsetof(pcu_mlx_c_manifest, c_source_revision) == 72, "source revision");
