// Cold ABI and exact header/source facts, independent of runtime object creation.
#include "pcu_mlx/manifest.h"
#include "reporting/reporting.hpp"
#include "mlx/c/array.h"
#include "mlx/c/device.h"
#include "mlx/c/stream.h"
#include "mlx/c/closure.h"
#include "mlx/c/vector.h"
#include "mlx/c/string.h"
#include <mlx/version.h>

static_assert(MLX_VERSION_NUMERIC == 32003, "direct C requires exact MLX0.32.3 headers");

extern "C" uint32_t pcu_mlx_c_safety_abi(void) { return 1; }
extern "C" int pcu_mlx_c_manifest_get(pcu_mlx_c_manifest* result) {
  if (!result) { pcu::mlx_c::detail::report("null direct-C manifest output"); return 1; }
  *result = {};
  result->safety_abi = 1;
  result->native_version_numeric = MLX_VERSION_NUMERIC;
  result->array_handle_size = sizeof(mlx_array);
  result->device_handle_size = sizeof(mlx_device);
  result->stream_handle_size = sizeof(mlx_stream);
  result->closure_handle_size = sizeof(mlx_closure);
  result->vector_array_handle_size = sizeof(mlx_vector_array);
  result->string_handle_size = sizeof(mlx_string);
  result->float32_dtype = MLX_FLOAT32;
  result->flags = 1;
  pcu::mlx_c::detail::copy_text(result->native_header_version, sizeof(result->native_header_version), "0.32.3");
  pcu::mlx_c::detail::copy_text(result->c_source_revision, sizeof(result->c_source_revision), PCU_MLX_C_SOURCE_REVISION);
  return 0;
}
