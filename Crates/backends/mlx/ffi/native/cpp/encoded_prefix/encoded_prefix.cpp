// Exact immutable logical carrier prefix replacement, executed by MLX.
#include "pcu_mlx/encoded_prefix.h"
#include "reporting/reporting.hpp"
#include "mlx/c/private/mlx.h"
#include <mlx/ops.h>
#include <mlx/primitives.h>
#include <exception>
#include <stdexcept>
#include <limits>
extern "C" int pcu_mlx_c_encoded_prefix_merge(
    mlx_array* output, mlx_array prefix, mlx_array previous, mlx_stream stream) {
  try {
    namespace mx = mlx::core;
    if (!output) throw std::invalid_argument("nil encoded prefix output");
    const auto& update = mlx_array_get_(prefix);
    const auto& old = mlx_array_get_(previous);
    const auto s = mlx_stream_get_(stream);
    if (s.device.type != mx::Device::gpu || old.ndim() != 1 || update.ndim() != 1
        || update.shape(0) <= 0 || update.shape(0) >= old.shape(0)
        || old.dtype() != update.dtype()
        || (old.dtype() != mx::uint8 && old.dtype() != mx::uint16 && old.dtype() != mx::uint32))
      throw std::invalid_argument("invalid exact encoded prefix profile");
    // Lazy SliceUpdate retains the real immutable input descriptors. Rust retains
    // separate official holders as well, so descriptor/data uniqueness cannot
    // permit donation of the previous escaped owner's backing during evaluation.
    mlx_array_set_(*output, mx::slice_update(old, update,
        mx::Shape{0}, mx::Shape{update.shape(0)}, mx::Shape{1}, s));
    return 0;
  } catch (const std::exception& error) {
    pcu::mlx_c::detail::report(error.what());
  } catch (...) {
    pcu::mlx_c::detail::report("unknown exception in immutable MLX encoded prefix");
  }
  return 1;
}
