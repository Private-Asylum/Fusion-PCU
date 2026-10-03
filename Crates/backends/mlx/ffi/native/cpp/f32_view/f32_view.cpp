// Exact immutable same-width views, owned and scheduled by MLX.
#include "pcu_mlx/f32_view.h"
#include "reporting/reporting.hpp"
#include "mlx/c/private/mlx.h"
#include <mlx/ops.h>
#include <exception>
#include <limits>
#include <stdexcept>
extern "C" int pcu_mlx_c_f32_view(mlx_array* output, mlx_array input,
    mlx_stream stream, int32_t rows, int32_t columns, int to_matrix) {
  try {
    namespace mx = mlx::core;
    if (!output || rows <= 0 || columns <= 0 || (to_matrix != 0 && to_matrix != 1))
      throw std::invalid_argument("invalid exact F32 view extent");
    const auto& in = mlx_array_get_(input);
    const auto s = mlx_stream_get_(stream);
    const size_t count = static_cast<size_t>(rows) * static_cast<size_t>(columns);
    if (s.device.type != mx::Device::gpu || count > static_cast<size_t>(std::numeric_limits<int>::max())
        || in.size() != count || in.nbytes() != count * sizeof(uint32_t))
      throw std::invalid_argument("invalid exact F32 view storage");
    if (to_matrix) {
      if (in.dtype() != mx::uint32 || in.ndim() != 1 || in.strides()[0] != 1)
        throw std::invalid_argument("expected dense UInt32 logical F32 carrier");
      mlx_array_set_(*output, mx::reshape(mx::view(in, mx::float32, s), mx::Shape{rows, columns}, s));
    } else {
      if (in.dtype() != mx::float32 || in.ndim() != 2 || in.shape(0) != rows
          || in.shape(1) != columns || in.strides()[0] != static_cast<size_t>(columns)
          || in.strides()[1] != 1)
        throw std::invalid_argument("expected dense exact F32 matrix");
      mlx_array_set_(*output, mx::reshape(mx::view(in, mx::uint32, s), mx::Shape{static_cast<int>(count)}, s));
    }
    return 0;
  } catch (const std::exception& error) {
    pcu::mlx_c::detail::report(error.what());
  } catch (...) {
    pcu::mlx_c::detail::report("unknown exception in immutable MLX F32 view");
  }
  return 1;
}

extern "C" int pcu_mlx_c_f32_view_validate_shared(mlx_array input, mlx_array output) {
  try {
    const auto& in = mlx_array_get_(input);
    const auto& out = mlx_array_get_(output);
    if (!in.is_available() || !out.is_available() || !in.data_shared_ptr()
        || in.data_shared_ptr() != out.data_shared_ptr() || in.offset() != out.offset()
        || in.nbytes() != out.nbytes() || !out.flags().row_contiguous)
      throw std::invalid_argument("F32 view did not retain exact terminal shared storage");
    return 0;
  } catch (const std::exception& error) {
    pcu::mlx_c::detail::report(error.what());
  } catch (...) {
    pcu::mlx_c::detail::report("unknown exception validating terminal F32 shared view");
  }
  return 1;
}
