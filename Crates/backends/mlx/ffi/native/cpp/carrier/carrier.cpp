// Fixed integer-limb transport owned, allocated and scheduled exclusively by MLX.
#include "pcu_mlx/carrier.h"
#include "reporting/reporting.hpp"
#include "mlx/c/private/mlx.h"
#include <mlx/fast.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>
#include <mlx/version.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
static_assert(MLX_VERSION_NUMERIC == 32003, "carrier primitive requires exact MLX0.32.3");
namespace {
namespace mx = mlx::core;
struct Carrier {
  std::shared_ptr<mx::Primitive> primitive;
  mx::Shape input, output;
  mx::Dtype dtype;
  mx::Stream stream;
  std::thread::id thread;
};
template<class F> int checked(F&& function) noexcept {
  try { function(); return 0; }
  catch (const std::exception& error) { pcu::mlx_c::detail::report(error.what()); }
  catch (...) { pcu::mlx_c::detail::report("unknown exception in MLX-owned integer carrier"); }
  return 1;
}
}
static int prepare_carrier(pcu_mlx_c_carrier* owner, mlx_stream stream,
    uint32_t dtype, uint32_t input_count, uint32_t output_count,
    uint32_t scalar_lanes, int broadcast, bool prefix) {
  return checked([&] {
    if (!owner || owner->ctx || dtype < 1 || dtype > 3 || !input_count || !output_count
        || output_count > uint32_t(std::numeric_limits<int>::max())
        || input_count > uint32_t(std::numeric_limits<int>::max())
        || (broadcast != 0 && broadcast != 1)
        || (broadcast ? !scalar_lanes || scalar_lanes > 16 || input_count < scalar_lanes
            || output_count % scalar_lanes != 0 || input_count % scalar_lanes != 0
            : input_count < output_count || scalar_lanes != 0)
        || (!prefix && (broadcast ? input_count != scalar_lanes : input_count != output_count)))
      throw std::invalid_argument("invalid fixed integer carrier profile/owner");
    const auto s = mlx_stream_get_(stream);
    if (s.device.type != mx::Device::gpu)
      throw std::invalid_argument("carrier requires an explicit MLX GPU stream");
    const auto carrier = dtype == 1 ? mx::uint8 : dtype == 2 ? mx::uint16 : mx::uint32;
    const mx::Shape input_shape{int(input_count)}, output_shape{int(output_count)};
    // Only bounded unsigned literals are generated. Every physical lane is owned by one
    // invocation; scalar broadcast repeats whole logical limb sequences without conversion.
    const std::string index = broadcast ? "i % " + std::to_string(scalar_lanes) + "u" : "i";
    const std::string source = "uint i = thread_position_in_grid.x;\nif (i >= "
        + std::to_string(output_count) + "u) return;\noutput[i] = input[" + index + "];\n";
    auto input = mx::broadcast_to(mx::array(uint32_t(0), carrier), input_shape, s);
    auto kernel = mx::fast::metal_kernel("pcu_mlx_integer_carrier", {"input"}, {"output"}, source);
    // Baseline group size one divides the exact physical grid, with no nonuniform-group dependency.
    auto outputs = kernel({input}, {output_shape}, {carrier},
        {int(output_count),1,1}, {1,1,1}, {}, std::nullopt, false, s);
    if (outputs.size() != 1 || !outputs[0].has_primitive()
        || outputs[0].primitive().stream() != s || outputs[0].dtype() != carrier)
      throw std::runtime_error("carrier custom primitive ownership/stream mismatch");
    owner->ctx = new Carrier{outputs[0].primitive_ptr(), input_shape, output_shape,
                             carrier, s, std::this_thread::get_id()};
  });
}
extern "C" int pcu_mlx_c_carrier_new(pcu_mlx_c_carrier* owner, mlx_stream stream,
    uint32_t dtype, uint32_t input_count, uint32_t output_count, int broadcast) {
  return prepare_carrier(owner, stream, dtype, input_count, output_count,
      broadcast ? input_count : 0, broadcast, false);
}
extern "C" int pcu_mlx_c_carrier_prefix_new(pcu_mlx_c_carrier* owner, mlx_stream stream,
    uint32_t dtype, uint32_t input_count, uint32_t output_count,
    uint32_t scalar_lanes, int broadcast) {
  return prepare_carrier(owner, stream, dtype, input_count, output_count,
      scalar_lanes, broadcast, true);
}
extern "C" int pcu_mlx_c_carrier_apply(mlx_array* output, pcu_mlx_c_carrier owner, mlx_array input) {
  return checked([&] {
    if (!output || !owner.ctx) throw std::invalid_argument("invalid integer carrier output/owner");
    const auto& state = *static_cast<Carrier*>(owner.ctx);
    if (state.thread != std::this_thread::get_id())
      throw std::invalid_argument("integer carrier used from another thread");
    const auto& value = mlx_array_get_(input);
    if (value.dtype() != state.dtype || value.shape() != state.input)
      throw std::invalid_argument("integer carrier exact physical dtype/shape mismatch");
    // Warm calls instantiate result descriptors from the already retained primitive; they
    // perform no source generation, fast::metal_kernel preparation, shape lowering or ranking.
    auto result = mx::array::make_arrays({state.output}, {state.dtype}, state.primitive, {value});
    mlx_array_set_(*output, result[0]);
  });
}
extern "C" int pcu_mlx_c_carrier_free(pcu_mlx_c_carrier owner) {
  return checked([&] {
    auto* state = static_cast<Carrier*>(owner.ctx);
    if (state && state->thread != std::this_thread::get_id())
      throw std::invalid_argument("integer carrier released from another thread");
    delete state;
  });
}
