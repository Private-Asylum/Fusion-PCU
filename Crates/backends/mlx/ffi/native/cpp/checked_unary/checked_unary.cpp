// Fixed checked encoding kernels compiled, allocated and scheduled exclusively by MLX.
#include "pcu_mlx/checked_unary.h"
#include "reporting/reporting.hpp"
#include "mlx/c/private/mlx.h"
#include <mlx/fast.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>
#include <mlx/version.h>
#include <limits>
#include <cstring>
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
static_assert(MLX_VERSION_NUMERIC == 32003, "checked primitive requires exact MLX0.32.3");
namespace {
namespace mx = mlx::core;
struct CheckedUnary {
  std::shared_ptr<mx::Primitive> primitive;
  mx::Shape input, output, records;
  mx::Dtype carrier;
  mx::Stream stream;
  std::thread::id thread;
};
template<class F> int checked(F&& function) noexcept {
  try { function(); return 0; }
  catch (const std::exception& error) { pcu::mlx_c::detail::report(error.what()); }
  catch (...) { pcu::mlx_c::detail::report("unknown exception in MLX-owned checked unary"); }
  return 1;
}
std::string shader(uint32_t format, uint32_t operation, uint32_t underflow,
                   uint32_t range, uint32_t count, bool broadcast) {
  if (format == 5) {
    const std::string index = broadcast ? "0u" : "2u*i";
    const auto literal = [](uint32_t value) { return std::to_string(value) + "u"; };
    return "uint i = thread_position_in_grid.x;\nif (i >= " + literal(count) + ") return;\n"
      "uint low = input[" + index + "], high = input[" + index + "+1u];\n"
      "if ((high & 0x7ff00000u) == 0x7ff00000u) { output[2u*i] = 0u; output[2u*i+1u] = 0u; records[i] = 4u; return; }\n"
      "uint out_low = low, out_high = high;\n"
      + (operation == 0 ? std::string("out_high ^= 0x80000000u;\n") : std::string("if ((high & 0x80000000u) != 0u || ((high & 0x7fffffffu) | low) == 0u) { out_low = 0u; out_high = 0u; }\n"))
      + "bool rejected = " + std::string(underflow == 1 ? "true" : "false")
      + " && (out_high & 0x7ff00000u) == 0u && ((out_high & 0x7fffffffu) | out_low) != 0u;\n"
      "output[2u*i] = rejected && " + std::string(range == 0 ? "true" : "false") + " ? 0u : out_low;\n"
      "output[2u*i+1u] = rejected && " + std::string(range == 0 ? "true" : "false") + " ? 0u : out_high;\n"
      "records[i] = rejected ? " + literal(range == 0 ? 3 : 0x103) + " : 0u;\n";
  }
  const uint32_t fraction[] = {10, 7, 3, 2, 23};
  const uint32_t sign[] = {0x8000, 0x8000, 0x80, 0x80, 0x80000000};
  const uint32_t maximum[] = {0x7bff, 0x7f7f, 0x7e, 0x7b, 0x7f7fffff};
  // Only bounded integer literals are emitted, never caller-provided source or identifiers.
  const auto literal = [](uint32_t value) { return std::to_string(value) + "u"; };
  return "uint i = thread_position_in_grid.x;\nif (i >= " + literal(count) + ") return;\n"
      "uint bits = uint(input[" + (broadcast ? std::string("0") : std::string("i")) + "]);\n"
      "uint magnitude = bits & (" + literal(sign[format]) + " - 1u);\n"
      "if (magnitude > " + literal(maximum[format]) + ") { output[i] = 0; records[i] = 4u; return; }\n"
      "uint result = " + (operation == 0 ? "bits ^ " + literal(sign[format]) :
          "((bits & " + literal(sign[format]) + ") != 0u || magnitude == 0u ? 0u : bits)") + ";\n"
      "uint out_magnitude = result & (" + literal(sign[format]) + " - 1u);\n"
      "bool rejected = " + std::string(underflow == 1 ? "true" : "false") +
          " && out_magnitude != 0u && out_magnitude < (1u << " + literal(fraction[format]) + ");\n"
      "output[i] = rejected && " + std::string(range == 0 ? "true" : "false") + " ? 0u : result;\n"
      "records[i] = rejected ? " + literal(range == 0 ? 3 : 0x103) + " : 0u;\n";
}
}
static int prepare_checked_unary(
    pcu_mlx_c_checked_unary* owner, mlx_stream stream, uint32_t format,
    uint32_t operation, uint32_t underflow, uint32_t range, uint32_t count,
    uint32_t input_count, int broadcast, bool prefix) {
  return checked([&] {
    if (!owner || owner->ctx || format > 5 || operation > 1 || underflow > 2 || range > 1
        || !count || !input_count || count > uint32_t(std::numeric_limits<int>::max())
        || (broadcast != 0 && broadcast != 1)
        || input_count < (broadcast ? 1u : count)
        || (!prefix && input_count != (broadcast ? 1u : count)))
      throw std::invalid_argument("invalid fixed checked unary profile/owner");
    const auto s = mlx_stream_get_(stream);
    if (s.device.type != mx::Device::gpu)
      throw std::invalid_argument("checked unary requires an explicit MLX GPU stream");
    const auto carrier = format < 2 ? mx::uint16 : format < 4 ? mx::uint8 : mx::uint32;
    const uint32_t limbs = format == 5 ? 2 : 1;
    if (count > uint32_t(std::numeric_limits<int>::max()) / limbs
        || input_count > uint32_t(std::numeric_limits<int>::max()) / limbs)
      throw std::invalid_argument("checked unary physical carrier extent overflow");
    const mx::Shape input_shape{int(input_count*limbs)},
        output_shape{int(count*limbs)}, record_shape{int(count)};
    // Descriptor preparation is lazy: no GPU work is issued before the Rust pending-owner
    // protocol retains this primitive, its explicit stream and the SDK input/output holders.
    auto input = mx::array(uint32_t(0), carrier);
    input = mx::broadcast_to(input, input_shape, s);
    auto kernel = mx::fast::metal_kernel("pcu_mlx_checked_unary", {"input"},
        {"output", "records"}, shader(format,operation,underflow,range,count,broadcast != 0));
    // Group size one is supported by every admitted pipeline and divides the grid exactly;
    // no partial/nonuniform threadgroup capability is assumed by this bounded realization.
    auto outputs = kernel({input}, {output_shape,record_shape}, {carrier,mx::uint32},
        {int(count),1,1}, {1,1,1}, {}, std::nullopt, false, s);
    if (outputs.size() != 2 || !outputs[0].has_primitive()
        || outputs[0].primitive().stream() != s || outputs[0].dtype() != carrier
        || outputs[1].dtype() != mx::uint32 || outputs[0].primitive_ptr() != outputs[1].primitive_ptr())
      throw std::runtime_error("checked custom primitive ownership/stream mismatch");
    owner->ctx = new CheckedUnary{outputs[0].primitive_ptr(), input_shape, output_shape, record_shape,
                                  carrier, s, std::this_thread::get_id()};
  });
}
extern "C" int pcu_mlx_c_checked_unary_new(
    pcu_mlx_c_checked_unary* owner, mlx_stream stream, uint32_t format,
    uint32_t operation, uint32_t underflow, uint32_t range, uint32_t count, int broadcast) {
  return prepare_checked_unary(owner,stream,format,operation,underflow,range,count,
      broadcast ? 1u : count,broadcast,false);
}
extern "C" int pcu_mlx_c_checked_unary_prefix_new(
    pcu_mlx_c_checked_unary* owner, mlx_stream stream, uint32_t format,
    uint32_t operation, uint32_t underflow, uint32_t range, uint32_t count,
    uint32_t input_count, int broadcast) {
  return prepare_checked_unary(owner,stream,format,operation,underflow,range,count,
      input_count,broadcast,true);
}
extern "C" int pcu_mlx_c_checked_upload(mlx_array* output, const void* bytes, size_t count, uint32_t format) {
  return checked([&] {
    if (!output || !bytes || !count || count > size_t(std::numeric_limits<int>::max()) || format > 6)
      throw std::invalid_argument("invalid exact encoding upload");
    if (format < 2 || format == 5) {
      // memcpy accepts unaligned ordinary RAM. MLX's typed constructor receives an aligned,
      // initialized temporary and copies into its own storage before this temporary is freed.
      std::vector<uint16_t> aligned(count);
      std::memcpy(aligned.data(), bytes, count * sizeof(uint16_t));
      mlx_array_set_(*output, mx::array(aligned.data(), {int(count)}, mx::uint16));
    } else if (format == 6) {
      // Wide logical encodings use a flat exact UInt32 limb span. This copies raw
      // bytes into aligned storage without numerical conversion or aliasing.
      std::vector<uint32_t> aligned(count);
      std::memcpy(aligned.data(), bytes, count * sizeof(uint32_t));
      mlx_array_set_(*output, mx::array(aligned.data(), {int(count)}, mx::uint32));
    } else {
      mlx_array_set_(*output, mx::array(static_cast<const uint8_t*>(bytes), {int(count)}, mx::uint8));
    }
  });
}
extern "C" int pcu_mlx_c_checked_unary_apply(
    mlx_array* output, mlx_array* records, pcu_mlx_c_checked_unary owner, mlx_array input) {
  return checked([&] {
    if (!output || !records || output == records || !owner.ctx)
      throw std::invalid_argument("invalid checked unary output/status/owner");
    const auto& state = *static_cast<CheckedUnary*>(owner.ctx);
    if (state.thread != std::this_thread::get_id())
      throw std::invalid_argument("checked unary used from another thread");
    const auto& value = mlx_array_get_(input);
    if (value.dtype() != state.carrier || value.shape() != state.input)
      throw std::invalid_argument("checked unary exact physical carrier/shape mismatch");
    auto result = mx::array::make_arrays({state.output,state.records},
        {state.carrier,mx::uint32}, state.primitive, {value});
    mlx_array_set_(*output, result[0]);
    mlx_array_set_(*records, result[1]);
  });
}
extern "C" int pcu_mlx_c_checked_unary_free(pcu_mlx_c_checked_unary owner) {
  return checked([&] {
    auto* state = static_cast<CheckedUnary*>(owner.ctx);
    if (state && state->thread != std::this_thread::get_id())
      throw std::invalid_argument("checked unary released from another thread");
    delete state;
  });
}
