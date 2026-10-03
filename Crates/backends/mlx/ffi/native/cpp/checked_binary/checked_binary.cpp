// Fixed integer kernels compiled, allocated and scheduled exclusively by MLX.
#include "pcu_mlx/checked_binary.h"
#include "reporting/reporting.hpp"
#include "shader/low_binary.hpp"
#include "shader/f32_binary.hpp"
#include "shader/f64_binary.hpp"
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
static_assert(MLX_VERSION_NUMERIC == 32003, "checked binary requires exact MLX0.32.3");
namespace {
namespace mx = mlx::core;
struct CheckedBinary {
  std::shared_ptr<mx::Primitive> primitive;
  mx::Shape left, right, output, records;
  mx::Dtype carrier;
  mx::Stream stream;
  std::thread::id thread;
};
template<class F> int contained(F&& function) noexcept {
  try { function(); return 0; }
  catch (const std::exception& error) { pcu::mlx_c::detail::report(error.what()); }
  catch (...) { pcu::mlx_c::detail::report("unknown exception in MLX checked binary"); }
  return 1;
}
std::string source(uint32_t format, uint32_t operation, uint32_t underflow,
                   uint32_t range, uint32_t count, uint32_t broadcast) {
  const auto literal = [](uint32_t value) { return std::to_string(value) + "u"; };
  const std::string prefix = "uint i = thread_position_in_grid.x;\nif (i >= " + literal(count) + ") return;\n";
  const auto left = broadcast & 1 ? "0u" : "i";
  const auto right = broadcast & 2 ? "0u" : "i";
  if (format == 4) {
    return prefix + "uint a = left[" + left + "], b = right[" + right + "];\n"
      "Binary arithmetic = {" + literal(underflow) + "," + (range ? "true" : "false") + "};\nResult result;\n"
      "if (((a >> 23) & 255u) == 255u || ((b >> 23) & 255u) == 255u) result = fault(1);\n"
      "else " + (operation == 0 ? std::string("result = arithmetic.add(a,b,false);") : operation == 1 ? std::string("result = arithmetic.add(a,b,true);") : operation == 2 ? std::string("result = arithmetic.multiply(a,b);") : std::string("result = arithmetic.divide(a,b);"))
      + "\noutput[i] = result.bits; records[i] = pcu_diagnostic(result.status);\n";
  }
  if (format == 5) {
    return prefix + "uint a_index = 2u * " + left + ", b_index = 2u * " + right + ";\n"
      "ulong a = ulong(left[a_index]) | (ulong(left[a_index+1u]) << 32);\n"
      "ulong b = ulong(right[b_index]) | (ulong(right[b_index+1u]) << 32);\nResult result;\n"
      "if (((a >> 52) & 2047ul) == 2047ul || ((b >> 52) & 2047ul) == 2047ul) result = fault(4);\n"
      "else " + (operation == 0 ? std::string("result = add(a,b,false,") : operation == 1 ? std::string("result = add(a,b,true,") : operation == 2 ? std::string("result = multiply(a,b,") : std::string("result = divide(a,b,"))
      + literal(underflow | (range ? 0x100u : 0u)) + ");\n"
      "output[2u*i] = uint(result.bits); output[2u*i+1u] = uint(result.bits >> 32); records[i] = result.status;\n";
  }
  const uint32_t fraction[] = {10, 7, 3, 2};
  const uint32_t sign[] = {0x8000, 0x8000, 0x80, 0x80};
  const uint32_t maximum[] = {0x7bff, 0x7f7f, 0x7e, 0x7b};
  const uint32_t bias[] = {15, 127, 7, 15};
  // Only bounded profile literals and fixed variable names reach the compiler.
  return "uint i = thread_position_in_grid.x;\nif (i >= " + literal(count) + ") return;\n"
      "PcuBinaryArithmetic arithmetic = {{" + literal(fraction[format]) + ","
      + literal(sign[format]) + "," + literal(maximum[format]) + ","
      + std::to_string(bias[format]) + "}," + literal(underflow) + ","
      + (range ? "true" : "false") + "};\n"
      "PcuBinaryResult result = arithmetic.evaluate(uint(left["
      + (broadcast & 1 ? "0" : "i") + "]),uint(right["
      + (broadcast & 2 ? "0" : "i") + "])," + literal(operation) + ");\n"
      "output[i] = result.bits; records[i] = pcu_diagnostic(result.status);\n";
}
}
static int prepare_checked_binary(
    pcu_mlx_c_checked_binary* owner, mlx_stream stream, uint32_t format,
    uint32_t operation, uint32_t underflow, uint32_t range, uint32_t count,
    uint32_t left_count, uint32_t right_count, uint32_t broadcast_mask, bool prefix) {
  return contained([&] {
    const auto valid_extent = [count,prefix](uint32_t size, bool broadcast) {
      return prefix ? size >= (broadcast ? 1u : count)
                    : size == count || (broadcast && size == 1);
    };
    if (!owner || owner->ctx || format > 5 || operation > 3 || underflow > 2
        || range > 1 || !count || count > uint32_t(std::numeric_limits<int>::max())
        || broadcast_mask > 3 || !valid_extent(left_count, broadcast_mask & 1)
        || !valid_extent(right_count, broadcast_mask & 2))
      throw std::invalid_argument("invalid checked binary profile/owner");
    const auto s = mlx_stream_get_(stream);
    if (s.device.type != mx::Device::gpu)
      throw std::invalid_argument("checked binary requires explicit MLX GPU stream");
    const auto carrier = format < 2 ? mx::uint16 : format < 4 ? mx::uint8 : mx::uint32;
    const uint32_t limbs = format == 5 ? 2 : 1;
    if (count > uint32_t(std::numeric_limits<int>::max()) / limbs
        || left_count > uint32_t(std::numeric_limits<int>::max()) / limbs
        || right_count > uint32_t(std::numeric_limits<int>::max()) / limbs)
      throw std::invalid_argument("checked binary physical carrier extent overflow");
    const mx::Shape a{int(left_count*limbs)}, b{int(right_count*limbs)}, out{int(count*limbs)}, records{int(count)};
    auto left = mx::broadcast_to(mx::array(uint32_t(0),carrier), a, s);
    auto right = mx::broadcast_to(mx::array(uint32_t(0),carrier), b, s);
    auto kernel = mx::fast::metal_kernel("pcu_mlx_checked_binary", {"left","right"},
        {"output","records"}, source(format,operation,underflow,range,count,broadcast_mask),
        format == 4 ? pcu_mlx_f32_binary_header : format == 5 ? pcu_mlx_f64_binary_header : pcu_mlx_low_binary_header);
    // Baseline group size one divides every positive logical grid exactly.
    auto result = kernel({left,right}, {out,records}, {carrier,mx::uint32},
        {int(count),1,1}, {1,1,1}, {}, std::nullopt, false, s);
    if (result.size() != 2 || !result[0].has_primitive()
        || result[0].primitive().stream() != s || result[0].dtype() != carrier
        || result[1].dtype() != mx::uint32
        || result[0].primitive_ptr() != result[1].primitive_ptr())
      throw std::runtime_error("checked binary primitive ownership/stream mismatch");
    owner->ctx = new CheckedBinary{result[0].primitive_ptr(),a,b,out,records,carrier,s,
        std::this_thread::get_id()};
  });
}
extern "C" int pcu_mlx_c_checked_binary_new(
    pcu_mlx_c_checked_binary* owner, mlx_stream stream, uint32_t format,
    uint32_t operation, uint32_t underflow, uint32_t range, uint32_t count,
    uint32_t left_count, uint32_t right_count, uint32_t broadcast_mask) {
  return prepare_checked_binary(owner,stream,format,operation,underflow,range,count,
      left_count,right_count,broadcast_mask,false);
}
extern "C" int pcu_mlx_c_checked_binary_prefix_new(
    pcu_mlx_c_checked_binary* owner, mlx_stream stream, uint32_t format,
    uint32_t operation, uint32_t underflow, uint32_t range, uint32_t count,
    uint32_t left_count, uint32_t right_count, uint32_t broadcast_mask) {
  return prepare_checked_binary(owner,stream,format,operation,underflow,range,count,
      left_count,right_count,broadcast_mask,true);
}
extern "C" int pcu_mlx_c_checked_binary_apply(
    mlx_array* output, mlx_array* records, pcu_mlx_c_checked_binary owner,
    mlx_array left, mlx_array right) {
  return contained([&] {
    if (!output || !records || output == records || !owner.ctx)
      throw std::invalid_argument("invalid checked binary output/status/owner");
    const auto& state = *static_cast<CheckedBinary*>(owner.ctx);
    const auto& a = mlx_array_get_(left);
    const auto& b = mlx_array_get_(right);
    if (state.thread != std::this_thread::get_id()
        || a.dtype() != state.carrier || b.dtype() != state.carrier
        || a.shape() != state.left || b.shape() != state.right)
      throw std::invalid_argument("checked binary exact carrier/shape/thread mismatch");
    auto result = mx::array::make_arrays({state.output,state.records},
        {state.carrier,mx::uint32}, state.primitive, {a,b});
    mlx_array_set_(*output,result[0]);
    mlx_array_set_(*records,result[1]);
  });
}
extern "C" int pcu_mlx_c_checked_binary_free(pcu_mlx_c_checked_binary owner) {
  return contained([&] {
    auto* state = static_cast<CheckedBinary*>(owner.ctx);
    if (state && state->thread != std::this_thread::get_id())
      throw std::invalid_argument("checked binary released from another thread");
    delete state;
  });
}
