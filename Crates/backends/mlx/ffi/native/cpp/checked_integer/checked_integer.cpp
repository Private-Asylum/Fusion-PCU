// Fixed integer kernels compiled, allocated and scheduled exclusively by MLX.
#include "pcu_mlx/checked_integer.h"
#include "reporting/reporting.hpp"
#include "shader/checked_integer.hpp"
#include "shader/body.hpp"
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
static_assert(MLX_VERSION_NUMERIC == 32003, "checked integer requires exact MLX0.32.3");
namespace {
namespace mx = mlx::core;
struct CheckedInteger {
  std::shared_ptr<mx::Primitive> primitive;
  mx::Shape left, right, output, records;
  mx::Dtype carrier;
  mx::Stream stream;
  std::thread::id thread;
};
template<class F> int contained(F&& function) noexcept {
  try { function(); return 0; }
  catch (const std::exception& error) { pcu::mlx_c::detail::report(error.what()); }
  catch (...) { pcu::mlx_c::detail::report("unknown exception in MLX checked integer"); }
  return 1;
}
std::string header(uint32_t width, uint32_t is_signed, uint32_t operation,
                   uint32_t range, uint32_t count, uint32_t broadcast) {
  const auto define = [](const char* name, uint32_t value) {
    return std::string("#define ") + name + " " + std::to_string(value) + "u\n";
  };
  const uint32_t mask = width < 32 ? (uint32_t(1) << width) - 1 : UINT32_MAX;
  const uint32_t sign = uint32_t(1) << (width < 32 ? width - 1 : 31);
  return define("PCU_WIDTH",width) + define("PCU_SIGNED",is_signed)
      + define("PCU_OPERATION",operation) + define("PCU_CLAMP",range)
      + define("PCU_COUNT",count) + define("PCU_LEFT_BROADCAST",broadcast & 1)
      + define("PCU_RIGHT_BROADCAST",broadcast & 2)
      + define("PCU_MASK",mask) + define("PCU_SIGN",sign) + pcu_mlx_integer_header;
}
}
static int prepare_checked_integer(
    pcu_mlx_c_checked_integer* owner, mlx_stream stream, uint32_t width,
    uint32_t is_signed, uint32_t operation, uint32_t range, uint32_t count,
    uint32_t left_count, uint32_t right_count, uint32_t broadcast_mask, bool prefix) {
  return contained([&] {
    const auto valid_extent = [count,prefix](uint32_t size, bool broadcast) {
      return prefix ? size >= (broadcast ? 1u : count)
                    : size == count || (broadcast && size == 1);
    };
    if (!owner || owner->ctx || !(width == 8 || width == 16 || width == 32 || width == 64
            || width == 128 || width == 256 || width == 512)
        || is_signed > 1 || operation > 2
        || range > 1 || !count || count > uint32_t(std::numeric_limits<int>::max())
        || broadcast_mask > 3 || !valid_extent(left_count, broadcast_mask & 1)
        || !valid_extent(right_count, broadcast_mask & 2))
      throw std::invalid_argument("invalid checked integer profile/owner");
    const auto s = mlx_stream_get_(stream);
    if (s.device.type != mx::Device::gpu)
      throw std::invalid_argument("checked integer requires explicit MLX GPU stream");
    const auto carrier = width == 8 ? mx::uint8 : width == 16 ? mx::uint16 : mx::uint32;
    const uint32_t limbs = (width + 31) / 32;
    if (count > uint32_t(std::numeric_limits<int>::max()) / limbs
        || left_count > uint32_t(std::numeric_limits<int>::max()) / limbs
        || right_count > uint32_t(std::numeric_limits<int>::max()) / limbs)
      throw std::invalid_argument("checked integer physical carrier extent overflow");
    const mx::Shape a{int(left_count*limbs)}, b{int(right_count*limbs)}, out{int(count*limbs)}, records{int(count)};
    auto left = mx::broadcast_to(mx::array(uint32_t(0),carrier), a, s);
    auto right = mx::broadcast_to(mx::array(uint32_t(0),carrier), b, s);
    auto kernel = mx::fast::metal_kernel("pcu_mlx_checked_integer", {"left","right"},
        {"output","records"}, pcu_mlx_integer_body,
        header(width,is_signed,operation,range,count,broadcast_mask));
    // Baseline group size one divides every positive logical grid exactly.
    auto result = kernel({left,right}, {out,records}, {carrier,mx::uint32},
        {int(count),1,1}, {1,1,1}, {}, std::nullopt, false, s);
    if (result.size() != 2 || !result[0].has_primitive()
        || result[0].primitive().stream() != s || result[0].dtype() != carrier
        || result[1].dtype() != mx::uint32
        || result[0].primitive_ptr() != result[1].primitive_ptr())
      throw std::runtime_error("checked integer primitive ownership/stream mismatch");
    owner->ctx = new CheckedInteger{result[0].primitive_ptr(),a,b,out,records,carrier,s,
        std::this_thread::get_id()};
  });
}
extern "C" int pcu_mlx_c_checked_integer_new(
    pcu_mlx_c_checked_integer* owner, mlx_stream stream, uint32_t width,
    uint32_t is_signed, uint32_t operation, uint32_t range, uint32_t count,
    uint32_t left_count, uint32_t right_count, uint32_t broadcast_mask) {
  return prepare_checked_integer(owner,stream,width,is_signed,operation,range,count,
      left_count,right_count,broadcast_mask,false);
}
extern "C" int pcu_mlx_c_checked_integer_prefix_new(
    pcu_mlx_c_checked_integer* owner, mlx_stream stream, uint32_t width,
    uint32_t is_signed, uint32_t operation, uint32_t range, uint32_t count,
    uint32_t left_count, uint32_t right_count, uint32_t broadcast_mask) {
  return prepare_checked_integer(owner,stream,width,is_signed,operation,range,count,
      left_count,right_count,broadcast_mask,true);
}
extern "C" int pcu_mlx_c_checked_integer_apply(
    mlx_array* output, mlx_array* records, pcu_mlx_c_checked_integer owner,
    mlx_array left, mlx_array right) {
  return contained([&] {
    if (!output || !records || output == records || !owner.ctx)
      throw std::invalid_argument("invalid checked integer output/status/owner");
    const auto& state = *static_cast<CheckedInteger*>(owner.ctx);
    const auto& a = mlx_array_get_(left);
    const auto& b = mlx_array_get_(right);
    if (state.thread != std::this_thread::get_id()
        || a.dtype() != state.carrier || b.dtype() != state.carrier
        || a.shape() != state.left || b.shape() != state.right)
      throw std::invalid_argument("checked integer exact carrier/shape/thread mismatch");
    auto result = mx::array::make_arrays({state.output,state.records},
        {state.carrier,mx::uint32}, state.primitive, {a,b});
    mlx_array_set_(*output,result[0]);
    mlx_array_set_(*records,result[1]);
  });
}
extern "C" int pcu_mlx_c_checked_integer_free(pcu_mlx_c_checked_integer owner) {
  return contained([&] {
    auto* state = static_cast<CheckedInteger*>(owner.ctx);
    if (state && state->thread != std::this_thread::get_id())
      throw std::invalid_argument("checked integer released from another thread");
    delete state;
  });
}
