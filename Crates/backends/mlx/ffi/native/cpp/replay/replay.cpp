// Retain the verified native primitive; operators remain upstream mlx-c bodies.
#include "pcu_mlx/replay.h"
#include "reporting/reporting.hpp"
#include "mlx/c/private/mlx.h"
#include <mlx/primitives.h>
#include <mlx/version.h>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <thread>

static_assert(MLX_VERSION_NUMERIC == 32003, "direct C replay requires exact MLX0.32.3");

namespace {
namespace mx = mlx::core;
struct Replay {
  std::shared_ptr<mx::Primitive> primitive;
  mx::Shape left, right, output;
  mx::Stream stream;
  std::thread::id thread;
};
template<class F> int checked(F&& function) noexcept {
  try { function(); return 0; }
  catch (const std::exception& error) { pcu::mlx_c::detail::report(error.what()); }
  catch (...) { pcu::mlx_c::detail::report("unknown exception in direct-C retained replay"); }
  return 1;
}
void matrix(const mx::array& value) {
  if (value.dtype() != mx::float32 || value.ndim() != 2 || value.shape(0) <= 0
      || value.shape(1) <= 0) throw std::invalid_argument("retained replay requires positive rank-two F32");
}
}
extern "C" int pcu_mlx_c_replay_new(
    mlx_array output, mlx_array left, mlx_array right, mlx_stream stream, void** owner) {
  return checked([&] {
    if (!owner || *owner) throw std::invalid_argument("replay owner output must be empty");
    const auto& a = mlx_array_get_(left);
    const auto& b = mlx_array_get_(right);
    const auto& c = mlx_array_get_(output);
    const auto s = mlx_stream_get_(stream);
    matrix(a); matrix(b); matrix(c);
    if (a.shape(1) != b.shape(0) || c.shape(0) != a.shape(0) || c.shape(1) != b.shape(1)
        || !c.has_primitive() || !c.siblings().empty()
        || std::strcmp(c.primitive().name(), "Matmul") || c.primitive().stream() != s
        || c.inputs().size() != 2 || c.inputs()[0].id() != a.id()
        || c.inputs()[1].id() != b.id())
      throw std::invalid_argument("compiled graph exceeds retained single MatMul profile");
    *owner = new Replay{c.primitive_ptr(), a.shape(), b.shape(), c.shape(), s,
                        std::this_thread::get_id()};
  });
}
extern "C" int pcu_mlx_c_replay_apply(
    mlx_array* output, void* owner, mlx_array left, mlx_array right) {
  return checked([&] {
    if (!output || !owner) throw std::invalid_argument("null retained replay output/owner");
    const auto& replay = *static_cast<Replay*>(owner);
    if (replay.thread != std::this_thread::get_id())
      throw std::invalid_argument("retained replay used from another thread");
    const auto& a = mlx_array_get_(left);
    const auto& b = mlx_array_get_(right);
    if (a.dtype() != mx::float32 || b.dtype() != mx::float32
        || a.shape() != replay.left || b.shape() != replay.right)
      throw std::invalid_argument("retained replay exact F32 shapes mismatch");
    mlx_array_set_(*output, mx::array(replay.output, mx::float32, replay.primitive, {a, b}));
  });
}
extern "C" int pcu_mlx_c_replay_free(void* owner) {
  return checked([&] { delete static_cast<Replay*>(owner); });
}
