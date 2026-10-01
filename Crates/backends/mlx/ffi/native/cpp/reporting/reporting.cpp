// Calling-thread reporting/TLS scopes and the independent global client handler.
#include "pcu_mlx/error.h"
#include "reporting/reporting.hpp"
#include "mlx/c/error.h"
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <utility>

namespace {
thread_local pcu_mlx_c_error_scope* current = nullptr;
void default_handler(const char* msg, void*) {
  std::fprintf(stderr, "MLX error: %s\n", msg);
  std::exit(-1);
}
struct Handler {
  mlx_error_handler_func callback = default_handler;
  std::shared_ptr<void> data;
};
std::mutex handler_mutex;
Handler global_handler;
void capture_cleanup(const char* message) noexcept {
  if (current) {
    if (!current->failed)
      pcu::mlx_c::detail::copy_text(current->message, current->capacity, message);
    current->failed = 1;
  }
}
}

namespace pcu::mlx_c::detail {
void copy_text(char* dst, std::size_t capacity, const char* src) noexcept {
  if (!dst || !capacity) return;
  if (!src) src = "MLX error";
  std::size_t i = 0;
  while (i + 1 < capacity && src[i]) { dst[i] = src[i]; ++i; }
  dst[i] = '\0';
}
void report(const char* message) noexcept {
  if (current) { capture_cleanup(message); return; }
  // Copy the pair under the lock and invoke after unlocking: a callback may
  // replace its own handler or invoke C again without deadlock. Shared ownership
  // keeps its data alive until invocation completes, even with concurrent set.
  try {
    Handler snapshot;
    { std::lock_guard<std::mutex> lock(handler_mutex); snapshot = global_handler; }
    try { snapshot.callback(message, snapshot.data.get()); } catch (...) {}
  } catch (...) {}
}
}

extern "C" int pcu_mlx_c_error_scope_begin(
    pcu_mlx_c_error_scope* scope, char* message, size_t capacity) {
  if (!scope || scope->active || !message || !capacity) return 1;
  scope->previous = current;
  scope->message = message;
  scope->capacity = capacity;
  scope->failed = 0;
  scope->active = 1;
  message[0] = '\0';
  current = scope;
  return 0;
}
extern "C" int pcu_mlx_c_error_scope_end(pcu_mlx_c_error_scope* scope) {
  if (!scope || current != scope || !scope->active) return 1;
  current = scope->previous;
  scope->previous = nullptr;
  scope->active = 0;
  return 0;
}
extern "C" void mlx_set_error_handler(
    mlx_error_handler_func callback, void* data, void (*dtor)(void*)) {
  try {
    Handler replacement;
    replacement.callback = callback ? callback : default_handler;
    // Preserve upstream data ownership: data is retained only with a dtor.
    if (dtor) replacement.data = std::shared_ptr<void>(data, [dtor](void* ptr) noexcept {
      try { dtor(ptr); } catch (...) { capture_cleanup("exception in MLX error payload destructor"); }
    });
    { std::lock_guard<std::mutex> lock(handler_mutex);
      std::swap(replacement, global_handler); }
    // Old data is destroyed outside the lock, permitting reentrant destructors.
  } catch (...) { pcu::mlx_c::detail::report("failed to install MLX global error handler"); }
}
extern "C" void _pcu_mlx_error_text(const char*, int, const char* text) {
  pcu::mlx_c::detail::report(text ? text : "MLX error");
}
extern "C" void _pcu_mlx_error_cleanup_text(const char*, int, const char* text) {
  // Never snapshot global_handler while its payload is destroyed at teardown.
  capture_cleanup(text ? text : "MLX cleanup error");
}
extern "C" void _mlx_error(const char*, int, const char* format, ...) {
  char message[2048] = {};
  va_list args;
  va_start(args, format);
  const int written = std::vsnprintf(message, sizeof(message), format ? format : "MLX error", args);
  va_end(args);
  if (written < 0)
    pcu::mlx_c::detail::copy_text(message, sizeof(message), "MLX error formatting failed");
  message[sizeof(message) - 1] = '\0';
  pcu::mlx_c::detail::report(message);
}
