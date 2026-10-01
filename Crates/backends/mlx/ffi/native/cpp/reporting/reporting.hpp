#ifndef PCU_MLX_REPORTING_INTERNAL_HPP
#define PCU_MLX_REPORTING_INTERNAL_HPP
#include <cstddef>

namespace pcu::mlx_c::detail {
// Shared implementation helpers are hidden C++ symbols, never native ABI.
__attribute__((visibility("hidden")))
void copy_text(char*, std::size_t, const char*) noexcept;
__attribute__((visibility("hidden")))
void report(const char*) noexcept;
}
#endif
