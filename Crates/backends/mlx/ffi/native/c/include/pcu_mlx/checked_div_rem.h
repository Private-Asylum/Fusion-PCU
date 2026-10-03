#ifndef PCU_MLX_C_CHECKED_DIV_REM_H
#define PCU_MLX_C_CHECKED_DIV_REM_H
#include "mlx/c/array.h"
#include "mlx/c/stream.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct pcu_mlx_c_checked_div_rem { void* ctx; } pcu_mlx_c_checked_div_rem;
int pcu_mlx_c_checked_div_rem_new(pcu_mlx_c_checked_div_rem*, mlx_stream,
    uint32_t width, uint32_t is_signed, uint32_t count,
    uint32_t left_count, uint32_t right_count, uint32_t broadcast_mask);
/* Separately freezes full dense input capacities; apply still requires exact retained shapes. */
int pcu_mlx_c_checked_div_rem_prefix_new(pcu_mlx_c_checked_div_rem*, mlx_stream,
    uint32_t width, uint32_t is_signed, uint32_t count,
    uint32_t left_count, uint32_t right_count, uint32_t broadcast_mask);
int pcu_mlx_c_checked_div_rem_apply(mlx_array*, mlx_array*, mlx_array*,
    pcu_mlx_c_checked_div_rem, mlx_array left, mlx_array right);
int pcu_mlx_c_checked_div_rem_free(pcu_mlx_c_checked_div_rem);
#ifdef __cplusplus
}
#endif
#endif
