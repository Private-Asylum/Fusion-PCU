#ifndef PCU_MLX_C_CHECKED_INTEGER_H
#define PCU_MLX_C_CHECKED_INTEGER_H
#include "mlx/c/array.h"
#include "mlx/c/stream.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct pcu_mlx_c_checked_integer { void* ctx; } pcu_mlx_c_checked_integer;
int pcu_mlx_c_checked_integer_new(pcu_mlx_c_checked_integer*, mlx_stream,
    uint32_t width, uint32_t is_signed, uint32_t operation, uint32_t range,
    uint32_t count, uint32_t left_count, uint32_t right_count, uint32_t broadcast_mask);
/* Separate frozen full capacities; old exact constructor stays unchanged. */
int pcu_mlx_c_checked_integer_prefix_new(pcu_mlx_c_checked_integer*, mlx_stream,
    uint32_t width, uint32_t is_signed, uint32_t operation, uint32_t range,
    uint32_t count, uint32_t left_count, uint32_t right_count, uint32_t broadcast_mask);
int pcu_mlx_c_checked_integer_apply(mlx_array*, mlx_array*,
    pcu_mlx_c_checked_integer, mlx_array left, mlx_array right);
int pcu_mlx_c_checked_integer_free(pcu_mlx_c_checked_integer);
#ifdef __cplusplus
}
#endif
#endif
