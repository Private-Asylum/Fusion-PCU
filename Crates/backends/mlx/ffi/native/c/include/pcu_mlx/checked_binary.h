#ifndef PCU_MLX_C_CHECKED_BINARY_H
#define PCU_MLX_C_CHECKED_BINARY_H
#include "mlx/c/array.h"
#include "mlx/c/stream.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct pcu_mlx_c_checked_binary { void* ctx; } pcu_mlx_c_checked_binary;
int pcu_mlx_c_checked_binary_new(pcu_mlx_c_checked_binary*, mlx_stream,
    uint32_t format, uint32_t operation, uint32_t underflow, uint32_t range,
    uint32_t count, uint32_t left_count, uint32_t right_count, uint32_t broadcast_mask);
/* Exact full physical input shapes are frozen cold, independently of count/index roles. */
int pcu_mlx_c_checked_binary_prefix_new(pcu_mlx_c_checked_binary*, mlx_stream,
    uint32_t format, uint32_t operation, uint32_t underflow, uint32_t range,
    uint32_t count, uint32_t left_count, uint32_t right_count, uint32_t broadcast_mask);
int pcu_mlx_c_checked_binary_apply(mlx_array*, mlx_array*,
    pcu_mlx_c_checked_binary, mlx_array left, mlx_array right);
int pcu_mlx_c_checked_binary_free(pcu_mlx_c_checked_binary);
#ifdef __cplusplus
}
#endif
#endif
