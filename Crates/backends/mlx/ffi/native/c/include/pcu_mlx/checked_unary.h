#ifndef PCU_MLX_C_CHECKED_UNARY_H
#define PCU_MLX_C_CHECKED_UNARY_H
#include "mlx/c/array.h"
#include "mlx/c/stream.h"
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Separate logical formats: 0 binary16, 1 bfloat16, 2 E4M3FN, 3 E5M2,
 * 4 binary32, 5 binary64. Physical arrays are UInt16/UInt8/UInt32 limbs;
 * no native floating dtype is implied. */
typedef struct pcu_mlx_c_checked_unary { void* ctx; } pcu_mlx_c_checked_unary;
int pcu_mlx_c_checked_unary_new(pcu_mlx_c_checked_unary*, mlx_stream,
    uint32_t format, uint32_t operation, uint32_t underflow, uint32_t range,
    uint32_t count, int broadcast);
/* Additive full-input constructor. count is the logical output/fault domain;
 * input_count is the separately retained full logical resident shape. */
int pcu_mlx_c_checked_unary_prefix_new(pcu_mlx_c_checked_unary*, mlx_stream,
    uint32_t format, uint32_t operation, uint32_t underflow, uint32_t range,
    uint32_t count, uint32_t input_count, int broadcast);
/* Upload format0..3 preserves the low arithmetic carriers; format4/5/6 copies
 * UInt8/UInt16/UInt32 physical lanes. count is physical, not wide logical count.
 * The caller's initialized bytes are copied via aligned storage when required. */
int pcu_mlx_c_checked_upload(mlx_array*, const void*, size_t count, uint32_t format);
int pcu_mlx_c_checked_unary_apply(mlx_array*, mlx_array*, pcu_mlx_c_checked_unary, mlx_array);
int pcu_mlx_c_checked_unary_free(pcu_mlx_c_checked_unary);
#ifdef __cplusplus
}
#endif
#endif
