#ifndef PCU_MLX_C_COMPOSED_H
#define PCU_MLX_C_COMPOSED_H
#include "mlx/c/array.h"
#include "mlx/c/stream.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Exact cold typed input shapes, one/two private payload siblings and a distinct
 * UInt32 status sibling. Header/body are bounded adapter-generated native source.
 * No caller array or source pointer escapes a synchronous contained call. */
typedef struct { void* ctx; } pcu_mlx_c_composed;
int pcu_mlx_c_composed_new(pcu_mlx_c_composed*, mlx_stream, uint32_t dtype,
    const uint32_t* input_counts, uint32_t input_count, uint32_t logical_count,
    uint32_t payload_lanes, uint32_t payload_count, uint32_t status_lanes,
    const char* header, const char* body);
int pcu_mlx_c_composed_apply(mlx_array* outputs, pcu_mlx_c_composed,
    const mlx_array* inputs, uint32_t input_count);
int pcu_mlx_c_composed_free(pcu_mlx_c_composed);
#ifdef __cplusplus
}
#endif
#endif
