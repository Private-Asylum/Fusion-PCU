#ifndef PCU_MLX_C_TRANSPORT_H
#define PCU_MLX_C_TRANSPORT_H
#include "mlx/c/array.h"
#include "mlx/c/stream.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Cold validated representation-only source. All counts are physical UInt lanes.
 * Actual input count is1..4, output count1..2; no fabricated arrays or caller pointers. */
typedef struct { void* ctx; } pcu_mlx_c_transport;
int pcu_mlx_c_transport_new(pcu_mlx_c_transport*, mlx_stream, uint32_t dtype,
    const uint32_t* input_counts, uint32_t input_count, uint32_t output_lanes,
    uint32_t output_count, const char* source);
int pcu_mlx_c_transport_apply(mlx_array* outputs, pcu_mlx_c_transport,
    const mlx_array* inputs, uint32_t input_count);
int pcu_mlx_c_transport_free(pcu_mlx_c_transport);
#ifdef __cplusplus
}
#endif
#endif
