#ifndef PCU_MLX_C_CARRIER_H
#define PCU_MLX_C_CARRIER_H
#include "mlx/c/array.h"
#include "mlx/c/stream.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Exact integer-limb copy/broadcast. Prepared primitive retains its actual GPU stream.
 * Logical scalar semantics remain Rust-owned; these counts describe physical UInt lanes. */
typedef struct { void* ctx; } pcu_mlx_c_carrier;
int pcu_mlx_c_carrier_new(pcu_mlx_c_carrier*, mlx_stream, uint32_t dtype,
    uint32_t input_count, uint32_t output_count, int broadcast);
/* Additive full-input-shape profile. scalar_lanes is the physical width of one
 * logical scalar for broadcast, zero for dense prefix copy. No input view is built. */
int pcu_mlx_c_carrier_prefix_new(pcu_mlx_c_carrier*, mlx_stream, uint32_t dtype,
    uint32_t input_count, uint32_t output_count, uint32_t scalar_lanes, int broadcast);
int pcu_mlx_c_carrier_apply(mlx_array*, pcu_mlx_c_carrier, mlx_array);
int pcu_mlx_c_carrier_free(pcu_mlx_c_carrier);
#ifdef __cplusplus
}
#endif
#endif
