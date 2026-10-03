#ifndef PCU_MLX_C_ENCODED_PREFIX_H
#define PCU_MLX_C_ENCODED_PREFIX_H
#include "mlx/c/array.h"
#include "mlx/c/stream.h"
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Lazy immutable UInt8/UInt16/UInt32 prefix replacement. Callers retain both real
 * input holders through explicit stream completion; no host data is imported. */
int pcu_mlx_c_encoded_prefix_merge(mlx_array*, mlx_array prefix,
    mlx_array previous, mlx_stream);
#ifdef __cplusplus
}
#endif
#endif
