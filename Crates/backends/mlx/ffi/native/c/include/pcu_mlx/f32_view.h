#ifndef PCU_MLX_C_F32_VIEW_H
#define PCU_MLX_C_F32_VIEW_H
#include "mlx/c/array.h"
#include "mlx/c/stream.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Equal-item-size bit view plus canonical reshape on the retained GPU stream.
 * to_matrix=0 accepts a dense rank2 F32 matrix and returns rank1 UInt32.
 * to_matrix=1 accepts dense rank1 UInt32 and returns exact rows*columns F32.
 * Descriptor construction is lazy. Caller retains both real holders until terminal. */
int pcu_mlx_c_f32_view(mlx_array*, mlx_array, mlx_stream,
    int32_t rows, int32_t columns, int to_matrix);
/* Terminal metadata check only: exact shared Data identity and byte offset.
 * No host materialization, pointer import or device-copy operation occurs. */
int pcu_mlx_c_f32_view_validate_shared(mlx_array input, mlx_array output);
#ifdef __cplusplus
}
#endif
#endif
