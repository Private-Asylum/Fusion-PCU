#ifndef PCU_MLX_C_REPLAY_H
#define PCU_MLX_C_REPLAY_H
#include "mlx/c/array.h"
#include "mlx/c/stream.h"
#ifdef __cplusplus
extern "C" {
#endif

/* A frozen native Matmul primitive with exact positive rank-two F32 shapes,
 * input identity and explicit stream. Caller retains the library/session.
 * Application rebinds a descriptor; evaluation remains an upstream C call. */
int pcu_mlx_c_replay_new(mlx_array, mlx_array, mlx_array, mlx_stream, void**);
int pcu_mlx_c_replay_apply(mlx_array*, void*, mlx_array, mlx_array);
int pcu_mlx_c_replay_free(void*);

#ifdef __cplusplus
}
#endif
#endif
