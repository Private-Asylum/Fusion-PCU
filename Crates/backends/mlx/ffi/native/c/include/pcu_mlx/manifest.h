#ifndef PCU_MLX_C_MANIFEST_H
#define PCU_MLX_C_MANIFEST_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct pcu_mlx_c_manifest {
  uint32_t safety_abi, native_version_numeric;
  uint32_t array_handle_size, device_handle_size, stream_handle_size;
  uint32_t closure_handle_size, vector_array_handle_size, string_handle_size;
  int32_t float32_dtype;
  uint32_t flags;
  char native_header_version[32];
  char c_source_revision[64];
} pcu_mlx_c_manifest;

/* This zero-argument query precedes every layout-dependent extension call. */
uint32_t pcu_mlx_c_safety_abi(void);
int pcu_mlx_c_manifest_get(pcu_mlx_c_manifest*);

#ifdef __cplusplus
}
#endif
#endif
