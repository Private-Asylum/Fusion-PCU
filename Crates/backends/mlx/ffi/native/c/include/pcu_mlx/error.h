#ifndef PCU_MLX_C_ERROR_H
#define PCU_MLX_C_ERROR_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Caller storage must stay live on this thread until end. Initialize to zero.
 * Nesting is LIFO. An inner scope captures its own errors; the outer caller
 * must inspect the returned status. No callback or global handler is installed.
 * Only selected endpoints recorded in provenance have catch-all containment.
 * This does not contain SDK worker, driver or process failures. */
typedef struct pcu_mlx_c_error_scope {
  struct pcu_mlx_c_error_scope* previous;
  char* message;
  size_t capacity;
  int32_t failed;
  int32_t active;
} pcu_mlx_c_error_scope;

int pcu_mlx_c_error_scope_begin(pcu_mlx_c_error_scope*, char*, size_t);
/* end returns 0 on successful pop, 1 on misuse. Inspect failed separately. */
int pcu_mlx_c_error_scope_end(pcu_mlx_c_error_scope*);
/* Literal text, bounded copying, no dynamic allocation. Callback exceptions
 * are swallowed. Global callbacks retain upstream semantics outside scopes. */
void _pcu_mlx_error_text(const char*, int, const char*);
/* Cleanup captures only an active scope, and never invokes a global callback. */
void _pcu_mlx_error_cleanup_text(const char*, int, const char*);

#ifdef __cplusplus
}
#endif
#endif
