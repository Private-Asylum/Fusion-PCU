/* Compile as C11 against the actual pinned upstream C headers. The extension
 * interface exposes no C++ types or private implementation headers. */
#include "pcu_mlx/error.h"
#include "pcu_mlx/manifest.h"
#include "pcu_mlx/replay.h"
#include "pcu_mlx/safety.h"
#include "pcu_mlx/encoded_prefix.h"
#include "pcu_mlx/f32_view.h"
#include "pcu_mlx/carrier.h"
#include "pcu_mlx/checked_binary.h"
#include "pcu_mlx/checked_integer.h"
#include "pcu_mlx/checked_div_rem.h"
#include "pcu_mlx/composed.h"
#include "mlx/c/closure.h"
#include "mlx/c/device.h"
#include "mlx/c/string.h"
#include "mlx/c/vector.h"
#include <stddef.h>

_Static_assert(sizeof(void*) == 8, "qualified Apple arm64 pointer width");
typedef int (*pcu_composed_new_signature)(pcu_mlx_c_composed*,mlx_stream,uint32_t,
    const uint32_t*,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,const char*,const char*);
typedef int (*pcu_composed_apply_signature)(mlx_array*,pcu_mlx_c_composed,const mlx_array*,uint32_t);
typedef int (*pcu_composed_free_signature)(pcu_mlx_c_composed);
_Static_assert(sizeof(pcu_mlx_c_composed)==sizeof(void*), "composition handle width");
_Static_assert(_Generic(&pcu_mlx_c_composed_new,pcu_composed_new_signature:1,default:0), "composition new C ABI");
_Static_assert(_Generic(&pcu_mlx_c_composed_apply,pcu_composed_apply_signature:1,default:0), "composition apply C ABI");
_Static_assert(_Generic(&pcu_mlx_c_composed_free,pcu_composed_free_signature:1,default:0), "composition free C ABI");
typedef int (*pcu_shared_signature)(mlx_array,mlx_array);
_Static_assert(_Generic(&pcu_mlx_c_f32_view_validate_shared,pcu_shared_signature:1,default:0), "shared view ABI");
typedef int (*pcu_view_signature)(mlx_array*,mlx_array,mlx_stream,int32_t,int32_t,int);
_Static_assert(_Generic(&pcu_mlx_c_f32_view,pcu_view_signature:1,default:0), "F32 view ABI");
typedef int (*pcu_prefix_signature)(mlx_array*, mlx_array, mlx_array, mlx_stream);
typedef int (*pcu_carrier_new_signature)(pcu_mlx_c_carrier*,mlx_stream,uint32_t,uint32_t,uint32_t,int);
typedef int (*pcu_carrier_apply_signature)(mlx_array*,pcu_mlx_c_carrier,mlx_array);
typedef int (*pcu_carrier_free_signature)(pcu_mlx_c_carrier);
_Static_assert(sizeof(pcu_mlx_c_carrier) == sizeof(void*), "carrier handle width");
_Static_assert(_Generic(&pcu_mlx_c_carrier_new,pcu_carrier_new_signature:1,default:0), "carrier new ABI");
_Static_assert(_Generic(&pcu_mlx_c_carrier_apply,pcu_carrier_apply_signature:1,default:0), "carrier apply ABI");
_Static_assert(_Generic(&pcu_mlx_c_carrier_free,pcu_carrier_free_signature:1,default:0), "carrier free ABI");
_Static_assert(_Generic(&pcu_mlx_c_encoded_prefix_merge,
    pcu_prefix_signature: 1, default: 0), "encoded prefix C ABI signature");
_Static_assert(sizeof(mlx_array) == 8, "array opaque handle width");
_Static_assert(sizeof(mlx_device) == 8, "device opaque handle width");
_Static_assert(sizeof(mlx_stream) == 8, "stream opaque handle width");
_Static_assert(sizeof(mlx_closure) == 8, "closure opaque handle width");
_Static_assert(sizeof(mlx_vector_array) == 8, "vector opaque handle width");
_Static_assert(sizeof(mlx_string) == 8, "string opaque handle width");
_Static_assert(MLX_UINT8 == 1 && MLX_UINT16 == 2 && MLX_UINT32 == 3, "selected checked carrier vocabulary");
_Static_assert(sizeof(pcu_mlx_c_checked_unary) == sizeof(void*), "checked primitive handle width");
_Static_assert(sizeof(pcu_mlx_c_checked_binary) == sizeof(void*), "binary primitive handle width");
typedef int (*pcu_binary_new_signature)(pcu_mlx_c_checked_binary*, mlx_stream,
    uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);
typedef int (*pcu_binary_apply_signature)(mlx_array*,mlx_array*,
    pcu_mlx_c_checked_binary,mlx_array,mlx_array);
_Static_assert(_Generic(&pcu_mlx_c_checked_binary_new,
    pcu_binary_new_signature: 1, default: 0), "binary prepare C ABI signature");
_Static_assert(_Generic(&pcu_mlx_c_checked_binary_prefix_new,
    pcu_binary_new_signature:1,default:0), "binary prefix new ABI");
_Static_assert(_Generic(&pcu_mlx_c_checked_binary_apply,
    pcu_binary_apply_signature: 1, default: 0), "binary execute C ABI signature");
_Static_assert(MLX_FLOAT32 == 10, "selected F32 dtype vocabulary");
_Static_assert(sizeof(pcu_mlx_c_error_scope) == 32, "safety ABI1 scope size");
_Static_assert(offsetof(pcu_mlx_c_error_scope, previous) == 0, "scope previous");
_Static_assert(offsetof(pcu_mlx_c_error_scope, message) == 8, "scope message");
_Static_assert(offsetof(pcu_mlx_c_error_scope, capacity) == 16, "scope capacity");
_Static_assert(offsetof(pcu_mlx_c_error_scope, failed) == 24, "scope failed");
_Static_assert(offsetof(pcu_mlx_c_error_scope, active) == 28, "scope active");
_Static_assert(sizeof(pcu_mlx_c_manifest) == 136, "safety ABI1 manifest size");
_Static_assert(offsetof(pcu_mlx_c_manifest, float32_dtype) == 32, "manifest dtype");
_Static_assert(offsetof(pcu_mlx_c_manifest, flags) == 36, "manifest flags");
_Static_assert(offsetof(pcu_mlx_c_manifest, native_header_version) == 40, "header version");
_Static_assert(offsetof(pcu_mlx_c_manifest, c_source_revision) == 72, "source revision");

_Static_assert(sizeof(pcu_mlx_c_checked_integer) == sizeof(void*), "integer primitive handle width");
typedef int (*pcu_integer_new_signature)(pcu_mlx_c_checked_integer*, mlx_stream,
    uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);
typedef int (*pcu_integer_apply_signature)(mlx_array*,mlx_array*,
    pcu_mlx_c_checked_integer,mlx_array,mlx_array);
typedef int (*pcu_integer_free_signature)(pcu_mlx_c_checked_integer);
_Static_assert(_Generic(&pcu_mlx_c_checked_integer_prefix_new,pcu_integer_new_signature:1,default:0), "integer prefix new ABI");
_Static_assert(_Generic(&pcu_mlx_c_checked_integer_new,
    pcu_integer_new_signature: 1, default: 0), "integer prepare C ABI signature");
_Static_assert(_Generic(&pcu_mlx_c_checked_integer_apply,
    pcu_integer_apply_signature: 1, default: 0), "integer execute C ABI signature");
_Static_assert(_Generic(&pcu_mlx_c_checked_integer_free,
    pcu_integer_free_signature: 1, default: 0), "integer release C ABI signature");

_Static_assert(sizeof(pcu_mlx_c_checked_div_rem) == sizeof(void*), "div/rem handle width");
typedef int (*pcu_div_rem_new_signature)(pcu_mlx_c_checked_div_rem*,mlx_stream,
    uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);
typedef int (*pcu_div_rem_apply_signature)(mlx_array*,mlx_array*,mlx_array*,
    pcu_mlx_c_checked_div_rem,mlx_array,mlx_array);
typedef int (*pcu_div_rem_free_signature)(pcu_mlx_c_checked_div_rem);
_Static_assert(_Generic(&pcu_mlx_c_checked_div_rem_new,pcu_div_rem_new_signature:1,default:0), "div/rem new ABI");
_Static_assert(_Generic(&pcu_mlx_c_checked_div_rem_prefix_new,pcu_div_rem_new_signature:1,default:0), "div/rem prefix new ABI");
_Static_assert(_Generic(&pcu_mlx_c_checked_div_rem_apply,pcu_div_rem_apply_signature:1,default:0), "div/rem apply ABI");
_Static_assert(_Generic(&pcu_mlx_c_checked_div_rem_free,pcu_div_rem_free_signature:1,default:0), "div/rem free ABI");
