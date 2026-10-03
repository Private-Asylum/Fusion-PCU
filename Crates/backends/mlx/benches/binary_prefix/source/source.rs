//! Genuine annotated binary workload profiles, specialized only during cold construction.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedFloat,
    PcuScalar,
    PcuTensor,
    PcuExecutionError,
};
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn add_ieee_reject_0<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(clamp_range))]
pub fn add_ieee_clamp_0<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result))]
pub fn add_tight_reject_0<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result), flag(clamp_range))]
pub fn add_tight_clamp_0<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(allow_gradual_underflow))]
pub fn add_gradual_reject_0<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(allow_gradual_underflow), flag(clamp_range))]
pub fn add_gradual_clamp_0<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn sub_ieee_reject_0<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(clamp_range))]
pub fn sub_ieee_clamp_0<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result))]
pub fn sub_tight_reject_0<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result), flag(clamp_range))]
pub fn sub_tight_clamp_0<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(allow_gradual_underflow))]
pub fn sub_gradual_reject_0<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(allow_gradual_underflow), flag(clamp_range))]
pub fn sub_gradual_clamp_0<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn mul_ieee_reject_0<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(clamp_range))]
pub fn mul_ieee_clamp_0<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result))]
pub fn mul_tight_reject_0<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result), flag(clamp_range))]
pub fn mul_tight_clamp_0<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(allow_gradual_underflow))]
pub fn mul_gradual_reject_0<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(allow_gradual_underflow), flag(clamp_range))]
pub fn mul_gradual_clamp_0<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn div_ieee_reject_0<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] / right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(clamp_range))]
pub fn div_ieee_clamp_0<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] / right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result))]
pub fn div_tight_reject_0<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] / right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result), flag(clamp_range))]
pub fn div_tight_clamp_0<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] / right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(allow_gradual_underflow))]
pub fn div_gradual_reject_0<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] / right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(allow_gradual_underflow), flag(clamp_range))]
pub fn div_gradual_clamp_0<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] / right[id];
}
#[pcu(invocations = 1, crate_path = ::pcu_facade)]
pub fn add_ieee_reject_1<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] + right[id];
        id += stride;
    }
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(clamp_range))]
pub fn add_ieee_clamp_1<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] + right[id];
        id += stride;
    }
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result))]
pub fn add_tight_reject_1<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] + right[id];
        id += stride;
    }
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result), flag(clamp_range))]
pub fn add_tight_clamp_1<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] + right[id];
        id += stride;
    }
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(allow_gradual_underflow))]
pub fn add_gradual_reject_1<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] + right[id];
        id += stride;
    }
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(allow_gradual_underflow), flag(clamp_range))]
pub fn add_gradual_clamp_1<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] + right[id];
        id += stride;
    }
}
#[pcu(invocations = 1, crate_path = ::pcu_facade)]
pub fn sub_ieee_reject_1<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] - right[id];
        id += stride;
    }
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(clamp_range))]
pub fn sub_ieee_clamp_1<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] - right[id];
        id += stride;
    }
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result))]
pub fn sub_tight_reject_1<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] - right[id];
        id += stride;
    }
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result), flag(clamp_range))]
pub fn sub_tight_clamp_1<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] - right[id];
        id += stride;
    }
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(allow_gradual_underflow))]
pub fn sub_gradual_reject_1<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] - right[id];
        id += stride;
    }
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(allow_gradual_underflow), flag(clamp_range))]
pub fn sub_gradual_clamp_1<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] - right[id];
        id += stride;
    }
}
#[pcu(invocations = 1, crate_path = ::pcu_facade)]
pub fn mul_ieee_reject_1<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] * right[id];
        id += stride;
    }
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(clamp_range))]
pub fn mul_ieee_clamp_1<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] * right[id];
        id += stride;
    }
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result))]
pub fn mul_tight_reject_1<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] * right[id];
        id += stride;
    }
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result), flag(clamp_range))]
pub fn mul_tight_clamp_1<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] * right[id];
        id += stride;
    }
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(allow_gradual_underflow))]
pub fn mul_gradual_reject_1<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] * right[id];
        id += stride;
    }
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(allow_gradual_underflow), flag(clamp_range))]
pub fn mul_gradual_clamp_1<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] * right[id];
        id += stride;
    }
}
#[pcu(invocations = 1, crate_path = ::pcu_facade)]
pub fn div_ieee_reject_1<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] / right[id];
        id += stride;
    }
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(clamp_range))]
pub fn div_ieee_clamp_1<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] / right[id];
        id += stride;
    }
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result))]
pub fn div_tight_reject_1<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] / right[id];
        id += stride;
    }
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result), flag(clamp_range))]
pub fn div_tight_clamp_1<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] / right[id];
        id += stride;
    }
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(allow_gradual_underflow))]
pub fn div_gradual_reject_1<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] / right[id];
        id += stride;
    }
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(allow_gradual_underflow), flag(clamp_range))]
pub fn div_gradual_clamp_1<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] / right[id];
        id += stride;
    }
}
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn add_ieee_reject_2<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] + left[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(clamp_range))]
pub fn add_ieee_clamp_2<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] + left[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result))]
pub fn add_tight_reject_2<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] + left[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result), flag(clamp_range))]
pub fn add_tight_clamp_2<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] + left[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(allow_gradual_underflow))]
pub fn add_gradual_reject_2<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] + left[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(allow_gradual_underflow), flag(clamp_range))]
pub fn add_gradual_clamp_2<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] + left[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn sub_ieee_reject_2<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] - left[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(clamp_range))]
pub fn sub_ieee_clamp_2<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] - left[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result))]
pub fn sub_tight_reject_2<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] - left[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result), flag(clamp_range))]
pub fn sub_tight_clamp_2<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] - left[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(allow_gradual_underflow))]
pub fn sub_gradual_reject_2<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] - left[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(allow_gradual_underflow), flag(clamp_range))]
pub fn sub_gradual_clamp_2<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] - left[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn mul_ieee_reject_2<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] * left[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(clamp_range))]
pub fn mul_ieee_clamp_2<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] * left[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result))]
pub fn mul_tight_reject_2<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] * left[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result), flag(clamp_range))]
pub fn mul_tight_clamp_2<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] * left[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(allow_gradual_underflow))]
pub fn mul_gradual_reject_2<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] * left[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(allow_gradual_underflow), flag(clamp_range))]
pub fn mul_gradual_clamp_2<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] * left[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn div_ieee_reject_2<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] / left[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(clamp_range))]
pub fn div_ieee_clamp_2<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] / left[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result))]
pub fn div_tight_reject_2<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] / left[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result), flag(clamp_range))]
pub fn div_tight_clamp_2<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] / left[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(allow_gradual_underflow))]
pub fn div_gradual_reject_2<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] / left[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(allow_gradual_underflow), flag(clamp_range))]
pub fn div_gradual_clamp_2<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] / left[id];
}

#[pcu(crate_path = ::pcu_facade)]
pub fn retain<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
