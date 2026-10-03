//! Actual generic annotated unary workloads; all policy and load roles specialize cold.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedFloat,
    PcuScalar,
    PcuTensor,
    PcuExecutionError,
};
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn neg_ieeeafterrounding_reject_0<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
#[pcu(invocations = 1, crate_path = ::pcu_facade)]
pub fn neg_ieeeafterrounding_reject_1<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = -input[id];
        id += stride;
    }
}
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn neg_ieeeafterrounding_reject_2<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[0];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(clamp_range))]
pub fn neg_ieeeafterrounding_clamp_0<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(clamp_range))]
pub fn neg_ieeeafterrounding_clamp_1<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = -input[id];
        id += stride;
    }
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(clamp_range))]
pub fn neg_ieeeafterrounding_clamp_2<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[0];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(allow_gradual_underflow))]
pub fn neg_allowgradualunderflow_reject_0<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(allow_gradual_underflow))]
pub fn neg_allowgradualunderflow_reject_1<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = -input[id];
        id += stride;
    }
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(allow_gradual_underflow))]
pub fn neg_allowgradualunderflow_reject_2<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[0];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(allow_gradual_underflow), flag(clamp_range))]
pub fn neg_allowgradualunderflow_clamp_0<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(allow_gradual_underflow), flag(clamp_range))]
pub fn neg_allowgradualunderflow_clamp_1<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = -input[id];
        id += stride;
    }
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(allow_gradual_underflow), flag(clamp_range))]
pub fn neg_allowgradualunderflow_clamp_2<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[0];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result))]
pub fn neg_rejectsubnormalresult_reject_0<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result))]
pub fn neg_rejectsubnormalresult_reject_1<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = -input[id];
        id += stride;
    }
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result))]
pub fn neg_rejectsubnormalresult_reject_2<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[0];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result), flag(clamp_range))]
pub fn neg_rejectsubnormalresult_clamp_0<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result), flag(clamp_range))]
pub fn neg_rejectsubnormalresult_clamp_1<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = -input[id];
        id += stride;
    }
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result), flag(clamp_range))]
pub fn neg_rejectsubnormalresult_clamp_2<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[0];
}
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn relu_ieeeafterrounding_reject_0<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}
#[pcu(invocations = 1, crate_path = ::pcu_facade)]
pub fn relu_ieeeafterrounding_reject_1<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = pcu::relu(input[id]);
        id += stride;
    }
}
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn relu_ieeeafterrounding_reject_2<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[0]);
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(clamp_range))]
pub fn relu_ieeeafterrounding_clamp_0<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(clamp_range))]
pub fn relu_ieeeafterrounding_clamp_1<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = pcu::relu(input[id]);
        id += stride;
    }
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(clamp_range))]
pub fn relu_ieeeafterrounding_clamp_2<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[0]);
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(allow_gradual_underflow))]
pub fn relu_allowgradualunderflow_reject_0<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(allow_gradual_underflow))]
pub fn relu_allowgradualunderflow_reject_1<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = pcu::relu(input[id]);
        id += stride;
    }
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(allow_gradual_underflow))]
pub fn relu_allowgradualunderflow_reject_2<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[0]);
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(allow_gradual_underflow), flag(clamp_range))]
pub fn relu_allowgradualunderflow_clamp_0<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(allow_gradual_underflow), flag(clamp_range))]
pub fn relu_allowgradualunderflow_clamp_1<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = pcu::relu(input[id]);
        id += stride;
    }
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(allow_gradual_underflow), flag(clamp_range))]
pub fn relu_allowgradualunderflow_clamp_2<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[0]);
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result))]
pub fn relu_rejectsubnormalresult_reject_0<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result))]
pub fn relu_rejectsubnormalresult_reject_1<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = pcu::relu(input[id]);
        id += stride;
    }
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result))]
pub fn relu_rejectsubnormalresult_reject_2<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[0]);
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result), flag(clamp_range))]
pub fn relu_rejectsubnormalresult_clamp_0<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}
#[pcu(invocations = 1, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result), flag(clamp_range))]
pub fn relu_rejectsubnormalresult_clamp_1<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = pcu::relu(input[id]);
        id += stride;
    }
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result), flag(clamp_range))]
pub fn relu_rejectsubnormalresult_clamp_2<T: PcuCheckedFloat, const N: usize>(
    unread: &[T],
    output: &mut [T],
    scalar: &T,
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[0]);
}

#[pcu(crate_path = ::pcu_facade)]
pub fn retain<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
