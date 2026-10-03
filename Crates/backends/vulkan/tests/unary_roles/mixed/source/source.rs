//! Ordinary unary functions keep declaration order separate from actual resources.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedFloat,
};

pub(super) fn verify<T: PcuCheckedFloat>(request: pcu_facade::PcuImplementationRequirements) {
    macro_rules! profile {
        ($bindings:ident, $ir:ident) => {
            $ir::<T>(
                &$bindings::<T>(),
                request.float_underflow,
                request.range_policy,
                request,
            )
            .unwrap()
            .with_ir(|kernel| {
                assert_eq!(kernel.numerical_requirements, request);
                let description = pcu_facade::describe_checked_float_unary_map(kernel).unwrap();
                assert_eq!(description.requirements, request);
                assert_eq!(
                    description.input_binding,
                    pcu_facade::PcuBindingRef::new(0, 2)
                );
                assert_eq!(
                    description.output_binding,
                    pcu_facade::PcuBindingRef::new(0, 1)
                );
            });
        };
    }
    profile!(
        direct_neg_bindings,
        __direct_neg_ir_with_float_underflow_policy
    );
    profile!(
        direct_relu_bindings,
        __direct_relu_ir_with_float_underflow_policy
    );
    profile!(grid_neg_bindings, __grid_neg_ir_with_float_underflow_policy);
    profile!(
        grid_relu_bindings,
        __grid_relu_ir_with_float_underflow_policy
    );
    profile!(
        broadcast_neg_bindings,
        __broadcast_neg_ir_with_float_underflow_policy
    );
    profile!(
        broadcast_relu_bindings,
        __broadcast_relu_ir_with_float_underflow_policy
    );
}

#[pcu(crate_path=::pcu_facade, invocations = 7)]
pub(super) fn direct_neg<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

#[pcu(crate_path=::pcu_facade, invocations = 7)]
pub(super) fn direct_relu<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}

#[pcu(crate_path=::pcu_facade, invocations = 3)]
pub(super) fn grid_neg<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < 7 {
        output[id] = -input[id];
        id += stride;
    }
}

#[pcu(crate_path=::pcu_facade, invocations = 3)]
pub(super) fn grid_relu<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < 7 {
        output[id] = pcu::relu(input[id]);
        id += stride;
    }
}

#[pcu(crate_path=::pcu_facade, invocations = 7)]
pub(super) fn broadcast_neg<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[0];
}

#[pcu(crate_path=::pcu_facade, invocations = 7)]
pub(super) fn broadcast_relu<T: PcuCheckedFloat>(
    unread: &[T],
    output: &mut [T],
    input: &[T],
    unread_scalar: &T,
) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[0]);
}
