//! Actual captured source and independent unrewritten graph programs.
#[rustfmt::skip]
use std::sync::Arc;
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuCheckedFloat,
    PcuImplementationRequirements,
};
#[rustfmt::skip]
use pcu_facade::dialect::tensor::{
    Graph,
    TensorOwnedSelectedProgram,
    TensorArithmeticRewritePolicy,
    TensorArithmeticCapability,
    TensorPointwiseGroupingPolicy,
};
#[path = "../source/source.rs"]
mod source;
pub fn capture<T: PcuCheckedFloat>(
    profile: u8,
    count: usize,
    request: PcuImplementationRequirements,
) -> Arc<TensorOwnedSelectedProgram> {
    let captured = global::__pcu_capture_tensor_program::<T, 2, _>(
        [global::PcuSourceShape::Slice { length: count }; 2],
        request.float_underflow,
        request.numerical_mode,
        request.numerical_options,
        |capture, inputs| match profile {
            0 => source::add::__pcu_capture_entry::<T>(capture, inputs),
            1 => source::sub::__pcu_capture_entry::<T>(capture, inputs),
            2 => source::mul::__pcu_capture_entry::<T>(capture, inputs),
            3 => source::div::__pcu_capture_entry::<T>(capture, inputs),
            4 => source::unused::__pcu_capture_entry::<T>(capture, inputs),
            _ => source::repeat::__pcu_capture_entry::<T>(capture, inputs),
        },
    )
    .unwrap();
    Arc::clone(captured.program())
}
pub fn graph<T: PcuCheckedFloat>(
    profile: u8,
    count: usize,
    request: PcuImplementationRequirements,
) -> Arc<TensorOwnedSelectedProgram> {
    let mut graph = Graph::try_new().unwrap();
    let left = graph.input([count], T::TYPE).unwrap();
    let right = graph.input([count], T::TYPE).unwrap();
    graph.set_numerical_options(request.numerical_options);
    let effect = match profile {
        0 => graph.add(left, right),
        1 => graph.sub(right, left),
        2 => graph.mul(left, right),
        3 | 4 => graph.div(right, left),
        _ => graph.mul(right, right),
    }
    .unwrap();
    graph
        .set_value_float_underflow_policy(effect, request.float_underflow)
        .unwrap();
    Arc::new(
        graph
            .into_selected_program(
                &[if profile == 4 { right } else { effect }],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .unwrap(),
    )
}
