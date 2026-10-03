//! Actual captured source and independent unrewritten graph programs.
#[rustfmt::skip]
use std::sync::Arc;
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuCheckedInteger,
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
#[path = "../../../tests/tensor_integer/source/source.rs"]
pub mod source;
pub fn capture<T: PcuCheckedInteger>(
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
            _ => unreachable!(),
        },
    )
    .unwrap();
    Arc::clone(captured.program())
}
pub fn graph<T: PcuCheckedInteger>(
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
        _ => unreachable!(),
    }
    .unwrap();
    Arc::new(
        graph
            .into_selected_program(
                &[effect],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .unwrap(),
    )
}
