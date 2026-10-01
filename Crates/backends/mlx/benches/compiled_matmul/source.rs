#[rustfmt::skip]
use pcu_facade::{
    global,
    pcu,
    PcuExecutionError,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuTensor,
};

#[pcu(crate_path = pcu_facade, flag(native_compound), flag(backend_precision))]
fn matrix_source<const N: usize>(
    left: &[[f32; N]; N],
    right: &[[f32; N]; N],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::matmul(left, right)
}

pub fn capture<const N: usize>() -> global::PcuCapturedTensorProgram {
    global::__pcu_capture_tensor_program(
        [global::PcuSourceShape::FixedMatrix {
            rows: N,
            columns: N,
        }; 2],
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuNumericalMode::Boundary,
        PcuNumericalOptions::default(),
        matrix_source::__pcu_capture_entry::<N>,
    )
    .unwrap()
}
