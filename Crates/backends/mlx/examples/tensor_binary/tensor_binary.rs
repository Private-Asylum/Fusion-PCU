//! Captured annotated source with bounded authentic MLX-owned binary output.
#[rustfmt::skip]
use fusion_pcu_mlx::MlxRuntime;
#[path = "../../tests/tensor_binary/source/source.rs"]
mod source;
#[rustfmt::skip]
use fusion_pcu_mlx::MlxCheckedProgramInput;
#[rustfmt::skip]
use pcu_facade::{
    PcuHostArgument,
    PcuBindingRef,
    PcuScalarType,
    PcuImplementationRequirements,
    global,
};
fn main() {
    if !cfg!(target_os = "macos") {
        println!("SKIP: pinned MLX Apple execution unavailable");
        return;
    }
    let session = MlxRuntime::load_default().unwrap().open_gpu(0).unwrap();
    let captured = global::__pcu_capture_tensor_program(
        [global::PcuSourceShape::Slice { length: 5 }; 2],
        PcuImplementationRequirements::default().float_underflow,
        PcuImplementationRequirements::default().numerical_mode,
        PcuImplementationRequirements::default().numerical_options,
        source::add::__pcu_capture_entry::<f64>,
    )
    .unwrap();
    let prepared = session
        .prepare_tensor_binary_program(
            std::sync::Arc::clone(captured.program()),
            PcuImplementationRequirements::default(),
        )
        .unwrap();
    let left = PcuHostArgument::read(PcuBindingRef::new(0, 0), &[2.0_f64; 5]);
    let right = PcuHostArgument::read(PcuBindingRef::new(0, 1), &[4.0_f64; 5]);
    let ids = prepared.plan().input_values();
    let output = prepared
        .execute_mixed(&[
            (
                ids[0],
                MlxCheckedProgramInput::Host {
                    scalar: PcuScalarType::F64,
                    bytes: left.bytes(),
                },
            ),
            (
                ids[1],
                MlxCheckedProgramInput::Host {
                    scalar: PcuScalarType::F64,
                    bytes: right.bytes(),
                },
            ),
        ])
        .unwrap();
    drop(prepared);
    drop(session);
    let mut observed = [91.0_f64; 7];
    output.read_into(&mut observed).unwrap();
    assert_eq!(
        observed.map(f64::to_bits),
        [6.0, 6.0, 6.0, 6.0, 6.0, 91.0, 91.0].map(f64::to_bits)
    );
    println!("Escaped checked MLX F64 owner: {observed:?}");
}
