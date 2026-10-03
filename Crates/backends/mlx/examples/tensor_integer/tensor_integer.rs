//! Captured annotated source with bounded authentic fourteen-width MLX-owned integer output.
#[rustfmt::skip]
use fusion_pcu_mlx::MlxRuntime;
#[path = "../../tests/tensor_integer/source/source.rs"]
mod source;
#[rustfmt::skip]
use fusion_pcu_mlx::MlxCheckedProgramInput;
#[rustfmt::skip]
use pcu_facade::{
    PcuHostArgument,
    PcuBindingRef,
    PcuScalarType,
    PcuU512,
    PcuImplementationRequirements,
    global,
};
const fn small(value: u64) -> PcuU512 {
    let mut limbs = [0; 8];
    limbs[0] = value;
    PcuU512::from_limbs_le(limbs)
}
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
        source::add::__pcu_capture_entry::<PcuU512>,
    )
    .unwrap();
    let prepared = session
        .prepare_tensor_integer_program(
            std::sync::Arc::clone(captured.program()),
            PcuImplementationRequirements::default(),
        )
        .unwrap();
    let left_values = [small(2); 5];
    let right_values = [small(4); 5];
    let left = PcuHostArgument::read(PcuBindingRef::new(0, 0), &left_values);
    let right = PcuHostArgument::read(PcuBindingRef::new(0, 1), &right_values);
    let ids = prepared.plan().input_values();
    let output = prepared
        .execute_mixed(&[
            (
                ids[0],
                MlxCheckedProgramInput::Host {
                    scalar: PcuScalarType::U512,
                    bytes: left.bytes(),
                },
            ),
            (
                ids[1],
                MlxCheckedProgramInput::Host {
                    scalar: PcuScalarType::U512,
                    bytes: right.bytes(),
                },
            ),
        ])
        .unwrap();
    drop(prepared);
    drop(session);
    let mut observed = [small(91); 7];
    output.read_into(&mut observed).unwrap();
    assert_eq!(observed, [6, 6, 6, 6, 6, 91, 91].map(small));
    println!("Escaped checked MLX U512 owner: {observed:?}");
}
