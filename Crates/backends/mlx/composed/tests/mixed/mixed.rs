//! Actual immutable initial inputs only; private siblings remain distinct from facade publication.
#[rustfmt::skip]
use crate::{
    MlxCheckedMapInput,
    MlxRuntime,
    MlxSession,
};
use super::source;
#[rustfmt::skip]
use fusion_pcu::{
    PcuDispatchKernelIr,
    PcuHostKernelBackend,
    PcuScalar,
};
fn bytes<T: PcuScalar>(values: &[T]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.encode_le().as_ref().to_vec())
        .collect()
}
fn verify<T: PcuScalar>(
    session: &MlxSession,
    ir: &PcuDispatchKernelIr<'_>,
    input: &[T; 7],
    seed: T,
    sentinel: T,
    expected: &[T; 7],
) {
    let mut kernel = session
        .composed_host_backend()
        .prepare_host_kernel(ir)
        .unwrap();
    assert_eq!(kernel.argument_count(), 5);
    assert_eq!(kernel.input_bindings().len(), 2);
    assert_eq!(kernel.prepared_input_element_counts(), &[7, 1]);
    let targets: [_; 2] = core::array::from_fn(|slot| kernel.input_bindings()[slot]);
    let input_bytes = bytes(input);
    let mut host_input = input_bytes.clone();
    host_input.extend_from_slice(&bytes(&[sentinel; 3]));
    let seed_bytes = bytes(&[seed]);
    let input_owner = session
        .upload_transport_bytes(T::TYPE, 7, &input_bytes)
        .unwrap();
    let seed_owner = session
        .upload_transport_bytes(T::TYPE, 1, &seed_bytes)
        .unwrap();
    for mode in 0..3 {
        let input = if mode == 1 {
            MlxCheckedMapInput::HostBytes {
                target: targets[0],
                scalar: T::TYPE,
                bytes: &host_input,
            }
        } else {
            MlxCheckedMapInput::Resident {
                target: targets[0],
                array: &input_owner,
            }
        };
        let seed = if mode == 2 {
            MlxCheckedMapInput::HostBytes {
                target: targets[1],
                scalar: T::TYPE,
                bytes: &seed_bytes,
            }
        } else {
            MlxCheckedMapInput::Resident {
                target: targets[1],
                array: &seed_owner,
            }
        };
        let completed = kernel.execute_inputs(&[seed, input]).unwrap();
        let (outputs, notice) = completed.into_outputs();
        assert!(notice.is_none());
        assert!(!kernel.last_call_may_have_written());
        assert!(!kernel.last_call_completion_uncertain());
        let [Some(stage), Some(output)] = outputs else {
            panic!("missing actual private siblings")
        };
        let width = usize::from(T::TYPE.bit_width()) / 8;
        let mut stage_host = bytes(&[sentinel; 10]);
        let mut output_host = bytes(&[sentinel; 11]);
        stage.read_bytes_into(&mut stage_host[..7 * width]).unwrap();
        output
            .read_bytes_into(&mut output_host[..7 * width])
            .unwrap();
        assert_eq!(&stage_host[..7 * width], &input_bytes);
        assert_eq!(&stage_host[7 * width..], &bytes(&[sentinel; 3]));
        assert_eq!(&output_host[..7 * width], &bytes(expected));
        assert_eq!(&output_host[7 * width..], &bytes(&[sentinel; 4]));
        assert_eq!(stage.element_count(), 7);
        assert_eq!(output.element_count(), 7);
        let mut original = vec![0; input_bytes.len()];
        input_owner.read_bytes_into(&mut original).unwrap();
        assert_eq!(original, input_bytes);
        stage.release().unwrap();
        output.release().unwrap();
    }
    input_owner.release().unwrap();
    seed_owner.release().unwrap();
}
#[test]
#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    ignore = "requires real MLX actual mixed inputs and terminal private siblings"
)]
fn ten_integer_two_float_actual_unique_inputs_mixed_directions_saved_values_and_private_outputs() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    macro_rules! integer { ($($ty:ty),*) => { $(
        for phase in [0_u8, 1, 2] {
            let input: [$ty; 7] = core::array::from_fn(|index|
                <$ty>::try_from(1_u8 + u8::try_from(index).unwrap() + phase).unwrap());
            let expected: [$ty; 7] = core::array::from_fn(|lane|
                (input[lane] + 2) * input[lane] - input[lane]);
            source::integer_ir::<$ty, 7>(&source::integer_bindings::<$ty>()).unwrap().with_ir(|ir|
                verify(&session, ir, &input, 2, 19, &expected));
        }
    )* }; }
    integer!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);
    macro_rules! floating { ($($ty:ty),*) => { $(
        for phase in [0_u8, 1, 2] {
            let input: [$ty; 7] = core::array::from_fn(|index|
                <$ty>::from(1_u8 + u8::try_from(index).unwrap() + phase));
            let expected: [$ty; 7] = core::array::from_fn(|lane| {
                let value = u16::from(1_u8 + u8::try_from(lane).unwrap() + phase);
                <$ty>::from((value + 2) * value + value)
            });
            source::floating_ir::<$ty, 7>(&source::floating_bindings::<$ty>()).unwrap().with_ir(|ir|
                verify(&session, ir, &input, 2.0, 19.0, &expected));
        }
    )* }; }
    floating!(f32, f64);
}
#[path = "fault/fault.rs"]
mod fault;
