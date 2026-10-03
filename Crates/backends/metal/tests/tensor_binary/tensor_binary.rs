//! Actual source20-type binary owner lifetime and declaration/SSA role integration.
#[path = "../../benches/tensor_binary/support/sample/sample.rs"]
mod sample;
#[path = "source/source.rs"]
mod source;
use sample::Sample;
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuTensor,
    PcuExecutionError,
    PcuArgumentError,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuReproducibility,
    PcuU256,
    PcuI256,
    PcuU512,
    PcuI512,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
fn read<T: Sample>(owner: &PcuTensor<T>, expected: u8) {
    let mut actual = [T::literal(6); 9];
    owner.read_into(&mut actual).unwrap();
    for value in &actual[..7] {
        assert_eq!(
            value.encode_le().as_ref(),
            T::literal(expected).encode_le().as_ref()
        );
    }
    for value in &actual[7..] {
        assert_eq!(
            value.encode_le().as_ref(),
            T::literal(6).encode_le().as_ref()
        );
    }
}
fn case<T: Sample>() {
    for (a, b, c) in [(1, 2, 3), (2, 2, 4), (3, 3, 6)] {
        let mut left = [T::literal(a); 7];
        let mut right = [T::literal(b); 7];
        let lhs = source::retain(&left).unwrap();
        let rhs = source::retain(&right).unwrap();
        let host = source::add(&left, &right).unwrap();
        let mixed = source::add::<T>(&lhs, &right).unwrap();
        let resident = source::add::<T>(&lhs, &rhs).unwrap();
        read(&host, c);
        read(&mixed, c);
        read(&resident, c);
        read(&source::unused::<T>(&lhs, &rhs).unwrap(), b);
        let difference = b - a;
        read(&source::sub::<T>(&lhs, &rhs).unwrap(), difference);
        // A parameter1-only square uses exactly one selected input, even with empty param0.
        read(
            &source::right_square::<T>(&[], &source::retain(&[T::literal(2); 7]).unwrap()).unwrap(),
            4,
        );
        left.fill(T::literal(6));
        right.fill(T::literal(6));
        read(&lhs, a);
        read(&rhs, b);
        drop(mixed);
        drop(resident);
        drop(lhs);
        drop(rhs);
        global::clear_thread_cache().unwrap();
        read(&source::consume(host).unwrap(), c);
    }
    let lhs = source::retain(&[T::literal(2); 7]).unwrap();
    let rhs = source::retain(&[T::literal(3); 7]).unwrap();
    read(&source::mul::<T>(&lhs, &rhs).unwrap(), 6);
    if T::TYPE.binary_float_format().is_some() {
        read(
            &source::div::<T>(&lhs, &source::retain(&[T::literal(6); 7]).unwrap()).unwrap(),
            3,
        );
    }
}
fn width<T: Sample>() {
    for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound_arithmetic in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for float_underflow in [
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                ] {
                    global::configure(global::PcuExecutionPolicy {
                        backend: global::PcuBackendChoice::Metal,
                        numerical_mode,
                        float_underflow,
                        numerical_options: PcuNumericalOptions {
                            compound_arithmetic,
                            precision,
                            reproducibility: PcuReproducibility::Unspecified,
                        },
                        ..Default::default()
                    })
                    .unwrap();
                    case::<T>();
                }
            }
        }
    }
}
fn open_foreign() -> PcuTensor<f32> {
    #[rustfmt::skip]
    use pcu_facade::{
        PcuRuntimeDiscovery,
        PcuProviderDescriptor,
        PcuProviderId,
        PcuProviderReadiness,
        PcuProviderStatus,
        PcuObjectRef,
        PcuObjectKind,
        PcuMemoryPoolId,
    };
    let inventory = fusion_pcu_metal::MetalDiscovery::discover().unwrap();
    let mut providers = [PcuProviderDescriptor {
        id: PcuProviderId(0),
        generation: 0,
        backend: "",
        readiness: PcuProviderReadiness {
            status: PcuProviderStatus::Unavailable,
            reason: None,
        },
    }];
    inventory.providers(&mut providers).unwrap();
    let backend = inventory
        .open_owned_device(PcuObjectRef {
            provider: providers[0].id,
            generation: providers[0].generation,
            kind: PcuObjectKind::Device,
            id: 0,
        })
        .unwrap();
    let buffer = backend
        .upload_buffer(PcuMemoryPoolId(147), &[1.0_f32; 7])
        .unwrap();
    PcuTensor::from_device_buffer(backend, buffer, &[7]).unwrap()
}
fn unread_foreign_and_discarded_owner() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Metal,
        ..Default::default()
    })
    .unwrap();
    let active = source::retain(&[2.0_f32; 7]).unwrap();
    let mut foreign = open_foreign();
    read(&source::right_square::<f32>(&foreign, &active).unwrap(), 4);
    assert!(matches!(
        source::add::<f32>(&foreign, &active),
        Err(PcuExecutionError::Argument(
            PcuArgumentError::SessionMismatch
        ))
    ));
    assert!(source::invalidate(&[f32::NAN; 7], &mut foreign).is_err());
    let mut unchanged = [91.0; 9];
    assert!(matches!(
        foreign.read_into(&mut unchanged),
        Err(PcuExecutionError::Argument(
            PcuArgumentError::ResidentValueDiscarded
        ))
    ));
    assert_eq!(unchanged.map(f32::to_bits), [91.0_f32; 9].map(f32::to_bits));
    read(&source::right_square::<f32>(&foreign, &active).unwrap(), 4);
    read(&active, 2);
}
#[test]
#[ignore = "Requires actual Metal ordinary20-type binary ownership and unread foreign/discarded owner law."]
fn ordinary_twenty_type_binary_owners_and_actual_input_roles() {
    macro_rules! widths{($($ty:ty),+)=>{$(width::<$ty>();)+};}
    widths!(
        u8,
        i8,
        u16,
        i16,
        u32,
        i32,
        u64,
        i64,
        u128,
        i128,
        PcuU256,
        PcuI256,
        PcuU512,
        PcuI512,
        PcuF16Bits,
        PcuBf16Bits,
        PcuF8E4M3FnBits,
        PcuF8E5M2Bits,
        f32,
        f64
    );
    unread_foreign_and_discarded_owner();
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
