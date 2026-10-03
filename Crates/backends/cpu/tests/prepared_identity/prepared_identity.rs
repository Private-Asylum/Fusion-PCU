//! Exact generic source identity, all admitted representations and structural failure rollback.
use core::fmt::Debug;
#[rustfmt::skip]
use fusion_pcu_cpu::{PcuCpuIdentity, PcuCpuHostBackend, PcuCpuHostOffers, PcuCpuHostError};
#[rustfmt::skip]
use pcu_facade::{
    PcuBf16Bits, PcuF16Bits, PcuF128Bits, PcuF256Bits,
    PcuF8E4M3FnBits, PcuF8E5M2Bits,
    PcuU256, PcuI256, PcuU512, PcuI512,
    PcuBindingRef, PcuHostArgument, PcuHostKernelBackend, PcuPreparedHostKernel,
    PcuDispatchOp, PcuDispatchControlOp, PcuDispatchIndex, PcuDispatchDataOp,
    PcuScalar, PcuScalarType,
    PcuCostBoundary, PcuDeviceIdentity, PcuExecutorId, PcuProviderId,
    PcuObjectRef, PcuObjectKind, PcuImplementationOffers,
    PcuImplementationRequirements, PcuImplementationRequest,
    PcuNumericalMode, PcuReproducibility,
};
#[path = "source/source.rs"]
mod source;

fn check<T: PcuScalar + Debug>(values: [T; 5]) {
    let backend = PcuCpuHostBackend::scalar();
    let mut call = source::copy_prepare::<T, 5, _>(&backend).unwrap();
    let mut output = [values[0]; 7];
    call(&values, &mut output).unwrap();
    for index in 0..5 {
        assert_eq!(
            output[index].encode_le().as_ref(),
            values[index].encode_le().as_ref()
        );
    }
    assert_eq!(
        output[5].encode_le().as_ref(),
        values[0].encode_le().as_ref()
    );
    let before = output;
    assert!(call(&values[..4], &mut output).is_err());
    for index in 0..7 {
        assert_eq!(
            output[index].encode_le().as_ref(),
            before[index].encode_le().as_ref()
        );
    }
    let bindings = source::copy_bindings::<T>();
    let builder = source::copy_ir::<T, 5>(&bindings).unwrap();
    let kernel = builder.ir();
    let identity = PcuDeviceIdentity::from_device_ref(PcuObjectRef {
        provider: PcuProviderId(3),
        generation: 7,
        kind: PcuObjectKind::Device,
        id: 0,
    })
    .unwrap();
    let offers = PcuCpuHostOffers::new(backend, identity, PcuExecutorId(0));
    let mut request = PcuImplementationRequest {
        device: identity,
        executor: PcuExecutorId(0),
        operation: &kernel,
        requirements: PcuImplementationRequirements::default(),
        boundary: PcuCostBoundary::Host,
    };
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        request.requirements.numerical_mode = mode;
        let kernel = pcu_facade::PcuDispatchKernelIr {
            numerical_requirements: request.requirements,
            ..*request.operation
        };
        let mut request = PcuImplementationRequest {
            operation: &kernel,
            ..request
        };
        let mut output = [None];
        assert_eq!(offers.implementation_offers(&request, &mut output), Ok(1));
        assert_eq!(
            output[0].unwrap().implementation.local_id,
            64 + T::TYPE as u32
        );
        assert_eq!(output[0].unwrap().workspace_bytes, Some(0));
        request.requirements.numerical_options.reproducibility = PcuReproducibility::PortableV1;
        assert_eq!(offers.implementation_offers(&request, &mut output), Ok(0));
    }
}
macro_rules! integer {
    ($name:ident,$ty:ty) => {
        #[test]
        fn $name() {
            check::<$ty>([0, 1, <$ty>::MIN, <$ty>::MAX, 7]);
        }
    };
}
integer!(i8_copy, i8);
integer!(u8_copy, u8);
integer!(i16_copy, i16);
integer!(u16_copy, u16);
integer!(i32_copy, i32);
integer!(u32_copy, u32);
integer!(i64_copy, i64);
integer!(u64_copy, u64);
integer!(i128_copy, i128);
integer!(u128_copy, u128);
#[test]
fn f32_copy() {
    check([
        0.0_f32,
        -0.0,
        f32::from_bits(0x7f80_0001),
        f32::INFINITY,
        f32::from_bits(1),
    ]);
}
#[test]
fn f64_copy() {
    check([
        0.0_f64,
        -0.0,
        f64::from_bits(0x7ff0_0000_0000_0001),
        f64::INFINITY,
        f64::from_bits(1),
    ]);
}
#[test]
fn f16_copy() {
    check([0, 0x8000, 0x7c01, 0x7c00, 1].map(PcuF16Bits::from_bits));
}
#[test]
fn bf16_copy() {
    check([0, 0x8000, 0x7f81, 0x7f80, 1].map(PcuBf16Bits::from_bits));
}
macro_rules! wide {
    ($name:ident,$ty:ty,$n:expr) => {
        #[test]
        fn $name() {
            check(
                [
                    [0; $n],
                    [u64::MAX; $n],
                    [1; $n],
                    [1 << 63; $n],
                    [0x0123_4567_89ab_cdef; $n],
                ]
                .map(<$ty>::from_limbs_le),
            );
        }
    };
}
wide!(u256_copy, PcuU256, 4);
wide!(i256_copy, PcuI256, 4);
wide!(u512_copy, PcuU512, 8);
wide!(i512_copy, PcuI512, 8);
wide!(f128_bits_copy, PcuF128Bits, 2);
wide!(f256_bits_copy, PcuF256Bits, 4);

#[test]
fn detached_grid_and_shuffled_arguments_copy_exact_prefix() {
    let mut prepared = {
        let bindings = source::copy_bindings::<u128>();
        let builder = source::copy_ir::<u128, 5>(&bindings).unwrap();
        let kernel = builder.ir();
        let mut body = kernel.ops[..2].to_vec();
        for op in &mut body {
            if let PcuDispatchOp::Data(
                PcuDispatchDataOp::BindingLoad { index, .. }
                | PcuDispatchDataOp::BindingStore { index, .. },
            ) = op
            {
                *index = PcuDispatchIndex::GridStrideId;
            }
        }
        let ops = [
            PcuDispatchOp::GridStrideLoop {
                extent: 5,
                body: &body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let kernel = pcu_facade::PcuDispatchKernelIr {
            ops: &ops,
            entry: pcu_facade::PcuDispatchEntryPoint {
                logical_shape: [2, 1, 1],
                ..kernel.entry
            },
            ..kernel
        };
        PcuCpuIdentity.prepare_host_kernel(&kernel).unwrap()
    };
    let values = [0_u128, 1, u128::MAX, 1 << 100, 1 << 127];
    let mut output = [99_u128; 7];
    prepared
        .call(&mut [
            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &values),
        ])
        .unwrap();
    assert_eq!(output, [0, 1, u128::MAX, 1 << 100, 1 << 127, 99, 99]);
    assert!(
        prepared
            .call(&mut [
                PcuHostArgument::read(PcuBindingRef::new(0, 0), &values),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut [99_u64; 7])
            ])
            .is_err()
    );
    assert_eq!(output, [0, 1, u128::MAX, 1 << 100, 1 << 127, 99, 99]);
}
#[test]
fn subbyte_encodings_remain_explicitly_unsupported() {
    let bindings = source::copy_bindings::<u8>();
    let builder = source::copy_ir::<u8, 5>(&bindings).unwrap();
    let kernel = builder.ir();
    for scalar in [PcuScalarType::Bool, PcuScalarType::I4, PcuScalarType::U4] {
        let mut bindings = bindings;
        for binding in &mut bindings {
            binding.binding_type =
                pcu_facade::PcuBindingType::Value(pcu_facade::PcuValueType::Scalar(scalar));
        }
        let kernel = pcu_facade::PcuDispatchKernelIr {
            bindings: &bindings,
            type_caps: pcu_facade::PcuValueTypeCaps::for_scalar(scalar),
            ..kernel
        };
        assert_eq!(
            PcuCpuIdentity.prepare_host_kernel(&kernel).unwrap_err(),
            PcuCpuHostError::UnsupportedProfile
        );
    }
}

#[test]
fn fp8_e4m3fn_copy() {
    check([0, 0x80, 0x7f, 0xff, 1].map(PcuF8E4M3FnBits::from_bits));
}
#[test]
fn fp8_e5m2_copy() {
    check([0, 0x80, 0x7d, 0x7c, 1].map(PcuF8E5M2Bits::from_bits));
}
