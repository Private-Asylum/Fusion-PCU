//! Actual MLX binary source/graph/native ownership and transactional publication qualification.
#[path = "graph/graph.rs"]
mod graph;
#[path = "offers/offers.rs"]
mod offers;
#[path = "prefix/prefix.rs"]
mod prefix;
#[path = "source/source.rs"]
mod source;
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxRuntime,
    MlxError,
    MlxCheckedBinaryPlan,
    MlxBinaryInput,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuHostArgument,
    PcuBindingRef,
    PcuHostDispatchError,
    PcuCheckedFloat,
    PcuDispatchFloatBinaryOp as Op,
    PcuFloatUnderflowPolicy as Policy,
    PcuRangePolicy as Range,
    PcuReproducibility,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
trait Sample: PcuCheckedFloat + PartialEq + core::fmt::Debug {
    const MAX: u64;
    const SIGN: u64;
    fn from_bits(bits: u64) -> Self;
    fn value(value: f32) -> Self;
}
macro_rules! sample {
    ($ty:ty,$word:ty,$max:expr,$sign:expr,$convert:expr) => {
        impl Sample for $ty {
            const MAX: u64 = $max;
            const SIGN: u64 = $sign;
            fn from_bits(bits: u64) -> Self {
                Self::from_bits(<$word>::try_from(bits).unwrap())
            }
            fn value(value: f32) -> Self {
                ($convert)(value)
            }
        }
    };
}
sample!(PcuF16Bits, u16, 0x7bff, 0x8000, |value| {
    PcuF16Bits::pcu_checked_from_f32(value).unwrap()
});
sample!(PcuBf16Bits, u16, 0x7f7f, 0x8000, |value| {
    PcuBf16Bits::pcu_checked_from_f32(value).unwrap()
});
sample!(PcuF8E4M3FnBits, u8, 0x7e, 0x80, |value| {
    PcuF8E4M3FnBits::pcu_checked_from_f32(value).unwrap()
});
sample!(PcuF8E5M2Bits, u8, 0x7b, 0x80, |value| {
    PcuF8E5M2Bits::pcu_checked_from_f32(value).unwrap()
});
sample!(f32, u32, 0x7f7f_ffff, 0x8000_0000, |value| value);
sample!(
    f64,
    u64,
    0x7fef_ffff_ffff_ffff,
    0x8000_0000_0000_0000,
    f64::from
);
fn source_calls<T: Sample>(session: &fusion_pcu_mlx::MlxSession) {
    let backend = session.checked_binary_backend();
    let left = [T::value(2.0); 3];
    let right = [T::value(4.0); 3];
    let sentinel = T::from_bits(T::MAX);
    let mut out = [sentinel; 5];
    source::add_prepare::<T, 3, _>(&backend).unwrap()(&left, &right, &mut out).unwrap();
    assert_eq!(
        out,
        [
            T::value(6.0),
            T::value(6.0),
            T::value(6.0),
            sentinel,
            sentinel
        ]
    );
    source::sub_prepare::<T, 3, _>(&backend).unwrap()(&left, &right, &mut out).unwrap();
    assert_eq!(&out[..3], &[T::value(-2.0); 3]);
    source::mul_prepare::<T, 3, _>(&backend).unwrap()(&left, &right, &mut out).unwrap();
    assert_eq!(&out[..3], &[T::value(8.0); 3]);
    source::div_prepare::<T, 3, _>(&backend).unwrap()(&left, &right, &mut out).unwrap();
    assert_eq!(&out[..3], &[T::value(0.5); 3]);
    source::swapped_prepare::<T, 3, _>(&backend).unwrap()(&mut out, &right, &left).unwrap();
    assert_eq!(&out[..3], &[T::value(-2.0); 3]);
    source::repeated_prepare::<T, 3, _>(&backend).unwrap()(&left, &[], &mut out).unwrap();
    assert_eq!(&out[..3], &[T::value(4.0); 3]);
    source::broadcast_prepare::<T, 3, _>(&backend).unwrap()(&left[0], &right, &mut out).unwrap();
    assert_eq!(&out[..3], &[T::value(6.0); 3]);
    source::grid_prepare::<T, 3, _>(&backend).unwrap()(&left, &right, &mut out).unwrap();
    assert_eq!(&out[..3], &[T::value(0.5); 3]);
    let tiny = [T::from_bits(1), T::from_bits(T::MAX), left[0]];
    let rhs = [T::value(0.0), T::from_bits(T::MAX), right[0]];
    let mut clamp = source::add_tight_clamp_prepare::<T, 3, _>(&backend).unwrap();
    assert!(
        matches!(clamp(&tiny,&rhs,&mut out),Err(PcuHostDispatchError::Backend(MlxError::Arithmetic(f)))
        if f.recovered&&f.invocation_id==0)
    );
    assert_eq!(&out[..3], &[tiny[0], tiny[1], T::value(6.0)]);
    let before = out;
    let fatal = [tiny[0], T::from_bits(T::SIGN - 1), left[0]];
    assert!(
        matches!(clamp(&fatal,&rhs,&mut out),Err(PcuHostDispatchError::Backend(MlxError::Arithmetic(f)))
        if !f.recovered&&f.invocation_id==1)
    );
    assert_eq!(out, before);
    clamp(&left, &right, &mut out).unwrap();
    assert_eq!(&out[3..], &[sentinel; 2]);
}
fn qualify<T: Sample>() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let other = runtime.open_gpu(0).unwrap();
    source_calls::<T>(&session);
    let a = [T::value(2.0); 3];
    let b = [T::value(4.0); 3];
    let left = session.upload_encoded(&a).unwrap();
    let right = session.upload_encoded(&b).unwrap();
    let foreign = other.upload_encoded(&b).unwrap();
    for op in [Op::Add, Op::Sub, Op::Mul, Op::Div] {
        let expected = match op {
            Op::Add => T::value(6.0),
            Op::Sub => T::value(-2.0),
            Op::Mul => T::value(8.0),
            Op::Div => T::value(0.5),
        };
        for policy in [
            Policy::IeeeAfterRounding,
            Policy::RejectSubnormalResult,
            Policy::AllowGradualUnderflow,
        ] {
            for range in [Range::Reject, Range::Clamp] {
                for grid in [false, true] {
                    let backend = session.checked_binary_backend();
                    let mut prepared = graph::fixture_profile::<T, _>(
                        3,
                        op,
                        policy,
                        range,
                        grid,
                        [false; 2],
                        |ir| backend.prepare_host_kernel(ir),
                    )
                    .unwrap();
                    let mut out = [T::from_bits(17); 5];
                    prepared
                        .call(&mut [
                            PcuHostArgument::read_write(PcuBindingRef::new(4, 1), &mut out),
                            PcuHostArgument::read(PcuBindingRef::new(3, 2), &b),
                            PcuHostArgument::read(PcuBindingRef::new(2, 3), &a),
                        ])
                        .unwrap();
                    assert_eq!(&out[..3], &[expected; 3]);
                    assert_eq!(&out[3..], &[T::from_bits(17); 2]);
                    let completed = prepared.execute_resident(&[&left, &right]).unwrap();
                    let (old, _) = completed.into_parts();
                    let host = PcuHostArgument::read(PcuBindingRef::new(2, 3), &a);
                    let mixed = prepared
                        .execute_inputs(&[
                            MlxBinaryInput::Resident {
                                target: PcuBindingRef::new(3, 2),
                                array: &right,
                            },
                            MlxBinaryInput::HostBytes {
                                target: PcuBindingRef::new(2, 3),
                                scalar: T::TYPE,
                                bytes: host.bytes(),
                            },
                        ])
                        .unwrap();
                    let mut copied = [T::from_bits(17); 5];
                    mixed.output().read_into(&mut copied).unwrap();
                    assert_eq!(copied, out);
                    assert!(matches!(
                        prepared.execute_inputs(&[
                            MlxBinaryInput::HostBytes {
                                target: PcuBindingRef::new(2, 3),
                                scalar: T::TYPE,
                                bytes: host.bytes()
                            },
                            MlxBinaryInput::Resident {
                                target: PcuBindingRef::new(3, 2),
                                array: &foreign
                            },
                        ]),
                        Err(MlxError::ForeignSession)
                    ));
                    assert!(!prepared.last_call_may_have_written());

                    assert!(matches!(
                        prepared.execute_resident(&[&left, &foreign]),
                        Err(MlxError::ForeignSession)
                    ));
                    assert!(!prepared.last_call_may_have_written());
                    drop(prepared);
                    let mut values = [T::from_bits(17); 5];
                    old.read_into(&mut values).unwrap();
                    assert_eq!(values, out);
                }
            }
        }
    }
}
macro_rules! native {($name:ident,$ty:ty)=>{
    #[test]#[ignore="Requires actual MLX-owned checked binary GPU primitive and terminal source publication."]
    fn $name(){qualify::<$ty>();}
};}
native!(f16_binary_source_lifecycle, PcuF16Bits);
native!(bf16_binary_source_lifecycle, PcuBf16Bits);
native!(e4m3fn_binary_source_lifecycle, PcuF8E4M3FnBits);
native!(e5m2_binary_source_lifecycle, PcuF8E5M2Bits);
native!(f32_binary_source_lifecycle, f32);
native!(f64_binary_source_lifecycle, f64);
#[test]
fn unsupported_profiles_stay_detached() {
    graph::fixture_profile::<pcu_facade::PcuF128Bits, _>(
        3,
        Op::Add,
        Policy::IeeeAfterRounding,
        Range::Reject,
        false,
        [false; 2],
        |ir| assert!(MlxCheckedBinaryPlan::assess(ir).is_err()),
    );
    // Physical UInt32 limb arrays must fit MLX's signed shape extent before preparation.
    graph::fixture_profile::<f64, _>(
        u32::try_from(i32::MAX).unwrap() / 2 + 1,
        Op::Add,
        Policy::IeeeAfterRounding,
        Range::Reject,
        false,
        [false; 2],
        |ir| {
            assert!(matches!(
                MlxCheckedBinaryPlan::assess(ir),
                Err(MlxError::InvalidExtent)
            ));
        },
    );
    graph::fixture_profile::<PcuF16Bits, _>(
        3,
        Op::Add,
        Policy::IeeeAfterRounding,
        Range::Reject,
        false,
        [false; 2],
        |ir| {
            let mut ir = *ir;
            ir.numerical_requirements.numerical_options.reproducibility =
                PcuReproducibility::PortableV1;
            assert!(MlxCheckedBinaryPlan::assess(&ir).is_err());
            ir.numerical_requirements.numerical_options.reproducibility =
                PcuReproducibility::Unspecified;
            ir.numerical_requirements.range_policy = Range::Clamp;
            assert!(MlxCheckedBinaryPlan::assess(&ir).is_err());
        },
    );
}
struct Detached;
impl PcuHostKernelBackend for Detached {
    type Prepared = Self;
    type Error = fusion_pcu_mlx::MlxHostKernelError;
    fn prepare_host_kernel(
        &self,
        ir: &pcu_facade::PcuDispatchKernelIr<'_>,
    ) -> Result<Self, Self::Error> {
        MlxCheckedBinaryPlan::assess(ir).map_err(PcuHostDispatchError::Backend)?;
        Ok(Self)
    }
}
impl PcuPreparedHostKernel for Detached {
    type Error = fusion_pcu_mlx::MlxHostKernelError;
    fn call(&mut self, _: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        Err(PcuHostDispatchError::Backend(MlxError::InvalidRequest(
            "detached schema only".into(),
        )))
    }
}
#[test]
fn genuine_source_specializations_assess_without_runtime() {
    assert!(source::add_prepare::<PcuF16Bits, 3, _>(&Detached).is_ok());
    assert!(source::add_prepare::<f32, 3, _>(&Detached).is_ok());
    assert!(source::div_prepare::<f64, 3, _>(&Detached).is_ok());
    assert!(source::sub_prepare::<PcuBf16Bits, 3, _>(&Detached).is_ok());
    assert!(source::mul_prepare::<PcuF8E4M3FnBits, 3, _>(&Detached).is_ok());
    assert!(source::div_prepare::<PcuF8E5M2Bits, 3, _>(&Detached).is_ok());
    assert!(source::swapped_prepare::<PcuF16Bits, 3, _>(&Detached).is_ok());
    assert!(source::repeated_prepare::<PcuF16Bits, 3, _>(&Detached).is_ok());
    assert!(source::broadcast_prepare::<PcuF16Bits, 3, _>(&Detached).is_ok());
    assert!(source::grid_prepare::<PcuF16Bits, 3, _>(&Detached).is_ok());
    assert!(source::add_tight_clamp_prepare::<PcuF16Bits, 3, _>(&Detached).is_ok());
}

#[path = "ordinary/ordinary.rs"]
mod ordinary;
