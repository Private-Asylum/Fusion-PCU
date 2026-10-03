//! Observable checked range recovery across host, device, resident and mixed six-format calls.
#[rustfmt::skip]
use pcu_facade::{pcu,PcuCheckedFloat,PcuTensor,PcuExecutionError,PcuExecutionFaultKind as Kind,PcuArgumentError};
use fusion_pcu_metal::{MetalSession, MetalError};
use super::low_precision::{assert_bits, open_backend};
#[path = "../../../benches/checked_binary_range/source/source.rs"]
mod source;
trait Sample: PcuCheckedFloat {
    fn value(value: f32) -> Self;
    fn raw(bits: u64) -> Self;
    const MAX: u64;
    const NAN: u64;
}
macro_rules! low_sample {
    ($ty:ty,$word:ty,$max:expr,$nan:expr) => {
        impl Sample for $ty {
            fn value(value: f32) -> Self {
                Self::pcu_checked_from_f32(value).unwrap()
            }
            fn raw(bits: u64) -> Self {
                Self::from_bits(<$word>::try_from(bits).unwrap())
            }
            const MAX: u64 = $max;
            const NAN: u64 = $nan;
        }
    };
}
low_sample!(pcu_facade::PcuF16Bits, u16, 0x7bff, 0x7e00);
low_sample!(pcu_facade::PcuBf16Bits, u16, 0x7f7f, 0x7fc0);
low_sample!(pcu_facade::PcuF8E4M3FnBits, u8, 0x7e, 0x7f);
low_sample!(pcu_facade::PcuF8E5M2Bits, u8, 0x7b, 0x7f);
impl Sample for f32 {
    fn value(value: f32) -> Self {
        value
    }
    fn raw(bits: u64) -> Self {
        Self::from_bits(u32::try_from(bits).unwrap())
    }
    const MAX: u64 = 0x7f7f_ffff;
    const NAN: u64 = 0x7fc0_0000;
}
impl Sample for f64 {
    fn value(value: f32) -> Self {
        Self::from(value)
    }
    fn raw(bits: u64) -> Self {
        Self::from_bits(bits)
    }
    const MAX: u64 = 0x7fef_ffff_ffff_ffff;
    const NAN: u64 = 0x7ff8_0000_0000_0000;
}
#[pcu(invocations=3,flag(strict),flag(clamp_range),flag(reject_subnormal_result),crate_path=::pcu_facade)]
fn broadcast<T: PcuCheckedFloat>(left: &T, right: &[T], out: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    out[id] = *left + right[id];
}
#[pcu(invocations=1,flag(clamp_range),flag(reject_subnormal_result),crate_path=::pcu_facade)]
fn grid<T: PcuCheckedFloat>(left: &[T], right: &[T], out: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < 3 {
        out[id] = left[id] + right[id];
        id += stride;
    }
}
#[allow(clippy::too_many_lines, clippy::cognitive_complexity)] // One six-format source lifecycle pairs recovery, fatal precedence, rollback, affinity and retry.
fn qualify<T: Sample>() {
    let _guard = crate::source_policy_guard();
    let session = MetalSession::open(0).unwrap();
    let sentinel = T::value(91.0);
    let zero = T::value(0.0);
    let one = T::value(1.0);
    let maximum = T::raw(T::MAX);
    let tiny = T::raw(1);
    let a = [tiny, maximum, one];
    let b = [zero, maximum, one];
    let expected = [tiny, maximum, T::value(2.0)];
    let mut output = [sentinel; 5];
    let mut call = source::add_tight_clamp_prepare::<T, 3, _>(&session).unwrap();
    assert!(
        matches!(call(&a,&b,&mut output),Err(pcu_facade::PcuHostDispatchError::Backend(MetalError::Arithmetic(f))) if f.recovered&&f.invocation_id==0&&f.kind==Kind::ArithmeticUnderflow)
    );
    assert_bits(
        &output,
        &[expected[0], expected[1], expected[2], sentinel, sentinel],
    );
    let fatal = [tiny, maximum, T::raw(T::NAN)];
    let before = output;
    assert!(
        matches!(call(&fatal,&b,&mut output),Err(pcu_facade::PcuHostDispatchError::Backend(MetalError::Arithmetic(f))) if !f.recovered&&f.invocation_id==2&&f.kind==Kind::InvalidFloatingOperand)
    );
    assert_bits(&output, &before);
    assert!(call(&a[..1], &b, &mut output).is_err());
    assert_bits(&output, &before);
    let mut div = source::div_tight_clamp_prepare::<T, 3, _>(&session).unwrap();
    assert!(
        matches!(div(&[tiny,one,maximum],&[one,zero,one],&mut output),Err(pcu_facade::PcuHostDispatchError::Backend(MetalError::Arithmetic(f))) if !f.recovered&&f.invocation_id==1&&f.kind==Kind::DivideByZero)
    );
    assert_bits(&output, &before);
    call(&[one; 3], &[one; 3], &mut output).unwrap();
    assert_bits(
        &output,
        &[T::value(2.0); 3]
            .into_iter()
            .chain([sentinel; 2])
            .collect::<Vec<_>>(),
    );
    assert!(grid_prepare::<T, _>(&session).unwrap()(&a, &b, &mut output).is_err());
    assert_bits(
        &output,
        &[expected[0], expected[1], expected[2], sentinel, sentinel],
    );
    assert!(broadcast_prepare::<T, _>(&session).unwrap()(&tiny, &[zero; 3], &mut output).is_err());
    assert_bits(&output, &[tiny, tiny, tiny, sentinel, sentinel]);
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Metal,
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })
    .unwrap();
    assert!(
        matches!(source::add_tight_clamp::<T,3>(&a,&b,&mut output),Err(PcuExecutionError::ArithmeticFault(f)) if f.recovered&&f.invocation_id==0)
    );
    let backend = open_backend();
    let pool = pcu_facade::PcuMemoryPoolId(0);
    let left = backend.upload_buffer(pool, &a).unwrap();
    let right = backend.upload_buffer(pool, &b).unwrap();
    let mut raw = backend.upload_buffer(pool, &[sentinel; 5]).unwrap();
    let mut device = source::add_tight_clamp_prepare_device::<T, 3, _>(&backend).unwrap();
    assert!(device(&left, &right, &mut raw).is_err());
    let mut values = [sentinel; 5];
    backend.download_buffer(pool, &raw, &mut values).unwrap();
    assert_bits(
        &values,
        &[expected[0], expected[1], expected[2], sentinel, sentinel],
    );
    let left = PcuTensor::from_device_buffer(backend.clone(), left, &[3]).unwrap();
    let right = PcuTensor::from_device_buffer(backend.clone(), right, &[3]).unwrap();
    let out = backend.upload_buffer(pool, &[sentinel; 3]).unwrap();
    let mut resident = PcuTensor::from_device_buffer(backend.clone(), out, &[3]).unwrap();
    assert!(
        matches!(source::add_tight_clamp::<T,3>(&left,&right,&mut resident),Err(PcuExecutionError::ArithmeticFault(f)) if f.recovered)
    );
    let mut read = [sentinel; 3];
    resident.read_into(&mut read).unwrap();
    assert_bits(&read, &expected);
    assert!(source::add_tight_clamp::<T, 3>(&a, &right, &mut resident).is_err());
    resident.read_into(&mut read).unwrap();
    assert_bits(&read, &expected);
    assert!(source::add_tight_clamp::<T, 3>(&left, &b, &mut output).is_err());
    assert_bits(
        &output,
        &[expected[0], expected[1], expected[2], sentinel, sentinel],
    );
    let scalar_buffer = backend.upload_buffer(pool, &[tiny]).unwrap();
    let scalar = PcuTensor::from_device_buffer(backend.clone(), scalar_buffer, &[]).unwrap();
    assert!(
        matches!(broadcast::<T>(&scalar,&[zero;3],&mut resident),Err(PcuExecutionError::ArithmeticFault(f)) if f.recovered&&f.invocation_id==0)
    );
    resident.read_into(&mut read).unwrap();
    assert_bits(&read, &[tiny; 3]);
    assert!(
        matches!(source::add_tight_clamp::<T,3>(&fatal,&right,&mut resident),Err(PcuExecutionError::ArithmeticFault(f)) if !f.recovered&&f.invocation_id==2)
    );
    assert!(matches!(
        resident.read_into(&mut read),
        Err(PcuExecutionError::Argument(
            PcuArgumentError::ResidentValueDiscarded
        ))
    ));
    assert!(matches!(
        source::add_tight_clamp::<T, 3>(&resident, &b, &mut output),
        Err(PcuExecutionError::Argument(
            PcuArgumentError::ResidentValueDiscarded
        ))
    ));
    let fresh = backend.upload_buffer(pool, &[sentinel; 3]).unwrap();
    let mut fresh = PcuTensor::from_device_buffer(backend.clone(), fresh, &[3]).unwrap();
    source::add_tight_clamp::<T, 3>(&[one; 3], &[one; 3], &mut fresh).unwrap();
    fresh.read_into(&mut read).unwrap();
    assert_bits(&read, &[T::value(2.0); 3]);
    left.read_into(&mut read).unwrap();
    assert_bits(&read, &a);
    right.read_into(&mut read).unwrap();
    assert_bits(&read, &b);
    pcu_facade::global::use_defaults().unwrap();
}
macro_rules! formats {
    ($name:ident,$ty:ty) => {
        #[test]
        #[ignore = "Requires actual Metal six-format Clamp/broadcast/publication source proof."]
        fn $name() {
            qualify::<$ty>();
        }
    };
}
formats!(f32_range, f32);
formats!(f64_range, f64);
formats!(f16_range, pcu_facade::PcuF16Bits);
formats!(bf16_range, pcu_facade::PcuBf16Bits);
formats!(e4_range, pcu_facade::PcuF8E4M3FnBits);
formats!(e5_range, pcu_facade::PcuF8E5M2Bits);
