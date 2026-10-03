//! Genuine requested Portable source, reordered unused roles and actual mutable owner publication.
#[rustfmt::skip]
use fusion_pcu_metal::{
    MetalPortableUnaryPlan,
    MetalSession,
};
#[rustfmt::skip]
use pcu_facade::{
    global,
    pcu,
    PcuArgumentError,
    PcuBf16Bits,
    PcuCheckedFloat,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuF16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuMemoryPoolId,
    PcuReproducibility,
    PcuTensor,
};
#[rustfmt::skip]
use super::low_precision::{
    assert_bits,
    open_backend,
};
trait Sample: PcuCheckedFloat {
    const SIGN: u64;
    const NONFINITE: u64;
    fn raw(bits: u64) -> Self;
}
macro_rules! encoding {
    ($ty:ty,$word:ty,$sign:expr,$nan:expr) => {
        impl Sample for $ty {
            const SIGN: u64 = $sign;
            const NONFINITE: u64 = $nan;
            fn raw(bits: u64) -> Self {
                Self::from_bits(<$word>::try_from(bits).unwrap())
            }
        }
    };
}
encoding!(PcuF16Bits, u16, 0x8000, 0x7e00);
encoding!(PcuBf16Bits, u16, 0x8000, 0x7fc0);
encoding!(PcuF8E4M3FnBits, u8, 0x80, 0x7f);
encoding!(PcuF8E5M2Bits, u8, 0x80, 0x7e);
encoding!(f32, u32, 0x8000_0000, 0x7fc0_0000);
impl Sample for f64 {
    const SIGN: u64 = 0x8000_0000_0000_0000;
    const NONFINITE: u64 = 0x7ff8_0000_0000_0000;
    fn raw(bits: u64) -> Self {
        Self::from_bits(bits)
    }
}
#[pcu(invocations=3,flag(deterministic),crate_path=::pcu_facade)]
fn permuted<T: PcuCheckedFloat>(unread: &[T], output: &mut [T], input: &[T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
#[pcu(invocations=3,flag(deterministic),crate_path=::pcu_facade)]
fn negate<T: PcuCheckedFloat>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
#[pcu(invocations=3,flag(deterministic),flag(clamp_range),flag(reject_subnormal_result),crate_path=::pcu_facade)]
fn tight<T: PcuCheckedFloat>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
fn metadata<T: Sample>() {
    let bindings = permuted_bindings::<T>();
    permuted_ir::<T>(&bindings).unwrap().with_ir(|kernel| {
        let plan = MetalPortableUnaryPlan::assess(kernel).unwrap();
        assert_eq!(plan.requirements(), kernel.numerical_requirements);
        assert_eq!(
            plan.requirements().numerical_options.reproducibility,
            PcuReproducibility::PortableV1
        );
        assert_eq!(plan.description().input_binding, bindings[2].reference());
        assert_eq!(plan.description().output_binding, bindings[1].reference());
        let mut normal = *kernel;
        normal
            .numerical_requirements
            .numerical_options
            .reproducibility = PcuReproducibility::Unspecified;
        assert!(MetalPortableUnaryPlan::assess(&normal).is_err());
    });
}
fn case<T: Sample>() {
    metadata::<T>();
    let input = [T::raw(1), T::raw(T::SIGN), T::raw(T::SIGN | 1)];
    let expected = [T::raw(T::SIGN | 1), T::raw(0), T::raw(1)];
    let sentinel = T::raw(17);
    let session = MetalSession::open(0).unwrap();
    let mut host = [sentinel; 5];
    permuted_prepare::<T, _>(&session).unwrap()(&[], &mut host, &input).unwrap();
    assert_bits(
        &host,
        &[expected[0], expected[1], expected[2], sentinel, sentinel],
    );
    let backend = open_backend();
    let pool = PcuMemoryPoolId(0);
    let left = backend.upload_buffer(pool, &input).unwrap();
    let mut result = backend.upload_buffer(pool, &[sentinel; 5]).unwrap();
    negate_prepare_device::<T, _>(&backend).unwrap()(&left, &mut result).unwrap();
    backend.download_buffer(pool, &result, &mut host).unwrap();
    assert_bits(
        &host,
        &[expected[0], expected[1], expected[2], sentinel, sentinel],
    );
    let owner = PcuTensor::from_device_buffer(backend.clone(), left, &[3]).unwrap();
    let mut output = PcuTensor::from_device_buffer(backend.clone(), result, &[5]).unwrap();
    // A frontend unread ghost never acquires the foreign owner's shape, lease or session.
    let foreign = open_backend();
    let other = foreign
        .upload_buffer(pool, &[T::raw(T::NONFINITE)])
        .unwrap();
    let unread = PcuTensor::from_device_buffer(foreign, other, &[1]).unwrap();
    permuted::<T>(&unread, &mut output, &owner).unwrap();
    output.read_into(&mut host).unwrap();
    assert_bits(
        &host,
        &[expected[0], expected[1], expected[2], sentinel, sentinel],
    );
    permuted::<T>(&[], &mut host, &owner).unwrap();
    permuted::<T>(&[], &mut output, &input).unwrap();
    let recovered = tight::<T>(&owner, &mut output).unwrap_err();
    assert!(
        matches!(recovered,PcuExecutionError::ArithmeticFault(fault) if fault.recovered && fault.invocation_id==0 && fault.kind==PcuExecutionFaultKind::ArithmeticUnderflow)
    );
    output.read_into(&mut host).unwrap();
    assert_bits(
        &host,
        &[expected[0], expected[1], expected[2], sentinel, sentinel],
    );
    let fatal = [T::raw(1), T::raw(T::NONFINITE), T::raw(0)];
    let before = host;
    assert!(
        matches!(tight::<T>(&fatal,&mut host),Err(PcuExecutionError::ArithmeticFault(fault)) if !fault.recovered && fault.invocation_id==1)
    );
    assert_bits(&host, &before);
    assert!(
        matches!(tight::<T>(&fatal,&mut output),Err(PcuExecutionError::ArithmeticFault(fault)) if !fault.recovered && fault.invocation_id==1)
    );
    assert!(matches!(
        output.read_into(&mut host),
        Err(PcuExecutionError::Argument(
            PcuArgumentError::ResidentValueDiscarded
        ))
    ));
    assert!(matches!(
        permuted::<T>(&[], &mut host, &output),
        Err(PcuExecutionError::Argument(
            PcuArgumentError::ResidentValueDiscarded
        ))
    ));
    let fresh = backend.upload_buffer(pool, &[sentinel; 5]).unwrap();
    let mut output = PcuTensor::from_device_buffer(backend, fresh, &[5]).unwrap();
    negate::<T>(&owner, &mut output).unwrap();
    drop(owner);
    drop(unread);
    global::clear_thread_cache().unwrap();
    output.read_into(&mut host).unwrap();
    assert_bits(&host, &before);
}
macro_rules! six {
    ($function:ident) => {
        $function::<PcuF16Bits>();
        $function::<PcuBf16Bits>();
        $function::<PcuF8E4M3FnBits>();
        $function::<PcuF8E5M2Bits>();
        $function::<f32>();
        $function::<f64>();
    };
}
#[test]
fn six_format_portable_unary_actual_source_roles_are_detached() {
    six!(metadata);
}
#[test]
#[ignore = "Requires actual Metal requested Portable host/device/mixed/ordinary source and mutable publication."]
fn six_format_portable_unary_host_device_mixed_and_ordinary_roles() {
    let _guard = crate::source_policy_guard();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Metal,
        ..Default::default()
    })
    .unwrap();
    six!(case);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

#[path = "encodings/encodings.rs"]
mod encodings;
#[path = "graph/graph.rs"]
pub mod graph;
#[path = "oracle/oracle.rs"]
mod oracle;
