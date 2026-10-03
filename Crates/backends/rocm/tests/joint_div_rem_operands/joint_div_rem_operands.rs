//! Native joint roles, private results, first logical fault and terminal retry.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/joint_div_rem_operands/native/native.rs"]
mod native;
#[path = "../../benches/joint_div_rem_operands/oracle/oracle.rs"]
mod oracle;
#[path = "../../benches/strict_matmul/selection.rs"]
mod selection;
#[path = "../../benches/joint_div_rem_operands/source/source.rs"]
mod source;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuBindingRef,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuImplementationRequirements,
    PcuNumericalMode,
    PcuReproducibility,
};
use oracle::Format;
use fusion_pcu_rocm::RocmOwnedDispatchBackend;
fn prepared<T: Format, Prepared: PcuPreparedHostKernel>(
    explicit: &mut Prepared,
    kind: u32,
    a: &[T],
    b: &[T],
    q: &mut [T],
    r: &mut [T],
) -> Result<(), Prepared::Error>
where
    Prepared::Error: core::fmt::Debug,
{
    match kind {
        0 | 3 => explicit.call(&mut [
            PcuHostArgument::read_write(PcuBindingRef::new(0, 0), q),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), r),
            PcuHostArgument::read(PcuBindingRef::new(0, 2), a),
        ]),
        1 | 4 => explicit.call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &[] as &[T]),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), r),
            PcuHostArgument::read(PcuBindingRef::new(0, 2), a),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 3), q),
        ]),
        2 => explicit.call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), b),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), r),
            PcuHostArgument::read(PcuBindingRef::new(0, 2), a),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 3), q),
        ]),
        _ => unreachable!(),
    }
}

fn fault(error: &PcuExecutionError, lane: u64, kind: PcuExecutionFaultKind) {
    let actual = error.arithmetic_fault().unwrap();
    assert_eq!(
        (actual.invocation_id, actual.kind, actual.recovered),
        (lane, kind, false)
    );
}
#[allow(clippy::too_many_lines)] // All three real routes share the same independent bytes and domain transitions.
fn profile<T: Format>(
    backend: &RocmOwnedDispatchBackend,
    kind: u32,
    ir: &fusion_pcu::PcuDispatchKernelIr<'_>,
    mut host: impl FnMut(&[T], &[T], &mut [T], &mut [T]) -> Result<(), PcuExecutionError>,
) {
    let mut explicit = backend.prepare_host_kernel(ir).unwrap();
    let mut native = native::Native::new::<T, 17>(backend, ir, 1);
    let (mut a, mut b, wq, wr) = oracle::inputs::<T>(17, 11, kind);
    let mut quotient = vec![T::SENTINEL; 19];
    let mut remainder = quotient.clone();
    host(&a, &b, &mut quotient, &mut remainder).unwrap();
    oracle::verify(&wq, &wr, &quotient, &remainder);
    prepared(&mut explicit, kind, &a, &b, &mut quotient, &mut remainder).unwrap();
    oracle::verify(&wq, &wr, &quotient, &remainder);
    native.host(&a, &b, &mut quotient, &mut remainder);
    oracle::verify(&wq, &wr, &quotient, &remainder);
    let prior = (quotient.clone(), remainder.clone());
    let (lane, change) = if kind == 3 { (0, 0) } else { (2, 2) };
    if kind == 2 {
        b[change] = T::ZERO;
        b[6] = T::ZERO;
    } else {
        a[change] = T::ZERO;
        if kind != 3 {
            a[6] = T::ZERO;
        }
    }
    fault(
        &host(&a, &b, &mut quotient, &mut remainder).unwrap_err(),
        lane,
        PcuExecutionFaultKind::DivideByZero,
    );
    assert_eq!((&quotient, &remainder), (&prior.0, &prior.1));
    assert!(prepared(&mut explicit, kind, &a, &b, &mut quotient, &mut remainder).is_err());
    assert_eq!((&quotient, &remainder), (&prior.0, &prior.1));
    native.upload(0, &a, &b);
    assert_eq!(native.submit(0), (lane << 3) | 1);
    let (a, b, wq, wr) = oracle::inputs::<T>(17, 37, kind);
    host(&a, &b, &mut quotient, &mut remainder).unwrap();
    oracle::verify(&wq, &wr, &quotient, &remainder);
    prepared(&mut explicit, kind, &a, &b, &mut quotient, &mut remainder).unwrap();
    oracle::verify(&wq, &wr, &quotient, &remainder);
    native.host(&a, &b, &mut quotient, &mut remainder);
    oracle::verify(&wq, &wr, &quotient, &remainder);
    if kind == 2
        && let Some(minus_one) = T::NEGATIVE_ONE
    {
        let mut a = a;
        let mut b = b;
        a[2] = T::MIN;
        b[2] = minus_one;
        a[6] = T::MIN;
        b[6] = minus_one;
        let prior = (quotient.clone(), remainder.clone());
        fault(
            &host(&a, &b, &mut quotient, &mut remainder).unwrap_err(),
            2,
            PcuExecutionFaultKind::SignedDivisionOverflow,
        );
        assert_eq!((&quotient, &remainder), (&prior.0, &prior.1));
        native.upload(0, &a, &b);
        assert_eq!(native.submit(0), (2 << 3) | 2);
        let (a, b, wq, wr) = oracle::inputs::<T>(17, 71, kind);
        native.host(&a, &b, &mut quotient, &mut remainder);
        oracle::verify(&wq, &wr, &quotient, &remainder);
    }
}
fn width<T: Format>(
    backend: &RocmOwnedDispatchBackend,
    requirements: PcuImplementationRequirements,
) {
    macro_rules! entry {
        ($bindings:ident,$builder:ident,$kind:literal,$call:expr) => {{
            let bindings = source::$bindings::<T>();
            let builder = source::$builder::<T, 17>(
                &bindings,
                requirements.float_underflow,
                requirements.range_policy,
                requirements,
            )
            .unwrap();
            builder.with_ir(|ir| profile(backend, $kind, ir, $call));
        }};
    }
    entry!(
        repeated_bindings,
        __repeated_ir_with_float_underflow_policy,
        0,
        |a, _b, q, r| source::repeated::<T, 17>(q, r, a)
    );
    entry!(
        unread_bindings,
        __unread_ir_with_float_underflow_policy,
        1,
        |a, _b, q, r| source::unread::<T, 17>(&[], r, a, q)
    );
    entry!(
        reordered_bindings,
        __reordered_ir_with_float_underflow_policy,
        2,
        |a, b, q, r| source::reordered::<T, 17>(b, r, a, q)
    );
    entry!(
        indexed_zero_bindings,
        __indexed_zero_ir_with_float_underflow_policy,
        3,
        |a, _b, q, r| source::indexed_zero::<T, 17>(q, r, a)
    );
    entry!(
        grid_zero_bindings,
        __grid_zero_ir_with_float_underflow_policy,
        4,
        |a, _b, q, r| source::grid_zero::<T, 17>(&[], r, a, q)
    );
}
#[test]
#[ignore = "authorized native joint all14 source/prepared/native transaction qualification"]
fn fourteen_joint_roles_private_results_and_retry() {
    let (_, backend, _) = selection::selected_device();
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for reproducibility in [
            PcuReproducibility::Unspecified,
            PcuReproducibility::PortableV1,
        ] {
            let requirements = PcuImplementationRequirements {
                numerical_mode: mode,
                numerical_options: fusion_pcu::PcuNumericalOptions {
                    reproducibility,
                    ..Default::default()
                },
                ..PcuImplementationRequirements::DEFAULT
            };
            global::configure(global::PcuExecutionPolicy {
                backend: global::PcuBackendChoice::Rocm,
                device: Some(0),
                numerical_mode: mode,
                numerical_options: requirements.numerical_options,
                ..Default::default()
            })
            .unwrap();
            macro_rules! ty {
                ($t:ty) => {{
                    global::clear_thread_cache().unwrap();
                    width::<$t>(&backend, requirements);
                }};
            }
            ty!(i8);
            ty!(u8);
            ty!(i16);
            ty!(u16);
            ty!(i32);
            ty!(u32);
            ty!(i64);
            ty!(u64);
            ty!(i128);
            ty!(u128);
            ty!(fusion_pcu::PcuI256);
            ty!(fusion_pcu::PcuU256);
            ty!(fusion_pcu::PcuI512);
            ty!(fusion_pcu::PcuU512);
        }
    }
}

fn resident<T: Format>() {
    global::clear_thread_cache().unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    let raw = vec![T::MAX; 17];
    let foreign = source::identity(&raw).unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        device: Some(0),
        ..Default::default()
    })
    .unwrap();
    let input = vec![T::ONE; 17];
    let left = source::identity(&input).unwrap();
    let mut quotient = source::fresh_zero(&left).unwrap();
    let mut remainder = source::fresh_zero(&left).unwrap();
    let sibling = source::identity(&left).unwrap();
    let mut requirements = global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        device: Some(0),
        ..Default::default()
    };
    requirements.numerical_options.reproducibility = PcuReproducibility::PortableV1;
    global::configure(requirements).unwrap();
    global::clear_thread_cache().unwrap();
    source::unread::<T, 17>(&foreign, &mut remainder, &left, &mut quotient).unwrap();
    let mut host = vec![T::SENTINEL; 19];
    quotient.read_into(&mut host).unwrap();
    assert_eq!(&host[..17], &input);
    assert_eq!(&host[17..], &[T::SENTINEL; 2]);
    remainder.read_into(&mut host).unwrap();
    assert_eq!(&host[..17], &[T::ZERO; 17]);
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        device: Some(0),
        ..Default::default()
    })
    .unwrap();
    let bad = source::fresh_zero(&left).unwrap();
    global::configure(requirements).unwrap();
    fault(
        &source::unread::<T, 17>(&foreign, &mut remainder, &bad, &mut quotient).unwrap_err(),
        0,
        PcuExecutionFaultKind::DivideByZero,
    );
    assert!(quotient.read_into(&mut host).is_err());
    assert!(remainder.read_into(&mut host).is_err());
    sibling.read_into(&mut host).unwrap();
    assert_eq!(&host[..17], &input);
    foreign.read_into(&mut host).unwrap();
    assert_eq!(&host[..17], &raw);
    global::clear_thread_cache().unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        device: Some(0),
        ..Default::default()
    })
    .unwrap();
    let mut fresh_quotient = source::fresh_zero(&left).unwrap();
    let mut fresh_remainder = source::fresh_zero(&left).unwrap();
    global::configure(requirements).unwrap();
    // Both discarded private destinations may be declared unread without touching their owners.
    source::unread::<T, 17>(&quotient, &mut fresh_remainder, &left, &mut fresh_quotient).unwrap();
    source::unread::<T, 17>(&remainder, &mut fresh_remainder, &left, &mut fresh_quotient).unwrap();
    fresh_quotient.read_into(&mut host).unwrap();
    assert_eq!(&host[..17], &input);
    fresh_remainder.read_into(&mut host).unwrap();
    assert_eq!(&host[..17], &[T::ZERO; 17]);
    left.read_into(&mut host).unwrap();
    assert_eq!(&host[..17], &input);
    assert_eq!(&host[17..], &[T::SENTINEL; 2]);
}
#[test]
#[ignore = "authorized actual14 Portable joint mixed owners, ignored foreign/discarded owner, dual discard and same-session retry"]
fn fourteen_joint_resident_unread_owner_discard_and_fresh_retry() {
    macro_rules! ty {
        ($t:ty) => {
            resident::<$t>();
        };
    }
    ty!(i8);
    ty!(u8);
    ty!(i16);
    ty!(u16);
    ty!(i32);
    ty!(u32);
    ty!(i64);
    ty!(u64);
    ty!(i128);
    ty!(u128);
    ty!(fusion_pcu::PcuI256);
    ty!(fusion_pcu::PcuU256);
    ty!(fusion_pcu::PcuI512);
    ty!(fusion_pcu::PcuU512);
}
