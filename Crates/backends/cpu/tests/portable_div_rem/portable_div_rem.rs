//! Independent typed cold opt-in; Portable requests keep disjoint IDs and unchanged joint execution.
#[path = "../wide_div_rem/oracle/oracle.rs"]
#[allow(dead_code)]
// Independent byte arithmetic and endpoints, not text-golden allocation helpers.
mod oracle;
#[path = "source/source.rs"]
mod source;
#[rustfmt::skip]
use fusion_pcu_cpu::{PcuCpuCheckedDivRem,PcuCpuHostBackend,PcuCpuPreparedHost};
#[rustfmt::skip]
use pcu_facade::{PcuCheckedIntegerDivision,PcuDispatchKernelIr,PcuHostArgument,PcuHostKernelBackend,PcuPreparedHostKernel,PcuBindingRef,PcuReproducibility,PcuRangePolicy};
use oracle::Wide;
fn with_kernel<T: PcuCheckedIntegerDivision>(
    kind: usize,
    mut run: impl FnMut(PcuDispatchKernelIr<'_>),
) {
    macro_rules! visit {
        ($bindings:ident,$ir:ident) => {{
            let b = source::$bindings::<T>();
            let ir = source::$ir::<T, 5>(&b).unwrap();
            run(ir.ir());
        }};
    }
    match kind {
        0 => visit!(repeated_bindings, repeated_ir),
        1 => visit!(unused_bindings, unused_ir),
        2 => visit!(reordered_bindings, reordered_ir),
        3 => visit!(mixed_bindings, mixed_ir),
        4 => visit!(grid_bindings, grid_ir),
        _ => visit!(scalar_bindings, scalar_ir),
    }
}
fn call<T: PcuCheckedIntegerDivision>(
    kind: usize,
    plan: &mut impl PcuPreparedHostKernel<Error = fusion_pcu_cpu::PcuCpuHostError>,
    left: &[T],
    right: &[T],
    q: &mut [T],
    r: &mut [T],
) -> Result<(), fusion_pcu_cpu::PcuCpuHostError> {
    let binding = PcuBindingRef::new;
    match kind {
        0 => plan.call(&mut [
            PcuHostArgument::read_write(binding(0, 0), q),
            PcuHostArgument::read_write(binding(0, 1), r),
            PcuHostArgument::read(binding(0, 2), left),
        ]),
        1 => plan.call(&mut [
            PcuHostArgument::read(binding(0, 0), &[] as &[T]),
            PcuHostArgument::read(binding(0, 1), left),
            PcuHostArgument::read_write(binding(0, 2), r),
            PcuHostArgument::read_write(binding(0, 3), q),
        ]),
        2 => plan.call(&mut [
            PcuHostArgument::read(binding(0, 0), right),
            PcuHostArgument::read_write(binding(0, 1), r),
            PcuHostArgument::read(binding(0, 2), left),
            PcuHostArgument::read_write(binding(0, 3), q),
        ]),
        3 => plan.call(&mut [
            PcuHostArgument::read_write(binding(0, 0), q),
            PcuHostArgument::read(binding(0, 1), left),
            PcuHostArgument::read_write(binding(0, 2), r),
        ]),
        4 => plan.call(&mut [
            PcuHostArgument::read_write(binding(0, 0), r),
            PcuHostArgument::read(binding(0, 1), &[] as &[T]),
            PcuHostArgument::read_write(binding(0, 2), q),
            PcuHostArgument::read(binding(0, 3), left),
        ]),
        _ => plan.call(&mut [
            PcuHostArgument::read(binding(0, 0), &left[..1]),
            PcuHostArgument::read_write(binding(0, 1), q),
            PcuHostArgument::read(binding(0, 2), &right[..1]),
            PcuHostArgument::read_write(binding(0, 3), r),
        ]),
    }
}
fn verify<T: Wide + PcuCheckedIntegerDivision>(id: u32) {
    for kind in 0..6 {
        with_kernel::<T>(kind, |mut kernel| {
            kernel
                .numerical_requirements
                .numerical_options
                .reproducibility = PcuReproducibility::PortableV1;
            let mut plan = PcuCpuCheckedDivRem::<T>::new()
                .prepare_host_kernel(&kernel)
                .unwrap();
            assert_eq!((plan.local_id(), plan.implementation_revision()), (id, 1));
            let PcuCpuPreparedHost::DivRem(host) = PcuCpuHostBackend::scalar()
                .prepare_host_kernel(&kernel)
                .unwrap()
            else {
                panic!("joint host")
            };
            assert_eq!(host.local_id(), id);
            exact_offers(&kernel, id);
            let sentinel = oracle::small::<T>(99);
            let mut left = [oracle::small::<T>(7); 5];
            left[0] = oracle::small(3);
            left[2] = oracle::maximum();
            let mut right = [oracle::small::<T>(3); 5];
            let mut q = [sentinel; 8];
            let mut r = q;
            for phase in 0..3 {
                if phase == 1 {
                    if kind == 2 || kind == 5 {
                        right[if kind == 5 { 0 } else { 2 }] = oracle::small(0);
                    } else {
                        left[if kind == 3 { 0 } else { 2 }] = oracle::small(0);
                    }
                }
                if phase == 2 {
                    left.fill(oracle::small(7));
                    right.fill(oracle::small(3));
                }
                let mut expected_q = [sentinel; 8];
                let mut expected_r = expected_q;
                let mut fault = None;
                for lane in 0..5 {
                    let (a, b) = match kind {
                        0 | 1 => (left[lane], left[lane]),
                        2 => (left[lane], right[lane]),
                        3 => (left[lane], left[0]),
                        4 => (left[0], left[lane]),
                        _ => (left[0], right[0]),
                    };
                    match oracle::evaluate(a, b) {
                        Ok((a, b)) => {
                            expected_q[lane] = a;
                            expected_r[lane] = b;
                        }
                        Err(kind) => {
                            fault = Some((lane as u64, kind));
                            break;
                        }
                    }
                }
                q.fill(sentinel);
                r.fill(sentinel);
                let result = call(kind, &mut plan, &left, &right, &mut q, &mut r);
                if let Some((lane, kind)) = fault {
                    assert!(
                        matches!(result,Err(fusion_pcu_cpu::PcuCpuHostError::Integer(fusion_pcu_cpu::PcuCpuCheckedIntegerError::Fault(f)))if(f.invocation_id,f.kind,f.recovered)==(lane,kind,false))
                    );
                    assert_eq!((q, r), ([sentinel; 8], [sentinel; 8]));
                } else {
                    result.unwrap();
                    assert_eq!((q, r), (expected_q, expected_r));
                }
            }
            for invalid in 0..4 {
                let mut bad = kernel;
                match invalid {
                    0 => bad.numerical_requirements.range_policy = PcuRangePolicy::Clamp,
                    1 => bad.entry.logical_shape[1] = 2,
                    2 => bad.bindings = &[],
                    _ => bad.ops = &[],
                }
                assert!(
                    PcuCpuCheckedDivRem::<T>::new()
                        .prepare_host_kernel(&bad)
                        .is_err()
                );
            }
        });
    }
}
#[test]
fn exact_fourteen_width_portable_ids_roles_payloads_and_negatives() {
    verify::<i8>(12288);
    verify::<u8>(12289);
    verify::<i16>(12290);
    verify::<u16>(12291);
    verify::<i32>(12292);
    verify::<u32>(12293);
    verify::<i64>(12294);
    verify::<u64>(12295);
    verify::<i128>(12296);
    verify::<u128>(12297);
    verify::<pcu_facade::PcuI256>(12298);
    verify::<pcu_facade::PcuU256>(12299);
    verify::<pcu_facade::PcuI512>(12300);
    verify::<pcu_facade::PcuU512>(12301);
}

fn exact_offers(kernel: &PcuDispatchKernelIr<'_>, id: u32) {
    let device = pcu_facade::PcuDeviceIdentity::from_device_ref(pcu_facade::PcuObjectRef {
        provider: pcu_facade::PcuProviderId(3),
        generation: 7,
        kind: pcu_facade::PcuObjectKind::Device,
        id: 0,
    })
    .unwrap();
    let offers = fusion_pcu_cpu::PcuCpuHostOffers::new(
        PcuCpuHostBackend::scalar(),
        device,
        pcu_facade::PcuExecutorId(0),
    );
    let mut request = pcu_facade::PcuImplementationRequest {
        device,
        executor: pcu_facade::PcuExecutorId(0),
        operation: kernel,
        requirements: kernel.numerical_requirements,
        boundary: pcu_facade::PcuCostBoundary::Host,
    };
    let mut offered = [None];
    assert_eq!(
        pcu_facade::PcuImplementationOffers::implementation_offers(&offers, &request, &mut offered),
        Ok(1)
    );
    assert_eq!(
        (
            offered[0].unwrap().implementation.local_id,
            offered[0].unwrap().implementation.revision
        ),
        (id, 1)
    );
    assert_eq!(offered[0].unwrap().requirements, request.requirements);
    request.requirements.numerical_options.reproducibility = PcuReproducibility::Unspecified;
    assert_eq!(
        pcu_facade::PcuImplementationOffers::implementation_offers(&offers, &request, &mut offered),
        Ok(0)
    );
}

#[path = "source_proof/source_proof.rs"]
mod source_proof;

#[test]
fn reserved_total_division_flags_cannot_enter_portable_executor() {
    with_kernel::<i32>(0, |mut kernel| {
        kernel
            .numerical_requirements
            .numerical_options
            .reproducibility = PcuReproducibility::PortableV1;
        let mut operations =
            [pcu_facade::PcuDispatchOp::Control(pcu_facade::PcuDispatchControlOp::Return); 6];
        operations.copy_from_slice(kernel.ops);
        let pcu_facade::PcuDispatchOp::Data(pcu_facade::PcuDispatchDataOp::CheckedDivRem {
            flags,
            ..
        }) = &mut operations[2]
        else {
            panic!("joint operation");
        };
        *flags = pcu_facade::model::PcuIntegerDivFlags::DIV_OR_ZERO;
        let kernel = PcuDispatchKernelIr {
            ops: &operations,
            ..kernel
        };
        assert!(
            PcuCpuCheckedDivRem::<i32>::new()
                .prepare_host_kernel(&kernel)
                .is_err()
        );
        assert!(
            PcuCpuHostBackend::scalar()
                .prepare_host_kernel(&kernel)
                .is_err()
        );
    });
}
