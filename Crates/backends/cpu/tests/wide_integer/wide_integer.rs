//! Exact wide prepared and genuine source maps with independent base-256 full products.
#[path = "oracle/oracle.rs"]
mod oracle;
#[path = "source/source.rs"]
#[allow(dead_code)] // Each operation runs in semantic benchmark; layouts run here.
mod source;
use oracle::Wide;
#[rustfmt::skip]
use fusion_pcu_cpu::{PcuCpuCheckedInteger,PcuCpuCheckedIntegerError,PcuCpuHostBackend};
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuBindingRef,
    PcuDispatchIntegerBinaryOp,
    PcuDispatchDataOp,
    PcuDispatchOp,
    PcuExecutionFaultKind,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};
const OPS: [PcuDispatchIntegerBinaryOp; 3] = [
    PcuDispatchIntegerBinaryOp::Add,
    PcuDispatchIntegerBinaryOp::Sub,
    PcuDispatchIntegerBinaryOp::Mul,
];
fn pair<T: Wide>(
    prepared: &mut fusion_pcu_cpu::PcuCpuPreparedInteger<T>,
    left: T,
    right: T,
    op: PcuDispatchIntegerBinaryOp,
) {
    let sentinel = oracle::small::<T>(77);
    let mut output = [sentinel; 3];
    let expected = oracle::evaluate(left, right, op);
    let result = prepared.call(&mut [
        PcuHostArgument::read(PcuBindingRef::new(0, 0), &[left]),
        PcuHostArgument::read(PcuBindingRef::new(0, 1), &[right]),
        PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output),
    ]);
    match expected {
        Ok(value) => {
            result.unwrap();
            assert_eq!(output[0], value);
        }
        Err(kind) => {
            assert!(
                matches!(result,Err(PcuCpuCheckedIntegerError::Fault(fault)) if !fault.recovered && fault.kind==kind && fault.invocation_id==0)
            );
            assert_eq!(output[0], sentinel);
        }
    }
    assert_eq!(output[1..], [sentinel; 2]);
}
fn oracle_sweep<T: Wide>() {
    let bindings = source::add_bindings::<T>();
    let builder = source::add_ir::<T, 1>(&bindings).unwrap();
    let kernel = builder.ir();
    let mut ops = kernel.ops.to_vec();
    let mut negative_one = [255; 64];
    if !T::SIGNED {
        negative_one[0] = 1;
        negative_one[1..].fill(0);
    }
    let edges = [
        oracle::small::<T>(0),
        oracle::small::<T>(1),
        oracle::small::<T>(2),
        oracle::minimum::<T>(),
        oracle::maximum::<T>(),
        T::from_bytes(negative_one),
    ];
    for op in OPS {
        if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
            op: selected, ..
        }) = &mut ops[2]
        {
            *selected = op;
        }
        let kernel = pcu_facade::PcuDispatchKernelIr {
            ops: &ops,
            ..kernel
        };
        let mut prepared = PcuCpuCheckedInteger::<T>::new()
            .prepare_host_kernel(&kernel)
            .unwrap();
        for left in edges {
            for right in edges {
                pair(&mut prepared, left, right, op);
            }
        }
        for bit in 0..T::BYTES * 8 {
            let mut bytes = [0; 64];
            bytes[bit / 8] = 1 << (bit % 8);
            for right in edges {
                pair(&mut prepared, T::from_bytes(bytes), right, op);
            }
        }
        let mut state = 0x2bb6_c439_5691_759d_u64;
        for _ in 0..4096 {
            let mut left = [0; 64];
            let mut right = [0; 64];
            for byte in left[..T::BYTES]
                .iter_mut()
                .chain(right[..T::BYTES].iter_mut())
            {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *byte = state.to_le_bytes()[0];
            }
            pair(&mut prepared, T::from_bytes(left), T::from_bytes(right), op);
        }
    }
}
#[test]
fn i128_independent_oracle() {
    oracle_sweep::<i128>();
}
#[test]
fn u128_independent_oracle() {
    oracle_sweep::<u128>();
}
#[test]
fn i256_independent_oracle() {
    oracle_sweep::<PcuI256>();
}
#[test]
fn u256_independent_oracle() {
    oracle_sweep::<PcuU256>();
}
#[test]
fn i512_independent_oracle() {
    oracle_sweep::<PcuI512>();
}
#[test]
fn u512_independent_oracle() {
    oracle_sweep::<PcuU512>();
}
fn transaction<T: Wide>() {
    let one = oracle::small::<T>(1);
    let two = oracle::small::<T>(2);
    let max = oracle::maximum::<T>();
    let min = oracle::minimum::<T>();
    let sentinel = oracle::small::<T>(77);
    let mut output = [sentinel; 9];
    let left = [two; 7];
    let right = [one; 7];
    source::grid::<T, 7>(&left, &right, &mut output).unwrap();
    assert_eq!(output[..7], [oracle::small::<T>(3); 7]);
    assert_eq!(output[7..], [sentinel; 2]);
    for fault_lane in [0, 3, 6] {
        let mut bad = left;
        bad[fault_lane] = max;
        let before = output;
        assert!(
            matches!(source::add::<T,7>(&bad,&right,&mut output),Err(global::PcuExecutionError::ArithmeticFault(fault)) if !fault.recovered && fault.invocation_id==fault_lane as u64 && fault.kind==PcuExecutionFaultKind::ArithmeticOverflow)
        );
        assert_eq!(output, before);
        source::add::<T, 7>(&left, &right, &mut output).unwrap();
    }
    let mut bad = left;
    bad[1] = min;
    bad[5] = min;
    let before = output;
    assert!(
        matches!(source::sub::<T,7>(&bad,&right,&mut output),Err(global::PcuExecutionError::ArithmeticFault(fault)) if !fault.recovered && fault.invocation_id==1 && fault.kind==PcuExecutionFaultKind::ArithmeticUnderflow)
    );
    assert_eq!(output, before);
    source::sub::<T, 7>(&left, &right, &mut output).unwrap();
    source::scale::<T, 7>(&left, &two, &mut output).unwrap();
    assert_eq!(output[..7], [oracle::small::<T>(4); 7]);
    source::strict::<T, 7>(&left, &right, &mut output).unwrap();
    let bindings = source::sub_bindings::<T>();
    let builder = source::sub_ir::<T, 7>(&bindings).unwrap();
    let kernel = builder.ir();
    let mut prepared = PcuCpuHostBackend::scalar()
        .prepare_host_kernel(&kernel)
        .unwrap();
    for permutation in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let mut arguments = [
            Some(PcuHostArgument::read(PcuBindingRef::new(0, 0), &left)),
            Some(PcuHostArgument::read(PcuBindingRef::new(0, 1), &right)),
            Some(PcuHostArgument::read_write(
                PcuBindingRef::new(0, 2),
                &mut output,
            )),
        ];
        prepared
            .call(&mut permutation.map(|index| arguments[index].take().unwrap()))
            .unwrap();
        assert_eq!(output[..7], [one; 7]);
    }
    let before = output;
    assert!(
        prepared
            .call(&mut [
                PcuHostArgument::read(PcuBindingRef::new(0, 0), &left[..6]),
                PcuHostArgument::read(PcuBindingRef::new(0, 1), &right),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output)
            ])
            .is_err()
    );
    assert_eq!(output, before);
    let mut ops = kernel.ops.to_vec();
    if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary { lhs, rhs, .. }) =
        &mut ops[2]
    {
        *rhs = *lhs;
    }
    let kernel = pcu_facade::PcuDispatchKernelIr {
        ops: &ops,
        ..kernel
    };
    let mut repeated = PcuCpuHostBackend::scalar()
        .prepare_host_kernel(&kernel)
        .unwrap();
    repeated
        .call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &left),
            PcuHostArgument::read(PcuBindingRef::new(0, 1), &right),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output),
        ])
        .unwrap();
    assert_eq!(output[..7], [oracle::small::<T>(0); 7]);
    assert_eq!(output[7..], [sentinel; 2]);
}
#[test]
fn genuine_wide_ordinary_transaction_grid_broadcast_and_detached_schema() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    transaction::<i128>();
    transaction::<u128>();
    transaction::<PcuI256>();
    transaction::<PcuU256>();
    transaction::<PcuI512>();
    transaction::<PcuU512>();
}
#[allow(clippy::too_many_lines)] // Exact host/typed offer equivalence across all independent numerical axes.
fn offers<T: Wide>(base: u32) {
    #[rustfmt::skip]
    use pcu_facade::{PcuDeviceIdentity,PcuObjectRef,PcuObjectKind,PcuProviderId,PcuExecutorId,PcuImplementationRequest,PcuImplementationOffers,PcuCostBoundary,PcuNumericalMode,PcuCompoundArithmeticPolicy,PcuPrecisionPolicy,PcuReproducibility,PcuRangePolicy};
    let device = PcuDeviceIdentity::from_device_ref(PcuObjectRef {
        provider: PcuProviderId(3),
        generation: 7,
        kind: PcuObjectKind::Device,
        id: 0,
    })
    .unwrap();
    let backend = PcuCpuHostBackend::scalar();
    let host = fusion_pcu_cpu::PcuCpuHostOffers::new(backend, device, PcuExecutorId(0));
    let typed = fusion_pcu_cpu::PcuCpuIntegerOffers::<T>::new(device, PcuExecutorId(0));
    let bindings = source::add_bindings::<T>();
    let builder = source::add_ir::<T, 3>(&bindings).unwrap();
    let template = builder.ir();
    let mut operations = template.ops.to_vec();
    for (offset, op) in OPS.into_iter().enumerate() {
        if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
            op: selected, ..
        }) = &mut operations[2]
        {
            *selected = op;
        }
        for mode in [
            pcu_facade::PcuNumericalMode::Boundary,
            PcuNumericalMode::Strict,
        ] {
            for compound in [
                PcuCompoundArithmeticPolicy::Checked,
                PcuCompoundArithmeticPolicy::BackendDefined,
            ] {
                for precision in [
                    PcuPrecisionPolicy::Preserve,
                    PcuPrecisionPolicy::BackendOptimized,
                ] {
                    let mut kernel = pcu_facade::PcuDispatchKernelIr {
                        ops: &operations,
                        ..template
                    };
                    kernel.numerical_requirements.numerical_mode = mode;
                    kernel
                        .numerical_requirements
                        .numerical_options
                        .compound_arithmetic = compound;
                    kernel.numerical_requirements.numerical_options.precision = precision;
                    let mut request = PcuImplementationRequest {
                        device,
                        executor: PcuExecutorId(0),
                        operation: &kernel,
                        requirements: kernel.numerical_requirements,
                        boundary: PcuCostBoundary::Host,
                    };
                    let mut first = [None];
                    let mut second = [None];
                    assert_eq!(host.implementation_offers(&request, &mut first), Ok(1));
                    assert_eq!(typed.implementation_offers(&request, &mut second), Ok(1));
                    assert_eq!(first, second);
                    let offer = first[0].unwrap();
                    assert_eq!(
                        (offer.implementation.local_id, offer.implementation.revision),
                        (
                            base + u32::try_from(offset).unwrap(),
                            fusion_pcu_cpu::PCU_CPU_INTEGER_IMPLEMENTATION_REVISION
                        )
                    );
                    assert_eq!(offer.workspace_bytes, Some(0));
                    assert_eq!(offer.requirements, kernel.numerical_requirements);
                    request.requirements.numerical_mode = if mode == PcuNumericalMode::Strict {
                        PcuNumericalMode::Boundary
                    } else {
                        PcuNumericalMode::Strict
                    };
                    assert_eq!(host.implementation_offers(&request, &mut first), Ok(0));
                    assert_eq!(typed.implementation_offers(&request, &mut second), Ok(0));
                    request.requirements = kernel.numerical_requirements;
                    request.boundary = PcuCostBoundary::Resident;
                    assert_eq!(host.implementation_offers(&request, &mut first), Ok(0));
                    kernel
                        .numerical_requirements
                        .numerical_options
                        .reproducibility = PcuReproducibility::PortableV1;
                    assert!(backend.prepare_host_kernel(&kernel).is_ok());
                    let portable = PcuCpuCheckedInteger::<T>::new()
                        .prepare_host_kernel(&kernel)
                        .unwrap();
                    assert!((4096..=4179).contains(&portable.local_id()));
                    kernel
                        .numerical_requirements
                        .numerical_options
                        .reproducibility = PcuReproducibility::Unspecified;
                    kernel.numerical_requirements.range_policy = PcuRangePolicy::Clamp;
                    assert!(backend.prepare_host_kernel(&kernel).is_err());
                }
            }
        }
    }
}
#[test]
fn exact_wide_offer_ids_permissions_and_explicit_exclusions() {
    offers::<i128>(256);
    offers::<u128>(259);
    offers::<PcuI256>(262);
    offers::<PcuU256>(265);
    offers::<PcuI512>(268);
    offers::<PcuU512>(271);
}
