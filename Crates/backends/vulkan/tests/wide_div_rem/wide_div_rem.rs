//! Actual wide joint U32 division against independent full-width byte/BigInt goldens.
extern crate pcu_facade as fusion_pcu;
#[path = "../scalar_transport/device/device.rs"]
mod device;
#[path = "../../benches/checked_div_rem/ffi/ffi.rs"]
#[allow(dead_code)] // This fixture uses diagnostics and transactional calls, not allocator timing.
mod ffi;
#[path = "../../../spirv/tests/checked_div_rem/graph/graph.rs"]
mod graph;
#[path = "../../../cpu/tests/wide_div_rem/oracle/oracle.rs"]
#[allow(dead_code)] // Canonical carriers and full Q/R oracle/goldens are reused independently.
mod oracle;
#[path = "../../../cpu/tests/wide_div_rem/source/source.rs"]
mod source;
#[rustfmt::skip]
use pcu_facade::{global, PcuCheckedIntegerDivision, PcuHostArgument, PcuHostKernelBackend,
    PcuPreparedHostKernel, PcuExecutionFaultKind, PcuI256, PcuU256, PcuI512, PcuU512};
use oracle::Wide;
use fusion_pcu_vulkan::{PcuVulkanBackend, PcuVulkanError};
fn diagnostics<T: Wide>(backend: &PcuVulkanBackend, identity: pcu_facade::PcuStableDeviceIdentity) {
    let vectors = oracle::goldens::<T>();
    let left: Vec<_> = vectors.iter().map(|row| row.0).collect();
    let right: Vec<_> = vectors.iter().map(|row| row.1).collect();
    let sentinel = oracle::small::<T>(77);
    let mut q = vec![sentinel; vectors.len() + 3];
    let mut r = q.clone();
    let mut statuses = vec![u32::MAX; vectors.len()];
    let mut native = ffi::NativeDivRem::new(
        identity,
        u32::try_from(vectors.len()).unwrap(),
        T::SIGNED,
        T::HOST_SIZE,
    )
    .unwrap();
    native
        .diagnostics(
            ffi::bytes(&left),
            ffi::bytes(&right),
            ffi::bytes_mut(&mut q),
            ffi::bytes_mut(&mut r),
            &mut statuses,
        )
        .unwrap();
    let mut first = None;
    for (lane, (_, _, want)) in vectors.iter().enumerate() {
        assert_eq!(oracle::evaluate(left[lane], right[lane]), *want);
        match want {
            Ok(pair) => {
                assert_eq!(statuses[lane], 0);
                assert_eq!((q[lane], r[lane]), *pair);
            }
            Err(kind) => {
                assert_eq!(
                    statuses[lane],
                    if *kind == PcuExecutionFaultKind::DivideByZero {
                        4
                    } else {
                        5
                    }
                );
                first.get_or_insert((lane as u64, *kind));
            }
        }
    }
    assert_eq!(
        (&q[vectors.len()..], &r[vectors.len()..]),
        (&[sentinel; 3][..], &[sentinel; 3][..])
    );
    q.fill(sentinel);
    r.fill(sentinel);
    let mut plan = graph::Graph::new(T::TYPE, u32::try_from(vectors.len()).unwrap())
        .with(|k| backend.prepare_host_kernel(k).unwrap());
    let error = plan
        .call(&mut [
            PcuHostArgument::read(graph::INPUTS[0], &left),
            PcuHostArgument::read(graph::INPUTS[1], &right),
            PcuHostArgument::read_write(graph::OUTPUTS[0], &mut q),
            PcuHostArgument::read_write(graph::OUTPUTS[1], &mut r),
        ])
        .unwrap_err();
    assert!(
        matches!(error, PcuVulkanError::Fault(f) if Some((f.invocation_id,f.kind))==first && !f.recovered)
    );
    assert!(q.iter().chain(&r).all(|value| *value == sentinel));
}
fn source_proof<T: Wide + PcuCheckedIntegerDivision>(backend: &PcuVulkanBackend) {
    let valid: Vec<_> = oracle::goldens::<T>()
        .into_iter()
        .filter(|row| row.2.is_ok())
        .take(65)
        .collect();
    let sentinel = oracle::small::<T>(77);
    let mut q = vec![sentinel; 68];
    let mut r = q.clone();
    let mut prepared = source::direct_prepare::<T, 65, _>(backend).unwrap();
    for bank in 0..3 {
        let left: Vec<_> = (0..65).map(|i| valid[(i + bank) % 65].0).collect();
        let mut right: Vec<_> = (0..65).map(|i| valid[(i + bank) % 65].1).collect();
        for route in 0..3 {
            match route {
                0 => prepared(&left, &right, &mut q, &mut r).unwrap(),
                1 => source::direct::<T, 65>(&left, &right, &mut q, &mut r).unwrap(),
                _ => source::grid::<T, 65>(&left, &right, &mut q, &mut r).unwrap(),
            }
            for lane in 0..65 {
                assert_eq!(
                    (q[lane], r[lane]),
                    oracle::evaluate(left[lane], right[lane]).unwrap()
                );
            }
            assert_eq!(
                (&q[65..], &r[65..]),
                (&[sentinel; 3][..], &[sentinel; 3][..])
            );
        }
        for lane in [0, 2, 64] {
            let old = right[lane];
            right[lane] = oracle::small(0);
            let before = (q.clone(), r.clone());
            let fault = source::grid::<T, 65>(&left, &right, &mut q, &mut r)
                .unwrap_err()
                .arithmetic_fault()
                .unwrap();
            assert_eq!(
                (fault.kind, fault.invocation_id, fault.recovered),
                (PcuExecutionFaultKind::DivideByZero, lane as u64, false)
            );
            assert_eq!((&q, &r), (&before.0, &before.1));
            right[lane] = old;
            source::grid::<T, 65>(&left, &right, &mut q, &mut r).unwrap();
        }
        let before = (q.clone(), r.clone());
        assert!(source::direct::<T, 65>(&left, &right, &mut q, &mut r[..64]).is_err());
        assert_eq!((&q, &r), (&before.0, &before.1));
        prepared(&left, &right, &mut q, &mut r).unwrap();
    }
}
#[test]
#[ignore = "requires physical Vulkan device; full-width software division/transaction/source proof"]
fn six_wide_exact_quotient_remainder_and_joint_publication() {
    let (backend, identity) = device::selected();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })
    .unwrap();
    macro_rules! widths {($($ty:ty),+)=>{$(diagnostics::<$ty>(&backend,identity);source_proof::<$ty>(&backend);)+};}
    widths!(i128, u128, PcuI256, PcuU256, PcuI512, PcuU512);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
