//! Actual source-role borrows preserve both public owners until private division completes.
#[path = "../scalar_transport/device/device.rs"]
mod device;
#[path = "../../benches/checked_div_rem/ffi/ffi.rs"]
#[allow(dead_code)] // Only independent native resident publication is exercised by this fixture.
mod native;
#[path = "../../../cpu/tests/wide_div_rem/oracle/oracle.rs"]
#[allow(dead_code)] // Independent byte arithmetic is shared with the full-bit host proof.
mod oracle;
#[path = "../../../cpu/tests/scalar_tensor/source/source.rs"]
#[allow(dead_code)]
mod owned;
#[path = "../../../cpu/tests/div_rem_roles/source/source.rs"]
#[allow(dead_code)] // The scalar-source schema is covered by the prepared resident fixture.
mod source;
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuCheckedIntegerDivision,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuNumericalMode,
    PcuTensor,
    PcuBindingRef,
    PcuHostArgument,
};
#[rustfmt::skip]
use fusion_pcu_vulkan::{
    PcuVulkanArgument,
    PcuVulkanBackend,
    PcuVulkanError,
};
use oracle::Wide;

const N: usize = 5;
const STORAGE: usize = N + 3;

fn invoke<T: PcuCheckedIntegerDivision>(
    kind: usize,
    input: &PcuTensor<T>,
    right: &[T],
    quotient: &mut PcuTensor<T>,
    remainder: &mut PcuTensor<T>,
    unused: &PcuTensor<T>,
) -> Result<(), PcuExecutionError> {
    match kind {
        0 => source::repeated::<T, N>(quotient, remainder, input),
        1 => source::unused::<T, N>(unused, input, remainder, quotient),
        2 => source::reordered::<T, N>(right, remainder, input, quotient),
        3 => source::mixed::<T, N>(quotient, input, remainder),
        _ => source::grid::<T, N>(remainder, unused, quotient, input),
    }
}

fn expected<T: Wide>(kind: usize, input: &[T], right: &[T]) -> ([T; STORAGE], [T; STORAGE]) {
    let sentinel = oracle::small(99);
    let mut quotient = [sentinel; STORAGE];
    let mut remainder = quotient;
    for lane in 0..N {
        let (a, b) = match kind {
            0 | 1 => (input[lane], input[lane]),
            2 => (input[lane], right[lane]),
            3 => (input[lane], input[0]),
            _ => (input[0], input[lane]),
        };
        (quotient[lane], remainder[lane]) = oracle::evaluate(a, b).unwrap();
    }
    (quotient, remainder)
}

fn read<T: Wide>(owner: &PcuTensor<T>) -> [T; STORAGE] {
    let mut result = [oracle::small(99); STORAGE];
    owner.read_into(&mut result).unwrap();
    result
}

fn mixed_outputs<T: Wide + PcuCheckedIntegerDivision>(
    input: &PcuTensor<T>,
    right: &[T],
    bad_input: &PcuTensor<T>,
    bad_right: &[T],
    quotient: &mut PcuTensor<T>,
    remainder: &mut PcuTensor<T>,
    wanted: ([T; STORAGE], [T; STORAGE]),
) {
    let bank = [oracle::small(99); STORAGE];
    let mut host = bank;
    assert!(source::reordered::<T, N>(bad_right, &mut host, bad_input, quotient).is_err());
    assert_eq!(host, bank);
    assert_eq!(read(quotient), wanted.0);
    assert!(source::reordered::<T, N>(bad_right, remainder, bad_input, &mut host).is_err());
    assert_eq!(host, bank);
    assert_eq!(read(remainder), wanted.1);
    assert!(source::reordered::<T, N>(right, remainder, input, &mut host[..N - 1]).is_err());
    assert_eq!(host, bank);
    assert_eq!(read(remainder), wanted.1);
    source::reordered::<T, N>(right, remainder, input, &mut host).unwrap();
    assert_eq!(host, wanted.0);
    assert_eq!(read(remainder), wanted.1);
    host = bank;
    source::reordered::<T, N>(right, &mut host, input, quotient).unwrap();
    assert_eq!(host, wanted.1);
    assert_eq!(read(quotient), wanted.0);
}

fn overflow_bank<T: Wide>(kind: usize) -> ([T; STORAGE], [T; STORAGE]) {
    let mut input = [oracle::small(7); STORAGE];
    let mut right = [oracle::small(3); STORAGE];
    let minus_one = T::from_bytes([255; 64]);
    match kind {
        2 => {
            input[2] = oracle::minimum();
            right[2] = minus_one;
        }
        3 => {
            input[0] = minus_one;
            input[2] = oracle::minimum();
        }
        _ => {
            input[0] = oracle::minimum();
            input[2] = minus_one;
        }
    }
    (input, right)
}

fn verify<T: Wide + PcuCheckedIntegerDivision>(mode: PcuNumericalMode) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    // A real foreign owner is never consulted for an authoritative unread declaration.
    let foreign = owned::identity(&[oracle::small::<T>(37); STORAGE]).unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        numerical_mode: mode,
        ..Default::default()
    })
    .unwrap();
    for kind in 0..5 {
        verify_role::<T>(kind, &foreign);
    }
    drop(foreign);
    global::clear_thread_cache().unwrap();
}

fn verify_role<T: Wide + PcuCheckedIntegerDivision>(kind: usize, foreign: &PcuTensor<T>) {
    let sentinel = oracle::small(99);
    let bank = [sentinel; STORAGE];
    let mut values = [oracle::small(7); STORAGE];
    values[0] = oracle::small(3);
    values[2] = oracle::maximum();
    let right = [oracle::small(3); STORAGE];
    global::clear_thread_cache().unwrap();
    let input = owned::identity(&values).unwrap();
    let mut quotient = owned::identity(&bank).unwrap();
    let mut remainder = owned::identity(&bank).unwrap();
    let sibling = owned::identity(&quotient).unwrap();
    let mut bad = values;
    bad[if kind == 3 { 0 } else { 2 }] = oracle::small(0);
    let bad_input = owned::identity(&bad).unwrap();
    let (overflow_values, overflow_right) = overflow_bank::<T>(kind);
    let overflow_input = if T::SIGNED && kind >= 2 {
        Some(owned::identity(&overflow_values).unwrap())
    } else {
        None
    };
    let wanted = expected(kind, &values, &right);
    invoke(kind, &input, &right, &mut quotient, &mut remainder, foreign).unwrap();
    assert_eq!((read(&quotient), read(&remainder)), wanted);
    assert_eq!(read(&sibling), bank);
    if let Some(overflow_input) = overflow_input {
        let error = invoke(
            kind,
            &overflow_input,
            &overflow_right,
            &mut quotient,
            &mut remainder,
            foreign,
        )
        .unwrap_err();
        let fault = error.arithmetic_fault().unwrap();
        assert_eq!(
            (fault.invocation_id, fault.kind, fault.recovered),
            (2, PcuExecutionFaultKind::SignedDivisionOverflow, false)
        );
        assert_eq!((read(&quotient), read(&remainder)), wanted);
    }
    // Escaping owners survive cache destruction and retain the originating native root.
    global::clear_thread_cache().unwrap();
    invoke(kind, &input, &right, &mut quotient, &mut remainder, foreign).unwrap();
    assert_eq!((read(&quotient), read(&remainder)), wanted);
    let mut bad_right = right;
    if kind == 2 {
        bad_right[2] = oracle::small(0);
    }
    let error = invoke(
        kind,
        &bad_input,
        &bad_right,
        &mut quotient,
        &mut remainder,
        foreign,
    )
    .unwrap_err();
    let fault = error.arithmetic_fault().unwrap();
    assert_eq!(fault.kind, PcuExecutionFaultKind::DivideByZero);
    assert_eq!(fault.invocation_id, if kind == 3 { 0 } else { 2 });
    assert!(!fault.recovered);
    assert_eq!((read(&quotient), read(&remainder)), wanted);
    assert_eq!(read(&sibling), bank);
    // One host and one resident destination share the same whole-call publication boundary.
    if kind == 2 {
        mixed_outputs(
            &input,
            &right,
            &bad_input,
            &bad_right,
            &mut quotient,
            &mut remainder,
            wanted,
        );
    }
    invoke(kind, &input, &right, &mut quotient, &mut remainder, foreign).unwrap();
    assert_eq!((read(&quotient), read(&remainder)), wanted);
}

#[test]
#[ignore = "requires physical Vulkan GPU and exclusive untimed correctness window"]
fn fourteen_width_genuine_source_resident_roles_preserve_joint_publication() {
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        macro_rules! widths {($($ty:ty),+)=>{$(verify::<$ty>(mode);)+};}
        widths!(
            i8,
            u8,
            i16,
            u16,
            i32,
            u32,
            i64,
            u64,
            i128,
            u128,
            pcu_facade::PcuI256,
            pcu_facade::PcuU256,
            pcu_facade::PcuI512,
            pcu_facade::PcuU512
        );
    }
    global::use_defaults().unwrap();
}

fn scalar<T: Wide + PcuCheckedIntegerDivision>(backend: &PcuVulkanBackend) {
    let bindings = source::scalar_bindings::<T>();
    let graph = source::scalar_ir::<T, N>(&bindings).unwrap();
    let mut prepared = backend.prepare_mixed_kernel(&graph.ir()).unwrap();
    let bank = [oracle::small::<T>(99); STORAGE];
    let left = [oracle::maximum::<T>()];
    let right = [oracle::small::<T>(3)];
    let left_owner = backend.upload_owned(&left).unwrap();
    let right_owner = backend.upload_owned(&right).unwrap();
    let mut quotient = backend.upload_owned(&bank).unwrap();
    let mut remainder = backend.upload_owned(&bank).unwrap();
    let (q, r) = oracle::evaluate(left[0], right[0]).unwrap();
    let mut expected_q = bank;
    let mut expected_r = bank;
    expected_q[..N].fill(q);
    expected_r[..N].fill(r);
    prepared
        .call(&mut [
            remainder.write_argument(PcuBindingRef::new(0, 3)),
            left_owner.read_argument(PcuBindingRef::new(0, 0)),
            quotient.write_argument(PcuBindingRef::new(0, 1)),
            right_owner.read_argument(PcuBindingRef::new(0, 2)),
        ])
        .unwrap();
    assert!(prepared.last_call_may_have_written());
    assert!(!prepared.last_call_completion_uncertain());
    let mut q_read = bank;
    let mut r_read = bank;
    quotient.read_into(&mut q_read).unwrap();
    remainder.read_into(&mut r_read).unwrap();
    assert_eq!((q_read, r_read), (expected_q, expected_r));
    let zero = backend.upload_owned(&[oracle::small::<T>(0)]).unwrap();
    let error = prepared
        .call(&mut [
            left_owner.read_argument(PcuBindingRef::new(0, 0)),
            quotient.write_argument(PcuBindingRef::new(0, 1)),
            zero.read_argument(PcuBindingRef::new(0, 2)),
            remainder.write_argument(PcuBindingRef::new(0, 3)),
        ])
        .unwrap_err();
    assert!(matches!(error, PcuVulkanError::Fault(fault)
        if fault.invocation_id == 0 && fault.kind == PcuExecutionFaultKind::DivideByZero));
    assert!(!prepared.last_call_may_have_written());
    assert!(!prepared.last_call_completion_uncertain());
    quotient.read_into(&mut q_read).unwrap();
    remainder.read_into(&mut r_read).unwrap();
    assert_eq!((q_read, r_read), (expected_q, expected_r));
    let empty: [T; 0] = [];
    assert!(
        prepared
            .call(&mut [
                PcuVulkanArgument::host(PcuHostArgument::read(PcuBindingRef::new(0, 0), &empty)),
                quotient.write_argument(PcuBindingRef::new(0, 1)),
                right_owner.read_argument(PcuBindingRef::new(0, 2)),
                remainder.write_argument(PcuBindingRef::new(0, 3)),
            ])
            .is_err()
    );
    assert!(!prepared.last_call_may_have_written());
    let mut host = bank;
    prepared
        .call(&mut [
            left_owner.read_argument(PcuBindingRef::new(0, 0)),
            quotient.write_argument(PcuBindingRef::new(0, 1)),
            right_owner.read_argument(PcuBindingRef::new(0, 2)),
            PcuVulkanArgument::host(PcuHostArgument::read_write(
                PcuBindingRef::new(0, 3),
                &mut host,
            )),
        ])
        .unwrap();
    assert_eq!(host, expected_r);
    quotient.read_into(&mut q_read).unwrap();
    assert_eq!(q_read, expected_q);
}

#[test]
#[ignore = "requires physical Vulkan GPU and exclusive untimed correctness window"]
fn fourteen_width_scalar_native_broadcast_and_dual_commit() {
    let (backend, _) = device::selected();
    macro_rules! widths {($($ty:ty),+)=>{$(scalar::<$ty>(&backend);)+};}
    widths!(
        i8,
        u8,
        i16,
        u16,
        i32,
        u32,
        i64,
        u64,
        i128,
        u128,
        pcu_facade::PcuI256,
        pcu_facade::PcuU256,
        pcu_facade::PcuI512,
        pcu_facade::PcuU512
    );
}

fn native_publication<T: Wide>(identity: pcu_facade::PcuStableDeviceIdentity) {
    let mut input = [oracle::small::<T>(7); STORAGE];
    input[0] = oracle::small(3);
    input[2] = oracle::maximum();
    let right = [oracle::small::<T>(3); STORAGE];
    for kind in 0..5 {
        let mut bad = input;
        let mut bad_right = right;
        if kind == 2 {
            bad_right[2] = oracle::small(0);
        } else {
            bad[if kind == 3 { 0 } else { 2 }] = oracle::small(0);
        }
        native_failure::<T>(
            identity,
            kind,
            input,
            right,
            bad,
            bad_right,
            PcuExecutionFaultKind::DivideByZero,
        );
        if T::SIGNED && kind >= 2 {
            let (overflow_input, overflow_right) = overflow_bank::<T>(kind);
            native_failure::<T>(
                identity,
                kind,
                input,
                right,
                overflow_input,
                overflow_right,
                PcuExecutionFaultKind::SignedDivisionOverflow,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)] // Exact independent two-bank inputs and expected terminal fault are explicit fixture data.
fn native_failure<T: Wide>(
    identity: pcu_facade::PcuStableDeviceIdentity,
    kind: usize,
    input: [T; STORAGE],
    right: [T; STORAGE],
    bad: [T; STORAGE],
    bad_right: [T; STORAGE],
    fault_kind: PcuExecutionFaultKind,
) {
    let initial = [oracle::small::<T>(99); STORAGE];
    let wanted = expected(kind, &input, &right);
    let mut owner = native::resident::NativeResidentDivRem::new(
        identity,
        u32::try_from(N).unwrap(),
        T::SIGNED,
        T::HOST_SIZE,
        kind,
        [
            [native::bytes(&input), native::bytes(&bad)],
            [native::bytes(&right), native::bytes(&bad_right)],
        ],
        [native::bytes(&initial), native::bytes(&initial)],
    )
    .unwrap();
    let mut q = initial;
    let mut r = initial;
    assert!(owner.call(0).unwrap().is_none());
    owner
        .read([native::bytes_mut(&mut q), native::bytes_mut(&mut r)])
        .unwrap();
    assert_eq!((q, r), wanted);
    let fault = owner.call(1).unwrap().unwrap();
    assert_eq!(fault.kind, fault_kind);
    assert_eq!(
        fault.invocation_id,
        if kind == 3 && fault_kind == PcuExecutionFaultKind::DivideByZero {
            0
        } else {
            2
        }
    );
    assert!(!fault.recovered);
    owner
        .read([native::bytes_mut(&mut q), native::bytes_mut(&mut r)])
        .unwrap();
    assert_eq!((q, r), wanted);
    assert!(owner.call(0).unwrap().is_none());
    owner
        .read([native::bytes_mut(&mut q), native::bytes_mut(&mut r)])
        .unwrap();
    assert_eq!((q, r), wanted);
}

#[test]
#[ignore = "requires physical Vulkan GPU and exclusive untimed correctness window"]
fn fourteen_width_independent_native_dual_commit_faults_and_retry() {
    let (_, identity) = device::selected();
    macro_rules! widths {($($ty:ty),+)=>{$(native_publication::<$ty>(identity);)+};}
    widths!(
        i8,
        u8,
        i16,
        u16,
        i32,
        u32,
        i64,
        u64,
        i128,
        u128,
        pcu_facade::PcuI256,
        pcu_facade::PcuU256,
        pcu_facade::PcuI512,
        pcu_facade::PcuU512
    );
}
