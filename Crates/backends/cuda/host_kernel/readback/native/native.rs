//! Actual copies precede a deterministic late-transfer refusal; no driver loss is injected.
use super::super::*;
use crate::host_kernel::staging_tests::selected_device;
use crate::host_kernel::transport_tests::{oracle, source};
use oracle::Format;

const fn make_arguments<'a, T: Format>(
    input: &'a [T],
    ghost: &'a mut [T],
    stage: &'a mut [T],
    output: &'a mut [T],
) -> [PcuHostArgument<'a>; 4] {
    [
        PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
        PcuHostArgument::read_write(PcuBindingRef::new(0, 1), ghost),
        PcuHostArgument::read_write(PcuBindingRef::new(0, 2), stage),
        PcuHostArgument::read_write(PcuBindingRef::new(0, 3), output),
    ]
}
fn execute<T: Format>(backend: &crate::CudaOwnedDispatchBackend, ir: &PcuDispatchKernelIr<'_>) {
    let mut prepared = backend.prepare_host_kernel(ir).unwrap();
    let input = (0..17)
        .map(|i| T::pattern(u8::try_from(i * 7 + 3).unwrap()))
        .collect::<Vec<_>>();
    let mut ghost = [];
    let mut stage = vec![T::pattern(251); 19];
    let mut output = vec![T::pattern(251); 21];
    prepared.fail_readback_at = Some(2); // Physical ABI: input, stage, output.
    let mut arguments = make_arguments(&input, &mut ghost, &mut stage, &mut output);
    assert!(matches!(
        prepared.call(&mut arguments),
        Err(CudaHostKernelError::Memory(_))
    ));
    assert!(prepared.last_call_may_have_written());
    assert!(!prepared.last_call_completion_uncertain());
    assert!(!prepared.poisoned);
    assert!(
        stage
            .iter()
            .chain(&output)
            .all(|v| v.encode_le().as_ref() == T::pattern(251).encode_le().as_ref())
    );
    assert_eq!(
        prepared.slots[1]
            .resource
            .as_ref()
            .unwrap()
            .device_buffer()
            .allocation
            .host_transfer
            .borrow()[..17 * size_of::<T>()],
        input
            .iter()
            .flat_map(|v| v.encode_le().as_ref().to_vec())
            .collect::<Vec<_>>()
    );
    let pointers = ram_pointers(&prepared);
    prepared.fail_readback_at = None;
    let mut arguments = [
        PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
        PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut ghost),
        PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut stage),
        PcuHostArgument::read_write(PcuBindingRef::new(0, 3), &mut output),
    ];
    prepared.call(&mut arguments).unwrap();
    oracle::verify(&input, &stage);
    oracle::verify(&input, &output);
    // The second output had no readback before refusal: its first allocation is lawful.
    let initialized = ram_pointers(&prepared);
    assert_eq!(&initialized[..2], &pointers[..2]);
    prepared
        .call(&mut make_arguments(
            &input,
            &mut ghost,
            &mut stage,
            &mut output,
        ))
        .unwrap();
    assert_eq!(ram_pointers(&prepared), initialized);
    assert!(prepared.slots.iter().all(|slot| slot.readback.is_none()));
    for slot in &prepared.slots[1..] {
        assert_eq!(
            slot.resource
                .as_ref()
                .unwrap()
                .device_buffer()
                .allocation
                .host_transfer
                .borrow()
                .len(),
            17 * size_of::<T>()
        );
    }
}
fn ram_pointers(prepared: &CudaPreparedHostKernel) -> Vec<*const u8> {
    prepared
        .slots
        .iter()
        .map(|slot| {
            slot.resource
                .as_ref()
                .unwrap()
                .device_buffer()
                .allocation
                .host_transfer
                .borrow()
                .as_ptr()
        })
        .collect()
}
fn width<T: Format>(backend: &crate::CudaOwnedDispatchBackend) {
    let bindings = source::direct_bindings::<T>();
    source::direct_ir::<T, 17>(&bindings)
        .unwrap()
        .with_ir(|ir| execute::<T>(backend, ir));
    source::grid_ir::<T, 17>(&bindings)
        .unwrap()
        .with_ir(|ir| execute::<T>(backend, ir));
}
#[test]
#[ignore = "requires an authorized actual native GPU"]
fn all22_late_private_readback_refusal_preserves_all_host_outputs_and_retry() {
    let (_discovery, backend) = selected_device();
    macro_rules! widths {($($ty:ty),+)=>{$(width::<$ty>(&backend);)+};}
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
        pcu_facade::PcuU512,
        pcu_facade::PcuF16Bits,
        pcu_facade::PcuBf16Bits,
        pcu_facade::PcuF8E4M3FnBits,
        pcu_facade::PcuF8E5M2Bits,
        f32,
        f64,
        pcu_facade::PcuF128Bits,
        pcu_facade::PcuF256Bits
    );
}

#[test]
#[ignore = "requires an authorized actual native GPU"]
fn private_readback_quarantine_retains_destination_under_real_held_lease() {
    let (_discovery, backend) = selected_device();
    let bindings = source::direct_bindings::<f64>();
    source::direct_ir::<f64, 17>(&bindings)
        .unwrap()
        .with_ir(|ir| {
            let mut prepared = backend.prepare_host_kernel(ir).unwrap();
            let input = (0..17)
                .map(|i| f64::pattern(u8::try_from(i + 5).unwrap()))
                .collect::<Vec<_>>();
            let mut ghost = [];
            let mut stage = vec![f64::pattern(251); 19];
            let mut output = vec![f64::pattern(251); 21];
            prepared
                .call(&mut make_arguments(
                    &input,
                    &mut ghost,
                    &mut stage,
                    &mut output,
                ))
                .unwrap();
            let buffer = prepared.slots[2]
                .resource
                .as_ref()
                .unwrap()
                .device_buffer()
                .clone();
            buffer
                .with_access_lease_for_test(|| {
                    assert!(buffer.validate_access_available().is_err());
                    assert!(super::retire_after_transfer_failure(&mut prepared.slots[2]));
                    assert!(prepared.slots[2].resource.is_none());
                    assert!(prepared.slots[2].readback.is_none());
                    // This is the production retirement path under a real lease after known
                    // completion, not an injected driver-loss or genuinely pending-DMA proof.
                    drop(prepared);
                    assert!(buffer.validate_access_available().is_err());
                })
                .unwrap();
            assert!(buffer.validate_access_available().is_ok());
            let mut bytes = vec![0; 17 * size_of::<f64>()];
            buffer.copy_to(&mut bytes).unwrap();
            assert_eq!(
                bytes,
                input
                    .iter()
                    .flat_map(|v| v.to_bits().to_le_bytes())
                    .collect::<Vec<_>>()
            );
        });
}

#[test]
#[ignore = "requires an authorized actual native GPU"]
fn raw_owned_endpoints_block_clone_reuse_and_quarantine_complete_ram_runtime_roots() {
    let (_discovery, backend) = selected_device();
    let mut buffer = backend.allocate(17).unwrap();
    let bytes = std::array::from_fn::<_, 17, _>(|i| u8::try_from(i * 13).unwrap());
    buffer.copy_from(&bytes).unwrap();
    let ram = buffer.allocation.host_transfer.borrow().as_ptr();
    let ticket = buffer.readback_owned_at(0, 17).unwrap();
    assert_eq!(buffer.allocation.host_transfer.borrow().as_ptr(), ram);
    let mut cloned = buffer.clone();
    let mut destination = [251_u8; 17];
    assert!(cloned.copy_from(&[3; 17]).is_err());
    assert!(cloned.copy_to(&mut destination).is_err());
    assert_eq!(destination, [251; 17]);
    ticket.publish_to(&mut destination);
    assert_eq!(destination, bytes);
    drop(ticket);
    cloned.copy_from(&[3; 17]).unwrap();
    cloned.copy_to(&mut destination).unwrap();
    assert_eq!(destination, [3; 17]);
    let weak = std::rc::Rc::downgrade(&buffer.allocation);
    // Known quiescence above. Exercise the identical production unknown-retention path
    // deterministically with a real allocation lease; this does not inject driver loss.
    let lease = buffer.acquire_access().unwrap();
    lease.retain_after_unknown_completion();
    drop(buffer);
    drop(cloned);
    let retained = weak
        .upgrade()
        .expect("actual allocation/runtime/RAM retained by lease");
    assert_eq!(retained.host_transfer.borrow().as_ptr(), ram);
    assert_eq!(&retained.host_transfer.borrow()[..17], &[3; 17]);
    assert_ne!(
        retained.access.state.get(),
        crate::AllocationAccessState::Idle
    );
}
