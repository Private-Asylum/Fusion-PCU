//! Native ordered transport preserves bytes and private host publication boundaries.
use super::*;
use super::staging_tests::selected_device;
#[path = "../benches/ordered_transport/oracle/oracle.rs"]
pub(super) mod oracle;
#[path = "../benches/ordered_transport/source/source.rs"]
#[allow(dead_code)]
// Native fixture uses generated source IR; facade source has its own required gate.
pub(super) mod source;
use oracle::Format;
const fn arguments<'a, T: Format>(
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
fn execute<T: Format>(
    backend: &RocmOwnedDispatchBackend,
    ir: &fusion_pcu::PcuDispatchKernelIr<'_>,
) {
    let mut prepared = backend.prepare_host_kernel(ir).unwrap();
    assert!(!prepared.dispatch.requires_checked_fault_word());
    assert!(prepared.fault_word.is_none());
    assert_eq!(prepared.dispatch.binding_schema().len(), 3);
    assert_eq!(prepared.argument_requirements[1].min_required_bytes, 0);
    assert_eq!(
        prepared.argument_requirements[1].access,
        PcuBindingAccess::ReadWrite
    );
    let mut ghost = [];
    let mut stage = vec![T::pattern(251); 67];
    let mut output = vec![T::pattern(251); 67];
    for phase in [3_u8, 19, 127] {
        let input = (0..65)
            .map(|i| T::pattern(phase.wrapping_add(u8::try_from(i).unwrap())))
            .collect::<Vec<_>>();
        prepared
            .call(&mut arguments(&input, &mut ghost, &mut stage, &mut output))
            .unwrap();
        oracle::verify(&input, &stage);
        oracle::verify(&input, &output);
        for slot in [1, 2] {
            assert_eq!(
                prepared.slots[slot].resource.as_ref().unwrap().size_bytes(),
                u64::try_from(65 * size_of::<T>()).unwrap()
            );
        }
        let old_stage = stage.clone();
        let old_output = output.clone();
        assert!(
            prepared
                .call(&mut arguments(
                    &input,
                    &mut ghost,
                    &mut stage,
                    &mut output[..64]
                ))
                .is_err()
        );
        assert!(!prepared.last_call_may_have_written());
        assert!(!prepared.last_call_completion_uncertain());
        for (before, after) in old_stage
            .iter()
            .zip(&stage)
            .chain(old_output.iter().zip(&output))
        {
            assert_eq!(before.encode_le().as_ref(), after.encode_le().as_ref());
        }
        prepared
            .call(&mut arguments(&input, &mut ghost, &mut stage, &mut output))
            .unwrap();
        oracle::verify(&input, &stage);
        oracle::verify(&input, &output);
    }
}
fn width<T: Format>(backend: &RocmOwnedDispatchBackend) {
    let bindings = source::direct_bindings::<T>();
    source::direct_ir::<T, 65>(&bindings)
        .unwrap()
        .with_ir(|ir| execute::<T>(backend, ir));
    source::grid_ir::<T, 65>(&bindings)
        .unwrap()
        .with_ir(|ir| execute::<T>(backend, ir));
}
#[test]
#[ignore = "requires actual authorized AMD GPU"]
fn all22_ordered_transport_retains_exact_bits_and_host_transactions() {
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
