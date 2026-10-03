//! Full resident shapes are frozen cold; logical reads remain the exact prefix or first limb tile.
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxCarrierPlan,
    MlxError,
    MlxRuntime,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuBf16Bits,
    PcuF16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuF128Bits,
    PcuF256Bits,
    PcuHostArgument,
    PcuBindingRef,
    PcuI256,
    PcuI512,
    PcuU256,
    PcuU512,
};
#[rustfmt::skip]
use super::{
    graph,
    oracle::{
        compare,
        Sample,
    },
};
fn cold<T: Sample>() {
    for broadcast in [false, true] {
        graph::fixture::<T, _>(65, broadcast, true, |ir| {
            let plan = MlxCarrierPlan::assess(ir).unwrap();
            let minimum = if broadcast { 1 } else { 65 };
            assert_eq!(plan.assess_input_extents(&[minimum]).unwrap(), minimum);
            assert_eq!(plan.assess_input_extents(&[71]).unwrap(), 71);
            for extents in [
                &[][..],
                &[minimum - 1][..],
                &[71, 71][..],
                &[usize::MAX][..],
            ] {
                assert!(matches!(
                    plan.assess_input_extents(extents),
                    Err(MlxError::InvalidExtent)
                ));
            }
        });
    }
}
macro_rules! all_types {
    ($function:ident) => {
        $function::<u8>();
        $function::<i8>();
        $function::<u16>();
        $function::<i16>();
        $function::<u32>();
        $function::<i32>();
        $function::<u64>();
        $function::<i64>();
        $function::<u128>();
        $function::<i128>();
        $function::<PcuU256>();
        $function::<PcuI256>();
        $function::<PcuU512>();
        $function::<PcuI512>();
        $function::<PcuF16Bits>();
        $function::<PcuBf16Bits>();
        $function::<PcuF8E4M3FnBits>();
        $function::<PcuF8E5M2Bits>();
        $function::<f32>();
        $function::<f64>();
        $function::<PcuF128Bits>();
        $function::<PcuF256Bits>();
    };
}
#[test]
fn twenty_two_carrier_full_capacity_assessment_is_detached() {
    all_types!(cold);
}
#[allow(clippy::too_many_lines)] // One retained exact-shape case verifies cold admission, immutable results, rejection and post-drop lifetimes.
fn qualify<T: Sample>() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let foreign = runtime.open_gpu(0).unwrap();
    let sentinel = T::sample(91);
    for grid in [false, true] {
        for broadcast in [false, true] {
            for full in [68_usize, 71] {
                let mut prepared = graph::fixture::<T, _>(65, broadcast, grid, |ir| {
                    session.prepare_carrier_host_kernel_with_input_extents(ir, &[full])
                })
                .unwrap();
                let mut native = session
                    .prepare_carrier_control_with_input_extent(T::TYPE, 65, broadcast, full)
                    .unwrap();
                assert_eq!(prepared.prepared_input_element_count(), full);
                let minimum = if broadcast { 1 } else { 65 };
                assert_eq!(
                    prepared.input_byte_len(),
                    minimum * usize::from(T::TYPE.bit_width()) / 8
                );
                for phase in [0_u8, 23, 71] {
                    let input: Vec<T> = (0..full)
                        .map(|index| T::sample(u8::try_from(index).unwrap().wrapping_add(phase)))
                        .collect();
                    let resident = session.upload_encoded(&input).unwrap();
                    let expected = if broadcast {
                        vec![input[0]; 65]
                    } else {
                        input[..65].to_vec()
                    };
                    let output = prepared.execute_resident(&resident).unwrap().into_parts().0;
                    let sibling = native.execute_resident(&resident).unwrap().into_parts().0;
                    let mut destination = [sentinel; 68];
                    output.read_into(&mut destination).unwrap();
                    compare(&destination[..65], &expected);
                    compare(&destination[65..], &[sentinel; 3]);
                    sibling.read_into(&mut destination).unwrap();
                    compare(&destination[..65], &expected);
                    let other = foreign.upload_encoded(&input).unwrap();
                    assert!(matches!(
                        prepared.execute_resident(&other),
                        Err(MlxError::ForeignSession)
                    ));
                    assert!(!prepared.last_call_may_have_written());
                    let wrong = session.upload_encoded(&input[..full - 1]).unwrap();
                    assert!(matches!(
                        prepared.execute_resident(&wrong),
                        Err(MlxError::InvalidExtent)
                    ));
                    assert!(!prepared.last_call_may_have_written());
                    let source = PcuHostArgument::read(PcuBindingRef::new(2, 3), &input[..minimum]);
                    assert!(matches!(
                        prepared.execute_encoded_bytes(T::TYPE, source.bytes()),
                        Err(MlxError::InvalidExtent)
                    ));
                    assert!(!prepared.last_call_may_have_written());
                    let mut original = vec![sentinel; full + 3];
                    resident.read_into(&mut original).unwrap();
                    compare(&original[..full], &input);
                    compare(&original[full..], &[sentinel; 3]);
                    drop(resident);
                    output.read_into(&mut destination).unwrap();
                    compare(&destination[..65], &expected);
                    sibling.read_into(&mut destination).unwrap();
                    compare(&destination[..65], &expected);
                    compare(&destination[65..], &[sentinel; 3]);
                }
                let retained = session.upload_encoded(&vec![T::sample(2); full]).unwrap();
                let result = prepared.execute_resident(&retained).unwrap().into_parts().0;
                drop(prepared);
                drop(native);
                drop(retained);
                let mut destination = [sentinel; 68];
                result.read_into(&mut destination).unwrap();
                compare(&destination[..65], &[T::sample(2); 65]);
                compare(&destination[65..], &[sentinel; 3]);
            }
        }
    }
}
#[test]
#[ignore = "required actual MLX22 carrier full-shape prefix/tile kernels and immutable exact-session ownership"]
fn twenty_two_carrier_full_capacity_resident_prefixes_and_lifetimes() {
    all_types!(qualify);
}
