//! Native exact22 retained integer primitive source, schema, status-free bit publication and immutable lifetimes.
#[path = "graph/graph.rs"]
mod graph;
#[path = "../encoded_carrier/support/support.rs"]
mod oracle;
#[path = "prefix/prefix.rs"]
mod prefix;
#[path = "source/source.rs"]
mod source;
#[rustfmt::skip]
use pcu_facade::{PcuPreparedHostKernel,PcuHostArgument,PcuBindingRef,
    PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,PcuF128Bits,PcuF256Bits,
    PcuU256,PcuI256,PcuU512,PcuI512};
#[rustfmt::skip]
use fusion_pcu_mlx::{MlxRuntime,MlxCarrierPlan,MlxError};
use oracle::{Sample, compare};
fn qualify<T: Sample>() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let foreign = runtime.open_gpu(0).unwrap();
    let sentinel = T::sample(91);
    let mut dense = source::copy_prepare::<T, 65, _>(&session).unwrap();
    let mut grid = source::grid_prepare::<T, 65, _>(&session).unwrap();
    let mut broadcast = source::broadcast_prepare::<T, 65, _>(&session).unwrap();
    let mut output = [sentinel; 68];
    for phase in [0_u8, 23, 71] {
        let input: [T; 65] = std::array::from_fn(|index| {
            T::sample(u8::try_from(index).unwrap().wrapping_add(phase))
        });
        dense(&input, &mut output).unwrap();
        compare(&output[..65], &input);
        compare(&output[65..], &[sentinel; 3]);
        grid(&input, &mut output).unwrap();
        compare(&output[..65], &input);
        broadcast(&input[0], &mut output).unwrap();
        compare(&output[..65], &[input[0]; 65]);
        for scalar in [false, true] {
            let mut prepared = graph::fixture::<T, _>(65, scalar, true, |ir| {
                session.prepare_carrier_host_kernel(ir)
            })
            .unwrap();
            let values = if scalar { &input[..1] } else { &input[..] };
            prepared
                .call(&mut [
                    PcuHostArgument::read_write(PcuBindingRef::new(4, 1), &mut output),
                    PcuHostArgument::read(PcuBindingRef::new(2, 3), values),
                ])
                .unwrap();
            let expected = if scalar { [input[0]; 65] } else { input };
            compare(&output[..65], &expected);
            compare(&output[65..], &[sentinel; 3]);
            let resident = session.upload_encoded(values).unwrap();
            let old = prepared.execute_resident(&resident).unwrap().into_parts().0;
            let other = foreign.upload_encoded(values).unwrap();
            assert!(matches!(
                prepared.execute_resident(&other),
                Err(MlxError::ForeignSession)
            ));
            assert!(!prepared.last_call_may_have_written());
            let mut native = session
                .prepare_carrier_control(T::TYPE, 65, scalar)
                .unwrap();
            native.call(values, &mut output).unwrap();
            compare(&output[..65], &expected);
            let sibling = native.execute_resident(&resident).unwrap().into_parts().0;
            drop(prepared);
            drop(native);
            drop(resident);
            old.read_into(&mut output).unwrap();
            compare(&output[..65], &expected);
            sibling.read_into(&mut output).unwrap();
            compare(&output[..65], &expected);
        }
        let previous = output;
        assert!(dense(&input[..64], &mut output).is_err());
        compare(&output, &previous);
        assert!(broadcast(&input[0], &mut output[..64]).is_err());
        compare(&output, &previous);
        dense(&input, &mut output).unwrap();
        compare(&output[..65], &input);
    }
}
#[test]
#[ignore = "Requires genuine MLX GPU retained integer primitive for all22 logical carriers and scalar broadcasts."]
fn twenty_two_carrier_source_maps_and_lifetimes() {
    macro_rules! all {($($ty:ty),+) => {$(qualify::<$ty>();)+};}
    all!(
        u8,
        i8,
        u16,
        i16,
        u32,
        i32,
        u64,
        i64,
        u128,
        i128,
        PcuU256,
        PcuI256,
        PcuU512,
        PcuI512,
        PcuF16Bits,
        PcuBf16Bits,
        PcuF8E4M3FnBits,
        PcuF8E5M2Bits,
        f32,
        f64,
        PcuF128Bits,
        PcuF256Bits
    );
}
#[test]
fn cold_copy_schema_rejects_portable_and_unsafe_physical_extents() {
    graph::fixture::<PcuU512, _>(3, false, false, |ir| {
        assert!(MlxCarrierPlan::assess(ir).is_ok());
        let mut portable = *ir;
        portable
            .numerical_requirements
            .numerical_options
            .reproducibility = pcu_facade::PcuReproducibility::PortableV1;
        assert!(MlxCarrierPlan::assess(&portable).is_err());
    });
    graph::fixture::<PcuU512, _>(u32::MAX, false, false, |ir| {
        assert!(MlxCarrierPlan::assess(ir).is_err());
    });
}

fn offers<T: Sample>(discovery: &fusion_pcu_mlx::MlxDiscovery) {
    use pcu_facade::{
        PcuDeviceActivation, PcuDeviceIdentity, PcuImplementationOffers, PcuImplementationRequest,
        PcuCostBoundary, PcuHostKernelBackend,
    };
    let reference = discovery.device_reference(0).unwrap();
    let device = PcuDeviceIdentity::from_device_ref(reference).unwrap();
    let session = discovery.open_device(reference).unwrap();
    for broadcast in [false, true] {
        graph::fixture::<T, _>(3, broadcast, true, |ir| {
            let request = PcuImplementationRequest {
                device,
                executor: fusion_pcu_mlx::MLX_EXECUTOR,
                requirements: ir.numerical_requirements,
                boundary: PcuCostBoundary::Host,
                operation: ir,
            };
            let mut output = [None];
            assert_eq!(
                discovery
                    .implementation_offers(&request, &mut output)
                    .unwrap(),
                1
            );
            let offer = output[0].unwrap();
            offer.validate_request(&request).unwrap();
            assert_eq!(offer.implementation.revision, 0x0003_0020_0003_0400);
            assert_eq!(offer.workspace_bytes, None);
            assert_eq!(
                offer.cost,
                pcu_facade::PcuImplementationCost::unknown(PcuCostBoundary::Host)
            );
            for boundary in [
                PcuCostBoundary::Resident,
                PcuCostBoundary::HostInputsResidentOutput,
                PcuCostBoundary::MixedInputsResidentOutput,
            ] {
                let resident = PcuImplementationRequest {
                    boundary,
                    ..request
                };
                assert_eq!(
                    discovery
                        .implementation_offers(&resident, &mut output)
                        .unwrap(),
                    0
                );
                assert_eq!(output, [None]);
            }
            let mut portable = *ir;
            portable
                .numerical_requirements
                .numerical_options
                .reproducibility = pcu_facade::PcuReproducibility::PortableV1;
            let portable_request = PcuImplementationRequest {
                operation: &portable,
                requirements: portable.numerical_requirements,
                ..request
            };
            assert_eq!(
                discovery
                    .implementation_offers(&portable_request, &mut output)
                    .unwrap(),
                0
            );
            let mut prepared = session.prepare_host_kernel(ir).unwrap();
            assert!(matches!(
                prepared,
                fusion_pcu_mlx::MlxPreparedDispatchKernel::Carrier(_)
            ));
            assert_eq!(prepared.input_bindings(), [PcuBindingRef::new(2, 3)]);
            let input = [T::sample(71); 3];
            let input = if broadcast { &input[..1] } else { &input[..] };
            let sentinel = T::sample(91);
            let mut result = [sentinel; 5];
            prepared
                .call(&mut [
                    PcuHostArgument::read_write(PcuBindingRef::new(4, 1), &mut result),
                    PcuHostArgument::read(PcuBindingRef::new(2, 3), input),
                ])
                .unwrap();
            compare(&result[..3], &[input[0]; 3]);
            compare(&result[3..], &[sentinel; 2]);
        });
    }
}
#[test]
#[ignore = "Requires actual MLX inventory and retained GPU carrier aggregate across all22 exact tags."]
fn twenty_two_carrier_inventory_and_session_aggregate() {
    let discovery = fusion_pcu_mlx::MlxDiscovery::discover_default().unwrap();
    macro_rules! all {($($ty:ty),+) => {$(offers::<$ty>(&discovery);)+};}
    all!(
        u8,
        i8,
        u16,
        i16,
        u32,
        i32,
        u64,
        i64,
        u128,
        i128,
        PcuU256,
        PcuI256,
        PcuU512,
        PcuI512,
        PcuF16Bits,
        PcuBf16Bits,
        PcuF8E4M3FnBits,
        PcuF8E5M2Bits,
        f32,
        f64,
        PcuF128Bits,
        PcuF256Bits
    );
}
