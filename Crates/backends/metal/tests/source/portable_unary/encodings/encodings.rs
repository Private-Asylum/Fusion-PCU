//! Batched full low-format encodings under explicitly requested Portable headers.
#[rustfmt::skip]
use fusion_pcu_metal::{
    MetalError,
    MetalSession,
    MetalHostKernelError,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuBf16Bits,
    PcuDispatchFloatUnaryOp as Op,
    PcuF16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuFloatUnderflowPolicy as Policy,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuHostDispatchError,
    PcuPreparedHostKernel,
    PcuRangePolicy as Range,
    PcuReproducibility,
};
#[rustfmt::skip]
use super::{
    graph,
    oracle::{
        self,
        Low,
    },
};
const LANES: usize = 257;
fn format<T: Low>() {
    let session = MetalSession::open(0).unwrap();
    let limit = if T::SIGN == 0x80 { 256 } else { 65536 };
    for op in [Op::Neg, Op::Relu] {
        for policy in [
            Policy::IeeeAfterRounding,
            Policy::AllowGradualUnderflow,
            Policy::RejectSubnormalResult,
        ] {
            for range in [Range::Reject, Range::Clamp] {
                let mut prepared = graph::fixture_profile::<T, _>(
                    u32::try_from(LANES).unwrap(),
                    op,
                    policy,
                    range,
                    false,
                    false,
                    |ir| {
                        let mut request = *ir;
                        request
                            .numerical_requirements
                            .numerical_options
                            .reproducibility = PcuReproducibility::PortableV1;
                        session.prepare_host_kernel(&request)
                    },
                )
                .unwrap();
                for offset in (0..limit).step_by(LANES) {
                    let input: [T; LANES] = core::array::from_fn(|lane| {
                        T::from_bits(
                            u16::try_from(if offset + lane < limit {
                                offset + lane
                            } else {
                                0
                            })
                            .unwrap(),
                        )
                    });
                    verify(&mut prepared, &input, op, policy, range);
                }
                // Fatal banks intentionally publish nothing. Independently visit every finite
                // accepted payload in successful banks so a neighboring NaN cannot hide its bits.
                let accepted: Vec<T> = (0..limit)
                    .filter_map(|bits| {
                        let bits = u16::try_from(bits).unwrap();
                        oracle::evaluate::<T>(bits, op, policy)
                            .ok()
                            .and_then(|(_, tiny)| {
                                (!(tiny && range == Range::Reject)).then(|| T::from_bits(bits))
                            })
                    })
                    .collect();
                for chunk in accepted.chunks(LANES) {
                    let input = core::array::from_fn(|lane| {
                        chunk.get(lane).copied().unwrap_or_else(|| T::from_bits(0))
                    });
                    verify(&mut prepared, &input, op, policy, range);
                }
            }
        }
    }
}
fn verify<T: Low>(
    prepared: &mut impl PcuPreparedHostKernel<Error = MetalHostKernelError>,
    input: &[T; LANES],
    op: Op,
    policy: Policy,
    range: Range,
) {
    let sentinel = T::from_bits(17);
    let mut output = [sentinel; LANES + 3];
    let mut expected = output;
    let expected_fault = oracle::native::<T, LANES>(input, &mut expected, op, policy, range);
    let actual = prepared
        .call(&mut [
            PcuHostArgument::read(pcu_facade::PcuBindingRef::new(2, 3), input),
            PcuHostArgument::read_write(pcu_facade::PcuBindingRef::new(4, 1), &mut output),
        ])
        .map_err(|error| match error {
            PcuHostDispatchError::Backend(MetalError::Arithmetic(fault)) => fault,
            other => panic!("unexpected encoding matrix failure {other:?}"),
        });
    assert_eq!(actual, expected_fault, "encoding bank");
    assert_eq!(
        output, expected,
        "exact payload/rollback/tails encoding bank"
    );
}
#[test]
#[ignore = "Requires actual Metal Portable full low-format encoding banks, private statuses and terminal host publication."]
fn four_low_format_portable_full_encoding_payloads_and_earliest_faults() {
    format::<PcuF16Bits>();
    format::<PcuBf16Bits>();
    format::<PcuF8E4M3FnBits>();
    format::<PcuF8E5M2Bits>();
}
