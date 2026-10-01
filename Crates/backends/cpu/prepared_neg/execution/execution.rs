//! Cold-selected concrete float checking and transactional publication.

#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedFloat,
    PcuExecutionFault,
    PcuFloatUnderflowPolicy,
};
#[rustfmt::skip]
use super::{
    PcuCpuCheckedNeg,
    PcuCpuPreparedNegError,
};

macro_rules! execute {
    ($name:ident, $scalar:ty, $bits:ty, $bytes:literal, $exponent:literal, $negate:ident) => {
        pub(super) fn $name(
            backend: PcuCpuCheckedNeg,
            input: &[u8],
            output: &mut [u8],
            policy: PcuFloatUnderflowPolicy,
        ) -> Result<(), PcuCpuPreparedNegError> {
            // Exact Neg cannot round. Normals cannot fault; the integer-only core oracle
            // resolves zero, nonfinite and subnormal policy before the first output store.
            for (invocation, value) in input.as_chunks::<$bytes>().0.iter().enumerate() {
                let bits = <$bits>::from_ne_bytes(*value);
                let exponent = bits & $exponent;
                if exponent == 0 || exponent == $exponent {
                    <$scalar>::from_bits(bits)
                        .pcu_checked_neg_with_policy(policy)
                        .map_err(|kind| {
                            PcuCpuPreparedNegError::Fault(PcuExecutionFault {
                                recovered: false,
                                kind,
                                invocation_id: u64::try_from(invocation)
                                    .expect("admitted u32 extent"),
                            })
                        })?;
                }
            }
            crate::processor::$negate(backend.processor, backend.implementation, input, output);
            Ok(())
        }
    };
}

execute!(f32, f32, u32, 4, 0x7f80_0000, negate_bytes);
execute!(f64, f64, u64, 8, 0x7ff0_0000_0000_0000, negate_f64_bytes);
