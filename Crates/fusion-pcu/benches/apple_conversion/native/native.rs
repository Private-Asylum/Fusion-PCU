//! Detached controls share the qualified numerical helper, not source admission.
//! These are NOT independently handwritten arithmetic oracles. Host transport,
//! terminal completion and private readback are included in both controls.
#[rustfmt::skip]
use fusion_pcu::{PcuDispatchCheckedFloatConversion, PcuFloatUnderflowPolicy,
    PcuRangePolicy, PcuScalar, PcuHostArgument, PcuScalarType};
#[rustfmt::skip]
use fusion_pcu_metal::{MetalPreparedFloatConversion, MetalSession};
#[rustfmt::skip]
use fusion_pcu_mlx::{MlxPreparedFloatConversion, MlxSession};
use super::graph::INPUT;
use super::graph::OUTPUT;

pub enum Native {
    Metal {
        session: MetalSession,
        conversion: MetalPreparedFloatConversion,
    },
    Mlx {
        session: MlxSession,
        conversion: MlxPreparedFloatConversion,
    },
}
impl Native {
    pub fn metal(session: &MetalSession, conversion: PcuDispatchCheckedFloatConversion) -> Self {
        Self::Metal {
            session: session.clone(),
            conversion: session
                .prepare_float_conversion(
                    conversion,
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuRangePolicy::Reject,
                    false,
                )
                .unwrap(),
        }
    }
    pub fn mlx(
        session: &MlxSession,
        conversion: PcuDispatchCheckedFloatConversion,
        count: usize,
    ) -> Self {
        Self::Mlx {
            session: session.clone(),
            conversion: session
                .prepare_float_conversion(
                    conversion,
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuRangePolicy::Reject,
                    count,
                    false,
                )
                .unwrap(),
        }
    }
    pub fn call<S: PcuScalar, D: PcuScalar>(
        &self,
        input: &[S],
        output: &mut [D],
        scratch: &mut [u8],
    ) {
        let source = PcuHostArgument::read(INPUT, input);
        let mut destination = PcuHostArgument::read_write(OUTPUT, output);
        match self {
            Self::Metal {
                session,
                conversion,
            } => {
                let input = session.upload_bytes(source.bytes()).unwrap();
                let (result, notice) = conversion
                    .execute_completed(&input, input_count::<D>(scratch))
                    .unwrap();
                assert!(notice.is_none());
                result.read_into_bytes(scratch).unwrap();
            }
            Self::Mlx {
                session,
                conversion,
            } => {
                let input = session.upload_encoded(input).unwrap();
                let (result, notice) = conversion.execute_resident(&input).unwrap().into_parts();
                assert!(notice.is_none());
                input.release().unwrap();
                result.read_bytes_into(scratch).unwrap();
                result.release().unwrap();
            }
        }
        destination.bytes_mut().unwrap()[..scratch.len()].copy_from_slice(scratch);
    }
}
fn input_count<D: PcuScalar>(scratch: &[u8]) -> usize {
    let width = match D::TYPE {
        PcuScalarType::F32 => 4,
        PcuScalarType::F64 => 8,
        _ => unreachable!("fixed F32/F64 benchmark"),
    };
    scratch.len() / width
}
