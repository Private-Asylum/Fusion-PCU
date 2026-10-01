//! Common cold subnormal admission for explicitly backend-defined compound arithmetic.
//!
//! Native compound permission accepts this implementation's documented exception/subnormal
//! behavior under the ordinary default. It does not prove a separately requested gradual
//! result or tight subnormal diagnostic. IEEE754 gradual underflow is a behavior guarantee,
//! distinct from permitting a numerical exception; see `PcuFloatUnderflowPolicy`'s IEEE7.5
//! references. Neither final finiteness nor a successful library call proves that guarantee.

use fusion_pcu::PcuFloatUnderflowPolicy;
use super::TensorUnsupportedReason;

pub(super) const fn assess_underflow(
    policy: Option<PcuFloatUnderflowPolicy>,
) -> Result<(), TensorUnsupportedReason> {
    match policy {
        None | Some(PcuFloatUnderflowPolicy::IeeeAfterRounding) => Ok(()),
        Some(policy) => Err(TensorUnsupportedReason::UnderflowPolicy(policy)),
    }
}
