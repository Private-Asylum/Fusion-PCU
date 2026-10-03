//! Independent binary64 midpoint bit oracle, with core used only for fault-free fixture filtering.
#[path = "../../checked_low_precision/oracle/oracle.rs"]
#[allow(dead_code)] // The common oracle's full public fault verifier serves other fixtures.
mod shared;
pub use shared::Format;
use fusion_pcu::PcuFloatUnderflowPolicy;
pub fn inputs<T: Format>(
    count: usize,
    phase: u32,
    op: u32,
    policy: PcuFloatUnderflowPolicy,
) -> (Vec<T>, Vec<T>, Vec<T>) {
    let (mut a, mut b, mut out) = shared::inputs::<T>(count, phase, op.min(3));
    for lane in 0..count {
        if op == 4 {
            let mag = a[lane].bits() & (T::SIGN - 1);
            if policy == PcuFloatUnderflowPolicy::RejectSubnormalResult
                && a[lane].bits() & T::SIGN == 0
                && mag != 0
                && mag < (1 << T::FRACTION)
            {
                a[lane] = T::one();
            }
            let bits = a[lane].bits();
            out[lane] = T::from(if bits & T::SIGN == 0 && bits != 0 {
                bits
            } else {
                0
            });
        } else {
            if shared::reference(a[lane], b[lane], op, policy).is_err() {
                a[lane] = T::one();
                b[lane] = T::one();
            }
            out[lane] = T::from(shared::independent(a[lane], b[lane], op));
        }
    }
    (a, b, out)
}
