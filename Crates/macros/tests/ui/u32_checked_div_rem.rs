use fusion_pcu_macros::pcu;

extern crate pcu_alias as renamed_pcu;

#[pcu(invocations = N, crate_path = ::renamed_pcu)]
fn checked<const N: usize>(a: &[u32], b: &[u32], q: &mut [u32], r: &mut [u32]) {
    let id = context.global_invocation_id;
    let (quotient, remainder) = pcu::checked_div_rem(a[id], b[id]);
    q[id] = quotient;
    r[id] = remainder;
}

fn main() {}
