use fusion_pcu_macros::pcu;

extern crate pcu_alias as renamed_pcu;

#[pcu(invocations = 8, crate_path = ::renamed_pcu)]
fn checked(a: &[u32], b: &[u32], q: &mut [u32], r: &mut [u32]) {
    let id = context.global_invocation_id;
    let (quotient, remainder) = pcu::checked_div_rem(a[id], b[id]);
    q[id] = quotient;
    q[id] = remainder;
}

fn main() {}
