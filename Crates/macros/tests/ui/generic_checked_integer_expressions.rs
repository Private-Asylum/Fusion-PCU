use fusion_pcu_macros::pcu;
extern crate pcu_alias;

#[pcu(invocations = 4, crate_path = ::pcu_alias)]
fn literal<T: pcu_alias::PcuCheckedInteger>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = input[id] + 1;
}

#[pcu(invocations = 4, crate_path = ::pcu_alias)]
fn cast<T: pcu_alias::PcuCheckedInteger>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = input[id] as T;
}

#[pcu(invocations = 4, crate_path = ::pcu_alias)]
fn negate<T: pcu_alias::PcuCheckedInteger>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = -input[id];
}

#[pcu(invocations = 4, crate_path = ::pcu_alias, flag(native_compound), flag(backend_precision))]
fn divide<T: pcu_alias::PcuCheckedInteger>(left: &[T], right: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] / right[id];
}

fn helper<T>(value: T) -> T {
    value
}

#[pcu(invocations = 4, crate_path = ::pcu_alias)]
fn call<T: pcu_alias::PcuCheckedInteger>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = helper(input[id]);
}

fn main() {}
