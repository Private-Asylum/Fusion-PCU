//! Repeated used input and unread foreign carrier retain exact owner/publication laws.
#[path = "../../../../cpu/tests/scalar_tensor/source/source.rs"]
#[allow(dead_code)] // Consumed-owner companion is covered by canonical transport fixtures.
mod owned;
#[rustfmt::skip]
use super::{
    global,
    oracle,
    source,
    Wide,
    N,
    PcuExecutionFaultKind,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};

#[pcu_facade::pcu(invocations=N,crate_path=::pcu_facade,flag(deterministic))]
fn portable_square<T: pcu_facade::PcuCheckedInteger, const N: usize>(
    unused: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = context.global_invocation_id;
    output[id] = input[id] * input[id];
}
#[pcu_facade::pcu(invocations=N,crate_path=::pcu_facade,flag(deterministic))]
fn portable_double<T: pcu_facade::PcuCheckedInteger, const N: usize>(
    output: &mut [T],
    input: &[T],
) {
    let id = context.global_invocation_id;
    output[id] = input[id] + input[id];
}
#[pcu_facade::pcu(invocations=N,crate_path=::pcu_facade,flag(deterministic),flag(clamp_range))]
fn portable_clamp<T: pcu_facade::PcuCheckedInteger, const N: usize>(output: &mut [T], input: &[T]) {
    let id = context.global_invocation_id;
    output[id] = input[id] + input[id];
}
#[pcu_facade::pcu(invocations=3,crate_path=::pcu_facade,flag(deterministic))]
fn portable_grid<T: pcu_facade::PcuCheckedInteger, const N: usize>(
    unused: &[T],
    output: &mut [T],
    input: &[T],
) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = input[0] - input[id];
        id += stride;
    }
}

fn width<T: Wide>() {
    let one = [oracle::small::<T>(1); N];
    let max = [oracle::maximum::<T>(); N];
    let sentinel = oracle::small::<T>(77);
    let bank = [sentinel; N + 3];
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    let ignored = owned::identity(&one).unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })
    .unwrap();
    let input = owned::identity(&one).unwrap();
    let large = owned::identity(&max).unwrap();
    let mut output = owned::identity(&bank).unwrap();
    let sibling = owned::identity(&output).unwrap();
    let mut observed = bank;
    if super::portable() {
        portable_square::<T, N>(&ignored, &mut output, &input).unwrap();
    } else {
        source::squared::<T, N>(&ignored, &mut output, &input).unwrap();
    }
    output.read_into(&mut observed).unwrap();
    assert_eq!(&observed[..N], &one);
    assert_eq!(&observed[N..], &bank[N..]);
    let fatal = if super::portable() {
        portable_double::<T, N>(&mut output, &large)
    } else {
        source::doubled::<T, N>(&mut output, &large)
    }
    .unwrap_err()
    .arithmetic_fault()
    .unwrap();
    assert!(!fatal.recovered);
    assert_eq!(fatal.invocation_id, 0);
    assert_eq!(fatal.kind, PcuExecutionFaultKind::ArithmeticOverflow);
    output.read_into(&mut observed).unwrap();
    assert_eq!(&observed[..N], &one);
    let notice = if super::portable() {
        portable_clamp::<T, N>(&mut output, &large)
    } else {
        source::doubled_clamp::<T, N>(&mut output, &large)
    }
    .unwrap_err()
    .arithmetic_fault()
    .unwrap();
    assert!(notice.recovered);
    assert_eq!(notice.invocation_id, 0);
    output.read_into(&mut observed).unwrap();
    assert_eq!(&observed[..N], &max);
    assert_eq!(&observed[N..], &bank[N..]);
    if super::portable() {
        portable_grid::<T, N>(&ignored, &mut output, &input).unwrap();
    } else {
        source::grid::<T, N>(&ignored, &mut output, &input).unwrap();
    }
    output.read_into(&mut observed).unwrap();
    assert_eq!(&observed[..N], &[oracle::small::<T>(0); N]);
    assert_eq!(&observed[N..], &bank[N..]);
    sibling.read_into(&mut observed).unwrap();
    assert_eq!(observed, bank);
    drop(ignored);
    drop(input);
    global::clear_thread_cache().unwrap();
    output.read_into(&mut observed).unwrap();
    assert_eq!(&observed[..N], &[oracle::small::<T>(0); N]);
}
#[test]
#[ignore = "requires actual Vulkan GPU and coordinated exclusive correctness window"]
fn fourteen_width_repeated_resident_ignored_cpu_owner_and_private_fatal() {
    macro_rules! widths {($($t:ty),+)=>{$(width::<$t>();)+};}
    widths!(
        i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, PcuI256, PcuU256, PcuI512, PcuU512
    );
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
