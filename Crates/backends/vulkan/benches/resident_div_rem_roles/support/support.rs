//! Same retained input banks, exact prefix dual publication, completed readback and preserved tails.
#[rustfmt::skip]
use super::{ffi,oracle,owned,source,SCORES};
use oracle::Wide;
use criterion::Criterion;
#[rustfmt::skip]
use fusion_pcu_vulkan::{PcuVulkanArgument,PcuVulkanBackend,PcuVulkanOwnedBuffer,PcuVulkanPreparedMixed};
#[rustfmt::skip]
use pcu_facade::{
    global,PcuBindingRef,PcuCheckedIntegerDivision,PcuHostArgument,PcuNumericalMode,
    PcuStableDeviceIdentity,PcuTensor,
};
use std::sync::atomic::Ordering;
const N: usize = 5;
const STORAGE: usize = N + 3;
type Banks<T> = [[T; STORAGE]; 2];
type Pair<T> = ([T; STORAGE], [T; STORAGE]);

fn prepare<T: PcuCheckedIntegerDivision>(
    backend: &PcuVulkanBackend,
    kind: usize,
    mode: PcuNumericalMode,
) -> PcuVulkanPreparedMixed {
    macro_rules! plan {
        ($bindings:ident,$ir:ident) => {{
            let bindings = source::$bindings::<T>();
            let builder = source::$ir::<T, N>(&bindings).unwrap();
            let mut kernel = builder.ir();
            kernel.numerical_requirements.numerical_mode = mode;
            backend.prepare_mixed_kernel(&kernel).unwrap()
        }};
    }
    match kind {
        0 => plan!(repeated_bindings, repeated_ir),
        1 => plan!(unused_bindings, unused_ir),
        2 => plan!(reordered_bindings, reordered_ir),
        3 => plan!(mixed_bindings, mixed_ir),
        _ => plan!(grid_bindings, grid_ir),
    }
}
fn ordinary<T: PcuCheckedIntegerDivision>(
    kind: usize,
    left: &PcuTensor<T>,
    right: &PcuTensor<T>,
    q: &mut PcuTensor<T>,
    r: &mut PcuTensor<T>,
) {
    match kind {
        0 => source::repeated::<T, N>(q, r, left),
        1 => source::unused::<T, N>(&[] as &[T], left, r, q),
        2 => source::reordered::<T, N>(right, r, left, q),
        3 => source::mixed::<T, N>(q, left, r),
        _ => source::grid::<T, N>(r, &[] as &[T], q, left),
    }
    .unwrap();
}
fn graph<T: PcuCheckedIntegerDivision>(
    kind: usize,
    prepared: &mut PcuVulkanPreparedMixed,
    left: &PcuVulkanOwnedBuffer<T>,
    right: &PcuVulkanOwnedBuffer<T>,
    q: &mut PcuVulkanOwnedBuffer<T>,
    r: &mut PcuVulkanOwnedBuffer<T>,
) {
    let binding = |index| PcuBindingRef::new(0, index);
    match kind {
        0 => prepared.call(&mut [
            q.write_argument(binding(0)),
            r.write_argument(binding(1)),
            left.read_argument(binding(2)),
        ]),
        1 => prepared.call(&mut [
            PcuVulkanArgument::host(PcuHostArgument::read(binding(0), &[] as &[T])),
            left.read_argument(binding(1)),
            r.write_argument(binding(2)),
            q.write_argument(binding(3)),
        ]),
        2 => prepared.call(&mut [
            right.read_argument(binding(0)),
            r.write_argument(binding(1)),
            left.read_argument(binding(2)),
            q.write_argument(binding(3)),
        ]),
        3 => prepared.call(&mut [
            q.write_argument(binding(0)),
            left.read_argument(binding(1)),
            r.write_argument(binding(2)),
        ]),
        _ => prepared.call(&mut [
            r.write_argument(binding(0)),
            PcuVulkanArgument::host(PcuHostArgument::read(binding(1), &[] as &[T])),
            q.write_argument(binding(2)),
            left.read_argument(binding(3)),
        ]),
    }
    .unwrap();
}
fn expected<T: Wide>(
    kind: usize,
    left: &[T; STORAGE],
    right: &[T; STORAGE],
    sentinel: T,
) -> Pair<T> {
    let mut q = [sentinel; STORAGE];
    let mut r = q;
    for lane in 0..N {
        let (a, b) = match kind {
            0 | 1 => (left[lane], left[lane]),
            2 => (left[lane], right[lane]),
            3 => (left[lane], left[0]),
            _ => (left[0], left[lane]),
        };
        (q[lane], r[lane]) = oracle::evaluate(a, b).unwrap();
    }
    (q, r)
}
#[allow(clippy::too_many_lines)] // One matched group owns three independent native/session roots and identical public readback boundaries.
pub fn compare<T: Wide + PcuCheckedIntegerDivision>(
    criterion: &mut Criterion,
    backend: &PcuVulkanBackend,
    identity: PcuStableDeviceIdentity,
    kind: usize,
    mode: PcuNumericalMode,
) {
    global::clear_thread_cache().unwrap();
    let sentinel = oracle::small::<T>(99);
    let initial = [sentinel; STORAGE];
    let left: Banks<T> = core::array::from_fn(|bank| {
        let mut values = [oracle::small(7 + u8::try_from(bank).unwrap()); STORAGE];
        values[0] = oracle::small(3 + u8::try_from(bank).unwrap());
        values[2] = oracle::maximum();
        values
    });
    let right: Banks<T> =
        core::array::from_fn(|bank| [oracle::small(3 + u8::try_from(bank).unwrap()); STORAGE]);
    let source_left = left.each_ref().map(|bank| owned::identity(bank).unwrap());
    let source_right = right.each_ref().map(|bank| owned::identity(bank).unwrap());
    let mut source_q = owned::identity(&initial).unwrap();
    let mut source_r = owned::identity(&initial).unwrap();
    let graph_left = left
        .each_ref()
        .map(|bank| backend.upload_owned(bank).unwrap());
    let graph_right = right
        .each_ref()
        .map(|bank| backend.upload_owned(bank).unwrap());
    let mut graph_q = backend.upload_owned(&initial).unwrap();
    let mut graph_r = backend.upload_owned(&initial).unwrap();
    let mut plan = prepare::<T>(backend, kind, mode);
    let mut native = ffi::resident::NativeResidentDivRem::new(
        identity,
        u32::try_from(N).unwrap(),
        T::SIGNED,
        T::HOST_SIZE,
        kind,
        [
            [ffi::bytes(&left[0]), ffi::bytes(&left[1])],
            [ffi::bytes(&right[0]), ffi::bytes(&right[1])],
        ],
        [ffi::bytes(&initial), ffi::bytes(&initial)],
    )
    .unwrap();
    let wanted: [Pair<T>; 2] =
        core::array::from_fn(|bank| expected(kind, &left[bank], &right[bank], sentinel));
    let mut q = initial;
    let mut r = initial;
    let mut run = |route, bank: usize| {
        match route {
            0 => {
                ordinary(
                    kind,
                    &source_left[bank],
                    &source_right[bank],
                    &mut source_q,
                    &mut source_r,
                );
                source_q.read_into(&mut q).unwrap();
                source_r.read_into(&mut r).unwrap();
            }
            1 => {
                graph(
                    kind,
                    &mut plan,
                    &graph_left[bank],
                    &graph_right[bank],
                    &mut graph_q,
                    &mut graph_r,
                );
                graph_q.read_into(&mut q).unwrap();
                graph_r.read_into(&mut r).unwrap();
            }
            _ => {
                assert!(native.call(bank).unwrap().is_none());
                native
                    .read([ffi::bytes_mut(&mut q), ffi::bytes_mut(&mut r)])
                    .unwrap();
            }
        }
        assert_eq!((q, r), wanted[bank]);
    };
    {
        let mut group = criterion.benchmark_group(format!(
            "vulkan_resident_div_rem_roles/{mode:?}/{}/kind{kind}",
            core::any::type_name::<T>()
        ));
        for (route, label) in [
            "source_ordinary",
            "explicit_graph_diagnostic",
            "native_checked_dual_commit",
        ]
        .into_iter()
        .enumerate()
        {
            run(route, 0);
            run(route, 1);
            let scores = SCORES.load(Ordering::Relaxed);
            let counts = ffi::count_heap(|| {
                for call in 0..64 {
                    run(route, call & 1);
                }
            });
            assert_eq!(
                (counts.allocations, counts.reallocations, counts.frees),
                (0, 0, 0)
            );
            assert_eq!(SCORES.load(Ordering::Relaxed), scores);
            println!(
                "Resident joint census type={} mode{mode:?} kind{kind} route{route} calls64 alloc0 realloc0 free0; retained banks/private status/exact dual commits/readback/tails; SDK counts unknown",
                core::any::type_name::<T>()
            );
            group.bench_function(label, |bench| {
                let mut bank = 0;
                bench.iter(|| {
                    bank ^= 1;
                    run(route, std::hint::black_box(bank));
                });
            });
        }
        group.finish();
    }
    global::clear_thread_cache().unwrap();
}
