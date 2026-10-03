//! Actual ordinary host/resident/mixed integer sources and immutable prefix publication.
#[rustfmt::skip]
use pcu_facade::{
    global,
    pcu,
    PcuScalar,
    PcuTensor,
    PcuExecutionError,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuRangePolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
};
#[rustfmt::skip]
use super::{
    prepared,
    Sample,
    same,
    Range,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};
#[pcu(crate_path=::pcu_facade)]
fn retain<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
fn read<T: Sample>(owner: &PcuTensor<T>, wanted: &[T]) {
    let sentinel = T::small(17);
    let mut output = vec![sentinel; wanted.len() + 2];
    owner.read_into(&mut output).unwrap();
    same(&output[..wanted.len()], wanted);
    same(&output[wanted.len()..], &[sentinel; 2]);
}
fn leaf_policy(policy: global::PcuExecutionPolicy) {
    global::configure(global::PcuExecutionPolicy {
        range_policy: Range::Reject,
        ..policy
    })
    .unwrap();
}
fn publish<T: Sample>(policy: global::PcuExecutionPolicy) {
    // Owned-return graph Clamp is not an admitted carrier; construct exact leaves under Reject.
    leaf_policy(policy);
    let left = [T::small(3); 5];
    let right = [T::small(2); 5];
    let a = retain(&left).unwrap();
    let b = retain(&right).unwrap();
    let sentinel = T::small(7);
    let mut output = retain(&[sentinel; 7]).unwrap();
    let sibling = retain(&output).unwrap();
    let bad = retain(&[T::max(); 5]).unwrap();
    let ones = retain(&[T::small(1); 5]).unwrap();
    let mut short = retain(&[sentinel; 4]).unwrap();
    global::configure(policy).unwrap();
    let wanted = [
        T::small(5),
        T::small(5),
        T::small(5),
        T::small(5),
        T::small(5),
        sentinel,
        sentinel,
    ];
    prepared::source::add::<T, 5>(&a, &right, &mut output).unwrap();
    read(&output, &wanted);
    prepared::source::add::<T, 5>(&left, &b, &mut output).unwrap();
    prepared::source::add::<T, 5>(&a, &b, &mut output).unwrap();
    read(&output, &wanted);
    read(&a, &left);
    read(&b, &right);
    read(&sibling, &[sentinel; 7]);
    let mut host = [sentinel; 7];
    prepared::source::swapped::<T, 5>(&mut host, &b, &a).unwrap();
    same(
        &host,
        &[
            T::small(1),
            T::small(1),
            T::small(1),
            T::small(1),
            T::small(1),
            sentinel,
            sentinel,
        ],
    );
    let fault = PcuExecutionFault {
        invocation_id: 0,
        kind: PcuExecutionFaultKind::ArithmeticOverflow,
        recovered: policy.range_policy == Range::Clamp,
    };
    assert!(
        matches!(prepared::source::add::<T, 5>(&bad, &ones, &mut output), Err(PcuExecutionError::ArithmeticFault(actual)) if actual == fault)
    );
    if fault.recovered {
        read(
            &output,
            &[
                T::max(),
                T::max(),
                T::max(),
                T::max(),
                T::max(),
                sentinel,
                sentinel,
            ],
        );
    } else {
        read(&output, &wanted);
    }
    // An explicit local Clamp remains independent from the global Reject setting.
    assert!(
        matches!(prepared::source::add_clamp::<T, 5>(&bad, &ones, &mut output), Err(PcuExecutionError::ArithmeticFault(actual)) if actual.recovered && actual.invocation_id == 0)
    );
    read(
        &output,
        &[
            T::max(),
            T::max(),
            T::max(),
            T::max(),
            T::max(),
            sentinel,
            sentinel,
        ],
    );
    read(&sibling, &[sentinel; 7]);
    assert!(prepared::source::add::<T, 5>(&a, &b, &mut short).is_err());
    read(&short, &[sentinel; 4]);
    prepared::source::add::<T, 5>(&a, &b, &mut output).unwrap();
    drop(a);
    drop(b);
    global::clear_thread_cache().unwrap();
    read(&output, &wanted);
    read(&sibling, &[sentinel; 7]);
    // Escaped owners retain the actual session; clearing compiled code must not split affinity.
    leaf_policy(policy);
    let a = retain(&left).unwrap();
    global::configure(policy).unwrap();
    prepared::source::add::<T, 5>(&a, &right, &mut output).unwrap();
    read(&output, &wanted);
}
fn qualify<T: Sample>() {
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                    publish::<T>(global::PcuExecutionPolicy {
                        backend: global::PcuBackendChoice::Mlx,
                        numerical_mode: mode,
                        numerical_options: PcuNumericalOptions {
                            compound_arithmetic: compound,
                            precision,
                            ..Default::default()
                        },
                        range_policy: range,
                        ..Default::default()
                    });
                }
            }
        }
    }
}
#[test]
#[ignore = "Requires actual ordinary MLX fourteen-width mixed and prefix publication."]
fn fourteen_width_ordinary_mixed_prefix_and_range_publication() {
    qualify::<u8>();
    qualify::<i8>();
    qualify::<u16>();
    qualify::<i16>();
    qualify::<u32>();
    qualify::<i32>();
    qualify::<u64>();
    qualify::<i64>();
    qualify::<u128>();
    qualify::<i128>();
    qualify::<PcuU256>();
    qualify::<PcuI256>();
    qualify::<PcuU512>();
    qualify::<PcuI512>();
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
