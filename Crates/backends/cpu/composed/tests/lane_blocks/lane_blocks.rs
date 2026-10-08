//! Lane-block execution compared with forced scalar execution and independent host oracles.
use super::*;
use alloc::vec;
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedInteger,
    PcuDispatchIntegerBinaryOp,
};

#[pcu(crate_path = ::pcu_facade, invocations = N)]
fn tiled<T: PcuCheckedInteger, const N: usize>(input: &[T], seed: &T, output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = (input[id] + *seed) * input[id] - input[id];
}

fn small_oracle<T: PcuCheckedInteger, const N: usize>(from_small: impl Fn(u8) -> T) {
    let bindings = tiled_bindings::<T>();
    tiled_ir::<T, N>(&bindings).unwrap().with_ir(|kernel| {
        let mut block = prepare_erased(kernel).unwrap();
        let mut scalar = block.clone();
        scalar.program.use_scalar_integer_executor();
        let input: [T; N] =
            core::array::from_fn(|lane| from_small(u8::try_from(lane % 7 + 1).unwrap()));
        for seed in [1_u8, 2, 3, 1] {
            let mut actual = vec![from_small(99); N + 3];
            let mut reference = actual.clone();
            let seed_argument = [from_small(seed)];
            for (plan, output) in [(&mut block, &mut actual), (&mut scalar, &mut reference)] {
                plan.call(&mut [
                    PcuHostArgument::read(bindings[0].reference(), &input),
                    PcuHostArgument::read(bindings[1].reference(), &seed_argument),
                    PcuHostArgument::read_write(bindings[2].reference(), output),
                ])
                .unwrap();
            }
            for (lane, (actual, reference)) in actual.iter().zip(&reference).enumerate() {
                let expected = if lane < N {
                    let value = u8::try_from(lane % 7 + 1).unwrap();
                    from_small((value + seed) * value - value)
                } else {
                    from_small(99)
                };
                assert_eq!(actual.encode_le().as_ref(), expected.encode_le().as_ref());
                assert_eq!(actual.encode_le().as_ref(), reference.encode_le().as_ref());
            }
        }
    });
}

#[test]
fn all_fourteen_widths_match_scalar_and_small_host_oracle_with_tails() {
    macro_rules! primitive {
        ($($ty:ty),+) => {$({
            small_oracle::<$ty, 17>(<$ty>::from);
            small_oracle::<$ty, 33>(<$ty>::from);
            small_oracle::<$ty, 65>(<$ty>::from);
        })+};
    }
    primitive!(u8, i16, u16, i32, u32, i64, u64, i128, u128);
    for_extent_i8();
    macro_rules! wide {
        ($($ty:ty),+) => {$({
            let convert = |value| {
                let mut limbs = [0; core::mem::size_of::<$ty>() / 8];
                limbs[0] = u64::from(value);
                <$ty>::from_limbs_le(limbs)
            };
            small_oracle::<$ty, 17>(convert);
            small_oracle::<$ty, 33>(convert);
            small_oracle::<$ty, 65>(convert);
        })+};
    }
    wide!(
        fusion_pcu::PcuI256,
        fusion_pcu::PcuU256,
        fusion_pcu::PcuI512,
        fusion_pcu::PcuU512
    );
}

fn for_extent_i8() {
    small_oracle::<i8, 17>(|value| i8::try_from(value).unwrap());
    small_oracle::<i8, 33>(|value| i8::try_from(value).unwrap());
    small_oracle::<i8, 65>(|value| i8::try_from(value).unwrap());
}

fn integer_binary(
    result: u16,
    left: u16,
    right: u16,
    op: PcuDispatchIntegerBinaryOp,
    range_policy: PcuRangePolicy,
) -> PcuDispatchOp<'static> {
    PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
        result: PcuDispatchValueId(result),
        lhs: PcuDispatchValueId(left),
        rhs: PcuDispatchValueId(right),
        op,
        range_policy,
        value_type: PcuValueType::Scalar(PcuScalarType::U64),
    })
}

fn integer_plan(body: &[PcuDispatchOp<'_>], writable: bool) -> PcuCpuPreparedComposedMap {
    let bindings: [_; 4] = core::array::from_fn(|index| {
        PcuBinding::value(
            None,
            0,
            u32::try_from(index).unwrap(),
            PcuBindingStorageClass::Storage,
            if index >= 2 || (index == 0 && writable) {
                PcuBindingAccess::ReadWrite
            } else {
                PcuBindingAccess::ReadOnly
            },
            PcuValueType::Scalar(PcuScalarType::U64),
        )
    });
    prepare_erased(&PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(81),
        entry: PcuDispatchEntryPoint {
            name: "integer_lane_order",
            logical_shape: [65, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: body,
        type_caps: PcuValueTypeCaps::empty(),
        feature_caps: PcuDispatchFeatureCaps::empty(),
    })
    .unwrap()
}

fn run(
    plan: &mut PcuCpuPreparedComposedMap,
    input: &[u64],
    rhs: &[u64],
    first: &mut [u64],
    second: &mut [u64],
) -> Option<PcuExecutionFault> {
    plan.call(&mut [
        PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
        PcuHostArgument::read(PcuBindingRef::new(0, 1), rhs),
        PcuHostArgument::read_write(PcuBindingRef::new(0, 2), first),
        PcuHostArgument::read_write(PcuBindingRef::new(0, 3), second),
    ])
    .err()
    .and_then(PcuCpuComposedMapError::fault)
}

#[test]
fn earlier_lane_late_unused_fault_wins_and_retry_refreshes_every_tile() {
    let body = [
        load(0, 0, false),
        load(1, 1, false),
        integer_binary(
            2,
            0,
            1,
            PcuDispatchIntegerBinaryOp::Add,
            PcuRangePolicy::Reject,
        ),
        store(2, 2),
        store(3, 0),
        integer_binary(
            3,
            0,
            1,
            PcuDispatchIntegerBinaryOp::Sub,
            PcuRangePolicy::Reject,
        ),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    for (earlier, later) in [(3, 4), (15, 16), (16, 32), (31, 64)] {
        let mut block = integer_plan(&body, false);
        let mut scalar = block.clone();
        scalar.program.use_scalar_integer_executor();
        let mut input = [4; 65];
        let rhs = [2; 65];
        input[earlier] = 1;
        input[later] = u64::MAX;
        for plan in [&mut block, &mut scalar] {
            let mut first = [91; 68];
            let mut second = [92; 68];
            assert_eq!(
                run(plan, &input, &rhs, &mut first, &mut second),
                Some(PcuExecutionFault {
                    invocation_id: u64::try_from(earlier).unwrap(),
                    kind: PcuExecutionFaultKind::ArithmeticUnderflow,
                    recovered: false,
                })
            );
            assert_eq!(first, [91; 68]);
            assert_eq!(second, [92; 68]);
            assert_eq!(run(plan, &[4; 65], &rhs, &mut first, &mut second), None);
            assert_eq!(&first[..65], &[6; 65]);
            assert_eq!(&second[..65], &[4; 65]);
            assert_eq!(&first[65..], &[91; 3]);
            assert_eq!(&second[65..], &[92; 3]);
        }
    }
}

#[test]
fn clamp_publication_uses_earliest_recovery_and_later_fatal_rolls_back_both_outputs() {
    let body = [
        load(0, 0, false),
        load(1, 1, false),
        integer_binary(
            2,
            0,
            1,
            PcuDispatchIntegerBinaryOp::Add,
            PcuRangePolicy::Clamp,
        ),
        store(2, 2),
        integer_binary(
            3,
            0,
            1,
            PcuDispatchIntegerBinaryOp::Sub,
            PcuRangePolicy::Reject,
        ),
        store(3, 3),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let mut block = integer_plan(&body, false);
    let mut scalar = block.clone();
    scalar.program.use_scalar_integer_executor();
    for plan in [&mut block, &mut scalar] {
        let mut input = [4; 65];
        input[15] = u64::MAX;
        input[32] = u64::MAX;
        let mut first = [91; 68];
        let mut second = [92; 68];
        assert_eq!(
            run(plan, &input, &[2; 65], &mut first, &mut second),
            Some(PcuExecutionFault {
                invocation_id: 15,
                kind: PcuExecutionFaultKind::ArithmeticOverflow,
                recovered: true,
            })
        );
        for lane in 0..65 {
            assert_eq!(first[lane], input[lane].saturating_add(2));
            assert_eq!(second[lane], input[lane] - 2);
        }
        let old_first = first;
        let old_second = second;
        input[64] = 1;
        assert_eq!(
            run(plan, &input, &[2; 65], &mut first, &mut second),
            Some(PcuExecutionFault {
                invocation_id: 64,
                kind: PcuExecutionFaultKind::ArithmeticUnderflow,
                recovered: false,
            })
        );
        assert_eq!(first, old_first);
        assert_eq!(second, old_second);
    }
}

#[test]
fn writable_load_store_fallback_observes_private_stores_and_rolls_back() {
    let body = [
        load(0, 0, false),
        load(1, 1, false),
        integer_binary(
            2,
            0,
            1,
            PcuDispatchIntegerBinaryOp::Add,
            PcuRangePolicy::Reject,
        ),
        store(0, 2),
        load(3, 0, false),
        integer_binary(
            4,
            3,
            1,
            PcuDispatchIntegerBinaryOp::Mul,
            PcuRangePolicy::Reject,
        ),
        store(2, 4),
        store(3, 3),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let mut plan = integer_plan(&body, true);
    let mut state = [3_u64; 68];
    let mut first = [91_u64; 68];
    let mut second = [92_u64; 68];
    for fail in [false, true, false] {
        let old_state = state;
        let old_first = first;
        let old_second = second;
        let mut rhs = [2; 65];
        if fail {
            rhs[64] = u64::MAX;
        }
        let result = plan.call(&mut [
            PcuHostArgument::read_write(PcuBindingRef::new(0, 0), &mut state),
            PcuHostArgument::read(PcuBindingRef::new(0, 1), &rhs),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut first),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 3), &mut second),
        ]);
        if fail {
            assert_eq!(
                result.unwrap_err().fault(),
                Some(PcuExecutionFault {
                    invocation_id: 64,
                    kind: PcuExecutionFaultKind::ArithmeticOverflow,
                    recovered: false,
                })
            );
            assert_eq!(state, old_state);
            assert_eq!(first, old_first);
            assert_eq!(second, old_second);
        } else {
            result.unwrap();
            for lane in 0..65 {
                assert_eq!(state[lane], old_state[lane] + 2);
                assert_eq!(first[lane], (old_state[lane] + 2) * 2);
                assert_eq!(second[lane], old_state[lane] + 2);
            }
            assert_eq!(&state[65..], &[3; 3]);
            assert_eq!(&first[65..], &[91; 3]);
            assert_eq!(&second[65..], &[92; 3]);
        }
    }
}

fn run_complete(
    plan: &mut PcuCpuPreparedComposedMap,
    input: &[u64],
    rhs: &[u64],
    first: &mut [u64],
    second: &mut [u64],
) -> Result<(), PcuCpuComposedMapError> {
    plan.call(&mut [
        PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
        PcuHostArgument::read(PcuBindingRef::new(0, 1), rhs),
        PcuHostArgument::read_write(PcuBindingRef::new(0, 2), first),
        PcuHostArgument::read_write(PcuBindingRef::new(0, 3), second),
    ])
}

fn random_word(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1);
    *state >> 32
}

#[test]
fn deterministic_dependency_dags_match_scalar_for_faults_publication_and_retries() {
    let mut random = 0x5eed_fade_cafe_beef_u64;
    for case in 0..32 {
        // Keep intermediate values small while selecting non-adjacent dependency edges.
        let mut body = vec![load(0, 0, false), load(1, 1, case % 2 == 0)];
        for result in 2_u16..10 {
            let left = u16::try_from(random_word(&mut random) % u64::from(result)).unwrap();
            let right = u16::try_from(random_word(&mut random) % 2).unwrap();
            let operation = match random_word(&mut random) % 3 {
                0 => PcuDispatchIntegerBinaryOp::Add,
                1 => PcuDispatchIntegerBinaryOp::Sub,
                _ => PcuDispatchIntegerBinaryOp::Mul,
            };
            body.push(integer_binary(
                result,
                left,
                right,
                operation,
                if random_word(&mut random) & 1 == 0 {
                    PcuRangePolicy::Clamp
                } else {
                    PcuRangePolicy::Reject
                },
            ));
            if result == 5 {
                body.push(store(2, result));
            }
        }
        body.push(store(3, 9));
        body.push(PcuDispatchOp::Control(PcuDispatchControlOp::Return));
        let mut block = integer_plan(&body, false);
        let mut scalar = block.clone();
        scalar.program.use_scalar_integer_executor();
        let mut actual_first = [91_u64; 68];
        let mut actual_second = [92_u64; 68];
        let mut reference_first = actual_first;
        let mut reference_second = actual_second;
        for attempt in 0..5 {
            let mut input =
                core::array::from_fn::<_, 65, _>(|lane| u64::try_from(lane % 7 + 2).unwrap());
            let rhs = [u64::try_from(attempt % 3).unwrap(); 65];
            if attempt == 1 {
                input[15] = u64::MAX;
                input[16] = 0;
            } else if attempt == 3 {
                input[32] = 0;
                input[64] = u64::MAX;
            }
            let old_first = actual_first;
            let old_second = actual_second;
            let actual = run_complete(
                &mut block,
                &input,
                &rhs,
                &mut actual_first,
                &mut actual_second,
            );
            let reference = run_complete(
                &mut scalar,
                &input,
                &rhs,
                &mut reference_first,
                &mut reference_second,
            );
            assert_eq!(actual, reference, "case {case}, attempt {attempt}");
            assert_eq!(
                actual_first, reference_first,
                "case {case}, attempt {attempt}"
            );
            assert_eq!(
                actual_second, reference_second,
                "case {case}, attempt {attempt}"
            );
            if actual
                .err()
                .and_then(PcuCpuComposedMapError::fault)
                .is_some_and(|fault| !fault.recovered)
            {
                assert_eq!(actual_first, old_first);
                assert_eq!(actual_second, old_second);
            }
            assert_eq!(&actual_first[65..], &[91; 3]);
            assert_eq!(&actual_second[65..], &[92; 3]);
        }
    }
}

fn width_overflow_retry<T: PcuCheckedInteger>(one: T, maximum: T, sentinel: T) {
    let bindings = tiled_bindings::<T>();
    tiled_ir::<T, 65>(&bindings).unwrap().with_ir(|kernel| {
        let mut block = prepare_erased(kernel).unwrap();
        let mut scalar = block.clone();
        scalar.program.use_scalar_integer_executor();
        for fault_lane in [15_usize, 16, 64] {
            let mut input = [one; 65];
            input[fault_lane] = maximum;
            let mut actual = [sentinel; 68];
            let mut reference = actual;
            for failing in [true, false] {
                if !failing {
                    input[fault_lane] = one;
                }
                let seed = [one];
                let invoke = |plan: &mut PcuCpuPreparedComposedMap, output: &mut [T]| {
                    plan.call(&mut [
                        PcuHostArgument::read(bindings[0].reference(), &input),
                        PcuHostArgument::read(bindings[1].reference(), &seed),
                        PcuHostArgument::read_write(bindings[2].reference(), output),
                    ])
                };
                let result = invoke(&mut block, &mut actual);
                assert_eq!(result, invoke(&mut scalar, &mut reference));
                if failing {
                    assert_eq!(
                        result.unwrap_err().fault(),
                        Some(PcuExecutionFault {
                            invocation_id: u64::try_from(fault_lane).unwrap(),
                            kind: PcuExecutionFaultKind::ArithmeticOverflow,
                            recovered: false,
                        })
                    );
                } else {
                    result.unwrap();
                }
                for (lane, (actual, reference)) in actual.iter().zip(&reference).enumerate() {
                    let expected = if !failing && lane < 65 { one } else { sentinel };
                    assert_eq!(actual.encode_le().as_ref(), reference.encode_le().as_ref());
                    assert_eq!(actual.encode_le().as_ref(), expected.encode_le().as_ref());
                }
            }
        }
    });
}

#[test]
fn all_fourteen_widths_overflow_at_block_boundaries_roll_back_and_retry() {
    macro_rules! primitive {
        ($($ty:ty),+) => {$(width_overflow_retry::<$ty>(1, <$ty>::MAX, 99);)+};
    }
    primitive!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);
    macro_rules! wide {
        ($($ty:ty),+) => {$({
            let small = |value| {
                let mut limbs = [0; core::mem::size_of::<$ty>() / 8];
                limbs[0] = value;
                <$ty>::from_limbs_le(limbs)
            };
            width_overflow_retry::<$ty>(small(1), <$ty>::MAX, small(99));
        })+};
    }
    wide!(
        fusion_pcu::PcuI256,
        fusion_pcu::PcuU256,
        fusion_pcu::PcuI512,
        fusion_pcu::PcuU512
    );
}
