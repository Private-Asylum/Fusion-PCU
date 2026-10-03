//! Full dense owner capacities are frozen cold; shader reads remain bounded to IR spans.
#[rustfmt::skip]
use super::super::{
    graph,
    same,
    Sample,
};
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxBinaryInput,
    MlxCheckedDivRemRolePlan,
    MlxError,
    MlxRuntime,
    MlxSession,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuI256,
    PcuI512,
    PcuU256,
    PcuU512,
};
#[allow(clippy::too_many_lines)]
// Each format/profile preflights both resources, verifies repeated terminal pairs and independently checks negative affinity/shape paths.
fn verify<T: Sample>(profile: usize, session: &MlxSession, foreign: &MlxSession) {
    graph::roles::fixture::<T, _>(5, profile, |ir| {
        let plan = MlxCheckedDivRemRolePlan::assess(ir).unwrap();
        let count = plan.input_bindings().len();
        let extents = [8, 9];
        let backend = session.checked_div_rem_role_backend();
        let mut prepared = backend
            .prepare_host_kernel_with_input_extents(ir, &extents[..count])
            .unwrap();
        assert_eq!(prepared.input_element_extents(), &extents[..count]);
        assert!(
            backend
                .prepare_host_kernel_with_input_extents(ir, &[])
                .is_err()
        );
        assert!(
            backend
                .prepare_host_kernel_with_input_extents(ir, &[0; 2][..count])
                .is_err()
        );
        let mut a = [
            T::raw(3),
            T::raw(2),
            T::minimum(),
            T::raw(6),
            T::raw(8),
            T::raw(0),
            T::negative_one(),
            T::raw(0),
        ];
        let mut b = [
            T::raw(3),
            T::raw(3),
            T::raw(3),
            T::raw(3),
            T::raw(3),
            T::raw(0),
            T::minimum(),
            T::raw(0),
            T::raw(0),
        ];
        if profile == 5 {
            a[1] = T::raw(0); // Both mathematical operands read only element zero.
        }
        if profile == 4 {
            b[1] = T::raw(0); // The scalar divisor's larger native suffix is not read.
        }
        let left = session.upload_encoded(&a).unwrap();
        let right = session.upload_encoded(&b).unwrap();
        let foreign_left = foreign.upload_encoded(&a).unwrap();
        let wrong_shape = session.upload_encoded(&a[..7]).unwrap();
        let targets = plan.input_bindings();
        let inputs = [
            MlxBinaryInput::Resident {
                target: targets[0],
                array: &left,
            },
            MlxBinaryInput::Resident {
                target: targets[count - 1],
                array: &right,
            },
        ];
        let old = backend.prepare_host_kernel(ir).unwrap();
        assert_ne!(
            old.input_element_extents(),
            prepared.input_element_extents()
        );
        for _ in 0..3 {
            let [q, r] = prepared.execute_inputs(&inputs[..count]).unwrap();
            let sentinel = T::raw(91);
            let mut q_read = [sentinel; 7];
            let mut r_read = [sentinel; 8];
            q.read_into(&mut q_read).unwrap();
            r.read_into(&mut r_read).unwrap();
            let values = [&a[..], &b[..]];
            let slots = plan.operand_inputs();
            let broadcast = plan.operand_broadcast();
            for lane in 0..5 {
                let lhs = values[slots[0]][if broadcast[0] { 0 } else { lane }];
                let rhs = values[slots[1]][if broadcast[1] { 0 } else { lane }];
                same(&q_read[lane..=lane], &[lhs.pcu_checked_div(rhs).unwrap()]);
                same(&r_read[lane..=lane], &[lhs.pcu_checked_rem(rhs).unwrap()]);
            }
            same(&q_read[5..], &[sentinel; 2]);
            same(&r_read[5..], &[sentinel; 3]);
            q.release().unwrap();
            r.release().unwrap();
        }
        for array in [&wrong_shape, &foreign_left] {
            let mut invalid = inputs;
            invalid[0] = MlxBinaryInput::Resident {
                target: targets[0],
                array,
            };
            assert!(prepared.execute_inputs(&invalid[..count]).is_err());
            assert!(!prepared.last_call_may_have_written());
        }
        let mut host = inputs;
        let bytes = PcuHostArgument::read(targets[0], &a);
        host[0] = MlxBinaryInput::HostBytes {
            target: targets[0],
            scalar: T::TYPE,
            bytes: bytes.bytes(),
        };
        assert!(matches!(
            prepared.execute_inputs(&host[..count]),
            Err(MlxError::InvalidExtent)
        ));
        assert!(!prepared.last_call_may_have_written());
        if count == 2 {
            let spans = plan.input_element_counts();
            let mut mixed = backend
                .prepare_host_kernel_with_input_extents(ir, &[8, spans[1]])
                .unwrap();
            let right_bytes = PcuHostArgument::read(targets[1], &b);
            let [q, r] = mixed
                .execute_inputs(&[
                    inputs[0],
                    MlxBinaryInput::HostBytes {
                        target: targets[1],
                        scalar: T::TYPE,
                        bytes: right_bytes.bytes(),
                    },
                ])
                .unwrap();
            let mut q_read = [T::raw(91); 7];
            let mut r_read = [T::raw(91); 7];
            q.read_into(&mut q_read).unwrap();
            r.read_into(&mut r_read).unwrap();
            let slots = plan.operand_inputs();
            let broadcast = plan.operand_broadcast();
            let values = [&a[..], &b[..]];
            for lane in 0..5 {
                let lhs = values[slots[0]][if broadcast[0] { 0 } else { lane }];
                let rhs = values[slots[1]][if broadcast[1] { 0 } else { lane }];
                same(&q_read[lane..=lane], &[lhs.pcu_checked_div(rhs).unwrap()]);
                same(&r_read[lane..=lane], &[lhs.pcu_checked_rem(rhs).unwrap()]);
            }
            same(&q_read[5..], &[T::raw(91); 2]);
            same(&r_read[5..], &[T::raw(91); 2]);
            q.release().unwrap();
            r.release().unwrap();
            if profile == 2 {
                let mut canonical = session
                    .checked_div_rem_backend()
                    .prepare_host_kernel_with_input_extents(ir, &[8, 9])
                    .unwrap();
                let [q, r] = canonical.execute_inputs(&inputs).unwrap();
                let mut q_read = [T::raw(91); 7];
                let mut r_read = [T::raw(91); 7];
                q.read_into(&mut q_read).unwrap();
                r.read_into(&mut r_read).unwrap();
                for lane in 0..5 {
                    same(
                        &q_read[lane..=lane],
                        &[a[lane].pcu_checked_div(b[lane]).unwrap()],
                    );
                    same(
                        &r_read[lane..=lane],
                        &[a[lane].pcu_checked_rem(b[lane]).unwrap()],
                    );
                }
                same(&q_read[5..], &[T::raw(91); 2]);
                same(&r_read[5..], &[T::raw(91); 2]);
                q.release().unwrap();
                r.release().unwrap();
            }
        }
        let mut unchanged = [T::raw(0); 8];
        left.read_into(&mut unchanged).unwrap();
        same(&unchanged, &a);
        left.release().unwrap();
        right.release().unwrap();
        foreign_left.release().unwrap();
        wrong_shape.release().unwrap();
    });
}
#[test]
#[ignore = "required native MLX full dense resident capacity and bounded division read-span proof"]
fn fourteen_width_full_resident_capacity_preserves_bounded_read_spans() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let foreign = runtime.open_gpu(0).unwrap();
    macro_rules! run {($($ty:ty),+)=>{$(for profile in 0..6 {verify::<$ty>(profile, &session, &foreign);})+};}
    run!(
        u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, PcuU256, PcuI256, PcuU512, PcuI512
    );
}

#[cfg(all(feature = "division-census", feature = "allocation-census"))]
#[path = "../../../../benches/tensor_binary/census/census.rs"]
mod heap;

#[cfg(all(feature = "division-census", feature = "allocation-census"))]
#[allow(clippy::too_many_lines)]
// Both independently prepared peers verify 64 changing owner banks outside captured allocator intervals.
fn warm_census<T: Sample>(profile: usize, session: &MlxSession) {
    graph::roles::fixture::<T, _>(5, profile, |ir| {
        let plan = MlxCheckedDivRemRolePlan::assess(ir).unwrap();
        let count = plan.input_bindings().len();
        let mut prepared = session
            .checked_div_rem_role_backend()
            .prepare_host_kernel_with_input_extents(ir, &[8, 9][..count])
            .unwrap();
        let mut native = session
            .prepare_checked_div_rem_control_with_input_extents(
                T::TYPE,
                5,
                plan.operand_inputs().map(|slot| [8, 9][slot]),
                plan.operand_broadcast(),
            )
            .unwrap();
        let banks: Vec<_> = (0..64)
            .map(|bank| {
                let mut a = [T::raw(1); 8];
                let mut b = [T::raw(3); 9];
                for lane in 0..5 {
                    a[lane] = T::raw(u64::try_from((bank ^ lane) % 89 + 1).unwrap());
                    b[lane] = T::raw(u64::try_from((bank + lane) % 7 + 1).unwrap());
                }
                a[2] = T::minimum();
                a[5..].fill(T::raw(0));
                b[5..].fill(T::raw(0));
                if profile == 5 {
                    a[1] = T::raw(0);
                }
                if profile == 4 {
                    b[1] = T::raw(0);
                }
                let left = session.upload_encoded(&a).unwrap();
                let right = (count == 2).then(|| session.upload_encoded(&b).unwrap());
                (left, right, a, b)
            })
            .collect();
        let execute = |prepared: &mut fusion_pcu_mlx::MlxPreparedDivRemRoleHostKernel,
                       native: &mut fusion_pcu_mlx::MlxCheckedDivRemControl,
                       route: usize,
                       bank: usize| {
            let (left, right, _, _) = &banks[bank];
            let targets = plan.input_bindings();
            let inputs = [
                MlxBinaryInput::Resident {
                    target: targets[0],
                    array: left,
                },
                MlxBinaryInput::Resident {
                    target: targets[count - 1],
                    array: right.as_ref().unwrap_or(left),
                },
            ];
            let [q, r] = if route == 0 {
                prepared.execute_inputs(&inputs[..count]).unwrap()
            } else {
                let owners = [left, right.as_ref().unwrap_or(left)];
                native
                    .execute_resident(plan.operand_inputs().map(|slot| owners[slot]))
                    .unwrap()
            };
            let mut q_read = [T::raw(91); 7];
            let mut r_read = [T::raw(91); 8];
            q.read_into(&mut q_read).unwrap();
            r.read_into(&mut r_read).unwrap();
            q.release().unwrap();
            r.release().unwrap();
            (q_read, r_read)
        };
        for (route, name) in ["explicit_ir", "direct_native"].into_iter().enumerate() {
            let _ = execute(&mut prepared, &mut native, route, 0);
            fusion_pcu_mlx::reset_div_rem_call_census();
            let mut total = heap::Census::default();
            for (bank, values_bank) in banks.iter().enumerate() {
                let ((q, r), counts) =
                    heap::measure(|| execute(&mut prepared, &mut native, route, bank));
                total.alloc_calls += counts.alloc_calls;
                total.realloc_calls += counts.realloc_calls;
                total.dealloc_calls += counts.dealloc_calls;
                total.requested_bytes += counts.requested_bytes;
                let values = [&values_bank.2[..], &values_bank.3[..]];
                let slots = plan.operand_inputs();
                let broadcast = plan.operand_broadcast();
                for lane in 0..5 {
                    let lhs = values[slots[0]][if broadcast[0] { 0 } else { lane }];
                    let rhs = values[slots[1]][if broadcast[1] { 0 } else { lane }];
                    same(&q[lane..=lane], &[lhs.pcu_checked_div(rhs).unwrap()]);
                    same(&r[lane..=lane], &[lhs.pcu_checked_rem(rhs).unwrap()]);
                }
                same(&q[5..], &[T::raw(91); 2]);
                same(&r[5..], &[T::raw(91); 3]);
            }
            let calls = fusion_pcu_mlx::div_rem_call_census();
            assert_eq!(calls.exact_constructor_calls, 0);
            assert_eq!(calls.prefix_constructor_calls, 0);
            assert_eq!(calls.prime_calls, 0);
            assert_eq!(calls.table_symbol_attempts, 0);
            assert_eq!(calls.apply_calls, 64);
            eprintln!(
                "MLX full-capacity adapter census/{:?}/profile{profile}/{name}: calls=64 alloc={} realloc={} dealloc={} requested_bytes={} adapter={calls:?}; full resident inputs -> two fresh private owners -> both terminal reads/checked release; SDK-internal heap/JIT unknown",
                T::TYPE,
                total.alloc_calls,
                total.realloc_calls,
                total.dealloc_calls,
                total.requested_bytes
            );
        }
    });
}
#[test]
#[ignore = "required untimed MLX full-capacity 64-changing-call adapter and caller census"]
#[cfg(all(feature = "division-census", feature = "allocation-census"))]
fn fourteen_width_full_capacity_warm_adapter_and_caller_census() {
    let session = MlxRuntime::load_default().unwrap().open_gpu(0).unwrap();
    macro_rules! run {($($ty:ty),+)=>{$(for profile in 0..6 {warm_census::<$ty>(profile, &session);})+};}
    run!(
        u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, PcuU256, PcuI256, PcuU512, PcuI512
    );
}

#[cfg(all(feature = "division-census", feature = "allocation-census"))]
#[path = "ordinary/ordinary.rs"]
mod ordinary;
