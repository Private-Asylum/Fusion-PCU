//! Genuine source preparation retains the same checked arithmetic and transactional host prefix.
#[rustfmt::skip]
use super::{
    Sample,
    same,
    expected,
    Op,
    Range,
    MlxSession,
    MlxError,
    graph,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuHostDispatchError,
    PcuHostKernelBackend,
    PcuHostArgument,
    PcuBindingRef,
};
use fusion_pcu_mlx::MlxBinaryInput;
#[path = "../source/source.rs"]
pub mod source;
#[allow(clippy::too_many_lines)] // Paired source Reject/Clamp fault, retry, prefix and shape peers.
pub fn qualify<T: Sample>(session: &MlxSession) {
    let backend = session.checked_integer_backend();
    let left = [T::zero(), T::small(1), T::max(), T::min(), T::max()];
    let right = [T::small(1), T::small(2), T::small(1), T::max(), T::small(2)];
    let sentinel = T::small(7);
    macro_rules! peer {
        ($prepare:ident,$op:expr,$range:expr) => {{
            let mut call = source::$prepare::<T, 5, _>(&backend).unwrap();
            let (wanted, fault) = expected(&left, &right, $op, $range, [false; 2]);
            let mut output = [sentinel; 7];
            assert_eq!(
                call(&left, &right, &mut output).err(),
                fault.map(|fault| PcuHostDispatchError::Backend(MlxError::Arithmetic(fault)))
            );
            if fault.is_some_and(|fault| !fault.recovered) {
                same(&output, &[sentinel; 7]);
            } else {
                same(&output[..5], &wanted);
                same(&output[5..], &[sentinel; 2]);
            }
            let mut short = [sentinel; 4];
            assert!(call(&left, &right, &mut short).is_err());
            same(&short, &[sentinel; 4]);
            let zero = [T::zero(); 5];
            call(&zero, &zero, &mut output).unwrap();
            same(&output[..5], &zero);
            same(&output[5..], &[sentinel; 2]);
        }};
    }
    peer!(add_prepare, Op::Add, Range::Reject);
    peer!(sub_prepare, Op::Sub, Range::Reject);
    peer!(mul_prepare, Op::Mul, Range::Reject);
    peer!(add_clamp_prepare, Op::Add, Range::Clamp);
    peer!(sub_clamp_prepare, Op::Sub, Range::Clamp);
    peer!(mul_clamp_prepare, Op::Mul, Range::Clamp);
    let a = [T::small(3); 5];
    let b = [T::small(2); 5];
    let mut output = [sentinel; 7];
    source::swapped_prepare::<T, 5, _>(&backend).unwrap()(&mut output, &b, &a).unwrap();
    same(&output[..5], &[T::small(1); 5]);
    source::broadcast_prepare::<T, 5, _>(&backend).unwrap()(&a[0], &b, &mut output).unwrap();
    same(&output[..5], &[T::small(5); 5]);
    source::repeated_prepare::<T, 5, _>(&backend).unwrap()(&[], &b, &mut output).unwrap();
    same(&output[..5], &[T::small(4); 5]);
    source::single_prepare::<T, 5, _>(&backend).unwrap()(&b, &mut output).unwrap();
    same(&output[..5], &[T::small(4); 5]);
    source::mixed_indices_prepare::<T, 5, _>(&backend).unwrap()(&b, &mut output).unwrap();
    same(&output[..5], &[T::small(4); 5]);
    source::grid_prepare::<T, 5, _>(&backend).unwrap()(&a, &b, &mut output).unwrap();
    same(&output[..5], &[T::small(6); 5]);
    same(&output[5..], &[sentinel; 2]);
}

pub fn mixed<T: Sample>(session: &MlxSession, foreign: &MlxSession) {
    let left = [T::small(3); 5];
    let right = [T::small(2); 5];
    let sentinel = T::small(7);
    let a = session.upload_encoded(&left).unwrap();
    let b = session.upload_encoded(&right).unwrap();
    let other = foreign.upload_encoded(&right).unwrap();
    let backend = session.checked_integer_backend();
    for op in [Op::Add, Op::Sub, Op::Mul] {
        for range in [Range::Reject, Range::Clamp] {
            let mut kernel =
                graph::fixture_profile::<T, _>(5, op, range, false, [false; 2], |ir| {
                    backend.prepare_host_kernel(ir)
                })
                .unwrap();
            let host = PcuHostArgument::read(PcuBindingRef::new(2, 3), &left);
            let mixed = kernel
                .execute_inputs(&[
                    MlxBinaryInput::Resident {
                        target: PcuBindingRef::new(3, 2),
                        array: &b,
                    },
                    MlxBinaryInput::HostBytes {
                        target: host.target(),
                        scalar: T::TYPE,
                        bytes: host.bytes(),
                    },
                ])
                .unwrap();
            let mut output = [sentinel; 7];
            mixed.output().read_into(&mut output).unwrap();
            same(
                &output[..5],
                &expected(&left, &right, op, range, [false; 2]).0,
            );
            same(&output[5..], &[sentinel; 2]);
            assert!(matches!(
                kernel.execute_inputs(&[
                    MlxBinaryInput::HostBytes {
                        target: host.target(),
                        scalar: T::TYPE,
                        bytes: host.bytes()
                    },
                    MlxBinaryInput::Resident {
                        target: PcuBindingRef::new(3, 2),
                        array: &other
                    }
                ]),
                Err(MlxError::ForeignSession)
            ));
            assert!(!kernel.last_call_may_have_written());
            let resident = kernel.execute_resident(&[&a, &b]).unwrap();
            resident.output().read_into(&mut output).unwrap();
            same(
                &output[..5],
                &expected(&left, &right, op, range, [false; 2]).0,
            );
        }
    }
}
