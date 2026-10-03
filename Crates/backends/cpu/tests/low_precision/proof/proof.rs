//! Independent complete encoding sweep shared by normal and actually requested Portable cuts.
use super::oracle::Low;
#[rustfmt::skip]
use fusion_pcu_cpu::{PcuCpuCheckedBinary, PcuCpuPreparedBinaryError};
#[rustfmt::skip]
use pcu_facade::{PcuBindingRef, PcuDispatchKernelIr, PcuDispatchDataOp, PcuDispatchOp, PcuHostArgument, PcuHostKernelBackend, PcuPreparedHostKernel};
use super::{OPS, POLICIES};
pub fn check<T: Low>(exhaustive: bool, kernel: PcuDispatchKernelIr<'_>) {
    let mut operations = kernel.ops.to_vec();
    for op in OPS {
        for policy in POLICIES {
            for operation in &mut operations {
                if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
                    op: selected,
                    underflow_policy,
                    ..
                }) = operation
                {
                    *selected = op;
                    *underflow_policy = policy;
                }
            }
            let kernel = pcu_facade::PcuDispatchKernelIr {
                ops: &operations,
                numerical_requirements: pcu_facade::PcuImplementationRequirements {
                    float_underflow: policy,
                    ..kernel.numerical_requirements
                },
                ..kernel
            };
            let mut prepared = PcuCpuCheckedBinary::<T>::new()
                .prepare_host_kernel(&kernel)
                .unwrap();
            let count = u32::from(T::FORMAT.sign) * 2;
            let edges = [
                0,
                T::FORMAT.sign,
                1,
                T::FORMAT.sign | 1,
                (1 << T::FORMAT.fraction) - 1,
                1 << T::FORMAT.fraction,
                T::FORMAT.max,
                T::FORMAT.max - 1,
                T::FORMAT.sign | T::FORMAT.max,
                T::FORMAT.max + 1,
                T::FORMAT.sign - 1,
            ];
            for left in 0..count {
                let left = u16::try_from(left).unwrap();
                let mut check_pair = |right: u16| {
                    let lhs = [T::from_bits(left)];
                    let rhs = [T::from_bits(right)];
                    let sentinel = T::from_bits(T::FORMAT.max);
                    let mut output = [sentinel; 3];
                    let result = prepared.call(&mut [
                        PcuHostArgument::read(PcuBindingRef::new(0, 0), &lhs),
                        PcuHostArgument::read(PcuBindingRef::new(0, 1), &rhs),
                        PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output),
                    ]);
                    match T::FORMAT.evaluate(left, right, op, policy) {
                        Ok(expected) => {
                            result.unwrap();
                            assert_eq!(
                                output[0].bits(),
                                expected,
                                "{op:?} {policy:?} {left:x} {right:x}"
                            );
                        }
                        Err(expected) => {
                            let Err(PcuCpuPreparedBinaryError::Fault(fault)) = result else {
                                panic!("expected fault {op:?} {policy:?} {left:x} {right:x}")
                            };
                            assert_eq!(
                                fault.kind, expected,
                                "{op:?} {policy:?} {left:x} {right:x}"
                            );
                            assert_eq!(fault.invocation_id, 0);
                            assert!(!fault.recovered);
                            assert_eq!(output[0], sentinel);
                        }
                    }
                    assert_eq!(output[1..], [sentinel; 2]);
                };
                if exhaustive {
                    for right in 0..count {
                        check_pair(u16::try_from(right).unwrap());
                    }
                } else {
                    for right in edges {
                        check_pair(right);
                    }
                    // All encodings plus a full-period affine partner cover exponent/cancellation gaps.
                    check_pair(left.wrapping_mul(40503).wrapping_add(17));
                }
            }
        }
    }
}
