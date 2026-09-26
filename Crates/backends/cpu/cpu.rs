//! Opt-in CPU execution oracle for the bounded scalar Dispatch and U32 Stream profiles.
//!
//! The implementation is split by semantic concern: integer maps, floating-point maps,
//! stream transforms, and f32 program validation/execution.

#![no_std]

#[cfg(test)]
extern crate std;

#[cfg(test)]
use fusion_pcu::{
    PcuExecutionFault,
    PcuExecutionFaultKind,
};

#[path = "cpu/floats.rs"]
mod floats;
#[path = "cpu/integer.rs"]
mod integer;
#[path = "cpu/stream.rs"]
mod stream;
#[path = "cpu/validation.rs"]
mod validation;

const VALUE_SLOTS: usize = 256;

pub use floats::{
    PcuF32Reference,
    PcuF32ReferenceError,
    PcuF64Reference,
    PcuF64ReferenceError,
};
pub use integer::{
    PcuI8MapReference,
    PcuI8MapReferenceError,
    PcuI16MapReference,
    PcuI16MapReferenceError,
    PcuI32MapReference,
    PcuI32MapReferenceError,
    PcuI64MapReference,
    PcuI64MapReferenceError,
    PcuU8MapReference,
    PcuU8MapReferenceError,
    PcuU16MapReference,
    PcuU16MapReferenceError,
    PcuU32MapReference,
    PcuU32MapReferenceError,
    PcuU64MapReference,
    PcuU64MapReferenceError,
};
pub use stream::{
    PcuU32StreamReference,
    PcuU32StreamReferenceError,
};

#[cfg(test)]
mod tests {
    use core::num::NonZeroU32;
    use std::boxed::Box;

    use super::{
        PcuF32Reference,
        PcuF32ReferenceError,
        PcuF64Reference,
        PcuI8MapReference,
        PcuI16MapReference,
        PcuI32MapReference,
        PcuI64MapReference,
        PcuU8MapReference,
        PcuU16MapReference,
        PcuU32MapReference,
        PcuU64MapReference,
        PcuExecutionFault,
        PcuExecutionFaultKind,
        PcuU32StreamReference,
        PcuU32StreamReferenceError,
    };

    #[test]
    fn u32_map_reference_wraps_add_sub_and_mul() {
        let binding_ir = [
            PcuBinding::scalar::<u32>(
                Some("a"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<u32>(
                Some("b"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<u32>(
                Some("out"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
            ),
        ];
        let a = [u32::MAX, 0, 0x1_0000];
        let b = [1, 1, 0x1_0000];
        for (alu, expected) in [
            (PcuDispatchAluOp::Add, [0, 1, 0x2_0000]),
            (PcuDispatchAluOp::Sub, [u32::MAX - 1, u32::MAX, 0]),
            (PcuDispatchAluOp::Mul, [u32::MAX, 0, 0]),
        ] {
            let ops = [
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result: PcuDispatchValueId(1),
                    binding: PcuBindingRef::new(0, 0),
                    index: PcuDispatchIndex::InvocationId,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result: PcuDispatchValueId(2),
                    binding: PcuBindingRef::new(0, 1),
                    index: PcuDispatchIndex::InvocationId,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                    value_type: PcuValueType::u32(),
                    result: PcuDispatchValueId(3),
                    op: alu,
                    lhs: PcuDispatchValueId(1),
                    rhs: PcuDispatchValueId(2),
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                    binding: PcuBindingRef::new(0, 2),
                    index: PcuDispatchIndex::InvocationId,
                    value: PcuDispatchValueId(3),
                }),
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ];
            let kernel = PcuDispatchKernelIr {
                id: PcuKernelId(42),
                entry: PcuDispatchEntryPoint {
                    name: "u32_map",
                    logical_shape: [3, 1, 1],
                },
                bindings: &binding_ir,
                ports: &[],
                parameters: &[],
                ops: &ops,
                type_caps: PcuValueTypeCaps::for_scalar(PcuScalarType::U32),
                feature_caps: PcuDispatchFeatureCaps::default(),
            };
            let submission = PcuDispatchSubmission {
                kernel: &kernel,
                shape: PcuInvocationShape::invocations(NonZeroU32::new(3).expect("nonzero")),
            };
            let mut output = [0; 3];
            let mut host = [
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 0),
                    slice: PcuHostScalarSlice::Read(&a),
                },
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 1),
                    slice: PcuHostScalarSlice::Read(&b),
                },
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 2),
                    slice: PcuHostScalarSlice::ReadWrite(&mut output),
                },
            ];
            PcuU32MapReference
                .run_host_direct(submission, &mut host, PcuInvocationParameters::empty())
                .expect("u32 operation executes");
            assert_eq!(output, expected);
        }
    }

    fn checked_div_rem_kernel(extent: Option<u32>) -> PcuDispatchKernelIr<'static> {
        let bindings = Box::leak(Box::new([
            PcuBinding::scalar::<u32>(
                Some("a"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<u32>(
                Some("b"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<u32>(
                Some("q"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
            ),
            PcuBinding::scalar::<u32>(
                Some("r"),
                0,
                3,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
            ),
        ]));
        let index = if extent.is_some() {
            PcuDispatchIndex::GridStrideId
        } else {
            PcuDispatchIndex::InvocationId
        };
        let body = Box::leak(Box::new([
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
                value_type: PcuValueType::u32(),
                flags: fusion_pcu::model::PcuIntegerDivFlags::CHECKED,
                quotient: PcuDispatchValueId(3),
                remainder: PcuDispatchValueId(4),
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 3),
                index,
                value: PcuDispatchValueId(4),
            }),
        ]));
        let ops = match extent {
            Some(extent) => Box::leak(Box::new([
                PcuDispatchOp::GridStrideLoop { extent, body },
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ])) as &'static [PcuDispatchOp<'static>],
            None => Box::leak(Box::new([
                body[0],
                body[1],
                body[2],
                body[3],
                body[4],
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ])),
        };
        PcuDispatchKernelIr {
            id: PcuKernelId(77),
            entry: PcuDispatchEntryPoint {
                name: "checked_divrem",
                logical_shape: [2, 1, 1],
            },
            bindings,
            ports: &[],
            parameters: &[],
            ops,
            type_caps: PcuValueTypeCaps::for_scalar(PcuScalarType::U32),
            feature_caps: PcuDispatchFeatureCaps::default(),
        }
    }

    #[test]
    fn checked_u32_divrem_cpu_direct_and_grid_and_zero_fault() {
        for extent in [None, Some(5)] {
            let count = extent.unwrap_or(2) as usize;
            let a: std::vec::Vec<u32> = (0..count)
                .map(|i| 100 + u32::try_from(i).expect("test index fits u32"))
                .collect();
            let b = if extent.is_some() {
                std::vec![3, 4, 5, 0, 2]
            } else {
                std::vec![3, 4]
            };
            let mut q = std::vec![0; count];
            let mut r = std::vec![0; count];
            let kernel = checked_div_rem_kernel(extent);
            let submission = PcuDispatchSubmission {
                kernel: &kernel,
                shape: PcuInvocationShape::invocations(NonZeroU32::new(2).expect("nonzero")),
            };
            let mut host = [
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 0),
                    slice: PcuHostScalarSlice::Read(&a),
                },
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 1),
                    slice: PcuHostScalarSlice::Read(&b),
                },
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 2),
                    slice: PcuHostScalarSlice::ReadWrite(&mut q),
                },
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 3),
                    slice: PcuHostScalarSlice::ReadWrite(&mut r),
                },
            ];
            let outcome = PcuU32MapReference.run_host_direct(
                submission,
                &mut host,
                PcuInvocationParameters::empty(),
            );
            if extent.is_some() {
                assert_eq!(
                    outcome,
                    Err(super::PcuU32MapReferenceError::Fault(
                        fusion_pcu::PcuExecutionFault {
                            kind: fusion_pcu::PcuExecutionFaultKind::DivideByZero,
                            invocation_id: 3
                        }
                    ))
                );
            } else {
                outcome.expect("valid direct DivRem executes");
                assert_eq!(q, [33, 25]);
                assert_eq!(r, [1, 1]);
            }
        }
    }

    fn checked_u64_div_rem_kernel(extent: Option<u32>) -> PcuDispatchKernelIr<'static> {
        let bindings = Box::leak(Box::new([
            PcuBinding::scalar::<u64>(
                Some("a"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<u64>(
                Some("b"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<u64>(
                Some("q"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
            ),
            PcuBinding::scalar::<u64>(
                Some("r"),
                0,
                3,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
            ),
        ]));
        let index = if extent.is_some() {
            PcuDispatchIndex::GridStrideId
        } else {
            PcuDispatchIndex::InvocationId
        };
        let body = Box::leak(Box::new([
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
                value_type: PcuValueType::u64(),
                flags: fusion_pcu::model::PcuIntegerDivFlags::CHECKED,
                quotient: PcuDispatchValueId(3),
                remainder: PcuDispatchValueId(4),
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 3),
                index,
                value: PcuDispatchValueId(4),
            }),
        ]));
        let ops = match extent {
            Some(extent) => Box::leak(Box::new([
                PcuDispatchOp::GridStrideLoop { extent, body },
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ])) as &'static [PcuDispatchOp<'static>],
            None => Box::leak(Box::new([
                body[0],
                body[1],
                body[2],
                body[3],
                body[4],
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ])),
        };
        PcuDispatchKernelIr {
            id: PcuKernelId(78),
            entry: PcuDispatchEntryPoint {
                name: "checked_divrem_u64",
                logical_shape: [2, 1, 1],
            },
            bindings,
            ports: &[],
            parameters: &[],
            ops,
            type_caps: PcuValueTypeCaps::for_scalar(PcuScalarType::U64),
            feature_caps: PcuDispatchFeatureCaps::default(),
        }
    }

    #[test]
    fn checked_u64_divrem_cpu_direct_and_grid_and_terminal_lowest_zero() {
        let kernel = checked_u64_div_rem_kernel(None);
        let a = [u64::MAX, 0x1234_5678_9abc_def0];
        let b = [2, 3];
        let mut q = [0; 2];
        let mut r = [0; 2];
        let submission = PcuDispatchSubmission {
            kernel: &kernel,
            shape: PcuInvocationShape::invocations(NonZeroU32::new(2).expect("nonzero")),
        };
        let mut host = [
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 0),
                slice: PcuHostScalarSlice::Read(&a),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 1),
                slice: PcuHostScalarSlice::Read(&b),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 2),
                slice: PcuHostScalarSlice::ReadWrite(&mut q),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 3),
                slice: PcuHostScalarSlice::ReadWrite(&mut r),
            },
        ];
        PcuU64MapReference
            .run_host_direct(submission, &mut host, PcuInvocationParameters::empty())
            .expect("u64 DivRem executes");
        assert_eq!(q, [u64::MAX / 2, 0x1234_5678_9abc_def0 / 3]);
        assert_eq!(r, [u64::MAX % 2, 0x1234_5678_9abc_def0 % 3]);

        let kernel = checked_u64_div_rem_kernel(Some(5));
        let a = [10, 20, 30, 40, 50];
        let b = [2, 4, 5, 0, 0];
        let mut q = [91; 5];
        let mut r = [92; 5];
        let submission = PcuDispatchSubmission {
            kernel: &kernel,
            shape: PcuInvocationShape::invocations(NonZeroU32::new(2).expect("nonzero")),
        };
        let mut host = [
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 0),
                slice: PcuHostScalarSlice::Read(&a),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 1),
                slice: PcuHostScalarSlice::Read(&b),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 2),
                slice: PcuHostScalarSlice::ReadWrite(&mut q),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 3),
                slice: PcuHostScalarSlice::ReadWrite(&mut r),
            },
        ];
        assert_eq!(
            PcuU64MapReference.run_host_direct(
                submission,
                &mut host,
                PcuInvocationParameters::empty()
            ),
            Err(super::PcuU64MapReferenceError::Fault(PcuExecutionFault {
                kind: PcuExecutionFaultKind::DivideByZero,
                invocation_id: 3,
            }))
        );
        assert_eq!(q, [91; 5]);
        assert_eq!(r, [92; 5]);
    }

    fn checked_i32_div_rem_kernel(extent: Option<u32>) -> PcuDispatchKernelIr<'static> {
        let bindings = Box::leak(Box::new([
            PcuBinding::scalar::<i32>(
                Some("a"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<i32>(
                Some("b"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<i32>(
                Some("q"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
            ),
            PcuBinding::scalar::<i32>(
                Some("r"),
                0,
                3,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
            ),
        ]));
        let index = if extent.is_some() {
            PcuDispatchIndex::GridStrideId
        } else {
            PcuDispatchIndex::InvocationId
        };
        let body = Box::leak(Box::new([
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
                value_type: PcuValueType::i32(),
                flags: fusion_pcu::model::PcuIntegerDivFlags::CHECKED,
                quotient: PcuDispatchValueId(3),
                remainder: PcuDispatchValueId(4),
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 3),
                index,
                value: PcuDispatchValueId(4),
            }),
        ]));
        let ops = match extent {
            Some(extent) => Box::leak(Box::new([
                PcuDispatchOp::GridStrideLoop { extent, body },
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ])) as &'static [PcuDispatchOp<'static>],
            None => Box::leak(Box::new([
                body[0],
                body[1],
                body[2],
                body[3],
                body[4],
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ])),
        };
        PcuDispatchKernelIr {
            id: PcuKernelId(79),
            entry: PcuDispatchEntryPoint {
                name: "checked_divrem_i32",
                logical_shape: [2, 1, 1],
            },
            bindings,
            ports: &[],
            parameters: &[],
            ops,
            type_caps: PcuValueTypeCaps::for_scalar(PcuScalarType::I32),
            feature_caps: PcuDispatchFeatureCaps::default(),
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn checked_i32_divrem_reports_both_fault_kinds_and_lowest_grid_id() {
        let kernel = checked_i32_div_rem_kernel(None);
        let a = [-13, 10];
        let b = [3, -2];
        let mut q = [0; 2];
        let mut r = [0; 2];
        let submission = PcuDispatchSubmission {
            kernel: &kernel,
            shape: PcuInvocationShape::invocations(NonZeroU32::new(2).expect("nonzero")),
        };
        let mut host = [
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 0),
                slice: PcuHostScalarSlice::Read(&a),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 1),
                slice: PcuHostScalarSlice::Read(&b),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 2),
                slice: PcuHostScalarSlice::ReadWrite(&mut q),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 3),
                slice: PcuHostScalarSlice::ReadWrite(&mut r),
            },
        ];
        PcuI32MapReference
            .run_host_direct(submission, &mut host, PcuInvocationParameters::empty())
            .expect("i32 DivRem executes");
        assert_eq!(q, [-4, -5]);
        assert_eq!(r, [-1, 0]);

        let kernel = checked_i32_div_rem_kernel(Some(5));
        let a = [8, 12, i32::MIN, 99, 20];
        let b = [2, 3, -1, 0, 0];
        let mut q = [91; 5];
        let mut r = [92; 5];
        let submission = PcuDispatchSubmission {
            kernel: &kernel,
            shape: PcuInvocationShape::invocations(NonZeroU32::new(2).expect("nonzero")),
        };
        let mut host = [
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 0),
                slice: PcuHostScalarSlice::Read(&a),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 1),
                slice: PcuHostScalarSlice::Read(&b),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 2),
                slice: PcuHostScalarSlice::ReadWrite(&mut q),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 3),
                slice: PcuHostScalarSlice::ReadWrite(&mut r),
            },
        ];
        assert_eq!(
            PcuI32MapReference.run_host_direct(
                submission,
                &mut host,
                PcuInvocationParameters::empty()
            ),
            Err(super::PcuI32MapReferenceError::Fault(PcuExecutionFault {
                kind: PcuExecutionFaultKind::SignedDivisionOverflow,
                invocation_id: 2,
            }))
        );
        assert_eq!(q, [91; 5]);
        assert_eq!(r, [92; 5]);

        let kernel = checked_i32_div_rem_kernel(None);
        let a = [i32::MIN, 0];
        let b = [-1, 1];
        let mut q = [91; 2];
        let mut r = [92; 2];
        let submission = PcuDispatchSubmission {
            kernel: &kernel,
            shape: PcuInvocationShape::invocations(NonZeroU32::new(2).expect("nonzero")),
        };
        let mut host = [
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 0),
                slice: PcuHostScalarSlice::Read(&a),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 1),
                slice: PcuHostScalarSlice::Read(&b),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 2),
                slice: PcuHostScalarSlice::ReadWrite(&mut q),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 3),
                slice: PcuHostScalarSlice::ReadWrite(&mut r),
            },
        ];
        assert_eq!(
            PcuI32MapReference.run_host_direct(
                submission,
                &mut host,
                PcuInvocationParameters::empty()
            ),
            Err(super::PcuI32MapReferenceError::Fault(PcuExecutionFault {
                kind: PcuExecutionFaultKind::SignedDivisionOverflow,
                invocation_id: 0,
            }))
        );
        assert_eq!(q, [91; 2]);
        assert_eq!(r, [92; 2]);

        let kernel = checked_i32_div_rem_kernel(None);
        let a = [4, 5];
        let b = [0, 1];
        let mut q = [91; 2];
        let mut r = [92; 2];
        let submission = PcuDispatchSubmission {
            kernel: &kernel,
            shape: PcuInvocationShape::invocations(NonZeroU32::new(2).expect("nonzero")),
        };
        let mut host = [
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 0),
                slice: PcuHostScalarSlice::Read(&a),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 1),
                slice: PcuHostScalarSlice::Read(&b),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 2),
                slice: PcuHostScalarSlice::ReadWrite(&mut q),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 3),
                slice: PcuHostScalarSlice::ReadWrite(&mut r),
            },
        ];
        assert_eq!(
            PcuI32MapReference.run_host_direct(
                submission,
                &mut host,
                PcuInvocationParameters::empty()
            ),
            Err(super::PcuI32MapReferenceError::Fault(PcuExecutionFault {
                kind: PcuExecutionFaultKind::DivideByZero,
                invocation_id: 0,
            }))
        );
        assert_eq!(q, [91; 2]);
        assert_eq!(r, [92; 2]);
    }

    #[test]
    fn u16_map_reference_wraps_add_sub_and_mul() {
        let binding_ir = [
            PcuBinding::scalar::<u16>(
                Some("a"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<u16>(
                Some("b"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<u16>(
                Some("out"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
            ),
        ];
        let a = [u16::MAX, 0, 20_000];
        let b = [1, 1, 3];
        for (alu, expected) in [
            (PcuDispatchAluOp::Add, [0, 1, 20_003]),
            (PcuDispatchAluOp::Sub, [u16::MAX - 1, u16::MAX, 19_997]),
            (PcuDispatchAluOp::Mul, [u16::MAX, 0, 60_000]),
        ] {
            let ops = [
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result: PcuDispatchValueId(1),
                    binding: PcuBindingRef::new(0, 0),
                    index: PcuDispatchIndex::InvocationId,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result: PcuDispatchValueId(2),
                    binding: PcuBindingRef::new(0, 1),
                    index: PcuDispatchIndex::InvocationId,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                    value_type: PcuValueType::u16(),
                    result: PcuDispatchValueId(3),
                    op: alu,
                    lhs: PcuDispatchValueId(1),
                    rhs: PcuDispatchValueId(2),
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                    binding: PcuBindingRef::new(0, 2),
                    index: PcuDispatchIndex::InvocationId,
                    value: PcuDispatchValueId(3),
                }),
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ];
            let kernel = PcuDispatchKernelIr {
                id: PcuKernelId(42),
                entry: PcuDispatchEntryPoint {
                    name: "u16_map",
                    logical_shape: [3, 1, 1],
                },
                bindings: &binding_ir,
                ports: &[],
                parameters: &[],
                ops: &ops,
                type_caps: PcuValueTypeCaps::for_scalar(PcuScalarType::U16),
                feature_caps: PcuDispatchFeatureCaps::default(),
            };
            let submission = PcuDispatchSubmission {
                kernel: &kernel,
                shape: PcuInvocationShape::invocations(NonZeroU32::new(3).expect("nonzero")),
            };
            let mut output = [0; 3];
            let mut host = [
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 0),
                    slice: PcuHostScalarSlice::Read(&a),
                },
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 1),
                    slice: PcuHostScalarSlice::Read(&b),
                },
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 2),
                    slice: PcuHostScalarSlice::ReadWrite(&mut output),
                },
            ];
            PcuU16MapReference
                .run_host_direct(submission, &mut host, PcuInvocationParameters::empty())
                .expect("u16 operation executes");
            assert_eq!(output, expected);
        }
    }

    #[test]
    fn u8_map_reference_wraps_add_sub_and_mul() {
        let binding_ir = [
            PcuBinding::scalar::<u8>(
                Some("a"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<u8>(
                Some("b"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<u8>(
                Some("out"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
            ),
        ];
        let a = [u8::MAX, 0, 20];
        let b = [1, 1, 3];
        for (alu, expected) in [
            (PcuDispatchAluOp::Add, [0, 1, 23]),
            (PcuDispatchAluOp::Sub, [u8::MAX - 1, u8::MAX, 17]),
            (PcuDispatchAluOp::Mul, [u8::MAX, 0, 60]),
        ] {
            let ops = [
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result: PcuDispatchValueId(1),
                    binding: PcuBindingRef::new(0, 0),
                    index: PcuDispatchIndex::InvocationId,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result: PcuDispatchValueId(2),
                    binding: PcuBindingRef::new(0, 1),
                    index: PcuDispatchIndex::InvocationId,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                    value_type: PcuValueType::u8(),
                    result: PcuDispatchValueId(3),
                    op: alu,
                    lhs: PcuDispatchValueId(1),
                    rhs: PcuDispatchValueId(2),
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                    binding: PcuBindingRef::new(0, 2),
                    index: PcuDispatchIndex::InvocationId,
                    value: PcuDispatchValueId(3),
                }),
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ];
            let kernel = PcuDispatchKernelIr {
                id: PcuKernelId(42),
                entry: PcuDispatchEntryPoint {
                    name: "u8_map",
                    logical_shape: [3, 1, 1],
                },
                bindings: &binding_ir,
                ports: &[],
                parameters: &[],
                ops: &ops,
                type_caps: PcuValueTypeCaps::for_scalar(PcuScalarType::U8),
                feature_caps: PcuDispatchFeatureCaps::default(),
            };
            let submission = PcuDispatchSubmission {
                kernel: &kernel,
                shape: PcuInvocationShape::invocations(NonZeroU32::new(3).expect("nonzero")),
            };
            let mut output = [0; 3];
            let mut host = [
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 0),
                    slice: PcuHostScalarSlice::Read(&a),
                },
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 1),
                    slice: PcuHostScalarSlice::Read(&b),
                },
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 2),
                    slice: PcuHostScalarSlice::ReadWrite(&mut output),
                },
            ];
            PcuU8MapReference
                .run_host_direct(submission, &mut host, PcuInvocationParameters::empty())
                .expect("u8 operation executes");
            assert_eq!(output, expected);
        }
    }

    #[test]
    fn u64_map_reference_wraps_add_sub_and_mul() {
        let binding_ir = [
            PcuBinding::scalar::<u64>(
                Some("a"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<u64>(
                Some("b"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<u64>(
                Some("out"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
            ),
        ];
        let a = [u64::MAX, 0, 0x1_0000_0000];
        let b = [1, 1, 0x1_0000_0000];
        for (alu, expected) in [
            (PcuDispatchAluOp::Add, [0, 1, 0x2_0000_0000]),
            (PcuDispatchAluOp::Sub, [u64::MAX - 1, u64::MAX, 0]),
            (PcuDispatchAluOp::Mul, [u64::MAX, 0, 0]),
        ] {
            let ops = [
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result: PcuDispatchValueId(1),
                    binding: PcuBindingRef::new(0, 0),
                    index: PcuDispatchIndex::InvocationId,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result: PcuDispatchValueId(2),
                    binding: PcuBindingRef::new(0, 1),
                    index: PcuDispatchIndex::InvocationId,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                    value_type: PcuValueType::u64(),
                    result: PcuDispatchValueId(3),
                    op: alu,
                    lhs: PcuDispatchValueId(1),
                    rhs: PcuDispatchValueId(2),
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                    binding: PcuBindingRef::new(0, 2),
                    index: PcuDispatchIndex::InvocationId,
                    value: PcuDispatchValueId(3),
                }),
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ];
            let kernel = PcuDispatchKernelIr {
                id: PcuKernelId(42),
                entry: PcuDispatchEntryPoint {
                    name: "u64_map",
                    logical_shape: [3, 1, 1],
                },
                bindings: &binding_ir,
                ports: &[],
                parameters: &[],
                ops: &ops,
                type_caps: PcuValueTypeCaps::for_scalar(PcuScalarType::U64),
                feature_caps: PcuDispatchFeatureCaps::default(),
            };
            let submission = PcuDispatchSubmission {
                kernel: &kernel,
                shape: PcuInvocationShape::invocations(NonZeroU32::new(3).expect("nonzero")),
            };
            let mut output = [0; 3];
            let mut host = [
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 0),
                    slice: PcuHostScalarSlice::Read(&a),
                },
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 1),
                    slice: PcuHostScalarSlice::Read(&b),
                },
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 2),
                    slice: PcuHostScalarSlice::ReadWrite(&mut output),
                },
            ];
            PcuU64MapReference
                .run_host_direct(submission, &mut host, PcuInvocationParameters::empty())
                .expect("u64 operation executes");
            assert_eq!(output, expected);
        }
    }

    #[test]
    fn i64_map_reference_wraps_add_sub_and_mul() {
        let binding_ir = [
            PcuBinding::scalar::<i64>(
                Some("a"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<i64>(
                Some("b"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<i64>(
                Some("out"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
            ),
        ];
        let a = [i64::MAX, i64::MIN, -2];
        let b = [1, -1, i64::MAX];
        for (alu, expected) in [
            (PcuDispatchAluOp::Add, [i64::MIN, i64::MAX, i64::MAX - 2]),
            (
                PcuDispatchAluOp::Sub,
                [i64::MAX - 1, i64::MIN + 1, i64::MAX],
            ),
            (PcuDispatchAluOp::Mul, [i64::MAX, i64::MIN, 2]),
        ] {
            let ops = [
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result: PcuDispatchValueId(1),
                    binding: PcuBindingRef::new(0, 0),
                    index: PcuDispatchIndex::InvocationId,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result: PcuDispatchValueId(2),
                    binding: PcuBindingRef::new(0, 1),
                    index: PcuDispatchIndex::InvocationId,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                    value_type: PcuValueType::i64(),
                    result: PcuDispatchValueId(3),
                    op: alu,
                    lhs: PcuDispatchValueId(1),
                    rhs: PcuDispatchValueId(2),
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                    binding: PcuBindingRef::new(0, 2),
                    index: PcuDispatchIndex::InvocationId,
                    value: PcuDispatchValueId(3),
                }),
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ];
            let kernel = PcuDispatchKernelIr {
                id: PcuKernelId(42),
                entry: PcuDispatchEntryPoint {
                    name: "i64_map",
                    logical_shape: [3, 1, 1],
                },
                bindings: &binding_ir,
                ports: &[],
                parameters: &[],
                ops: &ops,
                type_caps: PcuValueTypeCaps::for_scalar(PcuScalarType::I64),
                feature_caps: PcuDispatchFeatureCaps::default(),
            };
            let submission = PcuDispatchSubmission {
                kernel: &kernel,
                shape: PcuInvocationShape::invocations(NonZeroU32::new(3).expect("nonzero")),
            };
            let mut output = [0; 3];
            let mut host = [
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 0),
                    slice: PcuHostScalarSlice::Read(&a),
                },
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 1),
                    slice: PcuHostScalarSlice::Read(&b),
                },
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 2),
                    slice: PcuHostScalarSlice::ReadWrite(&mut output),
                },
            ];
            PcuI64MapReference
                .run_host_direct(submission, &mut host, PcuInvocationParameters::empty())
                .expect("i64 operation executes");
            assert_eq!(output, expected);
        }
    }

    #[test]
    fn i32_map_reference_wraps_add_sub_and_mul() {
        let binding_ir = [
            PcuBinding::scalar::<i32>(
                Some("a"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<i32>(
                Some("b"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<i32>(
                Some("out"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
            ),
        ];
        let a = [i32::MAX, i32::MIN, -2];
        let b = [1, -1, i32::MAX];
        for (alu, expected) in [
            (PcuDispatchAluOp::Add, [i32::MIN, i32::MAX, i32::MAX - 2]),
            (
                PcuDispatchAluOp::Sub,
                [i32::MAX - 1, i32::MIN + 1, i32::MAX],
            ),
            (PcuDispatchAluOp::Mul, [i32::MAX, i32::MIN, 2]),
        ] {
            let ops = [
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result: PcuDispatchValueId(1),
                    binding: PcuBindingRef::new(0, 0),
                    index: PcuDispatchIndex::InvocationId,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result: PcuDispatchValueId(2),
                    binding: PcuBindingRef::new(0, 1),
                    index: PcuDispatchIndex::InvocationId,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                    value_type: PcuValueType::i32(),
                    result: PcuDispatchValueId(3),
                    op: alu,
                    lhs: PcuDispatchValueId(1),
                    rhs: PcuDispatchValueId(2),
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                    binding: PcuBindingRef::new(0, 2),
                    index: PcuDispatchIndex::InvocationId,
                    value: PcuDispatchValueId(3),
                }),
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ];
            let kernel = PcuDispatchKernelIr {
                id: PcuKernelId(42),
                entry: PcuDispatchEntryPoint {
                    name: "i32_map",
                    logical_shape: [3, 1, 1],
                },
                bindings: &binding_ir,
                ports: &[],
                parameters: &[],
                ops: &ops,
                type_caps: PcuValueTypeCaps::for_scalar(PcuScalarType::I32),
                feature_caps: PcuDispatchFeatureCaps::default(),
            };
            let submission = PcuDispatchSubmission {
                kernel: &kernel,
                shape: PcuInvocationShape::invocations(NonZeroU32::new(3).expect("nonzero")),
            };
            let mut output = [0; 3];
            let mut host = [
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 0),
                    slice: PcuHostScalarSlice::Read(&a),
                },
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 1),
                    slice: PcuHostScalarSlice::Read(&b),
                },
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 2),
                    slice: PcuHostScalarSlice::ReadWrite(&mut output),
                },
            ];
            PcuI32MapReference
                .run_host_direct(submission, &mut host, PcuInvocationParameters::empty())
                .expect("i32 operation executes");
            assert_eq!(output, expected);
        }
    }

    #[test]
    fn i16_map_reference_wraps_add_sub_and_mul() {
        let binding_ir = [
            PcuBinding::scalar::<i16>(
                Some("a"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<i16>(
                Some("b"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<i16>(
                Some("out"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
            ),
        ];
        let a = [i16::MAX, i16::MIN, -2];
        let b = [1, -1, i16::MAX];
        for (alu, expected) in [
            (PcuDispatchAluOp::Add, [i16::MIN, i16::MAX, i16::MAX - 2]),
            (
                PcuDispatchAluOp::Sub,
                [i16::MAX - 1, i16::MIN + 1, i16::MAX],
            ),
            (PcuDispatchAluOp::Mul, [i16::MAX, i16::MIN, 2]),
        ] {
            let ops = [
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result: PcuDispatchValueId(1),
                    binding: PcuBindingRef::new(0, 0),
                    index: PcuDispatchIndex::InvocationId,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result: PcuDispatchValueId(2),
                    binding: PcuBindingRef::new(0, 1),
                    index: PcuDispatchIndex::InvocationId,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                    value_type: PcuValueType::i16(),
                    result: PcuDispatchValueId(3),
                    op: alu,
                    lhs: PcuDispatchValueId(1),
                    rhs: PcuDispatchValueId(2),
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                    binding: PcuBindingRef::new(0, 2),
                    index: PcuDispatchIndex::InvocationId,
                    value: PcuDispatchValueId(3),
                }),
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ];
            let kernel = PcuDispatchKernelIr {
                id: PcuKernelId(42),
                entry: PcuDispatchEntryPoint {
                    name: "i16_map",
                    logical_shape: [3, 1, 1],
                },
                bindings: &binding_ir,
                ports: &[],
                parameters: &[],
                ops: &ops,
                type_caps: PcuValueTypeCaps::for_scalar(PcuScalarType::I16),
                feature_caps: PcuDispatchFeatureCaps::default(),
            };
            let submission = PcuDispatchSubmission {
                kernel: &kernel,
                shape: PcuInvocationShape::invocations(NonZeroU32::new(3).expect("nonzero")),
            };
            let mut output = [0; 3];
            let mut host = [
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 0),
                    slice: PcuHostScalarSlice::Read(&a),
                },
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 1),
                    slice: PcuHostScalarSlice::Read(&b),
                },
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 2),
                    slice: PcuHostScalarSlice::ReadWrite(&mut output),
                },
            ];
            PcuI16MapReference
                .run_host_direct(submission, &mut host, PcuInvocationParameters::empty())
                .expect("i16 operation executes");
            assert_eq!(output, expected);
        }
    }

    #[test]
    fn i8_map_reference_wraps_add_sub_and_mul() {
        let binding_ir = [
            PcuBinding::scalar::<i8>(
                Some("a"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<i8>(
                Some("b"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<i8>(
                Some("out"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
            ),
        ];
        let a = [i8::MAX, i8::MIN, -2];
        let b = [1, -1, i8::MAX];
        for (alu, expected) in [
            (PcuDispatchAluOp::Add, [i8::MIN, i8::MAX, i8::MAX - 2]),
            (PcuDispatchAluOp::Sub, [i8::MAX - 1, i8::MIN + 1, i8::MAX]),
            (PcuDispatchAluOp::Mul, [i8::MAX, i8::MIN, 2]),
        ] {
            let ops = [
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result: PcuDispatchValueId(1),
                    binding: PcuBindingRef::new(0, 0),
                    index: PcuDispatchIndex::InvocationId,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result: PcuDispatchValueId(2),
                    binding: PcuBindingRef::new(0, 1),
                    index: PcuDispatchIndex::InvocationId,
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                    value_type: PcuValueType::i8(),
                    result: PcuDispatchValueId(3),
                    op: alu,
                    lhs: PcuDispatchValueId(1),
                    rhs: PcuDispatchValueId(2),
                }),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                    binding: PcuBindingRef::new(0, 2),
                    index: PcuDispatchIndex::InvocationId,
                    value: PcuDispatchValueId(3),
                }),
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ];
            let kernel = PcuDispatchKernelIr {
                id: PcuKernelId(42),
                entry: PcuDispatchEntryPoint {
                    name: "i8_map",
                    logical_shape: [3, 1, 1],
                },
                bindings: &binding_ir,
                ports: &[],
                parameters: &[],
                ops: &ops,
                type_caps: PcuValueTypeCaps::for_scalar(PcuScalarType::I8),
                feature_caps: PcuDispatchFeatureCaps::default(),
            };
            let submission = PcuDispatchSubmission {
                kernel: &kernel,
                shape: PcuInvocationShape::invocations(NonZeroU32::new(3).expect("nonzero")),
            };
            let mut output = [0; 3];
            let mut host = [
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 0),
                    slice: PcuHostScalarSlice::Read(&a),
                },
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 1),
                    slice: PcuHostScalarSlice::Read(&b),
                },
                PcuHostScalarBinding {
                    target: PcuBindingRef::new(0, 2),
                    slice: PcuHostScalarSlice::ReadWrite(&mut output),
                },
            ];
            PcuI8MapReference
                .run_host_direct(submission, &mut host, PcuInvocationParameters::empty())
                .expect("i8 operation executes");
            assert_eq!(output, expected);
        }
    }

    #[test]
    fn u64_map_reference_copies_identity_values() {
        let binding_ir = [
            PcuBinding::scalar::<u64>(
                Some("input"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<u64>(
                Some("output"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
            ),
        ];
        let ops = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(1),
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let kernel = PcuDispatchKernelIr {
            id: PcuKernelId(99),
            entry: PcuDispatchEntryPoint {
                name: "u64_identity",
                logical_shape: [4, 1, 1],
            },
            bindings: &binding_ir,
            ports: &[],
            parameters: &[],
            ops: &ops,
            type_caps: PcuValueTypeCaps::for_scalar(PcuScalarType::U64),
            feature_caps: PcuDispatchFeatureCaps::default(),
        };
        let submission = PcuDispatchSubmission {
            kernel: &kernel,
            shape: PcuInvocationShape::invocations(NonZeroU32::new(4).expect("nonzero")),
        };
        let input = [0, 1, 1_u64 << 63, u64::MAX];
        let mut output = [0; 4];
        let mut host = [
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 0),
                slice: PcuHostScalarSlice::Read(&input),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 1),
                slice: PcuHostScalarSlice::ReadWrite(&mut output),
            },
        ];
        PcuU64MapReference
            .run_host_direct(submission, &mut host, PcuInvocationParameters::empty())
            .expect("u64 identity copy executes");
        assert_eq!(output, input);
    }

    #[test]
    fn u32_map_reference_executes_chained_wrapping_alu() {
        let bindings = [
            PcuBinding::scalar::<u32>(
                Some("a"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<u32>(
                Some("b"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<u32>(
                Some("out"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
            ),
        ];
        let ops = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: PcuValueType::u32(),
                result: PcuDispatchValueId(3),
                op: PcuDispatchAluOp::Add,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: PcuValueType::u32(),
                result: PcuDispatchValueId(4),
                op: PcuDispatchAluOp::Mul,
                lhs: PcuDispatchValueId(3),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(4),
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let kernel = PcuDispatchKernelIr {
            id: PcuKernelId(43),
            entry: PcuDispatchEntryPoint {
                name: "u32_chain",
                logical_shape: [2, 1, 1],
            },
            bindings: &bindings,
            ports: &[],
            parameters: &[],
            ops: &ops,
            type_caps: PcuValueTypeCaps::for_scalar(PcuScalarType::U32),
            feature_caps: PcuDispatchFeatureCaps::default(),
        };
        let a = [u32::MAX, 3];
        let b = [2, 4];
        let mut output = [0; 2];
        let mut host = [
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 0),
                slice: PcuHostScalarSlice::Read(&a),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 1),
                slice: PcuHostScalarSlice::Read(&b),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 2),
                slice: PcuHostScalarSlice::ReadWrite(&mut output),
            },
        ];
        let submission = PcuDispatchSubmission {
            kernel: &kernel,
            shape: PcuInvocationShape::invocations(NonZeroU32::new(2).expect("nonzero")),
        };
        PcuU32MapReference
            .run_host_direct(submission, &mut host, PcuInvocationParameters::empty())
            .expect("chain executes");
        assert_eq!(output, [2, 28]);
    }
    use fusion_pcu::{
        PcuBinding,
        PcuBindingAccess,
        PcuBindingRef,
        PcuBindingStorageClass,
        PcuDispatchAluOp,
        PcuDispatchControlOp,
        PcuDispatchDataOp,
        PcuDispatchEntryPoint,
        PcuDispatchIndex,
        PcuDispatchKernelIr,
        PcuDispatchOpCaps,
        PcuDispatchOp,
        PcuDispatchSubmission,
        PcuDispatchValueId,
        PcuDispatchFeatureCaps,
        PcuHostScalarBinding,
        PcuHostScalarSlice,
        PcuInvocationParameters,
        PcuInvocationShape,
        PcuKernelId,
        PcuParameterValue,
        PcuScalarType,
        PcuSynchronousHostDispatchBackend,
        PcuValueType,
        PcuValueTypeCaps,
        PcuStreamPattern,
        F32MapBuilder,
        validate_host_scalar_bindings,
    };
    use fusion_pcu::model::PcuStreamKernelBuilder;

    #[test]
    fn f32_reference_broadcasts_single_element_binding() {
        assert!(PcuF32Reference::SUPPORTED_INSTRUCTIONS.contains(
            PcuDispatchOpCaps::BINDING_LOAD.union(PcuDispatchOpCaps::BINDING_LOAD_ELEMENT_ZERO)
        ));
        let bindings = [
            PcuBinding::scalar::<f32>(
                Some("scalar"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<f32>(
                Some("output"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
            ),
        ];
        let (builder, scalar) = F32MapBuilder::<3>::new(7, "broadcast", [4, 1, 1], &bindings)
            .load_f32_broadcast(PcuBindingRef::new(0, 0))
            .expect("scalar load");
        let builder = builder
            .store_f32(PcuBindingRef::new(0, 1), scalar)
            .expect("output store");
        let kernel = builder.ir();
        let submission = PcuDispatchSubmission {
            kernel: &kernel,
            shape: PcuInvocationShape::invocations(NonZeroU32::new(4).expect("nonzero")),
        };
        let mut rejected_output = [0.0_f32; 4];
        let rejected_bindings = [
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 0),
                slice: PcuHostScalarSlice::Read(&[]),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 1),
                slice: PcuHostScalarSlice::ReadWrite(&mut rejected_output),
            },
        ];
        assert!(validate_host_scalar_bindings::<f32, ()>(submission, &rejected_bindings).is_err());
        let scalar_input = [12.5_f32];
        let mut output = [0.0_f32; 4];
        let mut host_bindings = [
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 0),
                slice: PcuHostScalarSlice::Read(&scalar_input),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 1),
                slice: PcuHostScalarSlice::ReadWrite(&mut output),
            },
        ];

        PcuF32Reference
            .run_host_direct(
                submission,
                &mut host_bindings,
                PcuInvocationParameters::empty(),
            )
            .expect("broadcast executes");

        assert_eq!(output.map(f32::to_bits), [12.5_f32; 4].map(f32::to_bits));
    }

    #[test]
    fn u32_stream_reference_matches_common_profile_vectors() {
        let vectors = [
            (PcuStreamPattern::BitReverse, 0x0000_0001, 0x8000_0000),
            (PcuStreamPattern::BitInvert, 0x00ff_00ff, 0xff00_ff00),
            (PcuStreamPattern::Increment, u32::MAX, 0),
            (PcuStreamPattern::Decrement, 0, u32::MAX),
            (
                PcuStreamPattern::ShiftLeft { bits: 3 },
                0x8000_0003,
                0x0000_0018,
            ),
            (
                PcuStreamPattern::ShiftRight { bits: 4 },
                0x8000_003f,
                0x0800_0003,
            ),
            (PcuStreamPattern::ShiftLeft { bits: 32 }, 0x1234_5678, 0),
            (
                PcuStreamPattern::ExtractBits {
                    offset: 8,
                    width: 8,
                },
                0x1234_56ab,
                0x56,
            ),
            (PcuStreamPattern::MaskLower { bits: 12 }, 0xabcd_1234, 0x234),
            (PcuStreamPattern::ByteSwap32, 0x1234_56ab, 0xab56_3412),
        ];
        let reference = PcuU32StreamReference;
        for (pattern, input, expected) in vectors {
            let builder = PcuStreamKernelBuilder::<1>::words(7, "cpu_reference")
                .with_pattern(pattern)
                .expect("one pattern fits");
            assert_eq!(reference.transform(&builder.ir(), input), Ok(expected));
        }
    }

    #[test]
    fn u32_stream_reference_rejects_non_profile_patterns_and_composition() {
        let parameterized = PcuStreamKernelBuilder::<1>::words(8, "parameterized")
            .with_pattern(PcuStreamPattern::AddParameter {
                parameter: fusion_pcu::PcuParameterSlot(0),
            })
            .expect("one pattern fits");
        assert_eq!(
            PcuU32StreamReference.transform(&parameterized.ir(), 1),
            Err(PcuU32StreamReferenceError::UnsupportedPattern(
                PcuStreamPattern::AddParameter {
                    parameter: fusion_pcu::PcuParameterSlot(0),
                }
            ))
        );

        let composed = PcuStreamKernelBuilder::<2>::words(9, "composed")
            .increment()
            .expect("pattern fits")
            .decrement()
            .expect("pattern fits");
        assert_eq!(
            PcuU32StreamReference.transform(&composed.ir(), 1),
            Err(PcuU32StreamReferenceError::InvalidPatternCount(2))
        );

        let invalid_shift = PcuStreamKernelBuilder::<1>::words(10, "invalid_shift")
            .with_pattern(PcuStreamPattern::ShiftRight { bits: 0 })
            .expect("builder preserves the invalid pattern for backend validation");
        assert_eq!(
            PcuU32StreamReference.transform(&invalid_shift.ir(), 1),
            Err(PcuU32StreamReferenceError::InvalidPattern(
                PcuStreamPattern::ShiftRight { bits: 0 }
            ))
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn min_max_execute_and_unsupported_programs_preserve_outputs() {
        let bindings_ir = [
            PcuBinding::value(
                Some("lhs"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::f32(),
            ),
            PcuBinding::value(
                Some("rhs"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::f32(),
            ),
            PcuBinding::value(
                Some("min"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::f32(),
            ),
            PcuBinding::value(
                Some("max"),
                0,
                3,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::f32(),
            ),
        ];
        let id0 = PcuDispatchValueId(1);
        let id1 = PcuDispatchValueId(2);
        let id2 = PcuDispatchValueId(3);
        let id3 = PcuDispatchValueId(4);
        let ops = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: id0,
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: id1,
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: fusion_pcu::PcuValueType::f32(),
                result: id2,
                op: PcuDispatchAluOp::Min,
                lhs: id0,
                rhs: id1,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: fusion_pcu::PcuValueType::f32(),
                result: id3,
                op: PcuDispatchAluOp::Max,
                lhs: id0,
                rhs: id1,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::InvocationId,
                value: id2,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 3),
                index: PcuDispatchIndex::InvocationId,
                value: id3,
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let kernel = PcuDispatchKernelIr {
            id: PcuKernelId(1),
            entry: PcuDispatchEntryPoint {
                name: "min_max",
                logical_shape: [8, 1, 1],
            },
            bindings: &bindings_ir,
            ports: &[],
            parameters: &[],
            ops: &ops,
            type_caps: PcuValueTypeCaps::for_scalar(PcuScalarType::F32),
            feature_caps: PcuDispatchFeatureCaps::default(),
        };
        let submission = PcuDispatchSubmission {
            kernel: &kernel,
            shape: PcuInvocationShape::invocations(NonZeroU32::new(8).expect("nonzero")),
        };
        let lhs = [1.0, 9.0, -2.0, -0.0, f32::NAN, f32::NAN, 0.0, 2.0];
        let rhs = [3.0, 4.0, -5.0, 0.0, 2.0, f32::NAN, -0.0, f32::NAN];
        let mut minimum = [0.0; 8];
        let mut maximum = [0.0; 8];
        let mut host_bindings = [
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 0),
                slice: PcuHostScalarSlice::Read(&lhs),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 1),
                slice: PcuHostScalarSlice::Read(&rhs),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 2),
                slice: PcuHostScalarSlice::ReadWrite(&mut minimum),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 3),
                slice: PcuHostScalarSlice::ReadWrite(&mut maximum),
            },
        ];
        PcuF32Reference
            .run_host_direct(
                submission,
                &mut host_bindings,
                PcuInvocationParameters::empty(),
            )
            .expect("min and max execute");
        let expected_minimum = [1.0_f32, 4.0, -5.0, -0.0, 2.0, 0.0, -0.0, 2.0];
        let expected_maximum = [3.0_f32, 9.0, -2.0, 0.0, 2.0, 0.0, 0.0, 2.0];
        for index in [0, 1, 2, 3, 4, 6, 7] {
            assert_eq!(minimum[index].to_bits(), expected_minimum[index].to_bits());
            assert_eq!(maximum[index].to_bits(), expected_maximum[index].to_bits());
        }
        assert!(minimum[5].is_nan());
        assert!(maximum[5].is_nan());
        assert!(minimum[6].is_sign_negative());
        assert!(maximum[6].is_sign_positive());

        let unsupported_ops = [
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
                result: id0,
                value: PcuParameterValue::F32(0),
            }),
            PcuDispatchOp::Arithmetic(PcuDispatchAluOp::And),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::InvocationId,
                value: id0,
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let unsupported_kernel = PcuDispatchKernelIr {
            ops: &unsupported_ops,
            ..kernel
        };
        let unsupported_submission = PcuDispatchSubmission {
            kernel: &unsupported_kernel,
            ..submission
        };
        let mut untouched = [7.0; 8];
        let mut unsupported_bindings = [
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 0),
                slice: PcuHostScalarSlice::Read(&lhs),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 1),
                slice: PcuHostScalarSlice::Read(&rhs),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 2),
                slice: PcuHostScalarSlice::ReadWrite(&mut untouched),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 3),
                slice: PcuHostScalarSlice::ReadWrite(&mut maximum),
            },
        ];
        assert_eq!(
            PcuF32Reference.run_host_direct(
                unsupported_submission,
                &mut unsupported_bindings,
                PcuInvocationParameters::empty()
            ),
            Err(PcuF32ReferenceError::UnsupportedInstruction(1))
        );
        assert_eq!(untouched.map(f32::to_bits), [7.0_f32; 8].map(f32::to_bits));
    }

    #[test]
    fn grid_stride_reference_processes_extent_larger_than_launch_width() {
        use fusion_pcu::validate_f32_map_kernel;
        let bindings = [
            PcuBinding::value(
                Some("source"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::f32(),
            ),
            PcuBinding::value(
                Some("destination"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::f32(),
            ),
        ];
        let body = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::GridStrideId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
                result: PcuDispatchValueId(2),
                value: PcuParameterValue::F32(1.0_f32.to_bits()),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: fusion_pcu::PcuValueType::f32(),
                result: PcuDispatchValueId(3),
                op: PcuDispatchAluOp::Add,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::GridStrideId,
                value: PcuDispatchValueId(3),
            }),
        ];
        let ops = [
            PcuDispatchOp::GridStrideLoop {
                extent: 7,
                body: &body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let kernel = PcuDispatchKernelIr {
            id: PcuKernelId(18),
            entry: PcuDispatchEntryPoint {
                name: "grid_stride",
                logical_shape: [2, 1, 1],
            },
            bindings: &bindings,
            ports: &[],
            parameters: &[],
            ops: &ops,
            type_caps: PcuValueTypeCaps::for_scalar(PcuScalarType::F32),
            feature_caps: PcuDispatchFeatureCaps::default(),
        };
        assert_eq!(validate_f32_map_kernel(&kernel), Ok(()));
        let submission = PcuDispatchSubmission {
            kernel: &kernel,
            shape: PcuInvocationShape::invocations(NonZeroU32::new(2).expect("nonzero")),
        };
        // Exact normal values keep this vector inside the portable cross-backend numeric domain.
        let source = [1.25, 2.5, 3.75, 4.0, 5.0, 6.0, 7.0];
        let mut destination = [0.0; 7];
        let mut host_bindings = [
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 0),
                slice: PcuHostScalarSlice::Read(&source),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 1),
                slice: PcuHostScalarSlice::ReadWrite(&mut destination),
            },
        ];
        PcuF32Reference
            .run_host_direct(
                submission,
                &mut host_bindings,
                PcuInvocationParameters::empty(),
            )
            .expect("grid stride covers the logical extent");
        assert_eq!(
            destination.map(f32::to_bits),
            [2.25_f32, 3.5, 4.75, 5.0, 6.0, 7.0, 8.0].map(f32::to_bits)
        );
    }

    #[test]
    fn f64_reference_preserves_source_order_for_basic_arithmetic() {
        let bindings = [
            PcuBinding::scalar::<f64>(
                Some("a"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<f64>(
                Some("b"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<f64>(
                Some("out"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
            ),
        ];
        let ops = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: PcuValueType::f64(),
                result: PcuDispatchValueId(3),
                op: PcuDispatchAluOp::Add,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(3),
            }),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let kernel = PcuDispatchKernelIr {
            id: PcuKernelId(95),
            entry: PcuDispatchEntryPoint {
                name: "f64_add",
                logical_shape: [2, 1, 1],
            },
            bindings: &bindings,
            ports: &[],
            parameters: &[],
            ops: &ops,
            type_caps: PcuValueTypeCaps::for_scalar(PcuScalarType::F64),
            feature_caps: PcuDispatchFeatureCaps::default(),
        };
        let a = [1.25_f64, 2.5];
        let b = [0.5_f64, 4.0];
        let mut output = [0.0_f64; 2];
        let mut host = [
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 0),
                slice: PcuHostScalarSlice::Read(&a),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 1),
                slice: PcuHostScalarSlice::Read(&b),
            },
            PcuHostScalarBinding {
                target: PcuBindingRef::new(0, 2),
                slice: PcuHostScalarSlice::ReadWrite(&mut output),
            },
        ];
        PcuF64Reference
            .run_host_direct(
                PcuDispatchSubmission {
                    kernel: &kernel,
                    shape: PcuInvocationShape::invocations(NonZeroU32::new(2).expect("nonzero")),
                },
                &mut host,
                PcuInvocationParameters::empty(),
            )
            .expect("f64 profile executes");
        assert_eq!(output.map(f64::to_bits), [1.75_f64, 6.5].map(f64::to_bits));
    }
}
