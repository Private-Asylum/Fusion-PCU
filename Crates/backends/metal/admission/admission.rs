//! Structural admission of the existing neutral checked-u32 map profile.

#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchIntegerBinaryOp,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuValueType,
    PcuValueTypeCaps,
    validate_integer_checked_binary_kernel,
    validate_u32_identity_kernel,
};
#[rustfmt::skip]
use crate::{
    MetalBuffer,
    MetalError,
    MetalIntegerOp,
    MetalPreparedIntegerMap,
    MetalSession,
};

/// A snapshotted, structurally admitted neutral checked-u32 map.
///
/// This bounded adapter admits identity or one Add/Sub/Mul node with indexed loads and one output.
/// Broadcast, general graphs, raw ALU, floating operations, `MatMul` and training reject before
/// compilation. No caller IR or parameter borrow survives preparation.
pub struct MetalPreparedU32Kernel {
    map: MetalPreparedIntegerMap,
    input_bindings: [PcuBindingRef; 2],
    output_binding: PcuBindingRef,
    extent: usize,
}
impl MetalPreparedU32Kernel {
    pub(crate) fn execute_into(
        &self,
        inputs: [&MetalBuffer; 2],
        output: &MetalBuffer,
    ) -> Result<(), MetalError> {
        self.map.execute_into(inputs, output, self.extent)
    }
    pub(crate) fn execute_prefix(
        &self,
        inputs: [&MetalBuffer; 2],
    ) -> Result<MetalBuffer, MetalError> {
        self.map.execute_prefix(inputs[0], inputs[1], self.extent)
    }
    #[must_use]
    pub const fn element_count(&self) -> usize {
        self.extent
    }
    #[must_use]
    pub const fn input_bindings(&self) -> [PcuBindingRef; 2] {
        self.input_bindings
    }
    #[must_use]
    pub const fn output_binding(&self) -> PcuBindingRef {
        self.output_binding
    }
    /// Executes inputs in the order returned by `input_bindings` with fresh owned output.
    ///
    /// # Errors
    /// Returns extent/affinity, terminal runtime or checked arithmetic failure.
    pub fn execute(&self, inputs: [&MetalBuffer; 2]) -> Result<MetalBuffer, MetalError> {
        if inputs.iter().any(|input| input.len() != self.extent) {
            return Err(MetalError::InvalidExtent);
        }
        self.map.execute(inputs[0], inputs[1])
    }
}
impl MetalSession {
    /// Validates the existing neutral checked-u32 kernel and compiles its fixed profile.
    ///
    /// Grid-stride maps preserve logical extent and index fault priority; this implementation
    /// launches one Metal thread per logical element rather than replaying the physical stride.
    ///
    /// # Errors
    /// Unsupported IR rejects before native compilation; native compiler errors remain visible.
    pub fn prepare_u32_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<MetalPreparedU32Kernel, MetalError> {
        let profile = Profile::admit(kernel)?;
        Ok(MetalPreparedU32Kernel {
            map: self.prepare_integer_map(profile.operation)?,
            input_bindings: profile.inputs,
            output_binding: profile.output,
            extent: profile.extent,
        })
    }
}
struct Profile {
    operation: MetalIntegerOp,
    inputs: [PcuBindingRef; 2],
    output: PcuBindingRef,
    extent: usize,
}
impl Profile {
    fn admit(kernel: &PcuDispatchKernelIr<'_>) -> Result<Self, MetalError> {
        let (body, extent) = if let [PcuDispatchOp::GridStrideLoop { extent, body }, _] = kernel.ops
        {
            (*body, *extent)
        } else {
            if kernel.entry.logical_shape[1..] != [1, 1] {
                return Err(MetalError::Unsupported);
            }
            (kernel.ops, kernel.entry.logical_shape[0])
        };
        if extent == 0 {
            return Err(MetalError::InvalidExtent);
        }
        if validate_u32_identity_kernel(kernel).is_ok() {
            let input = kernel
                .bindings
                .iter()
                .find(|binding| binding.access == fusion_pcu::PcuBindingAccess::ReadOnly)
                .ok_or(MetalError::Unsupported)?
                .reference();
            let output = kernel
                .bindings
                .iter()
                .find(|binding| binding.access != fusion_pcu::PcuBindingAccess::ReadOnly)
                .ok_or(MetalError::Unsupported)?
                .reference();
            return Ok(Self {
                operation: MetalIntegerOp::Identity,
                inputs: [input, input],
                output,
                extent: extent as usize,
            });
        }
        let Some(PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
            op, lhs, rhs, ..
        })) = body.get(2)
        else {
            return Err(MetalError::Unsupported);
        };
        validate_integer_checked_binary_kernel(
            kernel,
            PcuValueType::u32(),
            *op,
            PcuValueTypeCaps::UINT32,
        )
        .map_err(|_| MetalError::Unsupported)?;
        let operation = match op {
            PcuDispatchIntegerBinaryOp::Add => MetalIntegerOp::Add,
            PcuDispatchIntegerBinaryOp::Sub => MetalIntegerOp::Subtract,
            PcuDispatchIntegerBinaryOp::Mul => MetalIntegerOp::Multiply,
        };
        let mut operands = [None; 2];
        for load in &body[..2] {
            let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result,
                binding,
                index,
            }) = load
            else {
                return Err(MetalError::Unsupported);
            };
            if *index == PcuDispatchIndex::BindingElementZero {
                return Err(MetalError::Unsupported);
            }
            if result == lhs {
                operands[0] = Some(*binding);
            }
            if result == rhs {
                operands[1] = Some(*binding);
            }
        }
        let [Some(left), Some(right)] = operands else {
            return Err(MetalError::Unsupported);
        };
        let PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { binding, .. }) = body[3] else {
            return Err(MetalError::Unsupported);
        };
        Ok(Self {
            operation,
            inputs: [left, right],
            output: binding,
            extent: extent as usize,
        })
    }
}

/// A bounded neutral encoding-only F32 unary map.
pub struct MetalPreparedF32Kernel {
    map: crate::MetalPreparedF32Unary,
    input: PcuBindingRef,
    output: PcuBindingRef,
    extent: usize,
}
impl MetalPreparedF32Kernel {
    pub(crate) fn execute_into(
        &self,
        input: &MetalBuffer,
        output: &MetalBuffer,
    ) -> Result<(), MetalError> {
        self.map.execute_into(input, output, self.extent)
    }
    pub(crate) fn execute_prefix(&self, input: &MetalBuffer) -> Result<MetalBuffer, MetalError> {
        self.map.execute_prefix(input, self.extent)
    }
    #[must_use]
    pub const fn input_binding(&self) -> PcuBindingRef {
        self.input
    }
    #[must_use]
    pub const fn output_binding(&self) -> PcuBindingRef {
        self.output
    }
    #[must_use]
    pub const fn element_count(&self) -> usize {
        self.extent
    }
    /// Executes an exact encoding map with the admitted shape and policy.
    ///
    /// # Errors
    /// Returns extent/affinity, numerical fault or operational failure.
    pub fn execute(&self, input: &MetalBuffer) -> Result<MetalBuffer, MetalError> {
        if input.len() != self.extent {
            return Err(MetalError::InvalidExtent);
        }
        self.map.execute(input)
    }
}
impl MetalSession {
    /// Prepares exactly one neutral checked F32 unary operation.
    ///
    /// # Errors
    /// Rejects unsupported widths, policies, broadcast, graphs and resource schemas before work.
    pub fn prepare_f32_unary_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<MetalPreparedF32Kernel, MetalError> {
        fusion_pcu::validate_checked_float_map_kernel(
            kernel,
            PcuValueType::f32(),
            PcuValueTypeCaps::FLOAT32,
        )
        .map_err(|_| MetalError::Unsupported)?;
        if kernel.bindings.len() != 2
            || kernel.bindings[0].access != fusion_pcu::PcuBindingAccess::ReadOnly
            || kernel.bindings[1].access == fusion_pcu::PcuBindingAccess::ReadOnly
        {
            return Err(MetalError::Unsupported);
        }
        let (body, extent) = if let [PcuDispatchOp::GridStrideLoop { extent, body }, _] = kernel.ops
        {
            (*body, *extent)
        } else {
            (
                &kernel.ops[..kernel.ops.len() - 1],
                kernel.entry.logical_shape[0],
            )
        };
        let [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                binding: input,
                index,
                result: loaded,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
                op,
                underflow_policy,
                range_policy,
                result,
                value,
                ..
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: output,
                value: stored,
                ..
            }),
        ] = body
        else {
            return Err(MetalError::Unsupported);
        };
        if *index == PcuDispatchIndex::BindingElementZero
            || loaded != value
            || stored != result
            || *range_policy != fusion_pcu::PcuRangePolicy::Reject
            || *input != kernel.bindings[0].reference()
            || *output != kernel.bindings[1].reference()
        {
            return Err(MetalError::Unsupported);
        }
        let map = match op {
            fusion_pcu::PcuDispatchFloatUnaryOp::Neg => self.prepare_f32_neg(*underflow_policy)?,
            fusion_pcu::PcuDispatchFloatUnaryOp::Relu => {
                self.prepare_f32_relu(*underflow_policy)?
            }
        };
        Ok(MetalPreparedF32Kernel {
            map,
            input: *input,
            output: *output,
            extent: extent as usize,
        })
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    #[rustfmt::skip]
    use fusion_pcu::{
        PcuBinding,
        PcuBindingAccess,
        PcuBindingStorageClass,
        PcuDispatchControlOp,
        PcuDispatchEntryPoint,
        PcuDispatchValueId,
        PcuKernelId,
    };

    pub fn fixture(grid: bool, broadcast: bool, visit: impl FnOnce(&PcuDispatchKernelIr<'_>)) {
        let bindings = [
            PcuBinding::scalar::<u32>(
                Some("a"),
                2,
                3,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<u32>(
                Some("b"),
                2,
                7,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<u32>(
                Some("out"),
                4,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
            ),
        ];
        let index = if grid {
            PcuDispatchIndex::GridStrideId
        } else {
            PcuDispatchIndex::InvocationId
        };
        let body = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: bindings[0].reference(),
                index: if broadcast {
                    PcuDispatchIndex::BindingElementZero
                } else {
                    index
                },
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: bindings[1].reference(),
                index,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
                value_type: PcuValueType::u32(),
                op: PcuDispatchIntegerBinaryOp::Sub,
                result: PcuDispatchValueId(3),
                lhs: PcuDispatchValueId(2),
                rhs: PcuDispatchValueId(1),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: bindings[2].reference(),
                index,
                value: PcuDispatchValueId(3),
            }),
        ];
        let direct = [
            body[0],
            body[1],
            body[2],
            body[3],
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let grid_ops = [
            PcuDispatchOp::GridStrideLoop {
                extent: 3,
                body: &body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let kernel = PcuDispatchKernelIr {
            id: PcuKernelId(7),
            entry: PcuDispatchEntryPoint {
                name: "checked-map",
                logical_shape: [3, 1, 1],
            },
            bindings: &bindings,
            ports: &[],
            parameters: &[],
            ops: if grid { &grid_ops } else { &direct },
            type_caps: PcuValueTypeCaps::UINT32,
            feature_caps: fusion_pcu::PcuDispatchFeatureCaps::empty(),
        };
        visit(&kernel);
    }

    #[test]
    fn neutral_admission_preserves_operands_and_rejects_unimplemented_broadcast() {
        for grid in [false, true] {
            fixture(grid, false, |kernel| {
                let profile = Profile::admit(kernel).unwrap();
                assert_eq!(
                    profile.inputs,
                    [PcuBindingRef::new(2, 7), PcuBindingRef::new(2, 3)]
                );
                assert_eq!(profile.extent, 3);
                assert_eq!(profile.operation, MetalIntegerOp::Subtract);
            });
            fixture(grid, true, |kernel| {
                assert!(matches!(
                    Profile::admit(kernel),
                    Err(MetalError::Unsupported)
                ));
            });
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "Requires actual macOS Metal device."]
    fn neutral_prepared_direct_and_grid_maps_execute_on_metal() {
        let session = MetalSession::open(0).unwrap();
        for grid in [false, true] {
            fixture(grid, false, |kernel| {
                let prepared = session.prepare_u32_kernel(kernel).unwrap();
                let left = session.upload_u32(&[9, 15, u32::MAX]).unwrap();
                let right = session.upload_u32(&[1, 4, u32::MAX]).unwrap();
                assert_eq!(
                    prepared
                        .execute([&left, &right])
                        .unwrap()
                        .download_u32()
                        .unwrap(),
                    [8, 11, 0]
                );
                assert!(matches!(
                    prepared.execute([&session.upload_u32(&[1]).unwrap(), &right]),
                    Err(MetalError::InvalidExtent)
                ));
            });
        }
    }
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "Requires actual macOS Metal device."]
    fn host_checked_fault_keeps_output_and_retry_preserves_tail() {
        use fusion_pcu::{PcuHostArgument, PcuHostKernelBackend, PcuPreparedHostKernel};
        let session = MetalSession::open(0).unwrap();
        for grid in [false, true] {
            fixture(grid, false, |kernel| {
                let mut prepared = session.prepare_host_kernel(kernel).unwrap();
                let mut output = [91_u32; 5];
                {
                    let mut args = [
                        PcuHostArgument::read(PcuBindingRef::new(2, 3), &[1_u32, 4, 9]),
                        PcuHostArgument::read(PcuBindingRef::new(2, 7), &[9_u32, 15, 10]),
                        PcuHostArgument::read_write(PcuBindingRef::new(4, 1), &mut output),
                    ];
                    prepared.call(&mut args).unwrap();
                }
                assert_eq!(output, [8, 11, 1, 91, 91]);
                {
                    let mut args = [
                        PcuHostArgument::read(PcuBindingRef::new(2, 3), &[1_u32, 16, 11]),
                        PcuHostArgument::read(PcuBindingRef::new(2, 7), &[9_u32, 15, 10]),
                        PcuHostArgument::read_write(PcuBindingRef::new(4, 1), &mut output),
                    ];
                    assert!(
                        matches!(prepared.call(&mut args), Err(fusion_pcu::PcuHostDispatchError::Backend(MetalError::Arithmetic(fault))) if fault.kind == fusion_pcu::PcuExecutionFaultKind::ArithmeticUnderflow && fault.invocation_id == 1)
                    );
                }
                assert_eq!(output, [8, 11, 1, 91, 91]);
                {
                    let mut args = [
                        PcuHostArgument::read(PcuBindingRef::new(2, 3), &[1_u32, 2, 3]),
                        PcuHostArgument::read(PcuBindingRef::new(2, 7), &[7_u32, 8, 9]),
                        PcuHostArgument::read_write(PcuBindingRef::new(4, 1), &mut output),
                    ];
                    prepared.call(&mut args).unwrap();
                }
                assert_eq!(output, [6, 6, 6, 91, 91]);
            });
        }
    }
}
