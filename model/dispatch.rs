//! Dispatch-model vocabulary and backend-neutral kernel builder.

use core::ops::{
    BitAnd,
    BitAndAssign,
    BitOr,
    BitOrAssign,
};

use crate::{
    PcuBinding,
    PcuDispatchPolicyCaps,
    PcuDispatchOpCaps,
    PcuError,
    PcuKernel,
    PcuKernelIrContract,
    PcuKernelId,
    PcuKernelSignature,
    PcuParameter,
    PcuPort,
    PcuInvocationModel,
    PcuIrKind,
    PcuScalarType,
    PcuValueType,
};

pub use crate::ir::{
    PcuAluOp as PcuDispatchAluOp,
    PcuBindingOp as PcuDispatchResourceOp,
    PcuControlOp as PcuDispatchControlOp,
    PcuPortOp as PcuDispatchPortOp,
    PcuSampleLevel,
    PcuSampleOp,
    PcuSyncOp as PcuDispatchSyncOp,
    PcuValueOp as PcuDispatchValueOp,
};
pub use crate::validation::PcuSampleValidationError;

const DEFAULT_OP_CAPACITY: usize = 32;

/// Coarse dispatch-profile capabilities required by one program unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PcuDispatchCapabilities(u32);

impl PcuDispatchCapabilities {
    pub const INT32: Self = Self(1 << 0);
    pub const UINT32: Self = Self(1 << 1);
    pub const FLOAT16: Self = Self(1 << 2);
    pub const FLOAT32: Self = Self(1 << 3);
    pub const MUTABLE_RESOURCES: Self = Self(1 << 4);
    pub const READ_ONLY_RESOURCES: Self = Self(1 << 5);
    pub const INLINE_PARAMETERS: Self = Self(1 << 6);
    pub const COOPERATIVE_SCRATCHPAD: Self = Self(1 << 7);
    pub const BOOL: Self = Self(1 << 8);
    pub const INT8: Self = Self(1 << 9);
    pub const UINT8: Self = Self(1 << 10);
    pub const INT16: Self = Self(1 << 11);
    pub const UINT16: Self = Self(1 << 12);
    pub const INT64: Self = Self(1 << 13);
    pub const UINT64: Self = Self(1 << 14);
    pub const FLOAT64: Self = Self(1 << 15);

    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    #[must_use]
    pub const fn for_scalar(scalar: PcuScalarType) -> Self {
        match scalar {
            PcuScalarType::Bool => Self::BOOL,
            PcuScalarType::I8 => Self::INT8,
            PcuScalarType::U8 => Self::UINT8,
            PcuScalarType::I16 => Self::INT16,
            PcuScalarType::U16 => Self::UINT16,
            PcuScalarType::I32 => Self::INT32,
            PcuScalarType::U32 => Self::UINT32,
            PcuScalarType::I64 => Self::INT64,
            PcuScalarType::U64 => Self::UINT64,
            PcuScalarType::F16 => Self::FLOAT16,
            PcuScalarType::F32 => Self::FLOAT32,
            PcuScalarType::F64 => Self::FLOAT64,
        }
    }

    #[must_use]
    pub const fn supports_value_type(self, value_type: PcuValueType) -> bool {
        self.contains(Self::for_scalar(value_type.scalar_type()))
    }
}

/// One dispatch-model instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuDispatchOp<'a> {
    Value(PcuDispatchValueOp),
    Arithmetic(PcuDispatchAluOp),
    Control(PcuDispatchControlOp),
    Resource(PcuDispatchResourceOp),
    Port(PcuDispatchPortOp),
    Sync(PcuDispatchSyncOp),
    Intrinsic { name: &'a str },
}

impl PcuDispatchOp<'_> {
    #[must_use]
    pub const fn support_flag(self) -> PcuDispatchOpCaps {
        match self {
            Self::Value(op) => op.support_flag(),
            Self::Arithmetic(op) => op.support_flag(),
            Self::Control(op) => op.support_flag(),
            Self::Resource(op) => op.support_flag(),
            Self::Port(op) => op.support_flag(),
            Self::Sync(op) => op.support_flag(),
            Self::Intrinsic { .. } => PcuDispatchOpCaps::INTRINSIC,
        }
    }
}

impl BitOr for PcuDispatchCapabilities {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for PcuDispatchCapabilities {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for PcuDispatchCapabilities {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for PcuDispatchCapabilities {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

/// One entry-point descriptor for one dispatch profile program unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuDispatchEntryPoint<'a> {
    pub name: &'a str,
    pub logical_shape: [u32; 3],
}

/// Dispatch-oriented program-unit profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuDispatchKernelIr<'a> {
    pub id: PcuKernelId,
    pub entry: PcuDispatchEntryPoint<'a>,
    pub bindings: &'a [PcuBinding<'a>],
    pub ports: &'a [PcuPort<'a>],
    pub parameters: &'a [PcuParameter<'a>],
    pub ops: &'a [PcuDispatchOp<'a>],
    pub capabilities: PcuDispatchCapabilities,
}

impl PcuDispatchKernelIr<'_> {
    /// Returns the dispatch-policy flags required to route this dispatch kernel honestly.
    #[must_use]
    pub const fn required_dispatch_policy(&self) -> PcuDispatchPolicyCaps {
        // Dispatch kernels stay agnostic about whether the backend routes logical invocations
        // serially, pipelined, or in parallel. What they do require is one honest finite
        // submission path, and ordered admission is the minimum portable guarantee here.
        PcuDispatchPolicyCaps::ORDERED_SUBMISSION
    }

    /// Returns the per-instruction support flags required to execute this dispatch kernel.
    #[must_use]
    pub fn required_instruction_support(&self) -> PcuDispatchOpCaps {
        let mut flags = PcuDispatchOpCaps::empty();
        for op in self.ops.iter().copied() {
            flags = flags.union(op.support_flag());
        }
        flags
    }
}

impl PcuKernelIrContract for PcuDispatchKernelIr<'_> {
    fn id(&self) -> PcuKernelId {
        self.id
    }

    fn kind(&self) -> PcuIrKind {
        PcuIrKind::Dispatch
    }

    fn entry_point(&self) -> &str {
        self.entry.name
    }

    fn signature(&self) -> PcuKernelSignature<'_> {
        PcuKernelSignature {
            bindings: self.bindings,
            ports: self.ports,
            parameters: self.parameters,
            invocation: PcuInvocationModel::indexed(self.entry.logical_shape),
        }
    }
}

/// Builder for one backend-neutral dispatch kernel.
#[derive(Debug, Clone, Copy)]
pub struct PcuDispatchKernelBuilder<'a, const MAX_OPS: usize = DEFAULT_OP_CAPACITY> {
    kernel_id: PcuKernelId,
    entry: PcuDispatchEntryPoint<'a>,
    bindings: &'a [PcuBinding<'a>],
    ports: &'a [PcuPort<'a>],
    parameters: &'a [PcuParameter<'a>],
    ops: [PcuDispatchOp<'a>; MAX_OPS],
    op_len: usize,
    capabilities: PcuDispatchCapabilities,
}

impl<'a, const MAX_OPS: usize> PcuDispatchKernelBuilder<'a, MAX_OPS> {
    /// Creates one dispatch-kernel builder.
    #[must_use]
    pub fn new(kernel_id: u32, entry_point: &'a str, logical_shape: [u32; 3]) -> Self {
        Self {
            kernel_id: PcuKernelId(kernel_id),
            entry: PcuDispatchEntryPoint {
                name: entry_point,
                logical_shape,
            },
            bindings: &[],
            ports: &[],
            parameters: &[],
            ops: [PcuDispatchOp::Control(PcuDispatchControlOp::Return); MAX_OPS],
            op_len: 0,
            capabilities: PcuDispatchCapabilities::empty(),
        }
    }

    /// Replaces the binding slice.
    #[must_use]
    pub const fn with_bindings(mut self, bindings: &'a [PcuBinding<'a>]) -> Self {
        self.bindings = bindings;
        self
    }

    /// Replaces the port slice.
    #[must_use]
    pub const fn with_ports(mut self, ports: &'a [PcuPort<'a>]) -> Self {
        self.ports = ports;
        self
    }

    /// Replaces the parameter slice.
    #[must_use]
    pub const fn with_parameters(mut self, parameters: &'a [PcuParameter<'a>]) -> Self {
        self.parameters = parameters;
        self
    }

    /// Replaces the required capability set.
    #[must_use]
    pub const fn with_capabilities(mut self, capabilities: PcuDispatchCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// Appends one dispatch operation.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder op capacity is exhausted.
    pub fn with_op(mut self, op: PcuDispatchOp<'a>) -> Result<Self, PcuError> {
        self.push_op(op)?;
        Ok(self)
    }

    /// Appends one value/construction operation.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder op capacity is exhausted.
    pub fn with_value_op(self, op: PcuDispatchValueOp) -> Result<Self, PcuError> {
        self.with_op(PcuDispatchOp::Value(op))
    }

    /// Appends one arithmetic/logical operation.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder op capacity is exhausted.
    pub fn with_arithmetic_op(self, op: PcuDispatchAluOp) -> Result<Self, PcuError> {
        self.with_op(PcuDispatchOp::Arithmetic(op))
    }

    /// Appends one control-flow operation.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder op capacity is exhausted.
    pub fn with_control_op(self, op: PcuDispatchControlOp) -> Result<Self, PcuError> {
        self.with_op(PcuDispatchOp::Control(op))
    }

    /// Appends one resource/binding operation.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder op capacity is exhausted.
    pub fn with_resource_op(self, op: PcuDispatchResourceOp) -> Result<Self, PcuError> {
        self.with_op(PcuDispatchOp::Resource(op))
    }

    /// Appends one port/dataflow operation.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder op capacity is exhausted.
    pub fn with_port_op(self, op: PcuDispatchPortOp) -> Result<Self, PcuError> {
        self.with_op(PcuDispatchOp::Port(op))
    }

    /// Appends one synchronization operation.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder op capacity is exhausted.
    pub fn with_sync_op(self, op: PcuDispatchSyncOp) -> Result<Self, PcuError> {
        self.with_op(PcuDispatchOp::Sync(op))
    }

    /// Appends one backend-defined intrinsic operation.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder op capacity is exhausted.
    pub fn with_intrinsic(self, name: &'a str) -> Result<Self, PcuError> {
        self.with_op(PcuDispatchOp::Intrinsic { name })
    }

    /// Appends several dispatch operations in order.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder op capacity is exhausted.
    pub fn with_ops(mut self, ops: &[PcuDispatchOp<'a>]) -> Result<Self, PcuError> {
        for op in ops.iter().copied() {
            self.push_op(op)?;
        }
        Ok(self)
    }

    /// Returns the configured dispatch operation slice.
    #[must_use]
    pub fn ops(&self) -> &[PcuDispatchOp<'a>] {
        &self.ops[..self.op_len]
    }

    /// Builds the dispatch-kernel IR payload.
    #[must_use]
    pub fn ir(&self) -> PcuDispatchKernelIr<'_> {
        PcuDispatchKernelIr {
            id: self.kernel_id,
            entry: self.entry,
            bindings: self.bindings,
            ports: self.ports,
            parameters: self.parameters,
            ops: &self.ops[..self.op_len],
            capabilities: self.capabilities,
        }
    }

    /// Builds the generic kernel wrapper.
    #[must_use]
    pub fn kernel(&self) -> PcuKernel<'_> {
        PcuKernel::Dispatch(self.ir())
    }

    fn push_op(&mut self, op: PcuDispatchOp<'a>) -> Result<(), PcuError> {
        if self.op_len == MAX_OPS {
            return Err(PcuError::resource_exhausted());
        }
        self.ops[self.op_len] = op;
        self.op_len += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PcuDispatchAluOp,
        PcuDispatchCapabilities,
        PcuDispatchKernelBuilder,
    };
    use crate::{
        PcuDispatchPolicyCaps,
        PcuIrKind,
        PcuKernel,
        PcuKernelIrContract,
        PcuValueType,
    };

    #[test]
    fn builder_synthesizes_dispatch_kernel_with_ops() {
        let builder = PcuDispatchKernelBuilder::<4>::new(0x21, "main", [32, 1, 1])
            .with_capabilities(PcuDispatchCapabilities::UINT32)
            .with_arithmetic_op(PcuDispatchAluOp::Add)
            .expect("builder should accept one op");
        let kernel = builder.ir();

        assert_eq!(kernel.id.0, 0x21);
        assert_eq!(kernel.kind(), PcuIrKind::Dispatch);
        assert_eq!(kernel.entry.name, "main");
        assert_eq!(kernel.entry.logical_shape, [32, 1, 1]);
        assert_eq!(kernel.ops.len(), 1);
        assert!(
            kernel
                .capabilities
                .contains(PcuDispatchCapabilities::UINT32)
        );
        assert_eq!(
            kernel.required_dispatch_policy(),
            PcuDispatchPolicyCaps::ORDERED_SUBMISSION
        );
    }

    #[test]
    fn builder_wraps_generic_dispatch_kernel() {
        let builder = PcuDispatchKernelBuilder::<2>::new(9, "main", [1, 1, 1]);
        let kernel = builder.kernel();

        match kernel {
            PcuKernel::Dispatch(dispatch) => {
                assert_eq!(dispatch.kind(), PcuIrKind::Dispatch);
                assert_eq!(dispatch.id.0, 9);
            }
            _ => panic!("expected dispatch kernel"),
        }
    }

    #[test]
    fn dispatch_capabilities_cover_core_scalar_types() {
        let caps = PcuDispatchCapabilities::BOOL
            | PcuDispatchCapabilities::INT8
            | PcuDispatchCapabilities::UINT8
            | PcuDispatchCapabilities::INT16
            | PcuDispatchCapabilities::UINT16
            | PcuDispatchCapabilities::INT32
            | PcuDispatchCapabilities::UINT32
            | PcuDispatchCapabilities::INT64
            | PcuDispatchCapabilities::UINT64
            | PcuDispatchCapabilities::FLOAT16
            | PcuDispatchCapabilities::FLOAT32
            | PcuDispatchCapabilities::FLOAT64;

        assert!(caps.supports_value_type(PcuValueType::bool()));
        assert!(caps.supports_value_type(PcuValueType::i8()));
        assert!(caps.supports_value_type(PcuValueType::u8()));
        assert!(caps.supports_value_type(PcuValueType::i16()));
        assert!(caps.supports_value_type(PcuValueType::u16()));
        assert!(caps.supports_value_type(PcuValueType::i32()));
        assert!(caps.supports_value_type(PcuValueType::u32()));
        assert!(caps.supports_value_type(PcuValueType::i64()));
        assert!(caps.supports_value_type(PcuValueType::u64()));
        assert!(caps.supports_value_type(PcuValueType::f16()));
        assert!(caps.supports_value_type(PcuValueType::f32()));
        assert!(caps.supports_value_type(PcuValueType::f64()));
        assert!(caps.supports_value_type(PcuValueType::Vector {
            scalar: crate::PcuScalarType::F64,
            lanes: 4,
        }));
    }
}
