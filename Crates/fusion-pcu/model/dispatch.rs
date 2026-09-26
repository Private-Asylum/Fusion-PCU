//! Dispatch-model vocabulary and backend-neutral kernel builder.

use core::ops::{
    BitAnd,
    BitAndAssign,
    BitOr,
    BitOrAssign,
};

use crate::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuDispatchPolicyCaps,
    PcuDispatchOpCaps,
    PcuValueTypeCaps,
    PcuError,
    PcuKernel,
    PcuKernelIrContract,
    PcuKernelId,
    PcuKernelSignature,
    PcuParameter,
    PcuParameterValue,
    PcuValueType,
    PcuPort,
    PcuInvocationModel,
    PcuIrKind,
};

pub use crate::ir::{
    PcuAluOp as PcuDispatchAluOp,
    PcuBindingOp as PcuDispatchResourceOp,
    PcuCoordinateOp as PcuDispatchCoordinateOp,
    PcuControlOp as PcuDispatchControlOp,
    PcuPortOp as PcuDispatchPortOp,
    PcuRayFlags,
    PcuRayTraceOp as PcuDispatchRayTraceOp,
    PcuSampleLevel,
    PcuSampleOp,
    PcuSyncOp as PcuDispatchSyncOp,
    PcuTraceRayOp,
    PcuValueOp as PcuDispatchValueOp,
};
pub use crate::validation::PcuSampleValidationError;

const DEFAULT_OP_CAPACITY: usize = 32;

const fn index_extent(index: PcuDispatchIndex, direct_extent: u32, fallback_extent: u32) -> u32 {
    match index {
        PcuDispatchIndex::InvocationId => direct_extent,
        PcuDispatchIndex::GridStrideId | PcuDispatchIndex::Value(_) => fallback_extent,
        PcuDispatchIndex::BindingElementZero => 1,
    }
}

/// Dispatch-only feature caps required by one program unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PcuDispatchFeatureCaps(u32);

impl PcuDispatchFeatureCaps {
    pub const MUTABLE_RESOURCES: Self = Self(1 << 0);
    pub const READ_ONLY_RESOURCES: Self = Self(1 << 1);
    pub const INLINE_PARAMETERS: Self = Self(1 << 2);
    pub const COOPERATIVE_SCRATCHPAD: Self = Self(1 << 3);

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
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// One dispatch-model instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuDispatchOp<'a> {
    Value(PcuDispatchValueOp),
    Arithmetic(PcuDispatchAluOp),
    Control(PcuDispatchControlOp),
    Resource(PcuDispatchResourceOp),
    Data(PcuDispatchDataOp),
    /// Execute a data-only map body at logical indices `base + n * stride` below `extent`.
    /// The stride is the actual submitted invocation count; `base` is each invocation id.
    /// The body may use `GridStrideId` for indexed accesses and its SSA values are region-local.
    GridStrideLoop {
        extent: u32,
        body: &'a [Self],
    },
    Coordinate(PcuDispatchCoordinateOp),
    RayTrace(PcuDispatchRayTraceOp),
    Port(PcuDispatchPortOp),
    Sync(PcuDispatchSyncOp),
    Intrinsic {
        name: &'a str,
    },
}

impl PcuDispatchOp<'_> {
    #[must_use]
    pub const fn support_flag(self) -> PcuDispatchOpCaps {
        match self {
            Self::Value(op) => op.support_flag(),
            Self::Arithmetic(op) => op.support_flag(),
            Self::Control(op) => op.support_flag(),
            Self::Resource(op) => op.support_flag(),
            Self::Data(op) => op.support_flag(),
            Self::GridStrideLoop { .. } => PcuDispatchOpCaps::CONTROL_LOOP,
            Self::Coordinate(op) => op.support_flag(),
            Self::RayTrace(op) => op.support_flag(),
            Self::Port(op) => op.support_flag(),
            Self::Sync(op) => op.support_flag(),
            Self::Intrinsic { .. } => PcuDispatchOpCaps::INTRINSIC,
        }
    }
}

/// Virtual value id inside one operandful dispatch program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuDispatchValueId(pub u16);

/// Index expression for binding loads/stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuDispatchIndex {
    InvocationId,
    /// Current logical element index inside a `GridStrideLoop` body.
    GridStrideId,
    /// Element zero of a resource binding, broadcast to every invocation.
    ///
    /// This is an indexed resource read, not a declaration that the binding is a dense
    /// invocation-sized buffer. The binding must contain at least one element.
    BindingElementZero,
    Value(PcuDispatchValueId),
}

/// Operandful dispatch instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuDispatchDataOp {
    BindingLoad {
        result: PcuDispatchValueId,
        binding: PcuBindingRef,
        index: PcuDispatchIndex,
    },
    Constant {
        result: PcuDispatchValueId,
        value: PcuParameterValue,
    },
    Alu {
        value_type: PcuValueType,
        result: PcuDispatchValueId,
        op: PcuDispatchAluOp,
        lhs: PcuDispatchValueId,
        rhs: PcuDispatchValueId,
    },
    /// Computes an exact-width integer quotient and remainder together.
    ///
    /// A zero divisor and signed `MIN / -1` are terminal execution faults reported through the
    /// submission completion result. The quotient and remainder payloads are undefined for a
    /// faulting invocation; consumers must not use either output after a failed completion.
    /// `flags` is currently required to be empty. The reserved `DIV_OR_ZERO` flag belongs to a
    /// future quotient-only operation and is not admitted or executed by this instruction.
    CheckedDivRem {
        value_type: PcuValueType,
        flags: PcuIntegerDivFlags,
        quotient: PcuDispatchValueId,
        remainder: PcuDispatchValueId,
        lhs: PcuDispatchValueId,
        rhs: PcuDispatchValueId,
    },
    BindingStore {
        binding: PcuBindingRef,
        index: PcuDispatchIndex,
        value: PcuDispatchValueId,
    },
}

impl PcuDispatchDataOp {
    #[must_use]
    pub const fn support_flag(self) -> PcuDispatchOpCaps {
        match self {
            Self::BindingLoad { index, .. } => {
                if matches!(index, PcuDispatchIndex::BindingElementZero) {
                    PcuDispatchOpCaps::BINDING_LOAD
                        .union(PcuDispatchOpCaps::BINDING_LOAD_ELEMENT_ZERO)
                } else {
                    PcuDispatchOpCaps::BINDING_LOAD
                }
            }
            Self::Constant { .. } => PcuDispatchOpCaps::VALUE_CONSTANT,
            Self::Alu { op, .. } => op.support_flag(),
            Self::CheckedDivRem { .. } => PcuDispatchOpCaps::ALU_CHECKED_DIV_REM,
            Self::BindingStore { .. } => PcuDispatchOpCaps::BINDING_STORE,
        }
    }
}

/// Semantic mode flags reserved for integer division instructions.
///
/// The empty set selects checked division, whose invalid domains fault completion. `DIV_OR_ZERO`
/// is reserved for a future quotient-only operation that opts into a zero result when the divisor
/// is zero; setting it on [`PcuDispatchDataOp::CheckedDivRem`] is not currently supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PcuIntegerDivFlags(u8);

impl PcuIntegerDivFlags {
    /// No exceptional-domain override: division faults on invalid lanes.
    pub const CHECKED: Self = Self(0);
    /// Reserved for future quotient-only division with a zero result for a zero divisor.
    pub const DIV_OR_ZERO: Self = Self(1 << 0);

    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// Instruction-support contract for one dispatch kernel.
pub trait PcuDispatchInstructionContract {
    fn required_dispatch_instruction_support(&self) -> PcuDispatchOpCaps;
}

/// Coordinate-instruction contract for one dispatch kernel used by a graphics composition layer.
pub trait PcuCoordinateInstructionContract {
    fn required_coordinate_instruction_support(&self) -> PcuDispatchOpCaps;
}

/// Ray-tracing instruction contract for one dispatch kernel used by a graphics composition layer.
pub trait PcuRayTraceInstructionContract {
    fn required_ray_trace_instruction_support(&self) -> PcuDispatchOpCaps;

    #[must_use]
    fn uses_ray_tracing(&self) -> bool {
        self.required_ray_trace_instruction_support().bits() != 0
    }
}

impl BitOr for PcuDispatchFeatureCaps {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for PcuDispatchFeatureCaps {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for PcuDispatchFeatureCaps {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for PcuDispatchFeatureCaps {
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
    pub type_caps: PcuValueTypeCaps,
    pub feature_caps: PcuDispatchFeatureCaps,
}

impl PcuDispatchKernelIr<'_> {
    /// Returns the minimum element count required for buffer bindings by this launch.
    /// A grid-stride map uses its semantic extent; direct maps use submitted launch width.
    #[must_use]
    pub fn minimum_binding_elements(&self, submitted_invocations: u32) -> u32 {
        self.ops
            .iter()
            .find_map(|op| match op {
                PcuDispatchOp::GridStrideLoop { extent, .. } => Some(*extent),
                _ => None,
            })
            .unwrap_or(submitted_invocations)
    }

    /// Returns the minimum element count required by one binding's indexed accesses.
    ///
    /// `BindingElementZero` requires one element, while invocation and grid-stride indices
    /// require the complete submitted or semantic extent. Bindings not referenced by data
    /// instructions retain the legacy kernel-wide minimum.
    #[must_use]
    pub fn minimum_binding_elements_for(
        &self,
        target: PcuBindingRef,
        submitted_invocations: u32,
    ) -> u32 {
        let default = self.minimum_binding_elements(submitted_invocations);
        let mut required = 0;
        let mut referenced = false;
        for op in self.ops.iter().copied() {
            match op {
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { binding, index, .. })
                    if binding == target =>
                {
                    referenced = true;
                    required = required.max(index_extent(index, submitted_invocations, default));
                }
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { binding, index, .. })
                    if binding == target =>
                {
                    referenced = true;
                    required = required.max(index_extent(index, submitted_invocations, default));
                }
                PcuDispatchOp::GridStrideLoop { extent, body } => {
                    for body_op in body.iter().copied() {
                        match body_op {
                            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                                binding,
                                index,
                                ..
                            }) if binding == target => {
                                referenced = true;
                                required = required.max(index_extent(index, extent, extent));
                            }
                            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                                binding,
                                index,
                                ..
                            }) if binding == target => {
                                referenced = true;
                                required = required.max(index_extent(index, extent, extent));
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        if referenced { required } else { default }
    }

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
            if let PcuDispatchOp::GridStrideLoop { body, .. } = op {
                for body_op in body.iter().copied() {
                    flags = flags.union(body_op.support_flag());
                }
            }
        }
        flags
    }

    /// ALU operation flags required for each exact scalar type, including grid-stride bodies.
    #[must_use]
    pub fn required_scalar_alu_support(&self) -> crate::PcuDispatchScalarAluSupport {
        let mut support = crate::PcuDispatchScalarAluSupport::empty();
        for op in self.ops.iter().copied() {
            collect_alu_support(op, &mut support);
            if let PcuDispatchOp::GridStrideLoop { body, .. } = op {
                for body_op in body.iter().copied() {
                    collect_alu_support(body_op, &mut support);
                }
            }
        }
        support
    }

    /// Reject vector and matrix ALU until dispatch advertises shape-specific ALU semantics.
    #[must_use]
    pub fn has_non_scalar_alu_type(&self) -> bool {
        self.ops.iter().copied().any(|op| {
            if is_non_scalar_alu_type(op) {
                return true;
            }
            matches!(op, PcuDispatchOp::GridStrideLoop { body, .. }
                if body.iter().copied().any(is_non_scalar_alu_type))
        })
    }

    /// Returns only the coordinate-instruction support flags required by this dispatch kernel.
    #[must_use]
    pub fn required_coordinate_instruction_support(&self) -> PcuDispatchOpCaps {
        let mut flags = PcuDispatchOpCaps::empty();
        for op in self.ops.iter().copied() {
            if let PcuDispatchOp::Coordinate(op) = op {
                flags = flags.union(op.support_flag());
            }
        }
        flags
    }

    /// Returns only the ray-tracing instruction support flags required by this dispatch kernel.
    #[must_use]
    pub fn required_ray_trace_instruction_support(&self) -> PcuDispatchOpCaps {
        let mut flags = PcuDispatchOpCaps::empty();
        for op in self.ops.iter().copied() {
            if let PcuDispatchOp::RayTrace(op) = op {
                flags = flags.union(op.support_flag());
            }
        }
        flags
    }

    /// Returns the value/type support floor derived from the kernel's typed interface and
    /// constants, plus any explicitly requested capabilities.
    #[must_use]
    pub fn required_type_support(&self) -> PcuValueTypeCaps {
        let mut required = self.type_caps;
        for binding in self.bindings {
            if let Some(value_type) = binding.value_type() {
                required = required.union(PcuValueTypeCaps::for_value_type(value_type));
            }
            if let Some(image_type) = binding.image_type() {
                required = required.union(PcuValueTypeCaps::for_value_type(image_type.texel_type));
                required = required.union(PcuValueTypeCaps::for_value_type(
                    image_type.coordinate_type(),
                ));
            }
        }
        for port in self.ports {
            required = required.union(PcuValueTypeCaps::for_value_type(port.value_type));
        }
        for parameter in self.parameters {
            required = required.union(PcuValueTypeCaps::for_value_type(parameter.value_type));
        }
        for op in self.ops {
            match op {
                PcuDispatchOp::Data(PcuDispatchDataOp::Constant { value, .. }) => {
                    required = required.union(PcuValueTypeCaps::for_value_type(value.value_type()));
                }
                PcuDispatchOp::Data(
                    PcuDispatchDataOp::Alu { value_type, .. }
                    | PcuDispatchDataOp::CheckedDivRem { value_type, .. },
                ) => {
                    required = required.union(PcuValueTypeCaps::for_value_type(*value_type));
                }
                PcuDispatchOp::GridStrideLoop { body, .. } => {
                    for body_op in *body {
                        if let PcuDispatchOp::Data(
                            PcuDispatchDataOp::Alu { value_type, .. }
                            | PcuDispatchDataOp::CheckedDivRem { value_type, .. },
                        ) = body_op
                        {
                            required =
                                required.union(PcuValueTypeCaps::for_value_type(*value_type));
                        }
                    }
                }
                _ => {}
            }
        }
        required
    }

    /// Returns the dispatch-only feature floor required to execute this kernel honestly.
    #[must_use]
    pub fn required_feature_support(&self) -> PcuDispatchFeatureCaps {
        self.derived_feature_support().union(self.feature_caps)
    }

    fn derived_feature_support(&self) -> PcuDispatchFeatureCaps {
        let mut features = PcuDispatchFeatureCaps::empty();

        if !self.parameters.is_empty() {
            features = features.union(PcuDispatchFeatureCaps::INLINE_PARAMETERS);
        }

        for binding in self.bindings.iter().copied() {
            if binding.storage == PcuBindingStorageClass::Shared {
                features = features.union(PcuDispatchFeatureCaps::COOPERATIVE_SCRATCHPAD);
            }

            if binding.builtin.is_some() {
                continue;
            }

            match binding.access {
                PcuBindingAccess::ReadOnly => {
                    features = features.union(PcuDispatchFeatureCaps::READ_ONLY_RESOURCES);
                }
                PcuBindingAccess::WriteOnly | PcuBindingAccess::ReadWrite => {
                    features = features.union(PcuDispatchFeatureCaps::MUTABLE_RESOURCES);
                }
            }
        }

        features
    }
}

const fn collect_alu_support(
    op: PcuDispatchOp<'_>,
    support: &mut crate::PcuDispatchScalarAluSupport,
) {
    if let PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
        value_type: actual,
        op,
        ..
    }) = op
    {
        let scalar = actual.scalar_type();
        *support = support.with(scalar, support.for_scalar(scalar).union(op.support_flag()));
    }
    if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem { value_type, .. }) = op {
        let scalar = value_type.scalar_type();
        *support = support.with(
            scalar,
            support
                .for_scalar(scalar)
                .union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
        );
    }
}

const fn is_non_scalar_alu_type(op: PcuDispatchOp<'_>) -> bool {
    matches!(op, PcuDispatchOp::Data(PcuDispatchDataOp::Alu { value_type, .. }
        | PcuDispatchDataOp::CheckedDivRem { value_type, .. })
        if !matches!(value_type, PcuValueType::Scalar(_)))
}

impl PcuDispatchInstructionContract for PcuDispatchKernelIr<'_> {
    fn required_dispatch_instruction_support(&self) -> PcuDispatchOpCaps {
        self.required_instruction_support()
    }
}

impl PcuCoordinateInstructionContract for PcuDispatchKernelIr<'_> {
    fn required_coordinate_instruction_support(&self) -> PcuDispatchOpCaps {
        PcuDispatchKernelIr::required_coordinate_instruction_support(self)
    }
}

impl PcuRayTraceInstructionContract for PcuDispatchKernelIr<'_> {
    fn required_ray_trace_instruction_support(&self) -> PcuDispatchOpCaps {
        PcuDispatchKernelIr::required_ray_trace_instruction_support(self)
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
    type_caps: PcuValueTypeCaps,
    feature_caps: PcuDispatchFeatureCaps,
}

impl<'a, const MAX_OPS: usize> PcuDispatchKernelBuilder<'a, MAX_OPS> {
    /// Creates one dispatch-kernel builder.
    #[must_use]
    pub const fn new(kernel_id: u32, entry_point: &'a str, logical_shape: [u32; 3]) -> Self {
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
            type_caps: PcuValueTypeCaps::empty(),
            feature_caps: PcuDispatchFeatureCaps::empty(),
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

    /// Replaces the required value/type support floor.
    #[must_use]
    pub const fn with_type_caps(mut self, type_caps: PcuValueTypeCaps) -> Self {
        self.type_caps = type_caps;
        self
    }

    /// Replaces the required dispatch-only feature floor.
    #[must_use]
    pub const fn with_feature_caps(mut self, feature_caps: PcuDispatchFeatureCaps) -> Self {
        self.feature_caps = feature_caps;
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

    /// Appends one operandful data operation.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder op capacity is exhausted.
    pub fn with_data_op(self, op: PcuDispatchDataOp) -> Result<Self, PcuError> {
        self.with_op(PcuDispatchOp::Data(op))
    }

    /// Appends one coordinate-oriented operation.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder op capacity is exhausted.
    pub fn with_coordinate_op(self, op: PcuDispatchCoordinateOp) -> Result<Self, PcuError> {
        self.with_op(PcuDispatchOp::Coordinate(op))
    }

    /// Appends one ray-tracing operation.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder op capacity is exhausted.
    pub fn with_ray_trace_op(self, op: PcuDispatchRayTraceOp) -> Result<Self, PcuError> {
        self.with_op(PcuDispatchOp::RayTrace(op))
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
            type_caps: self.type_caps,
            feature_caps: self.feature_caps,
        }
    }

    /// Builds the generic kernel wrapper.
    #[must_use]
    pub fn kernel(&self) -> PcuKernel<'_> {
        PcuKernel::Dispatch(self.ir())
    }

    const fn push_op(&mut self, op: PcuDispatchOp<'a>) -> Result<(), PcuError> {
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
        PcuDispatchCoordinateOp,
        PcuDispatchFeatureCaps,
        PcuDispatchKernelBuilder,
        PcuDispatchRayTraceOp,
        PcuRayTraceInstructionContract,
        PcuTraceRayOp,
        PcuIntegerDivFlags,
    };
    use crate::{
        PcuAccelerationStructureBindingType,
        PcuAccelerationStructureLevel,
        PcuBinding,
        PcuBindingAccess,
        PcuBindingRef,
        PcuBindingStorageClass,
        PcuDispatchOpCaps,
        PcuDispatchDataOp,
        PcuDispatchIndex,
        PcuDispatchEntryPoint,
        PcuDispatchKernelIr,
        PcuDispatchOp,
        PcuDispatchValueId,
        PcuDispatchPolicyCaps,
        PcuIrKind,
        PcuKernel,
        PcuKernelIrContract,
        PcuParameterValue,
        PcuValueType,
        PcuValueTypeCaps,
    };

    #[test]
    fn required_scalar_alu_support_tracks_every_scalar_variant_exactly() {
        let ops = crate::PcuScalarType::ALL.map(|scalar| {
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: PcuValueType::Scalar(scalar),
                result: PcuDispatchValueId(3),
                op: PcuDispatchAluOp::Add,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            })
        });
        let kernel = PcuDispatchKernelIr {
            id: crate::PcuKernelId(1),
            entry: PcuDispatchEntryPoint {
                name: "scalar-table",
                logical_shape: [1, 1, 1],
            },
            bindings: &[],
            ports: &[],
            parameters: &[],
            ops: &ops,
            type_caps: PcuValueTypeCaps::empty(),
            feature_caps: PcuDispatchFeatureCaps::empty(),
        };
        let required = kernel.required_scalar_alu_support();
        for scalar in crate::PcuScalarType::ALL {
            assert_eq!(required.for_scalar(scalar), PcuDispatchOpCaps::ALU_ADD);
        }
        assert!(!kernel.has_non_scalar_alu_type());
    }

    #[test]
    fn checked_div_rem_contributes_its_typed_requirement_for_direct_and_grid_ops() {
        let checked = PcuDispatchDataOp::CheckedDivRem {
            value_type: PcuValueType::i32(),
            flags: PcuIntegerDivFlags::CHECKED,
            quotient: PcuDispatchValueId(3),
            remainder: PcuDispatchValueId(4),
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
        };
        assert_eq!(
            checked.support_flag(),
            PcuDispatchOpCaps::ALU_CHECKED_DIV_REM
        );
        let ops = [PcuDispatchOp::Data(checked)];
        let kernel = PcuDispatchKernelIr {
            id: crate::PcuKernelId(2),
            entry: PcuDispatchEntryPoint {
                name: "checked-div-rem",
                logical_shape: [1, 1, 1],
            },
            bindings: &[],
            ports: &[],
            parameters: &[],
            ops: &ops,
            type_caps: PcuValueTypeCaps::empty(),
            feature_caps: PcuDispatchFeatureCaps::empty(),
        };

        assert!(
            kernel
                .required_instruction_support()
                .contains(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
        );
        let typed = kernel.required_scalar_alu_support();
        assert_eq!(
            typed.for_scalar(crate::PcuScalarType::I32),
            PcuDispatchOpCaps::ALU_CHECKED_DIV_REM
        );
        assert_eq!(
            typed.for_scalar(crate::PcuScalarType::U32),
            PcuDispatchOpCaps::empty()
        );
        assert!(
            kernel
                .required_type_support()
                .contains(PcuValueTypeCaps::INT32 | PcuValueTypeCaps::SCALAR_VALUES)
        );
        assert!(!kernel.has_non_scalar_alu_type());

        let body = [PcuDispatchOp::Data(checked)];
        let loop_ops = [
            PcuDispatchOp::GridStrideLoop {
                extent: 8,
                body: &body,
            },
            PcuDispatchOp::Control(crate::PcuDispatchControlOp::Return),
        ];
        let loop_kernel = PcuDispatchKernelIr {
            ops: &loop_ops,
            ..kernel
        };
        assert_eq!(
            loop_kernel
                .required_scalar_alu_support()
                .for_scalar(crate::PcuScalarType::I32),
            PcuDispatchOpCaps::ALU_CHECKED_DIV_REM
        );
        assert!(
            loop_kernel
                .required_instruction_support()
                .contains(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
        );
    }

    #[test]
    fn integer_division_flags_reserve_div_or_zero_without_enabling_it() {
        let checked = PcuIntegerDivFlags::default();
        assert_eq!(checked, PcuIntegerDivFlags::CHECKED);
        assert_eq!(checked.bits(), 0);
        assert!(!checked.contains(PcuIntegerDivFlags::DIV_OR_ZERO));
        assert!(PcuIntegerDivFlags::DIV_OR_ZERO.contains(PcuIntegerDivFlags::DIV_OR_ZERO));
        assert_eq!(
            checked.union(PcuIntegerDivFlags::DIV_OR_ZERO),
            PcuIntegerDivFlags::DIV_OR_ZERO
        );
        assert!(PcuDispatchOpCaps::all().contains(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM));
    }

    #[test]
    fn builder_synthesizes_dispatch_kernel_with_ops() {
        let builder = PcuDispatchKernelBuilder::<4>::new(0x21, "main", [32, 1, 1])
            .with_type_caps(PcuValueTypeCaps::UINT32 | PcuValueTypeCaps::SCALAR_VALUES)
            .with_arithmetic_op(PcuDispatchAluOp::Add)
            .expect("builder should accept one op");
        let kernel = builder.ir();

        assert_eq!(kernel.id.0, 0x21);
        assert_eq!(kernel.kind(), PcuIrKind::Dispatch);
        assert_eq!(kernel.entry.name, "main");
        assert_eq!(kernel.entry.logical_shape, [32, 1, 1]);
        assert_eq!(kernel.ops.len(), 1);
        assert!(kernel.type_caps.contains(PcuValueTypeCaps::UINT32));
        assert_eq!(
            kernel.required_dispatch_policy(),
            PcuDispatchPolicyCaps::ORDERED_SUBMISSION
        );
    }

    #[test]
    fn per_binding_extent_keeps_broadcast_scalar_compact() {
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
        let ops = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::BindingElementZero,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(1),
            }),
        ];
        let builder = PcuDispatchKernelBuilder::<2>::new(12, "broadcast", [64, 1, 1])
            .with_bindings(&bindings)
            .with_ops(&ops)
            .expect("operations fit");
        let kernel = builder.ir();

        assert_eq!(kernel.minimum_binding_elements(64), 64);
        assert!(
            kernel.required_instruction_support().contains(
                PcuDispatchOpCaps::BINDING_LOAD
                    .union(PcuDispatchOpCaps::BINDING_LOAD_ELEMENT_ZERO)
                    .union(PcuDispatchOpCaps::BINDING_STORE)
            )
        );
        assert_eq!(
            kernel.minimum_binding_elements_for(PcuBindingRef::new(0, 0), 64),
            1
        );
        assert_eq!(
            kernel.minimum_binding_elements_for(PcuBindingRef::new(0, 1), 64),
            64
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
    fn value_type_caps_cover_core_scalar_types() {
        let caps = PcuValueTypeCaps::BOOL
            | PcuValueTypeCaps::INT4
            | PcuValueTypeCaps::UINT4
            | PcuValueTypeCaps::INT8
            | PcuValueTypeCaps::UINT8
            | PcuValueTypeCaps::INT16
            | PcuValueTypeCaps::UINT16
            | PcuValueTypeCaps::INT32
            | PcuValueTypeCaps::UINT32
            | PcuValueTypeCaps::INT64
            | PcuValueTypeCaps::UINT64
            | PcuValueTypeCaps::FLOAT16
            | PcuValueTypeCaps::BFLOAT16
            | PcuValueTypeCaps::FLOAT32
            | PcuValueTypeCaps::FLOAT64
            | PcuValueTypeCaps::SCALAR_VALUES
            | PcuValueTypeCaps::VECTOR_VALUES
            | PcuValueTypeCaps::MATRIX_VALUES;

        assert!(caps.supports_value_type(PcuValueType::bool()));
        assert!(caps.supports_value_type(PcuValueType::i4()));
        assert!(caps.supports_value_type(PcuValueType::u4()));
        assert!(caps.supports_value_type(PcuValueType::i8()));
        assert!(caps.supports_value_type(PcuValueType::u8()));
        assert!(caps.supports_value_type(PcuValueType::i16()));
        assert!(caps.supports_value_type(PcuValueType::u16()));
        assert!(caps.supports_value_type(PcuValueType::i32()));
        assert!(caps.supports_value_type(PcuValueType::u32()));
        assert!(caps.supports_value_type(PcuValueType::i64()));
        assert!(caps.supports_value_type(PcuValueType::u64()));
        assert!(caps.supports_value_type(PcuValueType::f16()));
        assert!(caps.supports_value_type(PcuValueType::bf16()));
        assert!(caps.supports_value_type(PcuValueType::f32()));
        assert!(caps.supports_value_type(PcuValueType::f64()));
        assert!(caps.supports_value_type(PcuValueType::Vector {
            scalar: crate::PcuScalarType::F64,
            lanes: 4,
        }));
        assert!(caps.supports_value_type(PcuValueType::Matrix {
            scalar: crate::PcuScalarType::BF16,
            rows: 4,
            cols: 4,
        }));
    }

    #[test]
    fn kernel_required_type_support_is_fully_explicit() {
        let scalar_builder = PcuDispatchKernelBuilder::<1>::new(1, "main", [1, 1, 1])
            .with_type_caps(PcuValueTypeCaps::UINT32 | PcuValueTypeCaps::SCALAR_VALUES);
        let scalar = scalar_builder.ir();
        let matrix_builder = PcuDispatchKernelBuilder::<1>::new(2, "main", [1, 1, 1])
            .with_type_caps(PcuValueTypeCaps::BFLOAT16 | PcuValueTypeCaps::MATRIX_VALUES);
        let matrix = matrix_builder.ir();
        let unshaped_builder = PcuDispatchKernelBuilder::<1>::new(3, "main", [1, 1, 1])
            .with_type_caps(PcuValueTypeCaps::UINT32);
        let unshaped = unshaped_builder.ir();

        assert!(
            scalar
                .required_type_support()
                .contains(PcuValueTypeCaps::UINT32 | PcuValueTypeCaps::SCALAR_VALUES)
        );
        assert!(
            matrix
                .required_type_support()
                .contains(PcuValueTypeCaps::BFLOAT16 | PcuValueTypeCaps::MATRIX_VALUES)
        );
        assert!(
            !matrix
                .required_type_support()
                .contains(PcuValueTypeCaps::SCALAR_VALUES)
        );
        assert!(
            !unshaped
                .required_type_support()
                .contains(PcuValueTypeCaps::SCALAR_VALUES)
        );
    }

    #[test]
    fn dispatch_type_requirements_include_bindings_and_constants() {
        let bindings = [PcuBinding::value(
            Some("input"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::f32(),
        )];
        let builder = PcuDispatchKernelBuilder::<1>::new(6, "main", [1, 1, 1])
            .with_bindings(&bindings)
            .with_data_op(PcuDispatchDataOp::Constant {
                result: PcuDispatchValueId(1),
                value: PcuParameterValue::U32(1),
            })
            .expect("one constant fits");
        let kernel = builder.ir();

        assert!(kernel.required_type_support().contains(
            PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::UINT32 | PcuValueTypeCaps::SCALAR_VALUES
        ));
    }

    #[test]
    fn dispatch_feature_caps_remain_independent_from_type_caps() {
        let builder = PcuDispatchKernelBuilder::<1>::new(4, "main", [1, 1, 1])
            .with_type_caps(PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::VECTOR_VALUES)
            .with_feature_caps(
                PcuDispatchFeatureCaps::INLINE_PARAMETERS
                    | PcuDispatchFeatureCaps::COOPERATIVE_SCRATCHPAD,
            );
        let kernel = builder.ir();

        assert!(
            kernel
                .required_type_support()
                .contains(PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::VECTOR_VALUES)
        );
        assert!(kernel.required_feature_support().contains(
            PcuDispatchFeatureCaps::INLINE_PARAMETERS
                | PcuDispatchFeatureCaps::COOPERATIVE_SCRATCHPAD
        ));
    }

    #[test]
    fn dispatch_feature_support_is_derived_from_signature_shape() {
        let bindings = [
            PcuBinding::value(
                Some("readonly"),
                0,
                0,
                PcuBindingStorageClass::Uniform,
                PcuBindingAccess::ReadOnly,
                PcuValueType::u32(),
            ),
            PcuBinding::value(
                Some("shared"),
                0,
                1,
                PcuBindingStorageClass::Shared,
                PcuBindingAccess::ReadWrite,
                PcuValueType::u32(),
            ),
        ];
        let parameters = [crate::PcuParameter {
            slot: crate::PcuParameterSlot(0),
            name: Some("scale"),
            value_type: PcuValueType::u32(),
        }];
        let builder = PcuDispatchKernelBuilder::<1>::new(5, "main", [1, 1, 1])
            .with_bindings(&bindings)
            .with_parameters(&parameters);
        let kernel = builder.ir();

        assert!(kernel.required_feature_support().contains(
            PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
                | PcuDispatchFeatureCaps::MUTABLE_RESOURCES
                | PcuDispatchFeatureCaps::INLINE_PARAMETERS
                | PcuDispatchFeatureCaps::COOPERATIVE_SCRATCHPAD
        ));
    }

    #[test]
    fn coordinate_ops_are_dispatch_instruction_extensions() {
        let builder = PcuDispatchKernelBuilder::<3>::new(6, "coordinate", [1, 1, 1])
            .with_coordinate_op(PcuDispatchCoordinateOp::LoadCoordinate)
            .expect("builder should accept coordinate load")
            .with_coordinate_op(PcuDispatchCoordinateOp::DerivativeX)
            .expect("builder should accept derivative op")
            .with_coordinate_op(PcuDispatchCoordinateOp::StoreOutput)
            .expect("builder should accept output store");
        let kernel = builder.ir();

        assert!(kernel.required_instruction_support().contains(
            PcuDispatchOpCaps::COORDINATE_LOAD
                | PcuDispatchOpCaps::DERIVATIVE_X
                | PcuDispatchOpCaps::OUTPUT_STORE
        ));
        assert!(kernel.required_coordinate_instruction_support().contains(
            PcuDispatchOpCaps::COORDINATE_LOAD
                | PcuDispatchOpCaps::DERIVATIVE_X
                | PcuDispatchOpCaps::OUTPUT_STORE
        ));
        assert_eq!(
            kernel.required_ray_trace_instruction_support(),
            PcuDispatchOpCaps::empty()
        );
    }

    #[test]
    fn ray_trace_ops_are_dispatch_instruction_extensions() {
        let acceleration_structure = PcuBinding::acceleration_structure(
            Some("scene"),
            0,
            0,
            PcuBindingAccess::ReadOnly,
            PcuAccelerationStructureBindingType {
                level: PcuAccelerationStructureLevel::TopLevel,
                mutable: false,
            },
        );
        let trace = PcuTraceRayOp::new(acceleration_structure.reference())
            .with_payload_bytes(16)
            .with_max_recursion_depth(1);
        let bindings = [acceleration_structure];
        let builder = PcuDispatchKernelBuilder::<3>::new(7, "trace", [1, 1, 1])
            .with_bindings(&bindings)
            .with_ray_trace_op(PcuDispatchRayTraceOp::TraceRay(trace))
            .expect("builder should accept trace op")
            .with_ray_trace_op(PcuDispatchRayTraceOp::PayloadRead {
                byte_offset: 0,
                byte_len: 16,
            })
            .expect("builder should accept payload read")
            .with_ray_trace_op(PcuDispatchRayTraceOp::PayloadWrite {
                byte_offset: 0,
                byte_len: 16,
            })
            .expect("builder should accept payload write");
        let kernel = builder.ir();

        assert!(trace.validate(kernel.bindings).is_ok());
        assert!(kernel.uses_ray_tracing());
        assert!(kernel.required_instruction_support().contains(
            PcuDispatchOpCaps::RAY_TRACE
                | PcuDispatchOpCaps::RAY_PAYLOAD_READ
                | PcuDispatchOpCaps::RAY_PAYLOAD_WRITE
        ));
        assert!(kernel.required_ray_trace_instruction_support().contains(
            PcuDispatchOpCaps::RAY_TRACE
                | PcuDispatchOpCaps::RAY_PAYLOAD_READ
                | PcuDispatchOpCaps::RAY_PAYLOAD_WRITE
        ));
    }
}
