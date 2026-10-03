//! Explicit delegated runtime owners; no implicit device or numerical policy selection.

#[rustfmt::skip]
use std::{
    cell::{Cell,RefCell},
    fmt,
    path::Path,
    rc::Rc,
};
#[cfg(feature = "tensor")]
use std::sync::Arc;
use crate::ffi;

/// Operational or admission error; native numerical exceptions are explicitly permitted only
/// by the prepared primitive's selected policy, never inferred from this error type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MlxError {
    /// The explicitly selected native C library could not be loaded; no CPU tensor work is attempted.
    Unavailable(String),
    /// MLX execution is supported only on Apple silicon macOS.
    UnsupportedPlatform,
    /// The C safety ABI or pinned source/runtime identity does not satisfy the adapter contract.
    Abi(String),
    InvalidExtent,
    ForeignSession,
    /// Delegated storage requires the exact Rust F32 identity, not only an equal byte width.
    UnsupportedScalar(fusion_pcu::PcuScalarType),
    InvalidRequest(String),
    Runtime(String),
    /// Fixed checked synthesis failed; recovered range faults retain completed borrowed payloads.
    Arithmetic(fusion_pcu::PcuExecutionFault),
    /// Work might remain in flight; real SDK owners and the library are quarantined.
    CompletionUnknown(String),
    #[cfg(feature = "tensor")]
    Unsupported(fusion_pcu::dialect::tensor::TensorUnsupportedReason),
}
impl fmt::Display for MlxError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}
impl std::error::Error for MlxError {}

/// The actual GPU backend compiled into and available from the pinned MLX runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MlxGpuBackend {
    Metal,
}

/// SDK-reported cold facts. Names are descriptive and never establish physical identity.
/// No Metal registry ID or extra memory capacity is fabricated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlxDeviceFacts {
    pub index: usize,
    pub backend: MlxGpuBackend,
    pub name: String,
    pub architecture: String,
}

/// A loaded upstream C image retained by every session, array and prepared primitive.
///
/// Cargo prepares the pinned native runtime on Apple silicon. Loading itself performs no
/// build/install work; every session and array retains the checked loaded image.
#[derive(Clone)]
pub struct MlxRuntime(Rc<ffi::Api>);
impl MlxRuntime {
    /// Loads the trusted safety-patched upstream C image and checks its source/ABI/runtime family.
    ///
    /// # Errors
    /// Returns unsupported platform, unavailable library, missing symbol, ABI or SDK mismatch.
    pub fn load(library: impl AsRef<Path>) -> Result<Self, MlxError> {
        ffi::Api::load(library.as_ref()).map(Self)
    }

    /// Loads Cargo's matching runtime, honoring an optional `PCU_MLX_LIBRARY` override.
    ///
    /// # Errors
    /// Returns unsupported platform, unavailable runtime, or source/ABI/SDK mismatch.
    pub fn load_default() -> Result<Self, MlxError> {
        ffi::Api::load_default().map(Self)
    }
    #[must_use]
    pub fn version(&self) -> &str {
        self.0.version()
    }

    /// Reads actual GPU inventory without evaluating a tensor workload.
    ///
    /// # Errors
    /// Returns GPU/backend availability or SDK operational errors.
    pub fn devices(&self) -> Result<Vec<MlxDeviceFacts>, MlxError> {
        self.0.devices()
    }

    /// Opens an explicit new stream on an available GPU, leaving MLX defaults untouched.
    ///
    /// # Errors
    /// Returns absent GPU, invalid index or native activation error.
    pub fn open_gpu(&self, index: usize) -> Result<MlxSession, MlxError> {
        self.open_gpu_with_identity(index, None)
    }
    pub(super) fn open_gpu_with_identity(
        &self,
        index: usize,
        identity: Option<fusion_pcu::PcuDeviceIdentity>,
    ) -> Result<MlxSession, MlxError> {
        let facts = self.0.device(index)?;
        let native = self.0.open(index)?;
        Ok(MlxSession(Rc::new(Session {
            native,
            facts,
            identity,
        })))
    }
}

struct Session {
    native: ffi::Session,
    facts: MlxDeviceFacts,
    identity: Option<fusion_pcu::PcuDeviceIdentity>,
}

/// One explicit MLX GPU stream. Clones retain exact affinity and stay thread-confined.
#[derive(Clone)]
pub struct MlxSession(Rc<Session>);

/// Observable materialization state; MLX's internal allocator/cache is still delegated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlxArrayResidency {
    /// Host data was copied to MLX-owned initialized storage, not yet used in GPU work.
    HostCopied,
    /// Explicit GPU work or a checked official GPU view produced this terminal array.
    /// Its backing remains MLX-owned.
    GpuEvaluated,
    /// A host read materialized CPU-visible data from the MLX-owned backing.
    HostMaterialized,
}

struct Array {
    native: ffi::Array,
    residency: Cell<MlxArrayResidency>,
    encoded_view: RefCell<Option<Rc<encoded::EncodedBacking>>>,
}

/// Immutable owned initialized F32 matrix.
///
/// Host upload copies into MLX-owned storage. Official checked immutable views may share that owned storage after terminal validation.
/// Native Metal buffer import is unsupported. Clones share the same SDK holder and backing;
/// they do not duplicate physical allocations.
#[derive(Clone)]
pub struct MlxArray {
    session: MlxSession,
    array: Rc<Array>,
}

impl MlxSession {
    pub(crate) fn prepare_composed_native(
        &self,
        plan: crate::MlxCheckedMapPlan,
    ) -> Result<ffi::Composed, MlxError> {
        ffi::Composed::prepare(&self.0.native, plan)
    }
    #[cfg(feature = "benchmark-control")]
    pub(crate) fn prepare_native_checked_map_control_internal(
        &self,
        plan: crate::MlxCheckedMapPlan,
        header: String,
        body: String,
    ) -> Result<ffi::Composed, MlxError> {
        ffi::Composed::prepare_control(&self.0.native, plan, header, body)
    }
    pub(crate) fn prepare_transport_native(
        &self,
        plan: &crate::MlxTransportPlan,
    ) -> Result<ffi::Transport, MlxError> {
        ffi::Transport::prepare(&self.0.native, plan)
    }
    pub(crate) fn prepare_transport_native_with_input_extents(
        &self,
        plan: &crate::MlxTransportPlan,
        extents: &[usize],
    ) -> Result<ffi::Transport, MlxError> {
        ffi::Transport::prepare_with_input_extents(&self.0.native, plan, extents)
    }
    #[cfg(feature = "benchmark-control")]
    pub(crate) fn prepare_native_transport_control_internal(
        &self,
        scalar: fusion_pcu::PcuScalarType,
        element_count: usize,
        workload: crate::MlxNativeTransportWorkload,
        extents: &[usize],
    ) -> Result<ffi::Transport, MlxError> {
        ffi::Transport::prepare_native_control(
            &self.0.native,
            scalar,
            element_count,
            workload,
            extents,
        )
    }
    pub(crate) fn upload_transport_bytes(
        &self,
        scalar: fusion_pcu::PcuScalarType,
        count: usize,
        bytes: &[u8],
    ) -> Result<MlxEncodedArray, MlxError> {
        self.upload_encoded_bytes(scalar, count, bytes)
    }
    pub(crate) fn wrap_transport_output(&self, output: ffi::EncodedArray) -> MlxEncodedArray {
        self.wrap_encoded(output)
    }
    #[cfg(test)]
    pub(crate) fn native_for_binary_proof(&self) -> &ffi::Session {
        &self.0.native
    }
    pub(crate) fn prepare_checked_div_rem_native(
        &self,
        scalar: fusion_pcu::PcuScalarType,
        count: usize,
        inputs: [usize; 2],
        broadcast: [bool; 2],
    ) -> Result<ffi::CheckedDivRem, MlxError> {
        ffi::CheckedDivRem::prepare(&self.0.native, scalar, count, inputs, broadcast)
    }
    pub(crate) fn prepare_checked_div_rem_native_with_input_extents(
        &self,
        scalar: fusion_pcu::PcuScalarType,
        count: usize,
        inputs: [usize; 2],
        broadcast: [bool; 2],
    ) -> Result<ffi::CheckedDivRem, MlxError> {
        ffi::CheckedDivRem::prepare_with_input_extents(
            &self.0.native,
            scalar,
            count,
            inputs,
            broadcast,
        )
    }
    #[allow(clippy::too_many_arguments)] // Independent frozen integer operation, range and both operand extents/roles.
    pub(crate) fn prepare_checked_integer_native(
        &self,
        scalar: fusion_pcu::PcuScalarType,
        operation: fusion_pcu::PcuDispatchIntegerBinaryOp,
        range: fusion_pcu::PcuRangePolicy,
        count: usize,
        inputs: [usize; 2],
        broadcast: [bool; 2],
    ) -> Result<ffi::CheckedInteger, MlxError> {
        let operation = match operation {
            fusion_pcu::PcuDispatchIntegerBinaryOp::Add => 0,
            fusion_pcu::PcuDispatchIntegerBinaryOp::Sub => 1,
            fusion_pcu::PcuDispatchIntegerBinaryOp::Mul => 2,
        };
        ffi::CheckedInteger::prepare(
            &self.0.native,
            scalar,
            operation,
            range,
            count,
            inputs,
            broadcast,
        )
    }
    #[allow(clippy::too_many_arguments)] // Independent frozen integer operation, range and both operand extents/roles.
    pub(crate) fn prepare_checked_integer_native_with_input_extents(
        &self,
        scalar: fusion_pcu::PcuScalarType,
        operation: fusion_pcu::PcuDispatchIntegerBinaryOp,
        range: fusion_pcu::PcuRangePolicy,
        count: usize,
        inputs: [usize; 2],
        broadcast: [bool; 2],
        full: [usize; 2],
    ) -> Result<ffi::CheckedInteger, MlxError> {
        let operation = match operation {
            fusion_pcu::PcuDispatchIntegerBinaryOp::Add => 0,
            fusion_pcu::PcuDispatchIntegerBinaryOp::Sub => 1,
            fusion_pcu::PcuDispatchIntegerBinaryOp::Mul => 2,
        };
        ffi::CheckedInteger::prepare_with_input_extents(
            &self.0.native,
            scalar,
            operation,
            range,
            count,
            inputs,
            broadcast,
            full,
        )
    }
    pub(crate) fn prepare_carrier_native(
        &self,
        scalar: fusion_pcu::PcuScalarType,
        count: usize,
        broadcast: bool,
    ) -> Result<ffi::CarrierCopy, MlxError> {
        ffi::CarrierCopy::prepare(&self.0.native, scalar, count, broadcast)
    }
    pub(crate) fn prepare_carrier_native_with_input_extent(
        &self,
        scalar: fusion_pcu::PcuScalarType,
        count: usize,
        broadcast: bool,
        input_count: usize,
    ) -> Result<ffi::CarrierCopy, MlxError> {
        ffi::CarrierCopy::prepare_with_input_extent(
            &self.0.native,
            scalar,
            count,
            broadcast,
            input_count,
        )
    }
    #[allow(clippy::too_many_arguments)] // Both cold operand extents/broadcast roles are independent.
    pub(crate) fn prepare_checked_binary_native(
        &self,
        scalar: fusion_pcu::PcuScalarType,
        operation: fusion_pcu::PcuDispatchFloatBinaryOp,
        policy: fusion_pcu::PcuFloatUnderflowPolicy,
        range: fusion_pcu::PcuRangePolicy,
        count: usize,
        inputs: [usize; 2],
        broadcast: [bool; 2],
    ) -> Result<ffi::CheckedBinary, MlxError> {
        ffi::CheckedBinary::prepare(
            &self.0.native,
            scalar,
            operation,
            policy,
            range,
            count,
            inputs,
            broadcast,
        )
    }
    #[allow(clippy::too_many_arguments)] // Logical operand spans and physical full shapes are distinct cold dimensions.
    pub(crate) fn prepare_checked_binary_native_with_input_extents(
        &self,
        scalar: fusion_pcu::PcuScalarType,
        operation: fusion_pcu::PcuDispatchFloatBinaryOp,
        policy: fusion_pcu::PcuFloatUnderflowPolicy,
        range: fusion_pcu::PcuRangePolicy,
        count: usize,
        inputs: [usize; 2],
        broadcast: [bool; 2],
        full: [usize; 2],
    ) -> Result<ffi::CheckedBinary, MlxError> {
        ffi::CheckedBinary::prepare_with_input_extents(
            &self.0.native,
            scalar,
            operation,
            policy,
            range,
            count,
            inputs,
            broadcast,
            full,
        )
    }
    #[allow(clippy::too_many_arguments)] // Independent cold scalar operation/policy/extent dimensions.
    pub(crate) fn prepare_checked_unary_native(
        &self,
        scalar: fusion_pcu::PcuScalarType,
        op: fusion_pcu::PcuDispatchFloatUnaryOp,
        policy: fusion_pcu::PcuFloatUnderflowPolicy,
        range: fusion_pcu::PcuRangePolicy,
        count: usize,
        broadcast: bool,
    ) -> Result<ffi::CheckedUnary, MlxError> {
        ffi::CheckedUnary::prepare(&self.0.native, scalar, op, policy, range, count, broadcast)
    }
    #[allow(clippy::too_many_arguments)] // Full resident capacity is separate from operation/policy/read span.
    pub(crate) fn prepare_checked_unary_native_with_input_extent(
        &self,
        scalar: fusion_pcu::PcuScalarType,
        op: fusion_pcu::PcuDispatchFloatUnaryOp,
        policy: fusion_pcu::PcuFloatUnderflowPolicy,
        range: fusion_pcu::PcuRangePolicy,
        count: usize,
        broadcast: bool,
        input_extent: usize,
    ) -> Result<ffi::CheckedUnary, MlxError> {
        ffi::CheckedUnary::prepare_with_input_extent(
            &self.0.native,
            scalar,
            op,
            policy,
            range,
            count,
            broadcast,
            input_extent,
        )
    }
    /// Exact opaque runtime/stream affinity; equal physical GPU indices alone do not suffice.
    #[must_use]
    pub fn same_session(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }

    /// Checks whether this synchronous session's backing may be accessed or submitted again.
    ///
    /// # Errors
    /// Returns completion uncertainty when native work or host materialization was quarantined.
    pub fn validate_access_available(&self) -> Result<(), MlxError> {
        self.0.native.ensure_ready()
    }

    /// Stages sealed PCU scalar storage after proving the exact Rust `f32` identity.
    /// It adds no intermediate allocation or conversion to the canonical copied upload.
    ///
    /// # Errors
    /// Returns unsupported scalar, extent, quarantine or native upload failure.
    pub fn upload_typed<T: fusion_pcu::PcuScalar>(
        &self,
        shape: [usize; 2],
        data: &[T],
    ) -> Result<MlxArray, MlxError> {
        self.validate_access_available()?;
        self.upload_f32(shape, ffi::as_f32(data)?)
    }

    /// Prepares the native retained MLX primitive control for matched benchmark boundaries.
    /// This explicit measurement control does not claim PCU numerical/source admission.
    ///
    /// # Errors
    /// Returns incompatible/invalid shapes or pinned native compilation errors.
    #[cfg(feature = "benchmark-control")]
    pub fn prepare_native_matmul_control(
        &self,
        left_shape: [usize; 2],
        right_shape: [usize; 2],
    ) -> Result<MlxNativeMatmulControl, MlxError> {
        let native = self.0.native.prepare_matmul(left_shape, right_shape)?;
        Ok(MlxNativeMatmulControl {
            session: self.clone(),
            native: Rc::new(native),
        })
    }

    /// Executes the native control with the same owned output/eval/sync/wait boundary.
    ///
    /// # Errors
    /// Returns shape/session guards or native execution/quarantine errors.
    #[cfg(feature = "benchmark-control")]
    pub fn execute_native_matmul_control(
        &self,
        control: &MlxNativeMatmulControl,
        left: &MlxArray,
        right: &MlxArray,
    ) -> Result<MlxArray, MlxError> {
        if !Rc::ptr_eq(&self.0, &control.session.0) {
            return Err(MlxError::ForeignSession);
        }
        let native = control
            .native
            .execute(&left.array.native, &right.array.native)?;
        Ok(MlxArray {
            session: self.clone(),
            array: Rc::new(Array {
                native,
                residency: Cell::new(MlxArrayResidency::GpuEvaluated),
                encoded_view: RefCell::new(None),
            }),
        })
    }

    /// Executes the upstream C frontend for matched benchmark controls.
    /// Native graph descriptor construction remains part of this route.
    ///
    /// # Errors
    /// Returns foreign session, shape, native error or uncertain completion.
    #[cfg(feature = "benchmark-control")]
    pub fn execute_native_frontend_control(
        &self,
        left: &MlxArray,
        right: &MlxArray,
    ) -> Result<MlxArray, MlxError> {
        let native = self
            .0
            .native
            .matmul(&left.array.native, &right.array.native)?;
        Ok(MlxArray {
            session: self.clone(),
            array: Rc::new(Array {
                native,
                residency: Cell::new(MlxArrayResidency::GpuEvaluated),
                encoded_view: RefCell::new(None),
            }),
        })
    }

    /// Executes the public C compiled wrapper for matched benchmark controls.
    /// This diagnostic route performs the upstream vector/closure/cache work each call.
    ///
    /// # Errors
    /// Returns foreign session, changed shapes/traces, native error or uncertain completion.
    #[cfg(feature = "benchmark-control")]
    pub fn execute_native_compiled_control(
        &self,
        control: &MlxNativeMatmulControl,
        left: &MlxArray,
        right: &MlxArray,
    ) -> Result<MlxArray, MlxError> {
        if !self.same_session(&control.session) {
            return Err(MlxError::ForeignSession);
        }
        let native = control
            .native
            .execute_compiled(&left.array.native, &right.array.native)?;
        Ok(MlxArray {
            session: self.clone(),
            array: Rc::new(Array {
                native,
                residency: Cell::new(MlxArrayResidency::GpuEvaluated),
                encoded_view: RefCell::new(None),
            }),
        })
    }

    /// Discovery-generation affinity when opened through the validated discovery provider.
    #[must_use]
    pub fn identity(&self) -> Option<fusion_pcu::PcuDeviceIdentity> {
        self.0.identity
    }
    #[must_use]
    pub fn facts(&self) -> &MlxDeviceFacts {
        &self.0.facts
    }

    /// Copies an ordinary dense F32 matrix into MLX-owned storage without retaining the slice.
    /// MLX may transfer/materialize it on first GPU use; this is not a zero-copy import claim.
    ///
    /// # Errors
    /// Returns invalid shape/extent, quarantine or allocation/SDK failure.
    pub fn upload_f32(&self, shape: [usize; 2], data: &[f32]) -> Result<MlxArray, MlxError> {
        let native = self.0.native.upload(shape, data)?;
        Ok(MlxArray {
            session: self.clone(),
            array: Rc::new(Array {
                native,
                residency: Cell::new(MlxArrayResidency::HostCopied),
                encoded_view: RefCell::new(None),
            }),
        })
    }

    /// Freezes an authentic admitted matrix-product contract against this exact session.
    /// Cold-traces MLX's public compiler once, then retains the exact single matrix primitive
    /// on this session's explicit stream. No tensor data is evaluated during preparation.
    ///
    /// # Errors
    /// Returns unsupported contract, unavailable compilation or an unproved compiled topology.
    #[cfg(feature = "tensor")]
    pub fn prepare_matmul(
        &self,
        graph: &fusion_pcu::dialect::tensor::Graph,
        node: fusion_pcu::dialect::tensor::NodeDescriptor<'_>,
    ) -> Result<MlxPreparedMatmul, MlxError> {
        let plan = crate::MlxMatmulPlan::assess(graph, node).map_err(MlxError::Unsupported)?;
        self.prepare_plan(plan)
    }

    #[cfg(feature = "tensor")]
    fn prepare_plan(&self, plan: crate::MlxMatmulPlan) -> Result<MlxPreparedMatmul, MlxError> {
        let native = self
            .0
            .native
            .prepare_matmul(plan.left_shape, plan.right_shape)?;
        Ok(MlxPreparedMatmul {
            session: self.clone(),
            plan,
            native: Rc::new(native),
        })
    }

    /// Prepares an authentic captured/selected source program through the same matrix admission.
    /// The selected program and exact `ValueId` bindings remain owned, with no graph cloning.
    ///
    /// # Errors
    /// Returns unsupported source topology/contract or cold native compilation failure.
    #[cfg(feature = "tensor")]
    pub fn prepare_program(
        &self,
        program: Arc<fusion_pcu::dialect::tensor::TensorOwnedSelectedProgram>,
    ) -> Result<MlxPreparedProgram, MlxError> {
        let plan = crate::MlxMatmulPlan::assess_program(&program).map_err(MlxError::Unsupported)?;
        let matmul = self.prepare_plan(plan)?;
        Ok(MlxPreparedProgram { program, matmul })
    }

    /// Executes exact captured input bindings with retained compiled matrix replay.
    /// Binding validation completes before tensor work; the output owns terminal MLX backing.
    ///
    /// # Errors
    /// Returns missing/duplicate/foreign bindings, shape/session error or native execution failure.
    #[cfg(feature = "tensor")]
    pub fn execute_program(
        &self,
        prepared: &MlxPreparedProgram,
        inputs: &[(fusion_pcu::dialect::tensor::ValueId, &MlxArray)],
    ) -> Result<MlxArray, MlxError> {
        let [first, second] = inputs else {
            return Err(MlxError::InvalidRequest(
                "expected two captured input bindings".into(),
            ));
        };
        let [left, right] = prepared.matmul.plan.inputs();
        let (a, b) = if first.0 == left && second.0 == right {
            (first.1, second.1)
        } else if first.0 == right && second.0 == left {
            (second.1, first.1)
        } else {
            return Err(MlxError::InvalidRequest(
                "captured input identities mismatch".into(),
            ));
        };
        self.execute_matmul(&prepared.matmul, a, b)
    }

    /// Evaluates one prepared matrix product and establishes explicit-stream terminal completion.
    ///
    /// Rebinds one fresh logical result descriptor to the frozen compiled matrix primitive.
    /// Warm execution bypasses MLX's compiler cache/default stream and `matmul` frontend.
    /// MLX still owns result allocation, GPU kernel selection and evaluation scheduling.
    /// Numeric nonfinite/underflow behavior follows the explicit native/optimized permission.
    ///
    /// # Errors
    /// Returns foreign session/shape before work, operational SDK error, or poisoned/quarantined
    /// unknown completion. Caller inputs remain owned and no host output is published.
    #[cfg(feature = "tensor")]
    pub fn execute_matmul(
        &self,
        prepared: &MlxPreparedMatmul,
        left: &MlxArray,
        right: &MlxArray,
    ) -> Result<MlxArray, MlxError> {
        if !Rc::ptr_eq(&self.0, &prepared.session.0)
            || !left.array.native.same_session(&self.0.native)
            || !right.array.native.same_session(&self.0.native)
        {
            return Err(MlxError::ForeignSession);
        }
        if left.shape() != prepared.plan.left_shape || right.shape() != prepared.plan.right_shape {
            return Err(MlxError::InvalidExtent);
        }
        let native = prepared
            .native
            .execute(&left.array.native, &right.array.native)?;
        Ok(MlxArray {
            session: self.clone(),
            array: Rc::new(Array {
                native,
                residency: Cell::new(MlxArrayResidency::GpuEvaluated),
                encoded_view: RefCell::new(None),
            }),
        })
    }
}

/// Feature-gated actual native MLX control for paired measurements, with no PCU graph frontend.
#[cfg(feature = "benchmark-control")]
#[derive(Clone)]
pub struct MlxNativeMatmulControl {
    session: MlxSession,
    native: Rc<ffi::PreparedMatmul>,
}

#[cfg(feature = "benchmark-control")]
impl MlxNativeMatmulControl {
    #[must_use]
    pub fn compilation_trace_count(&self) -> usize {
        self.native.traces()
    }
}

impl MlxArray {
    /// Checks terminal or host-copied opaque backing access without exposing a native pointer.
    ///
    /// # Errors
    /// Returns the owning session's completion/quarantine error.
    pub fn validate_access_available(&self) -> Result<(), MlxError> {
        self.session.validate_access_available()
    }

    /// Reads an exact Rust `f32` scalar destination through the canonical transactional read.
    /// Same-width integer storage and other scalar identities reject before publication.
    ///
    /// # Errors
    /// Returns unsupported scalar, short extent, quarantine or native materialization failure.
    pub fn read_into_typed<T: fusion_pcu::PcuScalar>(
        &self,
        output: &mut [T],
    ) -> Result<(), MlxError> {
        self.validate_access_available()?;
        self.read_into_f32(ffi::as_f32_mut(output)?)
    }

    #[must_use]
    pub fn shape(&self) -> [usize; 2] {
        self.array.native.shape()
    }
    #[must_use]
    pub fn residency(&self) -> MlxArrayResidency {
        self.array.residency.get()
    }
    #[must_use]
    pub const fn session(&self) -> &MlxSession {
        &self.session
    }

    /// Reads a complete initialized terminal matrix prefix, preserving any host tail.
    /// All fallible SDK materialization precedes publication; a failed call leaves output intact.
    /// Host access is an explicit materialization boundary for delegated MLX storage.
    ///
    /// # Errors
    /// Returns short destination, quarantined session or native read/materialization error.
    pub fn read_into_f32(&self, output: &mut [f32]) -> Result<(), MlxError> {
        let shape = self.shape();
        let count = shape[0]
            .checked_mul(shape[1])
            .ok_or(MlxError::InvalidExtent)?;
        if output.len() < count {
            return Err(MlxError::InvalidExtent);
        }
        self.array.native.read(&mut output[..count])?;
        self.array
            .residency
            .set(MlxArrayResidency::HostMaterialized);
        Ok(())
    }
}

/// Immutable session/shape/source/policy identity; no warm discovery or environment scan.
#[cfg(feature = "tensor")]
#[derive(Clone)]
pub struct MlxPreparedMatmul {
    session: MlxSession,
    plan: crate::MlxMatmulPlan,
    native: Rc<ffi::PreparedMatmul>,
}

/// Retained captured source program and its admitted single compiled matrix primitive.
/// This is an opaque delegated array plan, with no generic device-buffer interchange.
#[cfg(feature = "tensor")]
#[derive(Clone)]
pub struct MlxPreparedProgram {
    program: Arc<fusion_pcu::dialect::tensor::TensorOwnedSelectedProgram>,
    matmul: MlxPreparedMatmul,
}

#[cfg(feature = "tensor")]
impl MlxPreparedProgram {
    #[must_use]
    pub fn program(&self) -> &fusion_pcu::dialect::tensor::TensorOwnedSelectedProgram {
        &self.program
    }
    #[must_use]
    pub const fn matmul(&self) -> &MlxPreparedMatmul {
        &self.matmul
    }
}
#[cfg(feature = "tensor")]
impl MlxPreparedMatmul {
    /// Actual cold source-callback invocations observed while preparing the retained primitive.
    /// Preparation succeeds only for exactly one tracer invocation. Warm replay cannot invoke
    /// the source callback or compiler wrapper; one logical output descriptor remains per call.
    #[must_use]
    pub fn compilation_trace_count(&self) -> usize {
        self.native.traces()
    }
    /// Frozen delegated implementation identity for discovery-activated sessions.
    /// Manually opened advanced sessions have no invented discovery generation.
    #[must_use]
    pub fn implementation_id(&self) -> Option<fusion_pcu::PcuImplementationId> {
        self.session
            .identity()
            .map(|device| fusion_pcu::PcuImplementationId {
                device,
                executor: crate::discovery::EXECUTOR,
                local_id: 1,
                revision: crate::discovery::MATMUL_REVISION,
            })
    }
    #[must_use]
    pub const fn plan(&self) -> &crate::MlxMatmulPlan {
        &self.plan
    }
    #[must_use]
    pub const fn session(&self) -> &MlxSession {
        &self.session
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

#[cfg(feature = "tensor")]
#[path = "program/program.rs"]
mod program;
#[cfg(feature = "tensor")]
pub use program::MlxProgramInput;

#[path = "encoded/encoded.rs"]
mod encoded;
pub use encoded::{MlxEncodedArray, MlxEncodedCompletion, MlxPreparedEncodedPrefix};

#[cfg(feature = "tensor")]
#[path = "checked_program/checked_program.rs"]
mod checked_program;
#[cfg(feature = "tensor")]
pub use checked_program::{MlxPreparedCheckedProgram, MlxCheckedProgramInput};

#[cfg(feature = "tensor")]
#[path = "tensor_binary/tensor_binary.rs"]
mod tensor_binary;
#[cfg(feature = "tensor")]
pub use tensor_binary::MlxPreparedTensorBinaryProgram;

#[cfg(feature = "tensor")]
#[path = "tensor_integer/tensor_integer.rs"]
mod tensor_integer;
#[cfg(feature = "tensor")]
pub use tensor_integer::MlxPreparedTensorIntegerProgram;
