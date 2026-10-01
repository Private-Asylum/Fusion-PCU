//! Explicit delegated runtime owners; no implicit device or numerical policy selection.

#[rustfmt::skip]
use std::{
    cell::Cell,
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
    /// The explicitly selected bridge could not be loaded; no CPU tensor work is attempted.
    Unavailable(String),
    /// MLX execution is supported only on Apple silicon macOS.
    UnsupportedPlatform,
    /// The private bridge or pinned SDK identity does not satisfy the adapter contract.
    Abi(String),
    InvalidExtent,
    ForeignSession,
    /// Delegated storage requires the exact Rust F32 identity, not only an equal byte width.
    UnsupportedScalar(fusion_pcu::PcuScalarType),
    InvalidRequest(String),
    Runtime(String),
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

/// A loaded private bridge retained by every session, array and prepared primitive.
///
/// There is no build/install side effect. Native validation builds `ffi/CMakeLists.txt` against
/// MLX0.32.3; the caller passes that trusted bridge path explicitly during cold initialization.
#[derive(Clone)]
pub struct MlxRuntime(Rc<ffi::Api>);
impl MlxRuntime {
    /// Loads the trusted shipped bridge and checks private ABI/header/runtime compatibility.
    ///
    /// # Errors
    /// Returns unsupported platform, unavailable library, missing symbol, ABI or SDK mismatch.
    pub fn load(bridge: impl AsRef<Path>) -> Result<Self, MlxError> {
        ffi::Api::load(bridge.as_ref()).map(Self)
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
    /// A GPU `MatMul` produced this terminal array; its backing remains MLX-owned.
    GpuEvaluated,
    /// A host read materialized CPU-visible data from the MLX-owned backing.
    HostMaterialized,
}

struct Array {
    native: ffi::Array,
    residency: Cell<MlxArrayResidency>,
}

/// Immutable owned initialized F32 matrix. No mutable/no-copy or native Metal buffer import.
/// Clones share the same SDK holder and backing; they do not duplicate physical allocations.
#[derive(Clone)]
pub struct MlxArray {
    session: MlxSession,
    array: Rc<Array>,
}

impl MlxSession {
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
            }),
        })
    }

    /// Measures the native C++ MatMul frontend with the ordinary terminal boundary.
    /// This diagnostic control constructs a fresh primitive on every call.
    ///
    /// # Errors
    /// Returns shape/session guards or native execution/quarantine errors.
    #[cfg(feature = "c-api-evaluation")]
    #[doc(hidden)]
    pub fn execute_native_direct_matmul_control(
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
