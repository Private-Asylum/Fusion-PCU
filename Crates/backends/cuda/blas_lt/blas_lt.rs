//! Cold, immutable real row-major cuBLASLt plans with terminal batch ownership.
#[rustfmt::skip]
use std::{
    any::Any,
    cell::Cell,
    ffi::c_void,
    mem::size_of,
    ptr,
    rc::Rc,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuPrecisionPolicy,
    PcuScalarType,
};
#[rustfmt::skip]
use crate::{
    CublasError,
    CublasNumericalConfig,
    CudaCompletionBatch,
    CudaRuntime,
    CudaStreamHandle,
    DeviceBuffer,
};
#[rustfmt::skip]
use crate::ffi::cublaslt::{
    self,
    Algorithm,
    Api,
    Handle,
    Heuristic,
};

const WORKSPACE_LIMIT: usize = 16 * 1024 * 1024;

/// Exact physical operand layouts and immutable numerical permission for one real `MatMul`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CublasLtMatmulDescriptor {
    config: CublasNumericalConfig,
    left: [usize; 2],
    right: [usize; 2],
    output: [usize; 2],
    transpose: [bool; 2],
    bytes: [usize; 3],
}
impl CublasLtMatmulDescriptor {
    /// Validate physical dense row-major matrices without loading CUDA.
    /// # Errors
    /// Rejects empty, incompatible, overflowing or unsupported numerical configurations.
    pub fn new(
        left: [usize; 2],
        right: [usize; 2],
        transpose: [bool; 2],
        config: CublasNumericalConfig,
    ) -> Result<Self, CublasError> {
        if config
            .environment()
            .value("CUBLAS_BATCH_INVARIANCE_FLAGS")
            .is_some()
        {
            return Err(CublasError::UnsupportedNumericalConfiguration(
                "CUBLAS_BATCH_INVARIANCE_FLAGS",
            ));
        }
        let logical_left = if transpose[0] {
            [left[1], left[0]]
        } else {
            left
        };
        let logical_right = if transpose[1] {
            [right[1], right[0]]
        } else {
            right
        };
        if left.contains(&0) || right.contains(&0) || logical_left[1] != logical_right[0] {
            return Err(CublasError::InvalidDimensions(
                "Lt requires nonempty rank-two compatible matrices",
            ));
        }
        let output = [logical_left[0], logical_right[1]];
        let width = match config.scalar_type() {
            PcuScalarType::F32 => 4,
            PcuScalarType::F64 => 8,
            _ => {
                return Err(CublasError::UnsupportedNumericalConfiguration(
                    "Lt requires F32 or F64",
                ));
            }
        };
        let mut bytes = [0; 3];
        for (index, shape) in [left, right, output].into_iter().enumerate() {
            for extent in shape {
                i64::try_from(extent).map_err(|_| CublasError::DimensionOverflow)?;
            }
            bytes[index] = shape[0]
                .checked_mul(shape[1])
                .and_then(|count| count.checked_mul(width))
                .ok_or(CublasError::DimensionOverflow)?;
        }
        Ok(Self {
            config,
            left,
            right,
            output,
            transpose,
            bytes,
        })
    }
    /// Immutable admitted precision and process snapshot.
    #[must_use]
    pub const fn numerical_config(&self) -> &CublasNumericalConfig {
        &self.config
    }
}

/// Cold selected algorithm identity. This records a within-Lt heuristic, not cross-library speed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CublasLtPlanIdentity {
    pub algorithm_id: i32,
    pub workspace_bytes: usize,
    pub compute_type: i32,
    pub library_version: [i32; 3],
    pub device_ordinal: i32,
    pub runtime_identity: usize,
    pub minimum_operand_alignment: usize,
}
struct Root {
    api: Api,
    runtime: CudaRuntime,
    stream: CudaStreamHandle,
    handle: Handle,
    operation: Handle,
    layouts: [Handle; 3],
    preference: Handle,
    workspace: Option<DeviceBuffer>,
    descriptor: CublasLtMatmulDescriptor,
    algorithm: Algorithm,
    identity: CublasLtPlanIdentity,
    active: Cell<usize>,
    marker: Rc<dyn Any>,
}
impl Drop for Root {
    fn drop(&mut self) {
        // A root is dropped only cold or after every retained operation proves terminal completion.
        // Unknown completion quarantines the operation owner on the established batch spine.
        let _ = self.runtime.cuda_set_device(self.runtime.0.ordinal);
        // SAFETY: the last Rc owner is released only after terminal proof; handles are unique.
        unsafe {
            if !self.preference.is_null() {
                let _ = self.api.preference_destroy(self.preference);
            }
            for layout in self.layouts {
                if !layout.is_null() {
                    let _ = self.api.layout_destroy(layout);
                }
            }
            if !self.operation.is_null() {
                let _ = self.api.desc_destroy(self.operation);
            }
            if !self.handle.is_null() {
                let _ = self.api.destroy(self.handle);
            }
        }
    }
}
/// An immutable retained Lt algorithm, descriptors, runtime, stream and selected workspace.
#[derive(Clone)]
pub struct CublasLtMatmulPlan {
    root: Rc<Root>,
}
struct Operation {
    root: Rc<Root>,
    single: [f32; 2],
    double: [f64; 2],
}
impl Drop for Operation {
    fn drop(&mut self) {
        self.root
            .active
            .set(self.root.active.get().saturating_sub(1));
    }
}
impl CublasLtMatmulPlan {
    /// Prepare descriptors and bounded heuristics exactly once on the selected device/stream.
    /// # Errors
    /// Returns cold SDK, alignment, heuristic or workspace errors. No execution fallback is implied.
    #[allow(clippy::too_many_lines)] // Keep cold partial ownership and heuristic admission together.
    pub fn prepare(
        runtime: &CudaRuntime,
        stream: &CudaStreamHandle,
        descriptor: CublasLtMatmulDescriptor,
    ) -> Result<Self, CublasError> {
        if !stream.belongs_to_runtime(runtime) {
            return Err(CublasError::DifferentRuntime);
        }
        runtime.cuda_set_device(runtime.0.ordinal)?;
        let api = Api::load()?;
        let wide = descriptor.config.scalar_type() == PcuScalarType::F64;
        let scalar = if wide {
            cublaslt::CUDA_R_64F
        } else {
            cublaslt::CUDA_R_32F
        };
        let compute = if wide {
            cublaslt::COMPUTE_64F
        } else if descriptor.config.precision() == PcuPrecisionPolicy::BackendOptimized {
            cublaslt::COMPUTE_32F_FAST_TF32
        } else {
            cublaslt::COMPUTE_32F
        };
        let mut root = Root {
            api,
            runtime: runtime.clone(),
            stream: stream.clone(),
            handle: ptr::null_mut(),
            operation: ptr::null_mut(),
            layouts: [ptr::null_mut(); 3],
            preference: ptr::null_mut(),
            workspace: None,
            descriptor,
            algorithm: Algorithm::default(),
            identity: CublasLtPlanIdentity {
                algorithm_id: -1,
                workspace_bytes: 0,
                compute_type: compute,
                library_version: [0; 3],
                device_ordinal: runtime.0.ordinal,
                runtime_identity: std::sync::Arc::as_ptr(&runtime.0).addr(),
                minimum_operand_alignment: if wide { 8 } else { 4 },
            },
            active: Cell::new(0),
            marker: Rc::new(()),
        };
        // SAFETY: handles start null, all output storage is typed and Root owns every successful creation.
        unsafe {
            root.api.create(&raw mut root.handle)?;
            for (property, version) in root.identity.library_version.iter_mut().enumerate() {
                root.api.property(
                    i32::try_from(property).map_err(|_| CublasError::DimensionOverflow)?,
                    version,
                )?;
            }
            if root.identity.library_version[0] != 13
                || !matches!(root.identity.library_version[1], 7 | 8)
            {
                return Err(CublasError::UnsupportedLibraryVersion {
                    major: root.identity.library_version[0],
                    minor: root.identity.library_version[1],
                    patch: root.identity.library_version[2],
                });
            }
            root.api
                .desc_create(&raw mut root.operation, compute, scalar)?;
            for (attribute, value) in [
                (cublaslt::DESC_POINTER_MODE, cublaslt::POINTER_MODE_HOST),
                (
                    cublaslt::DESC_TRANSA,
                    i32::from(root.descriptor.transpose[0]),
                ),
                (
                    cublaslt::DESC_TRANSB,
                    i32::from(root.descriptor.transpose[1]),
                ),
            ] {
                root.api.desc_set(
                    root.operation,
                    attribute,
                    ptr::from_ref(&value).cast(),
                    size_of::<i32>(),
                )?;
            }
            for (slot, shape) in root.layouts.iter_mut().zip([
                root.descriptor.left,
                root.descriptor.right,
                root.descriptor.output,
            ]) {
                root.api.layout_create(
                    slot,
                    scalar,
                    u64::try_from(shape[0]).map_err(|_| CublasError::DimensionOverflow)?,
                    u64::try_from(shape[1]).map_err(|_| CublasError::DimensionOverflow)?,
                    i64::try_from(shape[1]).map_err(|_| CublasError::DimensionOverflow)?,
                )?;
                let order = cublaslt::ORDER_ROW;
                root.api.layout_set(
                    *slot,
                    cublaslt::LAYOUT_ORDER,
                    ptr::from_ref(&order).cast(),
                    size_of::<i32>(),
                )?;
            }
            root.api.preference_create(&raw mut root.preference)?;
            root.api.preference_set(
                root.preference,
                cublaslt::PREF_MAX_WORKSPACE,
                ptr::from_ref(&WORKSPACE_LIMIT).cast(),
                size_of::<usize>(),
            )?;
            let alignment: u32 = if wide { 8 } else { 4 };
            for attribute in cublaslt::PREF_ALIGNMENTS {
                root.api.preference_set(
                    root.preference,
                    attribute,
                    ptr::from_ref(&alignment).cast(),
                    size_of::<u32>(),
                )?;
            }
            let mut choices = [Heuristic::default(); 4];
            let mut returned = 0_i32;
            root.api.heuristic(
                root.handle,
                root.operation,
                root.layouts[0],
                root.layouts[1],
                root.layouts[2],
                root.layouts[2],
                root.preference,
                4,
                choices.as_mut_ptr(),
                &raw mut returned,
            )?;
            let returned = usize::try_from(returned)
                .map_err(|_| CublasError::InvalidDimensions("invalid Lt heuristic count"))?;
            if returned > choices.len() {
                return Err(CublasError::InvalidDimensions("invalid Lt heuristic count"));
            }
            let choice = choices[..returned]
                .iter()
                .find(|choice| choice.status == 0 && choice.workspace <= WORKSPACE_LIMIT)
                .ok_or(CublasError::UnsupportedNumericalConfiguration(
                    "no successful bounded Lt heuristic",
                ))?;
            root.algorithm = choice.algorithm;
            root.identity.workspace_bytes = choice.workspace;
            let mut written = 0;
            root.api.config(
                &raw const root.algorithm,
                cublaslt::CONFIG_ID,
                ptr::from_mut(&mut root.identity.algorithm_id).cast(),
                size_of::<i32>(),
                &raw mut written,
            )?;
            if written != size_of::<i32>() || root.identity.algorithm_id < 0 {
                return Err(CublasError::InvalidDimensions(
                    "invalid Lt algorithm identity",
                ));
            }
            for attribute in cublaslt::CAP_ALIGNMENTS {
                let mut required = 0_u32;
                root.api.capability(
                    &raw const root.algorithm,
                    attribute,
                    ptr::from_mut(&mut required).cast(),
                    size_of::<u32>(),
                    &raw mut written,
                )?;
                if written != size_of::<u32>()
                    || required == 0
                    || !required.is_power_of_two()
                    || required > alignment
                {
                    return Err(CublasError::InvalidDimensions(
                        "Lt algorithm exceeds admitted scalar alignment",
                    ));
                }
            }
        }
        if root.identity.workspace_bytes != 0 {
            let workspace = runtime.allocate(root.identity.workspace_bytes)?;
            if workspace.allocation.pointer.addr() % 256 != 0 {
                return Err(CublasError::InvalidDimensions(
                    "Lt workspace requires 256-byte alignment",
                ));
            }
            root.workspace = Some(workspace);
        }
        Ok(Self {
            root: Rc::new(root),
        })
    }
    /// Exact selected algorithm and workspace diagnostics.
    #[must_use]
    pub fn identity(&self) -> CublasLtPlanIdentity {
        self.root.identity
    }
    /// Whether all previous operations have terminal completion proof.
    #[must_use]
    pub fn is_usable(&self) -> bool {
        self.root.active.get() == 0
    }
    /// Validate an execution assessor's exact retained runtime and stream before allocations.
    /// # Errors
    /// Rejects a runtime or stream different from the cold preparation identity.
    pub fn validate_for_stream(&self, stream: &CudaStreamHandle) -> Result<(), CublasError> {
        if !stream.belongs_to_runtime(&self.root.runtime) {
            return Err(CublasError::DifferentRuntime);
        }
        if !Rc::ptr_eq(&self.root.stream.inner, &stream.inner) {
            return Err(CublasError::DifferentStream);
        }
        Ok(())
    }
    /// Queue immutable alpha=1/beta=0 `MatMul`, retaining all leases and SDK owners before enqueue.
    /// # Errors
    /// Rejects mismatched streams, extents, runtime, aliases or concurrent unrelated batches.
    pub fn submit_into_batch(
        &self,
        batch: &mut CudaCompletionBatch,
        left: &DeviceBuffer,
        right: &DeviceBuffer,
        output: &DeviceBuffer,
    ) -> Result<(), CublasError> {
        let root = &self.root;
        if !Rc::ptr_eq(&root.stream.inner, &batch.stream_handle().inner) {
            return Err(CublasError::DifferentStream);
        }
        if root.active.get() != 0 && !batch.has_queue_marker(&root.marker) {
            return Err(CublasError::Busy);
        }
        for ((buffer, required), name) in [left, right, output]
            .into_iter()
            .zip(root.descriptor.bytes)
            .zip(["A", "B", "D"])
        {
            root.runtime
                .ensure_same_runtime(&buffer.allocation.runtime)
                .map_err(|_| CublasError::DifferentRuntime)?;
            if buffer.len() < required {
                return Err(CublasError::BufferTooSmall {
                    matrix: name,
                    allocation: buffer.len(),
                    required,
                });
            }
            let alignment = if root.descriptor.config.scalar_type() == PcuScalarType::F64 {
                8
            } else {
                4
            };
            if buffer.allocation.pointer.addr() % alignment != 0 {
                return Err(CublasError::InvalidDimensions("Lt operand alignment"));
            }
        }
        if Rc::ptr_eq(&left.allocation, &output.allocation)
            || Rc::ptr_eq(&right.allocation, &output.allocation)
        {
            return Err(CublasError::AliasedBuffers);
        }
        root.runtime.cuda_set_device(root.runtime.0.ordinal)?;
        let next = root.active.get().checked_add(1).ok_or(CublasError::Busy)?;
        root.active.set(next);
        let operation = Rc::new(Operation {
            root: Rc::clone(root),
            single: [1.0, 0.0],
            double: [1.0, 0.0],
        });
        let mut buffers: smallvec::SmallVec<[&DeviceBuffer; 4]> =
            smallvec::smallvec![left, right, output];
        if let Some(workspace) = root.workspace.as_ref() {
            buffers.push(workspace);
        }
        batch.retain_external_operation(&buffers, operation.clone())?;
        batch.register_queue_marker(Rc::clone(&root.marker))?;
        let scalars: [*const c_void; 2] =
            if root.descriptor.config.scalar_type() == PcuScalarType::F64 {
                [
                    ptr::from_ref(&operation.double[0]).cast(),
                    ptr::from_ref(&operation.double[1]).cast(),
                ]
            } else {
                [
                    ptr::from_ref(&operation.single[0]).cast(),
                    ptr::from_ref(&operation.single[1]).cast(),
                ]
            };
        // SAFETY: all descriptors, scalars and allocation leases are retained before this call,
        // including an error after a partial enqueue; the exact algorithm and stream are cold-bound.
        unsafe {
            root.api.matmul(
                root.handle,
                root.operation,
                scalars[0],
                left.allocation.pointer,
                root.layouts[0],
                right.allocation.pointer,
                root.layouts[1],
                scalars[1],
                output.allocation.pointer,
                root.layouts[2],
                output.allocation.pointer,
                root.layouts[2],
                &raw const root.algorithm,
                root.workspace
                    .as_ref()
                    .map_or(ptr::null_mut(), |workspace| workspace.allocation.pointer),
                root.identity.workspace_bytes,
                root.stream.inner.raw,
            )
        }
    }
    /// Execute through a private terminal batch on this plan's retained stream.
    /// # Errors
    /// Returns validation, submission or terminal completion errors, retaining uncertain owners.
    pub fn execute(
        &self,
        left: &DeviceBuffer,
        right: &DeviceBuffer,
        output: &DeviceBuffer,
    ) -> Result<(), CublasError> {
        let mut batch = CudaCompletionBatch::new(&self.root.stream);
        self.submit_into_batch(&mut batch, left, right, output)?;
        batch.finish()?.wait()?;
        Ok(())
    }
}

#[path = "tests.rs"]
#[cfg(test)]
mod tests;
