//! Private static provider adapters for resident storage and typed graph execution.
//! Every match arm calls a safe provider interface; incompatible domains fail before submission.

#[rustfmt::skip]
use crate::{
    PcuBindingRef,
    PcuDeviceArgument,
    PcuDeviceTensor,
    PcuMemoryResource,
    PcuScalar,
};
#[cfg(any(
    feature = "tensor",
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "cpu",
    feature = "mlx"
))]
use crate::PcuOwnedDispatchBackend;
#[cfg(all(feature = "tensor", any(feature = "rocm", feature = "cuda")))]
#[rustfmt::skip]
use crate::{
    PcuMemoryPoolId,
    PcuMemoryProvider,
};
use super::PcuExecutionError;
use super::PcuArgumentError;
use std::rc::Rc;
#[cfg(all(feature = "cuda", feature = "tensor"))]
use std::cell::OnceCell;

#[cfg_attr(not(feature = "tensor"), allow(dead_code))] // Only successful graph outputs construct storage.
pub(super) enum DeviceTensor<T: PcuScalar> {
    #[cfg(feature = "metal")]
    Metal(PcuDeviceTensor<T, fusion_pcu_metal::MetalMemoryResource>),
    #[cfg(feature = "rocm")]
    Rocm(PcuDeviceTensor<T, fusion_pcu_rocm::RocmMemoryResource>),
    #[cfg(feature = "cuda")]
    Cuda(PcuDeviceTensor<T, fusion_pcu_cuda::CudaMemoryResource>),
}
pub(super) enum DeviceArgument<'a> {
    #[cfg(feature = "metal")]
    Metal(PcuDeviceArgument<'a, fusion_pcu_metal::MetalMemoryResource>),
    #[cfg(feature = "rocm")]
    Rocm(PcuDeviceArgument<'a, fusion_pcu_rocm::RocmMemoryResource>),
    #[cfg(feature = "cuda")]
    Cuda(PcuDeviceArgument<'a, fusion_pcu_cuda::CudaMemoryResource>),
}
#[cfg_attr(not(feature = "tensor"), allow(dead_code))] // Only resident graphs construct execution roots.
pub(super) enum Session {
    #[cfg(feature = "metal")]
    Metal(Rc<MetalSession>),
    #[cfg(feature = "rocm")]
    Rocm(Rc<super::session::RocmSession>),
    #[cfg(feature = "cuda")]
    Cuda(Rc<CudaSession>),
}
#[cfg(feature = "metal")]
pub(super) struct MetalSession {
    backend: Rc<fusion_pcu_metal::MetalOwnedDispatchBackend>,
    block_size: u32,
}
#[cfg(feature = "cuda")]
pub(super) struct CudaSession {
    backend: Rc<fusion_pcu_cuda::CudaOwnedDispatchBackend>,
    block_size: u32,
    #[cfg(all(feature = "tensor", any(feature = "rocm", feature = "cuda")))]
    tensor: OnceCell<fusion_pcu_cuda::CudaOwnedTensorAssessor>,
}
#[cfg(all(feature = "tensor", any(feature = "rocm", feature = "cuda")))]
const fn mismatch() -> PcuExecutionError {
    PcuExecutionError::Argument(PcuArgumentError::SessionMismatch)
}
impl<T: PcuScalar> DeviceTensor<T> {
    pub(super) fn shape(&self) -> &[usize] {
        match self {
            #[cfg(feature = "rocm")]
            Self::Rocm(tensor) => tensor.shape(),
            #[cfg(feature = "cuda")]
            Self::Cuda(tensor) => tensor.shape(),
            #[cfg(feature = "metal")]
            Self::Metal(tensor) => tensor.shape(),
        }
    }
    pub(super) const fn len(&self) -> usize {
        match self {
            #[cfg(feature = "rocm")]
            Self::Rocm(tensor) => tensor.buffer().len(),
            #[cfg(feature = "cuda")]
            Self::Cuda(tensor) => tensor.buffer().len(),
            #[cfg(feature = "metal")]
            Self::Metal(tensor) => tensor.buffer().len(),
        }
    }
    pub(super) fn validate_access_available(&self) -> Result<(), PcuArgumentError> {
        match self {
            #[cfg(feature = "metal")]
            Self::Metal(tensor) => tensor
                .buffer()
                .resource()
                .validate_access_available()
                .map_err(|_| PcuArgumentError::ResidentCompletionUncertain),
            #[cfg(feature = "rocm")]
            Self::Rocm(tensor) => tensor
                .buffer()
                .resource()
                .validate_access_available()
                .map_err(|_| PcuArgumentError::ResidentCompletionUncertain),
            #[cfg(feature = "cuda")]
            Self::Cuda(tensor) => tensor
                .buffer()
                .resource()
                .validate_access_available()
                .map_err(|_| PcuArgumentError::ResidentCompletionUncertain),
        }
    }
    pub(super) const fn read_argument(&self, target: PcuBindingRef) -> DeviceArgument<'_> {
        match self {
            #[cfg(feature = "rocm")]
            Self::Rocm(tensor) => DeviceArgument::Rocm(tensor.read_argument(target)),
            #[cfg(feature = "cuda")]
            Self::Cuda(tensor) => DeviceArgument::Cuda(tensor.read_argument(target)),
            #[cfg(feature = "metal")]
            Self::Metal(tensor) => DeviceArgument::Metal(tensor.read_argument(target)),
        }
    }
    pub(super) const fn read_write_argument(
        &mut self,
        target: PcuBindingRef,
    ) -> DeviceArgument<'_> {
        match self {
            #[cfg(feature = "rocm")]
            Self::Rocm(tensor) => DeviceArgument::Rocm(tensor.read_write_argument(target)),
            #[cfg(feature = "cuda")]
            Self::Cuda(tensor) => DeviceArgument::Cuda(tensor.read_write_argument(target)),
            #[cfg(feature = "metal")]
            Self::Metal(tensor) => DeviceArgument::Metal(tensor.read_write_argument(target)),
        }
    }
}
impl Session {
    #[cfg(all(feature = "metal", feature = "tensor"))]
    #[cfg_attr(
        not(any(feature = "rocm", feature = "cuda")),
        allow(clippy::unnecessary_wraps)
    )] // Mixed provider builds reject non-Metal sessions through the same static accessor.
    pub(in crate::global) fn metal_backend(
        &self,
    ) -> Option<&fusion_pcu_metal::MetalOwnedDispatchBackend> {
        match self {
            Self::Metal(session) => Some(&session.backend),
            #[cfg(any(feature = "rocm", feature = "cuda"))]
            _ => None,
        }
    }

    #[cfg(all(feature = "metal", feature = "tensor"))]
    pub(in crate::global) fn from_metal_backend(
        backend: fusion_pcu_metal::MetalOwnedDispatchBackend,
        block_size: u32,
    ) -> Self {
        Self::Metal(Rc::new(MetalSession {
            backend: Rc::new(backend),
            block_size,
        }))
    }

    #[cfg(feature = "metal")]
    pub(super) fn shares_metal_session(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Metal(left), Self::Metal(right)) => left.backend.shares_session(&right.backend),
            #[cfg(any(feature = "rocm", feature = "cuda"))]
            _ => false,
        }
    }

    #[cfg(any(
        feature = "tensor",
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "cpu",
        feature = "mlx"
    ))]
    pub(super) fn device_id(&self) -> u32 {
        match self {
            #[cfg(feature = "rocm")]
            Self::Rocm(session) => session.backend().device_identity().device_id(),
            #[cfg(feature = "cuda")]
            Self::Cuda(session) => session.backend.device_identity().device_id(),
            #[cfg(feature = "metal")]
            Self::Metal(session) => session.backend.device_identity().device_id(),
        }
    }
    #[cfg(any(
        feature = "tensor",
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "cpu",
        feature = "mlx"
    ))]
    pub(super) fn block_size(&self) -> u32 {
        match self {
            #[cfg(feature = "rocm")]
            Self::Rocm(session) => session.block_size(),
            #[cfg(feature = "cuda")]
            Self::Cuda(session) => session.block_size,
            #[cfg(feature = "metal")]
            Self::Metal(session) => session.block_size,
        }
    }
    pub(super) fn download<T: PcuScalar>(
        &self,
        tensor: &DeviceTensor<T>,
        destination: &mut [T],
    ) -> Result<(), PcuExecutionError> {
        match (self, tensor) {
            #[cfg(feature = "metal")]
            (Self::Metal(session), DeviceTensor::Metal(tensor)) => session
                .backend
                .download_buffer(
                    tensor.buffer().resource().pool(),
                    tensor.buffer(),
                    destination,
                )
                .map_err(PcuExecutionError::from),
            #[cfg(feature = "rocm")]
            (Self::Rocm(session), DeviceTensor::Rocm(tensor)) => session
                .backend()
                .download_buffer(
                    tensor.buffer().resource().pool(),
                    tensor.buffer(),
                    destination,
                )
                .map_err(PcuExecutionError::from),
            #[cfg(feature = "cuda")]
            (Self::Cuda(session), DeviceTensor::Cuda(tensor)) => session
                .backend
                .download_buffer(
                    tensor.buffer().resource().pool(),
                    tensor.buffer(),
                    destination,
                )
                .map_err(PcuExecutionError::from),
            #[cfg(any(
                all(feature = "rocm", feature = "cuda"),
                all(feature = "metal", any(feature = "rocm", feature = "cuda"))
            ))]
            _ => Err(PcuExecutionError::Argument(
                super::PcuArgumentError::SessionMismatch,
            )),
        }
    }
}
#[cfg(feature = "cuda")]
impl From<fusion_pcu_cuda::CudaDeviceKernelError> for PcuExecutionError {
    fn from(error: fusion_pcu_cuda::CudaDeviceKernelError) -> Self {
        match error {
            fusion_pcu_cuda::CudaDeviceKernelError::CheckedExecutionFault(fault) => {
                Self::ArithmeticFault(fault)
            }
            other => Self::BackendFailure(format!("CUDA device execution: {other}")),
        }
    }
}
#[cfg(all(feature = "cuda", feature = "tensor"))]
impl From<fusion_pcu_cuda::CudaTensorExecutionError> for PcuExecutionError {
    fn from(error: fusion_pcu_cuda::CudaTensorExecutionError) -> Self {
        match error {
            fusion_pcu_cuda::CudaTensorExecutionError::ExecutionFault(fault) => {
                Self::ArithmeticFault(fault)
            }
            other => Self::CudaTensorExecution(other),
        }
    }
}
#[cfg(all(feature = "cuda", not(feature = "rocm")))]
impl From<crate::PcuDeviceTensorError> for PcuExecutionError {
    fn from(error: crate::PcuDeviceTensorError) -> Self {
        Self::TensorStorage(error)
    }
}

#[cfg(all(feature = "tensor", any(feature = "rocm", feature = "cuda")))]
mod graph {
    #[rustfmt::skip]
    use super::{
        DeviceTensor,
        PcuExecutionError,
        PcuMemoryPoolId,
        PcuMemoryProvider,
        PcuScalar,
        Session,
        mismatch,
    };
    #[cfg(feature = "cuda")]
    use std::rc::Rc;
    #[rustfmt::skip]
    use crate::dialect::tensor::{
        TensorOwnedSelectedProgram,
        ValueId,
    };
    use std::sync::Arc;
    use smallvec::SmallVec;
    pub(in crate::global) enum Prepared {
        #[cfg(feature = "rocm")]
        Rocm(fusion_pcu_rocm::RocmOwnedPreparedTensorGraph),
        #[cfg(feature = "cuda")]
        Cuda(fusion_pcu_cuda::CudaOwnedPreparedTensorGraph),
    }
    pub(in crate::global) enum Memory {
        #[cfg(feature = "rocm")]
        Rocm(fusion_pcu_rocm::RocmMemoryProvider),
        #[cfg(feature = "cuda")]
        Cuda(fusion_pcu_cuda::CudaMemoryProvider),
    }
    pub(in crate::global) enum Resource {
        #[cfg(feature = "rocm")]
        Rocm(fusion_pcu_rocm::RocmMemoryResource),
        #[cfg(feature = "cuda")]
        Cuda(fusion_pcu_cuda::CudaMemoryResource),
    }
    pub(in crate::global) enum Assessor<'a> {
        #[cfg(feature = "rocm")]
        Rocm(fusion_pcu_rocm::RocmTensorAssessor<'a>),
        #[cfg(feature = "cuda")]
        Cuda(fusion_pcu_cuda::CudaTensorAssessor<'a>),
    }
    pub(in crate::global) enum InputRef<'a> {
        #[cfg(feature = "rocm")]
        Rocm(fusion_pcu_rocm::RocmTensorInputRef<'a>),
        #[cfg(feature = "cuda")]
        Cuda(fusion_pcu_cuda::CudaTensorInputRef<'a>),
    }
    impl Memory {
        pub(in crate::global) fn transfer_to(
            &mut self,
            resource: &mut Resource,
            offset: u64,
            bytes: &[u8],
        ) -> Result<(), crate::PcuMemoryProviderError> {
            match (self, resource) {
                #[cfg(feature = "rocm")]
                (Self::Rocm(memory), Resource::Rocm(resource)) => {
                    memory.transfer_to(resource, offset, bytes)
                }
                #[cfg(feature = "cuda")]
                (Self::Cuda(memory), Resource::Cuda(resource)) => {
                    memory.transfer_to(resource, offset, bytes)
                }
                #[cfg(all(feature = "rocm", feature = "cuda"))]
                _ => unreachable!("staging resources are allocated by their entry memory provider"),
            }
        }
    }
    impl Session {
        pub(in crate::global) fn memory_provider(&self, pool: PcuMemoryPoolId) -> Memory {
            match self {
                #[cfg(feature = "metal")]
                Self::Metal(_) => {
                    unreachable!("Metal tensor assessor rejects before obtaining graph memory")
                }
                #[cfg(feature = "rocm")]
                Self::Rocm(session) => Memory::Rocm(session.backend().memory_provider(pool)),
                #[cfg(feature = "cuda")]
                Self::Cuda(session) => Memory::Cuda(session.backend.memory_provider(pool)),
            }
        }
        pub(in crate::global) fn upload_resource<T: PcuScalar>(
            &self,
            pool: PcuMemoryPoolId,
            values: &[T],
        ) -> Result<Resource, PcuExecutionError> {
            match self {
                #[cfg(feature = "metal")]
                Self::Metal(_) => Err(PcuExecutionError::TensorExecutionUnavailable),
                #[cfg(feature = "rocm")]
                Self::Rocm(session) => session
                    .backend()
                    .upload_buffer(pool, values)
                    .map(|buffer| Resource::Rocm(buffer.into_resource()))
                    .map_err(PcuExecutionError::from),
                #[cfg(feature = "cuda")]
                Self::Cuda(session) => session
                    .backend
                    .upload_buffer(pool, values)
                    .map(|buffer| Resource::Cuda(buffer.into_resource()))
                    .map_err(PcuExecutionError::from),
            }
        }
        pub(in crate::global) fn tensor_assessor(&self) -> Result<Assessor<'_>, PcuExecutionError> {
            match self {
                #[cfg(feature = "metal")]
                Self::Metal(_) => Err(PcuExecutionError::TensorExecutionUnavailable),
                #[cfg(feature = "rocm")]
                Self::Rocm(session) => session
                    .tensor_assessor()
                    .map(Assessor::Rocm)
                    .map_err(PcuExecutionError::TensorInitialization),
                #[cfg(feature = "cuda")]
                Self::Cuda(session) => {
                    if session.tensor.get().is_none() {
                        let state = fusion_pcu_cuda::CudaOwnedTensorAssessor::new(Rc::clone(
                            &session.backend,
                        ))
                        .map_err(|error| {
                            PcuExecutionError::BackendFailure(format!(
                                "CUDA tensor initialization: {error}"
                            ))
                        })?;
                        if session.tensor.set(state).is_err() {
                            unreachable!("thread-owned tensor state initialized concurrently");
                        }
                    }
                    Ok(Assessor::Cuda(
                        session
                            .tensor
                            .get()
                            .expect("tensor state initialized")
                            .assessor(),
                    ))
                }
            }
        }
    }
    impl Assessor<'_> {
        pub(in crate::global) fn prepare_shared_owned_program(
            &self,
            program: Arc<TensorOwnedSelectedProgram>,
        ) -> Result<Prepared, PcuExecutionError> {
            match self {
                #[cfg(feature = "rocm")]
                Self::Rocm(assessor) => assessor
                    .prepare_shared_owned_program(program)
                    .map(Prepared::Rocm)
                    .map_err(PcuExecutionError::from),
                #[cfg(feature = "cuda")]
                Self::Cuda(assessor) => assessor
                    .prepare_shared_owned_program(program)
                    .map(Prepared::Cuda)
                    .map_err(PcuExecutionError::from),
            }
        }
        pub(in crate::global) fn borrow_resource_input_ref<'a>(
            &'a self,
            resource: &'a Resource,
            dimensions: &'a [usize],
            scalar_type: crate::PcuScalarType,
            pool: PcuMemoryPoolId,
        ) -> Result<InputRef<'a>, PcuExecutionError> {
            match (self, resource) {
                #[cfg(feature = "rocm")]
                (Self::Rocm(assessor), Resource::Rocm(resource)) => assessor
                    .borrow_resource_input_ref(resource, dimensions, scalar_type, pool)
                    .map(InputRef::Rocm)
                    .map_err(PcuExecutionError::from),
                #[cfg(feature = "cuda")]
                (Self::Cuda(assessor), Resource::Cuda(resource)) => assessor
                    .borrow_resource_input_ref(resource, dimensions, scalar_type, pool)
                    .map(InputRef::Cuda)
                    .map_err(PcuExecutionError::from),
                #[cfg(all(feature = "rocm", feature = "cuda"))]
                _ => Err(mismatch()),
            }
        }
        pub(in crate::global) fn borrow_device_input_ref<'a, T: PcuScalar>(
            &'a self,
            tensor: &'a DeviceTensor<T>,
            pool: PcuMemoryPoolId,
        ) -> Result<InputRef<'a>, PcuExecutionError> {
            match (self, tensor) {
                #[cfg(feature = "rocm")]
                (Self::Rocm(assessor), DeviceTensor::Rocm(tensor)) => assessor
                    .borrow_device_input_ref(tensor, pool)
                    .map(InputRef::Rocm)
                    .map_err(PcuExecutionError::from),
                #[cfg(feature = "cuda")]
                (Self::Cuda(assessor), DeviceTensor::Cuda(tensor)) => assessor
                    .borrow_device_input_ref(tensor, pool)
                    .map(InputRef::Cuda)
                    .map_err(PcuExecutionError::from),
                #[cfg(any(all(feature = "rocm", feature = "cuda"), feature = "metal"))]
                _ => Err(mismatch()),
            }
        }
        pub(in crate::global) fn execute_owned_program_output_from_inputs<T: PcuScalar>(
            &self,
            prepared: &Prepared,
            inputs: &[(ValueId, &InputRef<'_>)],
            pool: PcuMemoryPoolId,
            memory: &mut Memory,
        ) -> Result<DeviceTensor<T>, PcuExecutionError> {
            match (self, prepared, memory) {
                #[cfg(feature = "rocm")]
                (Self::Rocm(assessor), Prepared::Rocm(prepared), Memory::Rocm(memory)) => {
                    let mut bindings: SmallVec<
                        [(ValueId, &fusion_pcu_rocm::RocmTensorInputRef<'_>); 8],
                    > = SmallVec::new();
                    for &(id, input) in inputs {
                        #[allow(irrefutable_let_patterns)]
                        let InputRef::Rocm(input) = input else {
                            return Err(mismatch());
                        };
                        bindings.push((id, input));
                    }
                    assessor
                        .execute_owned_program_output_from_inputs(prepared, &bindings, pool, memory)
                        .map(DeviceTensor::Rocm)
                        .map_err(PcuExecutionError::from)
                }
                #[cfg(feature = "cuda")]
                (Self::Cuda(assessor), Prepared::Cuda(prepared), Memory::Cuda(memory)) => {
                    let mut bindings: SmallVec<
                        [(ValueId, &fusion_pcu_cuda::CudaTensorInputRef<'_>); 8],
                    > = SmallVec::new();
                    for &(id, input) in inputs {
                        #[allow(irrefutable_let_patterns)]
                        let InputRef::Cuda(input) = input else {
                            return Err(mismatch());
                        };
                        bindings.push((id, input));
                    }
                    assessor
                        .execute_owned_program_output_from_inputs(prepared, &bindings, pool, memory)
                        .map(DeviceTensor::Cuda)
                        .map_err(PcuExecutionError::from)
                }
                #[cfg(all(feature = "rocm", feature = "cuda"))]
                _ => Err(mismatch()),
            }
        }
        pub(in crate::global) fn execute_owned_program_consuming_input<T: PcuScalar>(
            &self,
            prepared: &Prepared,
            tensor: DeviceTensor<T>,
            pool: PcuMemoryPoolId,
            memory: &mut Memory,
        ) -> Result<DeviceTensor<T>, PcuExecutionError> {
            match (self, prepared, tensor, memory) {
                #[cfg(feature = "rocm")]
                (
                    Self::Rocm(assessor),
                    Prepared::Rocm(prepared),
                    DeviceTensor::Rocm(tensor),
                    Memory::Rocm(memory),
                ) => assessor
                    .execute_owned_program_consuming_input(prepared, tensor, pool, memory)
                    .map(DeviceTensor::Rocm)
                    .map_err(PcuExecutionError::from),
                #[cfg(feature = "cuda")]
                (
                    Self::Cuda(assessor),
                    Prepared::Cuda(prepared),
                    DeviceTensor::Cuda(tensor),
                    Memory::Cuda(memory),
                ) => assessor
                    .execute_owned_program_consuming_input(prepared, tensor, pool, memory)
                    .map(DeviceTensor::Cuda)
                    .map_err(PcuExecutionError::from),
                #[cfg(any(all(feature = "rocm", feature = "cuda"), feature = "metal"))]
                _ => Err(mismatch()),
            }
        }
        pub(in crate::global) fn execute_owned_program_consuming_binary_input<T: PcuScalar>(
            &self,
            prepared: &Prepared,
            donor_value: ValueId,
            tensor: DeviceTensor<T>,
            inputs: &[(ValueId, &InputRef<'_>)],
            pool: PcuMemoryPoolId,
            memory: &mut Memory,
        ) -> Result<DeviceTensor<T>, PcuExecutionError> {
            match (self, prepared, tensor, memory) {
                #[cfg(feature = "rocm")]
                (
                    Self::Rocm(assessor),
                    Prepared::Rocm(prepared),
                    DeviceTensor::Rocm(tensor),
                    Memory::Rocm(memory),
                ) => {
                    let mut bindings: SmallVec<
                        [(ValueId, &fusion_pcu_rocm::RocmTensorInputRef<'_>); 8],
                    > = SmallVec::new();
                    for &(id, input) in inputs {
                        #[allow(irrefutable_let_patterns)]
                        let InputRef::Rocm(input) = input else {
                            return Err(mismatch());
                        };
                        bindings.push((id, input));
                    }
                    assessor
                        .execute_owned_program_consuming_binary_input(
                            prepared,
                            donor_value,
                            tensor,
                            &bindings,
                            pool,
                            memory,
                        )
                        .map(DeviceTensor::Rocm)
                        .map_err(PcuExecutionError::from)
                }
                #[cfg(feature = "cuda")]
                (
                    Self::Cuda(assessor),
                    Prepared::Cuda(prepared),
                    DeviceTensor::Cuda(tensor),
                    Memory::Cuda(memory),
                ) => {
                    let mut bindings: SmallVec<
                        [(ValueId, &fusion_pcu_cuda::CudaTensorInputRef<'_>); 8],
                    > = SmallVec::new();
                    for &(id, input) in inputs {
                        #[allow(irrefutable_let_patterns)]
                        let InputRef::Cuda(input) = input else {
                            return Err(mismatch());
                        };
                        bindings.push((id, input));
                    }
                    assessor
                        .execute_owned_program_consuming_binary_input(
                            prepared,
                            donor_value,
                            tensor,
                            &bindings,
                            pool,
                            memory,
                        )
                        .map(DeviceTensor::Cuda)
                        .map_err(PcuExecutionError::from)
                }
                #[cfg(any(all(feature = "rocm", feature = "cuda"), feature = "metal"))]
                _ => Err(mismatch()),
            }
        }
        pub(in crate::global) fn execute_owned_program_consuming_binary_pair<T: PcuScalar>(
            &self,
            prepared: &Prepared,
            inputs: [(ValueId, DeviceTensor<T>); 2],
            pool: PcuMemoryPoolId,
            memory: &mut Memory,
        ) -> Result<DeviceTensor<T>, PcuExecutionError> {
            match (self, prepared, inputs, memory) {
                #[cfg(feature = "rocm")]
                (
                    Self::Rocm(assessor),
                    Prepared::Rocm(prepared),
                    [
                        (id0, DeviceTensor::Rocm(tensor0)),
                        (id1, DeviceTensor::Rocm(tensor1)),
                    ],
                    Memory::Rocm(memory),
                ) => assessor
                    .execute_owned_program_consuming_binary_pair(
                        prepared,
                        [(id0, tensor0), (id1, tensor1)],
                        pool,
                        memory,
                    )
                    .map(DeviceTensor::Rocm)
                    .map_err(PcuExecutionError::from),
                #[cfg(feature = "cuda")]
                (
                    Self::Cuda(assessor),
                    Prepared::Cuda(prepared),
                    [
                        (id0, DeviceTensor::Cuda(tensor0)),
                        (id1, DeviceTensor::Cuda(tensor1)),
                    ],
                    Memory::Cuda(memory),
                ) => assessor
                    .execute_owned_program_consuming_binary_pair(
                        prepared,
                        [(id0, tensor0), (id1, tensor1)],
                        pool,
                        memory,
                    )
                    .map(DeviceTensor::Cuda)
                    .map_err(PcuExecutionError::from),
                #[cfg(any(all(feature = "rocm", feature = "cuda"), feature = "metal"))]
                _ => Err(mismatch()),
            }
        }
    }
}
#[cfg(all(feature = "tensor", any(feature = "rocm", feature = "cuda")))]
#[rustfmt::skip]
pub(super) use graph::{
    Assessor,
    InputRef,
    Memory,
    Prepared,
    Resource,
};

#[cfg(all(feature = "tensor", any(feature = "rocm", feature = "cuda")))]
#[derive(Default)]
struct Roots {
    generation: u64,
    // Device runtime paths are part of cold execution-domain identity.
    realm: Vec<Option<std::ffi::OsString>>,
    sessions: Vec<Rc<Session>>,
}
#[cfg(all(feature = "tensor", any(feature = "rocm", feature = "cuda")))]
std::thread_local! {
    static ROOTS: std::cell::RefCell<Roots> = std::cell::RefCell::new(Roots::default());
}

#[cfg(all(feature = "tensor", any(feature = "rocm", feature = "cuda")))]
pub(super) fn prepare_tensor<R>(
    snapshot: super::policy::PolicySnapshot,
    affinity: Option<&Rc<Session>>,
    mut prepare: impl FnMut(&Rc<Session>) -> Result<R, PcuExecutionError>,
) -> Result<(Rc<Session>, R, u64, usize), PcuExecutionError> {
    let policy = snapshot.policy;
    if let Some(session) = affinity {
        let provider_conflict = match session.as_ref() {
            #[cfg(feature = "metal")]
            Session::Metal(_) => return Err(PcuExecutionError::TensorExecutionUnavailable),
            #[cfg(feature = "rocm")]
            Session::Rocm(_) => !matches!(
                policy.backend,
                super::PcuBackendChoice::Automatic | super::PcuBackendChoice::Rocm
            ),
            #[cfg(feature = "cuda")]
            Session::Cuda(_) => !matches!(
                policy.backend,
                super::PcuBackendChoice::Automatic | super::PcuBackendChoice::Cuda
            ),
        };
        if provider_conflict
            || policy
                .device
                .is_some_and(|device| device != session.device_id())
            || policy.block_size != session.block_size()
        {
            return Err(PcuExecutionError::ResidentPolicyConflict);
        }
        let prepared = prepare(session)?;
        return Ok((
            Rc::clone(session),
            prepared,
            snapshot.generation,
            policy.cache_capacity,
        ));
    }
    ROOTS
        .try_with(|roots| {
            let mut roots = roots
                .try_borrow_mut()
                .map_err(|_| PcuExecutionError::ReentrantCall)?;
            let realm = [
                "CUDA_DRIVER_LIBRARY",
                "CUDA_RUNTIME_LIBRARY",
                "NVRTC_LIBRARY",
                "CUBLAS_LIBRARY",
                "HIP_RUNTIME_LIBRARY",
                "HIPRTC_LIBRARY",
                "ROCBLAS_LIBRARY",
                "NVCC",
                "CUDA_NVCC",
                "CUDA_HOME",
                "CUDA_PATH",
                "HIPCC",
                "PATH",
            ]
            .map(std::env::var_os)
            .to_vec();
            if roots.generation != snapshot.generation || roots.realm != realm {
                roots.sessions.clear();
                roots.generation = snapshot.generation;
                roots.realm = realm;
            }
            #[cfg(feature = "cuda")]
            {
                prepare_candidates(&mut roots, snapshot, prepare)
            }
            #[cfg(all(feature = "rocm", not(feature = "cuda")))]
            {
                super::hosted::prepare_tensor(snapshot, None, |native| {
                    let session = if let Some(existing) =
                        roots.sessions.iter().find(|root| match root.as_ref() {
                            Session::Rocm(retained) => Rc::ptr_eq(retained, native),
                        }) {
                        Rc::clone(existing)
                    } else {
                        let session = Rc::new(Session::Rocm(Rc::clone(native)));
                        if roots.sessions.len() == policy.cache_capacity {
                            roots.sessions.remove(0);
                        }
                        roots.sessions.push(Rc::clone(&session));
                        session
                    };
                    prepare(&session).map(|prepared| (session, prepared))
                })
                .map(|(_, (session, prepared), generation, capacity)| {
                    (session, prepared, generation, capacity)
                })
            }
        })
        .map_err(|_| PcuExecutionError::ThreadUnavailable)?
}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "cpu",
    feature = "mlx"
))]
impl Session {
    pub(super) fn validate_policy(
        &self,
        policy: super::PcuExecutionPolicy,
    ) -> Result<(), PcuExecutionError> {
        let provider_conflict = match self {
            #[cfg(feature = "rocm")]
            Self::Rocm(_) => !matches!(
                policy.backend,
                super::PcuBackendChoice::Automatic | super::PcuBackendChoice::Rocm
            ),
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => !matches!(
                policy.backend,
                super::PcuBackendChoice::Automatic | super::PcuBackendChoice::Cuda
            ),
            #[cfg(feature = "metal")]
            Self::Metal(_) => !matches!(
                policy.backend,
                super::PcuBackendChoice::Automatic | super::PcuBackendChoice::Metal
            ),
        };
        if provider_conflict
            || policy
                .device
                .is_some_and(|device| device != self.device_id())
            || policy.block_size != self.block_size()
        {
            Err(PcuExecutionError::ResidentPolicyConflict)
        } else {
            Ok(())
        }
    }
    pub(super) fn prepare_host_kernel(
        &self,
        kernel: &crate::PcuDispatchKernelIr<'_>,
    ) -> Result<super::provider_hosted::Prepared, PcuExecutionError> {
        use crate::PcuHostKernelBackend;
        match self {
            #[cfg(feature = "metal")]
            Self::Metal(session) => session
                .backend
                .prepare_host_kernel(kernel)
                .map(super::provider_hosted::Prepared::Metal)
                .map_err(|error| {
                    PcuExecutionError::BackendFailure(format!("Metal preparation: {error:?}"))
                }),
            #[cfg(feature = "rocm")]
            Self::Rocm(session) => session
                .backend()
                .prepare_host_kernel(kernel)
                .map(super::provider_hosted::Prepared::Rocm)
                .map_err(|error| {
                    PcuExecutionError::BackendFailure(format!("ROCm preparation: {error}"))
                }),
            #[cfg(feature = "cuda")]
            Self::Cuda(session) => session
                .backend
                .prepare_host_kernel(kernel)
                .map(super::provider_hosted::Prepared::Cuda)
                .map_err(|error| {
                    PcuExecutionError::BackendFailure(format!("CUDA preparation: {error}"))
                }),
        }
    }
}

#[cfg(all(feature = "tensor", any(feature = "rocm", feature = "cuda")))]
pub(super) fn clear_roots() -> Result<(), PcuExecutionError> {
    ROOTS
        .try_with(|roots| {
            roots
                .try_borrow_mut()
                .map_err(|_| PcuExecutionError::ReentrantCall)?
                .sessions
                .clear();
            Ok(())
        })
        .map_err(|_| PcuExecutionError::ThreadUnavailable)?
}

#[cfg(all(feature = "cuda", feature = "tensor"))]
fn prepare_candidates<R>(
    roots: &mut Roots,
    snapshot: super::policy::PolicySnapshot,
    mut prepare: impl FnMut(&Rc<Session>) -> Result<R, PcuExecutionError>,
) -> Result<(Rc<Session>, R, u64, usize), PcuExecutionError> {
    #[rustfmt::skip]
    use super::provider_hosted::{
        Provider,
        collect_cuda_candidates,
        rank_candidates,
    };
    let policy = snapshot.policy;
    if matches!(policy.backend, super::PcuBackendChoice::Rocm) && !cfg!(feature = "rocm") {
        return Err(PcuExecutionError::NoBackendEnabled);
    }
    let cuda = fusion_pcu_cuda::CudaDiscovery::new();
    #[cfg(feature = "rocm")]
    let rocm = fusion_pcu_rocm::RocmDiscovery::new();
    let mut candidates = Vec::new();
    let mut rejected = Vec::new();
    let mut discovery_errors = Vec::new();
    if matches!(
        policy.backend,
        super::PcuBackendChoice::Automatic | super::PcuBackendChoice::Cuda
    ) && let Err(error) = collect_cuda_candidates(&cuda, policy, None, &mut candidates)
    {
        discovery_errors.push(format!("CUDA discovery: {error}"));
    }
    #[cfg(feature = "rocm")]
    if matches!(
        policy.backend,
        super::PcuBackendChoice::Automatic | super::PcuBackendChoice::Rocm
    ) && let Err(error) =
        super::provider_hosted::collect_rocm_candidates(&rocm, policy, None, &mut candidates)
    {
        discovery_errors.push(format!("ROCm discovery: {error}"));
    }
    rank_candidates(&mut candidates);
    for candidate in candidates {
        let existing = roots.sessions.iter().find(|session| {
            let same_provider = match (candidate.provider, session.as_ref()) {
                (Provider::Cuda, Session::Cuda(_)) => true,
                #[cfg(feature = "rocm")]
                (Provider::Rocm, Session::Rocm(_)) => true,
                #[cfg(any(
                    feature = "rocm",
                    feature = "metal",
                    feature = "cpu",
                    feature = "vulkan"
                ))]
                _ => false,
            };
            same_provider
                && candidate.device.id == session.device_id()
                && policy.block_size == session.block_size()
        });
        let session = if let Some(existing) = existing {
            Rc::clone(existing)
        } else {
            let opened = open_candidate(
                &candidate,
                policy,
                &cuda,
                #[cfg(feature = "rocm")]
                &rocm,
            );
            match opened {
                Ok(session) => Rc::new(session),
                Err(error) => {
                    rejected.push((candidate.device, error));
                    continue;
                }
            }
        };
        match prepare(&session) {
            Ok(prepared) => {
                if !roots.sessions.iter().any(|root| Rc::ptr_eq(root, &session)) {
                    if roots.sessions.len() == policy.cache_capacity {
                        roots.sessions.remove(0);
                    }
                    roots.sessions.push(Rc::clone(&session));
                }
                return Ok((
                    session,
                    prepared,
                    snapshot.generation,
                    policy.cache_capacity,
                ));
            }
            Err(error) => rejected.push((candidate.device, error)),
        }
    }
    Err(PcuExecutionError::NoCompatibleDevice {
        rejected,
        discovery: discovery_errors,
    })
}

#[cfg(all(feature = "cuda", feature = "tensor"))]
fn open_candidate(
    candidate: &super::provider_hosted::Candidate,
    policy: super::PcuExecutionPolicy,
    cuda: &fusion_pcu_cuda::CudaDiscovery,
    #[cfg(feature = "rocm")] rocm: &fusion_pcu_rocm::RocmDiscovery,
) -> Result<Session, PcuExecutionError> {
    use super::provider_hosted::Provider;
    match candidate.provider {
        #[cfg(feature = "mlx")]
        Provider::Mlx => Err(PcuExecutionError::TensorExecutionUnavailable),
        #[cfg(feature = "metal")]
        Provider::Metal => Err(PcuExecutionError::TensorExecutionUnavailable),
        #[cfg(feature = "vulkan")]
        Provider::Vulkan => Err(PcuExecutionError::TensorExecutionUnavailable),
        #[cfg(feature = "cpu")]
        Provider::Cpu => Err(PcuExecutionError::TensorExecutionUnavailable),
        Provider::Cuda => fusion_pcu_cuda::CudaOwnedDispatchBackend::open(
            cuda,
            candidate.device,
            policy.block_size,
        )
        .map(|backend| {
            Session::Cuda(Rc::new(CudaSession {
                backend: Rc::new(backend),
                block_size: policy.block_size,
                tensor: OnceCell::new(),
            }))
        })
        .map_err(|error| {
            PcuExecutionError::BackendFailure(format!("CUDA initialization: {error}"))
        }),
        #[cfg(feature = "rocm")]
        Provider::Rocm => fusion_pcu_rocm::RocmOwnedDispatchBackend::open(
            rocm,
            candidate.device,
            policy.block_size,
        )
        .map(|backend| {
            Session::Rocm(Rc::new(super::session::RocmSession::new(
                backend,
                policy.block_size,
            )))
        })
        .map_err(PcuExecutionError::BackendInitialization),
    }
}

#[cfg(all(test, feature = "cuda", feature = "tensor"))]
mod tests {
    #[rustfmt::skip]
    use crate::{
        PcuExecutionFault,
        PcuExecutionFaultKind,
        PcuScalarType,
    };
    use super::PcuExecutionError;
    use fusion_pcu_cuda::CudaTensorExecutionError;

    #[test]
    fn cuda_tensor_fault_conversion_preserves_terminal_metadata() {
        let fault = PcuExecutionFault {
            kind: PcuExecutionFaultKind::ArithmeticOverflow,
            invocation_id: 13,
            recovered: false,
        };
        let error = PcuExecutionError::from(CudaTensorExecutionError::ExecutionFault(fault));
        assert!(matches!(error, PcuExecutionError::ArithmeticFault(actual) if actual == fault));
    }

    #[test]
    fn cuda_tensor_rejections_preserve_native_structure() {
        let error = PcuExecutionError::from(CudaTensorExecutionError::UnsupportedScalarType(
            PcuScalarType::Bool,
        ));
        assert!(matches!(
            error,
            PcuExecutionError::CudaTensorExecution(
                CudaTensorExecutionError::UnsupportedScalarType(PcuScalarType::Bool)
            )
        ));
    }
}

#[cfg(feature = "metal")]
impl From<fusion_pcu_metal::MetalOwnedDispatchError> for PcuExecutionError {
    fn from(error: fusion_pcu_metal::MetalOwnedDispatchError) -> Self {
        match error {
            fusion_pcu_metal::MetalOwnedDispatchError::Metal(
                fusion_pcu_metal::MetalError::Arithmetic(fault),
            ) => Self::ArithmeticFault(fault),
            fusion_pcu_metal::MetalOwnedDispatchError::Memory(error) => Self::Memory(error),
            other => Self::BackendFailure(format!("Metal resident execution: {other}")),
        }
    }
}

#[cfg(feature = "metal")]
pub(super) fn from_metal_buffer<T: PcuScalar>(
    backend: fusion_pcu_metal::MetalOwnedDispatchBackend,
    buffer: crate::PcuDeviceBuffer<T, fusion_pcu_metal::MetalMemoryResource>,
    dimensions: &[usize],
) -> Result<super::PcuTensor<T>, PcuExecutionError> {
    use crate::PcuOwnedDispatchMemorySession;
    // Initialization proof is specific to this sealed adapter: MetalMemoryResource fields are
    // private, native Metal allocation zero-clears every byte, and resource import is unsupported.
    // Apple documents this guarantee for makeBuffer(length:options:):
    // https://developer.apple.com/documentation/metal/mtldevice/makebuffer(length:options:)
    // Generic PcuDeviceBufferAllocator does not promise initialization; future adapters must
    // establish their own initialized payload before constructing a Ready logical owner.
    let scalar = match T::TYPE {
        // These exact representations have qualified Metal storage and transfer paths.
        // This is transport admission, not a claim that every arithmetic operation exists.
        crate::PcuScalarType::U8
        | crate::PcuScalarType::I8
        | crate::PcuScalarType::U16
        | crate::PcuScalarType::I16
        | crate::PcuScalarType::U128
        | crate::PcuScalarType::I128
        | crate::PcuScalarType::U256
        | crate::PcuScalarType::I256
        | crate::PcuScalarType::U512
        | crate::PcuScalarType::I512
        | crate::PcuScalarType::F128
        | crate::PcuScalarType::F256
        | crate::PcuScalarType::U32
        | crate::PcuScalarType::I32
        | crate::PcuScalarType::U64
        | crate::PcuScalarType::I64
        | crate::PcuScalarType::F16
        | crate::PcuScalarType::BF16
        | crate::PcuScalarType::F32
        | crate::PcuScalarType::F64
        | crate::PcuScalarType::F8E4M3FN
        | crate::PcuScalarType::F8E5M2 => crate::PcuValueType::Scalar(T::TYPE),
        _ => {
            return Err(PcuExecutionError::BackendFailure(
                "unsupported Metal resident scalar".into(),
            ));
        }
    };
    if buffer.is_empty() {
        return Err(PcuExecutionError::EmptyTensorInput);
    }
    backend
        .bind(
            crate::PcuBindingRef::new(0, 0),
            crate::PcuBindingAccess::ReadOnly,
            crate::PcuBindingType::Value(scalar),
            buffer.resource(),
        )
        .map_err(PcuExecutionError::from)?;
    let bytes = buffer
        .len()
        .checked_mul(T::HOST_SIZE)
        .and_then(|size| u64::try_from(size).ok())
        .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?;
    if buffer.resource().size_bytes() < bytes {
        return Err(PcuExecutionError::InvalidTensorSourcePlan);
    }
    let tensor =
        PcuDeviceTensor::new(dimensions, buffer).map_err(PcuExecutionError::TensorStorage)?;
    let block_size = super::policy::snapshot()?.policy.block_size;
    let session = Session::Metal(Rc::new(MetalSession {
        backend: Rc::new(backend),
        block_size,
    }));
    Ok(super::PcuTensor::from_successful_output(
        DeviceTensor::Metal(tensor),
        Rc::new(session),
    ))
}
