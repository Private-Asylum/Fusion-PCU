//! Terminal ownership after a provider failure following an already-enqueued native GEMM.

#[rustfmt::skip]
use fusion_pcu::{
    PcuCompoundArithmeticPolicy,
    PcuDeviceTensor,
    PcuMemoryAllocationRequest,
    PcuMemoryDisposition,
    PcuMemoryPoolId,
    PcuMemoryPoolSnapshot,
    PcuMemoryProvider,
    PcuMemoryProviderError,
    PcuMemoryProviderFailure,
    PcuMemoryProviderOperation,
    PcuMemoryRange,
    PcuNumericalOptions,
    PcuScalar,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    TensorArithmeticCapability,
    TensorArithmeticRewritePolicy,
    TensorPointwiseGroupingPolicy,
};
#[rustfmt::skip]
use crate::{
    RocmImportDescriptor,
    RocmMemoryMapping,
    RocmMemoryProvider,
    RocmMemoryResource,
    RocmOwnedDispatchBackend,
    RocmTensorAssessor,
    RocmTensorExecutionError,
};

struct FailThirdAllocation {
    inner: RocmMemoryProvider,
    allocations: usize,
    armed: bool,
}
impl PcuMemoryProvider for FailThirdAllocation {
    type Resource = RocmMemoryResource;
    type ImportDescriptor = RocmImportDescriptor;
    type Mapping<'a> = RocmMemoryMapping;
    fn snapshot(
        &self,
        pool: PcuMemoryPoolId,
    ) -> Result<PcuMemoryPoolSnapshot, PcuMemoryProviderError> {
        self.inner.snapshot(pool)
    }
    fn allocate(
        &mut self,
        request: PcuMemoryAllocationRequest,
    ) -> Result<Self::Resource, PcuMemoryProviderError> {
        self.allocations += 1;
        if self.armed && self.allocations == 3 {
            return Err(PcuMemoryProviderError {
                pool: request.pool,
                operation: PcuMemoryProviderOperation::Allocate,
                disposition: PcuMemoryDisposition::Reject,
                failure: PcuMemoryProviderFailure::OutOfMemory,
            });
        }
        self.inner.allocate(request)
    }
    fn import(
        &mut self,
        descriptor: Self::ImportDescriptor,
    ) -> Result<Self::Resource, PcuMemoryProviderError> {
        self.inner.import(descriptor)
    }
    fn map<'a>(
        &'a mut self,
        resource: &'a mut Self::Resource,
        range: PcuMemoryRange,
    ) -> Result<Self::Mapping<'a>, PcuMemoryProviderError> {
        self.inner.map(resource, range)
    }
    fn transfer_to(
        &mut self,
        resource: &mut Self::Resource,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), PcuMemoryProviderError> {
        self.inner.transfer_to(resource, offset, bytes)
    }
    fn transfer_from(
        &mut self,
        resource: &Self::Resource,
        offset: u64,
        bytes: &mut [u8],
    ) -> Result<(), PcuMemoryProviderError> {
        self.inner.transfer_from(resource, offset, bytes)
    }
    fn copy_resource(
        &mut self,
        destination: &mut Self::Resource,
        source: &Self::Resource,
        size: u64,
    ) -> Result<(), PcuMemoryProviderError> {
        self.inner.copy_resource(destination, source, size)
    }
}

fn check_width<T: PcuScalar + Default + From<i16> + core::fmt::Debug + PartialEq>(
    backend: &RocmOwnedDispatchBackend,
) {
    let assessor = RocmTensorAssessor::new(backend).unwrap();
    let pool = PcuMemoryPoolId(0);
    let mut graph = Graph::default();
    graph.set_numerical_options(PcuNumericalOptions {
        compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
        ..PcuNumericalOptions::default()
    });
    let input_ids = core::array::from_fn::<_, 4, _>(|_| graph.input([2, 2], T::TYPE).unwrap());
    let first = graph.matmul(input_ids[0], input_ids[1]).unwrap();
    let second = graph.matmul(first, input_ids[2]).unwrap();
    let output = graph.matmul(second, input_ids[3]).unwrap();
    let program = graph
        .into_selected_program(
            &[first, output],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
    let prepared = assessor.prepare_owned_program(program).unwrap();
    assert!(prepared.data.batch_native_matmuls);
    let input_values =
        [[1, 2, 3, 4], [2, 1, 0, 2], [1, 0, 1, 1], [2, 0, 0, 1]].map(|values| values.map(T::from));
    let inputs = input_values.each_ref().map(|values| {
        PcuDeviceTensor::new([2, 2], backend.upload_buffer(pool, values).unwrap()).unwrap()
    });
    let resources = core::array::from_fn::<_, 4, _>(|index| (input_ids[index], &inputs[index]));
    let mut memory = FailThirdAllocation {
        inner: backend.memory_provider(pool),
        allocations: 0,
        armed: true,
    };
    // Two fresh allocations retain the selected first intermediate and final output. The
    // third (the second intermediate) fails after the first GEMM enters the stream.
    let failure = assessor.execute_owned_program_outputs(&prepared, &resources, pool, &mut memory);
    assert!(matches!(
        failure,
        Err(RocmTensorExecutionError::Memory(PcuMemoryProviderError {
            operation: PcuMemoryProviderOperation::Allocate,
            failure: PcuMemoryProviderFailure::OutOfMemory,
            ..
        }))
    ));
    assert_eq!(memory.allocations, 3);
    for (input, expected) in inputs.iter().zip(input_values) {
        let mut actual = [T::default(); 4];
        backend
            .download_buffer(pool, input.buffer(), &mut actual)
            .unwrap();
        assert_eq!(actual, expected); // Access leases are terminally released, inputs unchanged.
    }
    memory.armed = false;
    let result = assessor
        .execute_owned_program_outputs(&prepared, &resources, pool, &mut memory)
        .unwrap();
    drop(inputs);
    // Independent hand arithmetic: A*B=[2,5;6,11], *C=[7,5;17,11], *D=[14,5;34,11].
    assert_eq!(result.len(), 2);
    for ((value, tensor), (expected_id, expected)) in result
        .iter()
        .zip([(first, [2, 5, 6, 11]), (output, [14, 5, 34, 11])])
    {
        assert_eq!(*value, expected_id);
        let mut actual = [T::default(); 4];
        backend
            .download_buffer(pool, tensor.buffer(), &mut actual)
            .unwrap();
        assert_eq!(actual, expected.map(T::from));
    }
}

#[test]
#[ignore = "requires idle ROCm hardware; run serially"]
fn native_chain_allocation_failure_quiesces_retained_owners_and_retries_both_widths() {
    let (_discovery, backend) = crate::tensor::tests::rocm_test_session();
    check_width::<f32>(&backend);
    check_width::<f64>(&backend);
}
