//! Ordered fixed-size output publication over the existing terminal one-graph scheduler.
#[rustfmt::skip]
use super::{
    RocmTensorAssessor,
    RocmOwnedPreparedTensorGraph,
    RocmTensorInputRef,
    RocmTensorExecutionError,
    RocmMemoryResource,
    PcuDeviceTensor,
    PcuMemoryPoolId,
    PcuMemoryProvider,
    ValueId,
    SmallVec,
    validate_owned_scalar_profile,
    scalar_layout,
    byte_len_for,
};

impl<'session> RocmTensorAssessor<'session> {
    /// Executes one prepared graph, returning its ordered independent output owners atomically.
    ///
    /// All selected outputs must have the requested scalar type, valid dense byte geometry,
    /// distinct value IDs and exactly `M` entries. These facts are checked before allocation or
    /// device work. The existing scheduler provides terminal completion and quarantines uncertain
    /// work; no output is published on error. Existing single-output execution is unchanged.
    ///
    /// The conversion uses a stack array and adds no Vec allocation. The existing scheduler may
    /// spill its output bookkeeping for multiple outputs, as in the established plural API.
    ///
    /// # Errors
    /// Returns cardinality/type/layout/graph, input validation, allocation/provider, operation,
    /// arithmetic or terminal completion errors. Empty output arrays are not admitted.
    pub fn execute_owned_program_output_array_from_inputs<'input, T, P, const M: usize>(
        &'input self,
        prepared: &RocmOwnedPreparedTensorGraph,
        inputs: &[(ValueId, &RocmTensorInputRef<'input>)],
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<[PcuDeviceTensor<T, RocmMemoryResource>; M], RocmTensorExecutionError>
    where
        T: fusion_pcu::PcuScalar,
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
        'session: 'input,
    {
        validate_output_array_plan::<T, M>(prepared)?;
        let outputs =
            self.execute_owned_program_outputs_inline_from_inputs(prepared, inputs, pool, memory)?;
        output_array::<_, M>(outputs).map(|ordered| ordered.map(|(_, tensor)| tensor))
    }
}

fn validate_output_array_plan<T: fusion_pcu::PcuScalar, const M: usize>(
    prepared: &RocmOwnedPreparedTensorGraph,
) -> Result<(), RocmTensorExecutionError> {
    let actual = prepared.data.outputs.len();
    if actual != M {
        return Err(RocmTensorExecutionError::OutputCountMismatch {
            expected: M,
            actual,
        });
    }
    if M == 0 {
        return Err(RocmTensorExecutionError::EmptyOutputs);
    }
    validate_owned_scalar_profile::<T>(prepared.data.scalar_type)?;
    scalar_layout(T::TYPE)?;
    for (index, &value) in prepared.data.outputs.iter().enumerate() {
        if prepared.data.outputs[..index].contains(&value) {
            return Err(RocmTensorExecutionError::DuplicateOutput(value));
        }
        let node = prepared.program.graph().node(value)?;
        if node.scalar_type != T::TYPE {
            return Err(RocmTensorExecutionError::UnsupportedScalarType(
                node.scalar_type,
            ));
        }
        byte_len_for::<T>(node.shape)?;
    }
    Ok(())
}

fn output_array<T, const M: usize>(
    outputs: SmallVec<[T; 1]>,
) -> Result<[T; M], RocmTensorExecutionError> {
    if outputs.len() != M {
        return Err(RocmTensorExecutionError::OutputCountMismatch {
            expected: M,
            actual: outputs.len(),
        });
    }
    let mut outputs = outputs.into_iter();
    // Exact length was checked above; from_fn requests exactly M entries. This invariant is
    // independent of caller data. No unchecked memory, extra collection or owner clone is needed.
    Ok(core::array::from_fn(|_| {
        outputs.next().expect("checked exact output cardinality")
    }))
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
