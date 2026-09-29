//! Safe consuming execution paths for selected resident tensor programs.

#[rustfmt::skip]
use super::{
    alignment_satisfies,
    byte_len_for,
    scalar_layout,
    validate_owned_scalar_profile,
    PreparedConsumingAction,
    PreparedConsumingBinary,
    RocmMemoryResource,
    RocmOwnedPreparedTensorGraph,
    RocmTensorAssessor,
    RocmTensorExecutionError,
    RocmTensorInputRef,
    RocmTensorInputDescriptor,
    TensorDispatchCacheKey,
    TensorPointwiseScalarType,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingType,
    PcuCompletionOutcome,
    PcuDeviceTensor,
    PcuMemoryAccess,
    PcuMemoryBackingOwnership,
    PcuMemoryPoolId,
    PcuMemoryProvider,
    PcuMemoryResource,
    PcuInvocationShape,
    PcuBindingRef,
    PcuOwnedCompletion,
    PcuOwnedDispatchMemorySession,
};
use core::num::NonZeroU32;
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    NodeDescriptor,
    TensorOwnedSelectedProgram,
    ValueId,
};
use super::pointwise;

impl RocmTensorAssessor<'_> {
    /// Executes a selected graph from one moved resident input and its other selected resident
    /// input. A proven terminal Add/Sub/Mul may overwrite the donor input when its physical
    /// resource is exclusive, quiescent, correctly typed, and disjoint from the borrowed input.
    /// Every other eligible graph uses the ordinary fresh-output scheduler.
    ///
    /// The borrowed input descriptor and the donor lease remain alive through terminal completion.
    /// Reuse preserves the operation's left/right operand order even when the donor is the right
    /// operand. Unknown overlap or non-exclusive backing takes the fresh-output path after the
    /// shared preflight confirms that both sources are readable and quiescent; unavailable or
    /// uncertain source access remains an error rather than triggering a read retry.
    ///
    /// # Errors
    ///
    /// Returns input validation, allocation/provider, dispatch, or completion errors. No result
    /// is published after an unsuccessful terminal dispatch.
    pub fn execute_owned_program_consuming_binary_input<T, P>(
        &self,
        prepared: &RocmOwnedPreparedTensorGraph,
        donor_value: ValueId,
        donor: PcuDeviceTensor<T, RocmMemoryResource>,
        other_inputs: &[(ValueId, &RocmTensorInputRef<'_>)],
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<PcuDeviceTensor<T, RocmMemoryResource>, RocmTensorExecutionError>
    where
        T: fusion_pcu::PcuScalar,
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        if other_inputs.is_empty() {
            if prepared.data.input_values.as_slice() != [donor_value] {
                return Err(RocmTensorExecutionError::InvalidPlan(donor_value));
            }
            return self.execute_owned_program_consuming_input(prepared, donor, pool, memory);
        }
        if other_inputs.len() != 1 || prepared.data.outputs.len() != 1 {
            return Err(RocmTensorExecutionError::InvalidPlan(prepared.data.output));
        }

        let other_input = other_inputs[0];
        let reuse = self.preflight_consuming_binary_donor(
            prepared,
            donor_value,
            &donor,
            other_input,
            pool,
        )?;
        match reuse {
            Some(reuse) => self.execute_consuming_binary_donor(reuse, donor, other_inputs[0].1),
            None => self.execute_fresh_owned_binary_input(
                prepared,
                donor_value,
                &donor,
                other_inputs[0],
                pool,
                memory,
            ),
        }
    }

    /// Executes a binary graph from two moved resident inputs.
    ///
    /// Both owners remain alive until the selected dispatch reaches terminal completion. If one
    /// input is a proven donor and its allocation is exclusive, quiescent, correctly typed, and
    /// disjoint from the other, that input is reused. The first eligible input wins; otherwise a
    /// fresh output is scheduled. Operand order always follows the graph, including subtraction.
    ///
    /// # Errors
    ///
    /// Returns an input validation, allocation/provider, dispatch, or completion error. Results
    /// are returned only after successful terminal completion.
    pub fn execute_owned_program_consuming_binary_pair<T, P>(
        &self,
        prepared: &RocmOwnedPreparedTensorGraph,
        inputs: [(ValueId, PcuDeviceTensor<T, RocmMemoryResource>); 2],
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<PcuDeviceTensor<T, RocmMemoryResource>, RocmTensorExecutionError>
    where
        T: fusion_pcu::PcuScalar,
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        if prepared.data.input_values.len() != 2 || prepared.data.outputs.len() != 1 {
            return Err(RocmTensorExecutionError::InvalidPlan(prepared.data.output));
        }
        let [(first_value, first), (second_value, second)] = inputs;
        if first_value == second_value {
            return Err(RocmTensorExecutionError::InvalidPlan(first_value));
        }

        let second_input = self.borrow_device_input_ref(&second, pool)?;
        let selected_donor = {
            let first_input = self.borrow_device_input_ref(&first, pool)?;
            let sources = [(first_value, &first_input), (second_value, &second_input)];
            self.validate_consuming_binary_sources::<T>(prepared, &sources, pool)?;
            if let Some(reuse) = Self::find_reusable_binary_donor(
                prepared,
                first_value,
                &first,
                second_value,
                second_input.resource(),
            )? {
                Some((0, reuse))
            } else {
                Self::find_reusable_binary_donor(
                    prepared,
                    second_value,
                    &second,
                    first_value,
                    first_input.resource(),
                )?
                .map(|reuse| (1, reuse))
            }
        };

        match selected_donor {
            Some((0, reuse)) => self.execute_consuming_binary_donor(reuse, first, &second_input),
            Some((1, reuse)) => {
                let other = self.borrow_device_input_ref(&first, pool)?;
                self.execute_consuming_binary_donor(reuse, second, &other)
            }
            _ => {
                let first_input = self.borrow_device_input_ref(&first, pool)?;
                let sources = [(first_value, &first_input), (second_value, &second_input)];
                self.execute_fresh_owned_from_resource_inputs(prepared, &sources, pool, memory)
            }
        }
    }

    fn validate_consuming_binary_sources<T>(
        &self,
        prepared: &RocmOwnedPreparedTensorGraph,
        inputs: &[(ValueId, &RocmTensorInputRef<'_>)],
        pool: PcuMemoryPoolId,
    ) -> Result<(), RocmTensorExecutionError>
    where
        T: fusion_pcu::PcuScalar,
    {
        let view = prepared.view();
        validate_owned_scalar_profile::<T>(prepared.data.scalar_type)?;
        self.validate_execution_sources_view(&view, &[], inputs, pool)
    }

    fn preflight_consuming_binary_donor<'prepared, T>(
        &self,
        prepared: &'prepared RocmOwnedPreparedTensorGraph,
        donor_value: ValueId,
        donor: &PcuDeviceTensor<T, RocmMemoryResource>,
        other_input: (ValueId, &RocmTensorInputRef<'_>),
        pool: PcuMemoryPoolId,
    ) -> Result<Option<&'prepared PreparedConsumingBinary>, RocmTensorExecutionError>
    where
        T: fusion_pcu::PcuScalar,
    {
        let donor_input = self.borrow_device_input_ref(donor, pool)?;
        let inputs = [(donor_value, &donor_input), other_input];
        self.validate_consuming_binary_sources::<T>(prepared, &inputs, pool)?;

        Self::find_reusable_binary_donor(
            prepared,
            donor_value,
            donor,
            other_input.0,
            other_input.1.resource(),
        )
    }

    fn find_reusable_binary_donor<'prepared, T>(
        prepared: &'prepared RocmOwnedPreparedTensorGraph,
        donor_value: ValueId,
        donor: &PcuDeviceTensor<T, RocmMemoryResource>,
        other_value: ValueId,
        other_resource: &RocmMemoryResource,
    ) -> Result<Option<&'prepared PreparedConsumingBinary>, RocmTensorExecutionError>
    where
        T: fusion_pcu::PcuScalar,
    {
        let reuse = prepared
            .data
            .consuming_action
            .as_ref()
            .and_then(|action| match action {
                PreparedConsumingAction::TerminalBinary(candidates) => candidates
                    .iter()
                    .find(|candidate| candidate.proof.donor() == donor_value),
                PreparedConsumingAction::IdentityTransfer(_)
                | PreparedConsumingAction::TerminalRelu(_) => None,
            });
        let Some(reuse) = reuse else {
            return Ok(None);
        };
        let proof = reuse.proof;
        let resource = donor.buffer().resource();
        let required_bytes = byte_len_for::<T>(donor.shape())?;
        let disjoint_from_other = resource.overlap(
            other_resource,
            fusion_pcu::PcuMemoryRange {
                offset_bytes: 0,
                size_bytes: resource.size_bytes(),
            },
            fusion_pcu::PcuMemoryRange {
                offset_bytes: 0,
                size_bytes: other_resource.size_bytes(),
            },
        ) == fusion_pcu::PcuMemoryOverlap::Disjoint;
        let can_reuse = proof.donor() == donor_value
            && proof.other() == other_value
            && proof.output() == prepared.data.output
            && proof.scalar_type() == T::TYPE
            && u64::try_from(required_bytes).ok() == Some(proof.bytes())
            && proof.alignment_bytes() <= resource.alignment_bytes()
            && alignment_satisfies(resource.alignment_bytes(), scalar_layout(T::TYPE)?.1)
            && resource.access() == PcuMemoryAccess::ReadWrite
            && resource.backing_ownership() == PcuMemoryBackingOwnership::Exclusive
            && resource.validate_access_available().is_ok()
            && usize::try_from(reuse.logical_count).ok() == Some(donor.buffer().len())
            && reuse.scalar_type.value_type() == reuse.value_type
            && disjoint_from_other;
        Ok(can_reuse.then_some(reuse))
    }

    fn execute_fresh_owned_binary_input<T, P>(
        &self,
        prepared: &RocmOwnedPreparedTensorGraph,
        donor_value: ValueId,
        donor: &PcuDeviceTensor<T, RocmMemoryResource>,
        other_input: (ValueId, &RocmTensorInputRef<'_>),
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<PcuDeviceTensor<T, RocmMemoryResource>, RocmTensorExecutionError>
    where
        T: fusion_pcu::PcuScalar,
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        let donor_input = self.borrow_device_input_ref(donor, pool)?;
        let inputs = [(donor_value, &donor_input), other_input];
        self.execute_fresh_owned_from_resource_inputs(prepared, &inputs, pool, memory)
    }

    fn execute_fresh_owned_from_resource_inputs<T, P>(
        &self,
        prepared: &RocmOwnedPreparedTensorGraph,
        inputs: &[(ValueId, &RocmTensorInputRef<'_>)],
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<PcuDeviceTensor<T, RocmMemoryResource>, RocmTensorExecutionError>
    where
        T: fusion_pcu::PcuScalar,
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        let outputs =
            self.execute_owned_program_outputs_from_inputs(prepared, inputs, pool, memory)?;
        if outputs.len() != 1 {
            return Err(RocmTensorExecutionError::InvalidPlan(prepared.data.output));
        }
        let (value, tensor) = outputs
            .into_iter()
            .next()
            .ok_or(RocmTensorExecutionError::InvalidPlan(prepared.data.output))?;
        if value != prepared.data.output {
            return Err(RocmTensorExecutionError::InvalidPlan(value));
        }
        Ok(tensor)
    }

    fn execute_consuming_binary_donor<T>(
        &self,
        reuse: &PreparedConsumingBinary,
        donor: PcuDeviceTensor<T, RocmMemoryResource>,
        other: &RocmTensorInputRef<'_>,
    ) -> Result<PcuDeviceTensor<T, RocmMemoryResource>, RocmTensorExecutionError>
    where
        T: fusion_pcu::PcuScalar,
    {
        let proof = reuse.proof;
        let cache_key = TensorDispatchCacheKey::ConsumingBinary(
            proof.operation(),
            proof.donor_operand(),
            reuse.scalar_type,
            reuse.logical_count,
        );
        self.ensure_dispatch_cached(
            cache_key.clone(),
            reuse.kernel,
            reuse.invocation_shape,
            false,
        )?;
        let donor_binding = self
            .session
            .bind(
                reuse.donor_binding,
                PcuBindingAccess::ReadWrite,
                PcuBindingType::Value(reuse.value_type),
                donor.buffer().resource(),
            )
            .map_err(RocmTensorExecutionError::Backend)?;
        let other_binding = self
            .session
            .bind(
                reuse.other_binding,
                PcuBindingAccess::ReadOnly,
                PcuBindingType::Value(reuse.value_type),
                other.resource(),
            )
            .map_err(RocmTensorExecutionError::Backend)?;
        let mut completion = {
            let cache = self.state().add_dispatches.borrow();
            let executable = cache
                .iter()
                .find(|(key, _)| *key == cache_key)
                .map(|(_, executable)| executable)
                .ok_or_else(|| RocmTensorExecutionError::InvalidPlan(proof.output()))?;
            executable
                .submit(&[donor_binding, other_binding])
                .map_err(RocmTensorExecutionError::Backend)?
        };
        match completion
            .wait()
            .map_err(RocmTensorExecutionError::Completion)?
        {
            PcuCompletionOutcome::Succeeded => Ok(donor),
            PcuCompletionOutcome::Failed => Err(RocmTensorExecutionError::FailedCompletion),
            PcuCompletionOutcome::Fault(fault) => {
                Err(RocmTensorExecutionError::ExecutionFault(fault))
            }
        }
    }
}

pub(super) fn prepare_consuming_binary_action(
    program: &TensorOwnedSelectedProgram,
    nodes: &[NodeDescriptor<'_>],
) -> Result<Option<PreparedConsumingAction>, RocmTensorExecutionError> {
    if program.input_values().len() != 2
        || program.output_values().len() != 1
        || program.selected_nodes().len() != 3
        || nodes.len() != 3
    {
        return Ok(None);
    }
    let inputs = program.input_values();
    let output = program.output_values()[0];
    let left_donor =
        prepare_consuming_binary_candidate(program, nodes, inputs[0], inputs[1], output)?;
    let right_donor =
        prepare_consuming_binary_candidate(program, nodes, inputs[1], inputs[0], output)?;
    match (left_donor, right_donor) {
        (Some(left_donor), Some(right_donor)) => {
            Ok(Some(PreparedConsumingAction::TerminalBinary(Box::new([
                left_donor,
                right_donor,
            ]))))
        }
        _ => Ok(None),
    }
}

fn prepare_consuming_binary_candidate(
    program: &TensorOwnedSelectedProgram,
    nodes: &[NodeDescriptor<'_>],
    donor: ValueId,
    other: ValueId,
    output: ValueId,
) -> Result<Option<PreparedConsumingBinary>, RocmTensorExecutionError> {
    let Ok(proof) = program.prove_consumed_binary_donor(donor, other, output) else {
        return Ok(None);
    };
    let Some(input_node) = nodes.iter().find(|node| node.value == donor) else {
        return Ok(None);
    };
    if input_node.scalar_type != proof.scalar_type() {
        return Ok(None);
    }
    let scalar_type = TensorPointwiseScalarType::try_from(proof.scalar_type())?;
    let logical_count = input_node.shape.iter().try_fold(1_u32, |count, &extent| {
        u32::try_from(extent)
            .ok()
            .and_then(|extent| count.checked_mul(extent))
    });
    let Some(logical_count) = logical_count.filter(|count| *count > 0) else {
        return Ok(None);
    };
    let invocation_count =
        NonZeroU32::new(logical_count).ok_or(RocmTensorExecutionError::SizeOverflow)?;
    Ok(Some(PreparedConsumingBinary {
        proof,
        kernel: pointwise::consuming_binary_kernel(
            scalar_type,
            proof.operation(),
            proof.donor_operand(),
            logical_count,
        ),
        invocation_shape: PcuInvocationShape::invocations(invocation_count),
        scalar_type,
        value_type: scalar_type.value_type(),
        donor_binding: PcuBindingRef::new(0, 0),
        other_binding: PcuBindingRef::new(0, 1),
        logical_count,
    }))
}

#[cfg(test)]
mod tests {
    use fusion_pcu::PcuScalarType;
    use fusion_pcu::dialect::tensor::{
        Graph, TensorArithmeticCapability, TensorArithmeticRewritePolicy, TensorBinaryOperand,
        TensorPointwiseGroupingPolicy,
    };

    use super::{PreparedConsumingAction, prepare_consuming_binary_action};

    #[test]
    fn selected_binary_action_captures_both_donor_orders_cold() {
        let mut graph = Graph::default();
        let left = graph.input([4], PcuScalarType::F64).unwrap();
        let right = graph.input([4], PcuScalarType::F64).unwrap();
        let output = graph.sub(left, right).unwrap();
        let program = graph
            .into_selected_program(
                &[output],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .unwrap();
        let nodes = program
            .selected_nodes()
            .iter()
            .map(|value| program.graph().node(*value).unwrap())
            .collect::<Vec<_>>();

        let Some(PreparedConsumingAction::TerminalBinary(candidates)) =
            prepare_consuming_binary_action(&program, &nodes).unwrap()
        else {
            panic!("both donor orders should be prepared");
        };
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].proof.donor(), left);
        assert_eq!(candidates[0].proof.other(), right);
        assert_eq!(
            candidates[0].proof.donor_operand(),
            TensorBinaryOperand::Left
        );
        assert_eq!(candidates[1].proof.donor(), right);
        assert_eq!(candidates[1].proof.other(), left);
        assert_eq!(
            candidates[1].proof.donor_operand(),
            TensorBinaryOperand::Right
        );
    }
}
