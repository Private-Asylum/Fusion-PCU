//! One cold source capture shared by ordinary execution and provider preparation.

#[rustfmt::skip]
use alloc::{
    sync::Arc,
    vec::Vec,
};
#[rustfmt::skip]
use super::{
    PcuExecutionError,
    PcuSourceShape,
    PcuTensorGraphCapture,
    PcuTensorGraphValue,
    PcuTensorShapeWitness,
};
#[rustfmt::skip]
use crate::{
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuScalar,
};
#[rustfmt::skip]
use crate::dialect::tensor::{
    TensorArithmeticCapability,
    TensorArithmeticRewritePolicy,
    TensorOwnedSelectedProgram,
    TensorPointwiseGroupingPolicy,
    ValueId,
};

/// Owned source IR and the mapping from its selected inputs to Rust argument positions.
///
/// This is a cold frontend artifact, not an executable, device owner or admission proof.
/// A provider must independently admit its full operation, numerical and storage contract.
#[doc(hidden)]
pub struct PcuCapturedTensorProgram {
    pub(super) program: Arc<TensorOwnedSelectedProgram>,
    pub(super) input_ids: Vec<ValueId>,
    pub(super) input_indices: Vec<usize>,
}

impl PcuCapturedTensorProgram {
    /// Retained selected IR; cloning its Arc does not recapture the function.
    #[must_use]
    pub const fn program(&self) -> &Arc<TensorOwnedSelectedProgram> {
        &self.program
    }

    /// Input IDs in the selected program's binding order.
    #[must_use]
    pub fn input_values(&self) -> &[ValueId] {
        &self.input_ids
    }

    /// Original argument positions corresponding to [`Self::input_values`].
    /// Unused arguments are absent, rather than uploaded or assigned invented bindings.
    #[must_use]
    pub fn argument_indices(&self) -> &[usize] {
        &self.input_indices
    }
}

/// Captures a generated per-function companion for independent provider preparation.
///
/// The ordinary call path uses the same builder. This performs no discovery, allocation of
/// device storage, upload, execution or readback. It can prove frontend-to-provider integration
/// without implying that the provider is integrated into ordinary global source calls.
/// Function/helper flags are resolved by the generated companion against these base policies.
///
/// # Errors
/// Returns empty executable input shapes, overflowing declared shapes, capture or
/// selected-program construction failures. Unused zero extents remain truthful
/// declarations and require no physical resource. Checked discarded effects still
/// contribute their actual executable inputs.
#[doc(hidden)]
pub fn __pcu_capture_tensor_program<T: PcuScalar, const N: usize, F>(
    shapes: [PcuSourceShape; N],
    float_underflow: PcuFloatUnderflowPolicy,
    numerical_mode: PcuNumericalMode,
    numerical_options: PcuNumericalOptions,
    capture: F,
) -> Result<PcuCapturedTensorProgram, PcuExecutionError>
where
    F: FnOnce(
        &mut PcuTensorGraphCapture,
        [PcuTensorGraphValue<T>; N],
    ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError>,
{
    if N == 0 {
        return Err(PcuExecutionError::EmptyTensorInput);
    }
    build(
        shapes.map(PcuTensorShapeWitness::Static),
        float_underflow,
        numerical_mode,
        numerical_options,
        capture,
    )
}

pub(super) fn build<T: PcuScalar, const N: usize, F>(
    shapes: [PcuTensorShapeWitness<'_>; N],
    float_underflow: PcuFloatUnderflowPolicy,
    numerical_mode: PcuNumericalMode,
    numerical_options: PcuNumericalOptions,
    capture_function: F,
) -> Result<PcuCapturedTensorProgram, PcuExecutionError>
where
    F: FnOnce(
        &mut PcuTensorGraphCapture,
        [PcuTensorGraphValue<T>; N],
    ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError>,
{
    let mut declared_elements = [0_usize; N];
    for (index, shape) in shapes.iter().copied().enumerate() {
        declared_elements[index] = element_count(shape)?;
    }
    let (mut capture, input_values) =
        PcuTensorGraphCapture::new_with_witnesses_and_policy::<T, N>(shapes, float_underflow)?;
    capture.numerical_mode.set(numerical_mode);
    capture.numerical_options.set(numerical_options);
    let captured_input_ids = input_values.map(|value| value.value.erase());
    let output = capture_function(&mut capture, input_values)?;
    let (graph, output_id) = capture.finish(output)?;
    let program = Arc::new(
        graph
            .into_selected_program(
                &[output_id],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .map_err(crate::global::tensor_build_error)?,
    );
    let mut input_ids = Vec::with_capacity(program.input_values().len());
    let mut input_indices = Vec::with_capacity(program.input_values().len());
    for &selected_id in program.input_values() {
        let argument_index = captured_input_ids
            .iter()
            .position(|captured_id| *captured_id == selected_id)
            .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?;
        if declared_elements[argument_index] == 0 {
            return Err(PcuExecutionError::EmptyTensorInput);
        }
        input_ids.push(selected_id);
        input_indices.push(argument_index);
    }
    if input_ids.is_empty() {
        return Err(PcuExecutionError::InvalidTensorSourcePlan);
    }
    Ok(PcuCapturedTensorProgram {
        program,
        input_ids,
        input_indices,
    })
}

pub(super) fn element_count(shape: PcuTensorShapeWitness<'_>) -> Result<usize, PcuExecutionError> {
    let count = match shape {
        PcuTensorShapeWitness::Static(PcuSourceShape::Scalar) => Some(1),
        PcuTensorShapeWitness::Static(
            PcuSourceShape::Slice { length } | PcuSourceShape::FixedArray { length },
        ) => Some(length),
        PcuTensorShapeWitness::Static(PcuSourceShape::FixedMatrix { rows, columns }) => {
            rows.checked_mul(columns)
        }
        PcuTensorShapeWitness::Dynamic(dimensions) => dimensions
            .iter()
            .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension)),
    };
    count.ok_or(PcuExecutionError::InvalidTensorSourcePlan)
}

#[cfg(test)]
mod tests;
