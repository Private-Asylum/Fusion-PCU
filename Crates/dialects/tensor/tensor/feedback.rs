//! Backend-neutral feedback bindings and reusable iteration schedule.

use super::{
    TensorError,
    TensorExecutionPlan,
    ValueId,
};
use std::num::NonZeroUsize;

/// One loop-carried value, mapping a selected output into a selected graph input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorFeedbackBinding {
    pub output: ValueId,
    pub input: ValueId,
}

/// Source to use for a graph input at a particular iteration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorFeedbackInput {
    /// This input is supplied by the caller at the beginning of every iteration.
    Initial { input: ValueId },
    /// This input reads the prior iteration's output from the indicated ping-pong bank.
    PreviousOutput {
        input: ValueId,
        output: ValueId,
        bank: usize,
    },
}

/// Validated feedback relationships attached to one selected tensor execution plan.
///
/// Bank zero is used for iteration zero's output; each subsequent iteration advances around the
/// configured ring of banks.
/// Query methods are allocation-free, so a backend can use this object in a reusable execution
/// loop without rebuilding a feedback map or doing host-side ping-pong bookkeeping.
#[derive(Debug)]
pub struct TensorFeedbackPlan<'p, 'g> {
    plan: &'p TensorExecutionPlan<'g>,
    bindings: Vec<TensorFeedbackBinding>,
    binding_by_input: Vec<Option<usize>>,
    bank_count: usize,
}

impl<'p, 'g> TensorFeedbackPlan<'p, 'g> {
    /// Validates and constructs a feedback plan for output-to-input pairs.
    ///
    /// Every output must be requested by the selected execution plan and every input must be in
    /// its dependency closure. Their shapes must match. Each input and output can appear at most
    /// once, which makes every carried value's source unambiguous.
    ///
    /// # Errors
    ///
    /// Returns a graph/value error when a pair does not identify a selected output and input,
    /// when a value is repeated, or when the paired shapes differ.
    pub fn new(
        plan: &'p TensorExecutionPlan<'g>,
        pairs: &[(ValueId, ValueId)],
    ) -> Result<Self, TensorError> {
        Self::new_with_bank_count(plan, pairs, 2)
    }

    /// Validates feedback bindings and configures an N-bank ring.
    ///
    /// At least two banks are required so an iteration never overwrites the prior step's value
    /// while reading it.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::FeedbackRequiresMultipleBanks`] for one bank, or a graph/value
    /// error when a pair does not identify a selected output and input, when a value is repeated,
    /// or when the paired shapes differ.
    pub fn new_with_banks(
        plan: &'p TensorExecutionPlan<'g>,
        pairs: &[(ValueId, ValueId)],
        bank_count: NonZeroUsize,
    ) -> Result<Self, TensorError> {
        Self::new_with_bank_count(plan, pairs, bank_count.get())
    }

    fn new_with_bank_count(
        plan: &'p TensorExecutionPlan<'g>,
        pairs: &[(ValueId, ValueId)],
        bank_count: usize,
    ) -> Result<Self, TensorError> {
        if bank_count < 2 {
            return Err(TensorError::FeedbackRequiresMultipleBanks(bank_count));
        }
        let graph = plan.graph;
        let mut bindings = Vec::with_capacity(pairs.len());
        let mut binding_by_input = vec![None; plan.node_order().len()];
        for &(output, input) in pairs {
            if !plan.output_values().contains(&output) || !plan.input_values().contains(&input) {
                return Err(TensorError::UnknownValue(
                    if plan.output_values().contains(&output) {
                        input
                    } else {
                        output
                    },
                ));
            }
            if bindings
                .iter()
                .any(|binding: &TensorFeedbackBinding| binding.output == output)
            {
                return Err(TensorError::DuplicateOutput(output));
            }
            if bindings.iter().any(|binding| binding.input == input) {
                return Err(TensorError::DuplicateInput(input));
            }
            let output_shape = graph.shape(output)?;
            let input_shape = graph.shape(input)?;
            if output_shape != input_shape {
                return Err(TensorError::ShapeMismatch {
                    left: output_shape.to_vec(),
                    right: input_shape.to_vec(),
                });
            }
            let binding_index = bindings.len();
            let Some(input_index) = plan.index_of(input) else {
                return Err(TensorError::UnknownValue(input));
            };
            binding_by_input[input_index] = Some(binding_index);
            bindings.push(TensorFeedbackBinding { output, input });
        }
        Ok(Self {
            plan,
            bindings,
            binding_by_input,
            bank_count,
        })
    }

    /// The exact selected execution plan this feedback schedule was validated against.
    #[must_use]
    pub const fn execution_plan(&self) -> &'p TensorExecutionPlan<'g> {
        self.plan
    }

    /// Stable feedback pairs in caller-provided order.
    #[must_use]
    pub fn bindings(&self) -> &[TensorFeedbackBinding] {
        &self.bindings
    }

    /// Number of banks in this schedule's rotation ring.
    #[must_use]
    pub const fn bank_count(&self) -> usize {
        self.bank_count
    }

    /// Returns the bank that receives outputs produced at `step`.
    #[must_use]
    pub const fn output_bank(&self, step: usize) -> usize {
        step % self.bank_count
    }

    /// Returns the bank containing feedback consumed at `step`, if `step > 0`.
    #[must_use]
    pub const fn previous_output_bank(&self, step: usize) -> Option<usize> {
        if step == 0 {
            None
        } else {
            Some((step - 1) % self.bank_count)
        }
    }

    /// Resolves the source for a selected graph input at `step`.
    ///
    /// Unmapped inputs remain caller supplied on every step. Mapped inputs read the matching
    /// output from the previous step, and the bank rotation is provided by this plan.
    #[must_use]
    pub fn input_source(&self, input: ValueId, step: usize) -> Option<TensorFeedbackInput> {
        if self.plan.index_of(input).is_none() || !self.plan.input_values().contains(&input) {
            return None;
        }
        let index = self.plan.index_of(input)?;
        match self.binding_by_input[index].and_then(|binding| self.bindings.get(binding)) {
            Some(binding) if step > 0 => Some(TensorFeedbackInput::PreviousOutput {
                input,
                output: binding.output,
                bank: (step - 1) % self.bank_count,
            }),
            _ => Some(TensorFeedbackInput::Initial { input }),
        }
    }

    /// Applies the precomputed schedule for one step without allocating.
    pub fn for_each_input(&self, step: usize, mut visit: impl FnMut(TensorFeedbackInput)) {
        for &input in self.plan.input_values() {
            if let Some(source) = self.input_source(input, step) {
                visit(source);
            }
        }
    }
}

impl<'g> TensorExecutionPlan<'g> {
    /// Adds output-to-input iteration feedback to this selected plan.
    ///
    /// # Errors
    ///
    /// Returns a graph/value error when a pair is outside this plan, is repeated, or has unequal
    /// input and output shapes.
    pub fn feedback_plan<'p>(
        &'p self,
        pairs: &[(ValueId, ValueId)],
    ) -> Result<TensorFeedbackPlan<'p, 'g>, TensorError> {
        TensorFeedbackPlan::new(self, pairs)
    }

    /// Adds output-to-input iteration feedback with an explicit ring size.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::FeedbackRequiresMultipleBanks`] for one bank, or a graph/value
    /// error when a pair is outside this plan, is repeated, or has unequal input and output
    /// shapes.
    pub fn feedback_plan_with_banks<'p>(
        &'p self,
        pairs: &[(ValueId, ValueId)],
        bank_count: NonZeroUsize,
    ) -> Result<TensorFeedbackPlan<'p, 'g>, TensorError> {
        TensorFeedbackPlan::new_with_banks(self, pairs, bank_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Graph;

    #[test]
    fn feedback_validates_selected_values_and_schedules_two_banks() {
        let mut graph = Graph::default();
        let state = graph.input(vec![2]).unwrap();
        let increment = graph.constant(crate::Tensor::new(vec![2], vec![1.0, 1.0]).unwrap());
        let next = graph.add(state, increment).unwrap();
        let plan = graph.execution_plan_for_outputs(&[next]).unwrap();
        let feedback = plan.feedback_plan(&[(next, state)]).unwrap();

        assert_eq!(feedback.output_bank(0), 0);
        assert_eq!(feedback.output_bank(1), 1);
        assert_eq!(feedback.output_bank(2), 0);
        assert_eq!(feedback.previous_output_bank(0), None);
        assert_eq!(feedback.previous_output_bank(1), Some(0));
        assert_eq!(
            feedback.input_source(state, 0),
            Some(TensorFeedbackInput::Initial { input: state })
        );
        assert_eq!(
            feedback.input_source(state, 1),
            Some(TensorFeedbackInput::PreviousOutput {
                input: state,
                output: next,
                bank: 0,
            })
        );
        assert_eq!(feedback.execution_plan().output_values(), &[next]);

        let mut visited = Vec::new();
        feedback.for_each_input(2, |source| visited.push(source));
        assert_eq!(
            visited,
            [TensorFeedbackInput::PreviousOutput {
                input: state,
                output: next,
                bank: 1
            }]
        );
    }

    #[test]
    fn feedback_rejects_shape_mismatch_duplicate_bindings_and_unselected_output() {
        let mut graph = Graph::default();
        let state = graph.input(vec![2]).unwrap();
        let other_state = graph.input(vec![3]).unwrap();
        let increment = graph.constant(crate::Tensor::new(vec![2], vec![0.0; 2]).unwrap());
        let next = graph.add(state, increment).unwrap();
        let other_increment = graph.constant(crate::Tensor::new(vec![3], vec![0.0; 3]).unwrap());
        let other_output = graph.add(other_state, other_increment).unwrap();
        let unselected = graph.constant(crate::Tensor::scalar(1.0));
        let plan = graph
            .execution_plan_for_outputs(&[next, other_output])
            .unwrap();
        assert_eq!(
            plan.feedback_plan(&[(next, state)])
                .unwrap()
                .bindings()
                .len(),
            1
        );
        assert_eq!(
            plan.feedback_plan(&[(next, other_state)]).unwrap_err(),
            TensorError::ShapeMismatch {
                left: vec![2],
                right: vec![3]
            }
        );
        assert_eq!(
            plan.feedback_plan(&[(next, state), (next, state)])
                .unwrap_err(),
            TensorError::DuplicateOutput(next)
        );
        assert_eq!(
            plan.feedback_plan(&[(unselected, state)]).unwrap_err(),
            TensorError::UnknownValue(unselected)
        );
    }

    #[test]
    fn feedback_supports_n_bank_rotation_and_rejects_one_bank() {
        let mut graph = Graph::default();
        let state = graph.input(vec![1]).unwrap();
        let increment = graph.constant(crate::Tensor::new(vec![1], vec![1.0]).unwrap());
        let next = graph.add(state, increment).unwrap();
        let plan = graph.execution_plan_for_outputs(&[next]).unwrap();
        let one = NonZeroUsize::new(1).unwrap();
        assert_eq!(
            plan.feedback_plan_with_banks(&[(next, state)], one)
                .unwrap_err(),
            TensorError::FeedbackRequiresMultipleBanks(1)
        );

        let three = NonZeroUsize::new(3).unwrap();
        let feedback = plan
            .feedback_plan_with_banks(&[(next, state)], three)
            .unwrap();
        assert_eq!(feedback.bank_count(), 3);
        assert_eq!(
            (0..7)
                .map(|step| feedback.output_bank(step))
                .collect::<Vec<_>>(),
            [0, 1, 2, 0, 1, 2, 0]
        );
        assert_eq!(feedback.previous_output_bank(0), None);
        assert_eq!(feedback.previous_output_bank(1), Some(0));
        assert_eq!(feedback.previous_output_bank(2), Some(1));
        assert_eq!(feedback.previous_output_bank(3), Some(2));
        assert_eq!(
            feedback.input_source(state, 4),
            Some(TensorFeedbackInput::PreviousOutput {
                input: state,
                output: next,
                bank: 0
            })
        );
    }
}
