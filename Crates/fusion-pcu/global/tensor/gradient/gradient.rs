//! Cold bounded reverse-mode source capture; no host derivative or warm IR work.
#[rustfmt::skip]
use super::{
    PcuExecutionError,
    PcuScalar,
    PcuTensorGraphCapture,
    PcuTensorGraphValue,
};
#[cfg(feature = "tensor")]
use core::marker::PhantomData;
#[cfg(feature = "tensor")]
use alloc::vec;

impl PcuTensorGraphCapture {
    /// Capture one shared reverse pass for an ordered set of MSE derivatives.
    ///
    /// The union of requested derivative paths is differentiated once. Unrequested
    /// branches cannot introduce arithmetic faults, and generated operations retain
    /// their own forward source policy. Repeating an already captured group reuses
    /// its logical results. This does no warm lowering or host derivative work.
    ///
    /// # Errors
    /// Rejects empty groups, foreign/disconnected targets and unsupported bounded
    /// differentiation contracts. All target identities are checked before mutation.
    #[doc(hidden)]
    #[cfg_attr(not(feature = "tensor"), allow(clippy::missing_const_for_fn))]
    pub fn gradients<T: PcuScalar, const M: usize>(
        &mut self,
        loss: PcuTensorGraphValue<T>,
        targets: [PcuTensorGraphValue<T>; M],
    ) -> Result<[PcuTensorGraphValue<T>; M], PcuExecutionError> {
        #[cfg(feature = "tensor")]
        {
            if M == 0 {
                return Err(PcuExecutionError::InvalidTensorSourcePlan);
            }
            self.validate_value(loss)?;
            let mut indices = [0; M];
            for (index, target) in targets.iter().copied().enumerate() {
                self.validate_value(target)?;
                indices[index] = self
                    .graph
                    .nodes()
                    .position(|node| node.value == target.value.erase())
                    .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?;
            }
            let loss_id = loss.value.erase();
            let cached = indices.map(|index| {
                self.gradient_sets
                    .iter()
                    .find(|(root, _)| *root == loss_id)
                    .and_then(|(_, gradients)| gradients.get(index))
                    .copied()
                    .flatten()
            });
            let values = if cached.iter().all(Option::is_some) {
                cached.map(|value| value.expect("the complete cached group was checked"))
            } else {
                let targets = targets.map(|target| target.value.erase());
                let values: [_; M] = self
                    .graph
                    .backward_mse_for_targets(loss_id, &targets)
                    .map_err(crate::global::tensor_build_error)?
                    .try_into()
                    .map_err(|_| PcuExecutionError::InvalidTensorSourcePlan)?;
                let required = indices.iter().copied().max().expect("nonempty group") + 1;
                let gradients = if let Some(position) = self
                    .gradient_sets
                    .iter()
                    .position(|(root, _)| *root == loss_id)
                {
                    &mut self.gradient_sets[position].1
                } else {
                    self.gradient_sets.push((loss_id, vec![None; required]));
                    &mut self
                        .gradient_sets
                        .last_mut()
                        .expect("inserted gradient group")
                        .1
                };
                gradients.resize(gradients.len().max(required), None);
                for (index, value) in indices.into_iter().zip(values) {
                    gradients[index] = Some(value);
                }
                values
            };
            let mut outputs = [None; M];
            for (output, value) in outputs.iter_mut().zip(values) {
                *output = Some(PcuTensorGraphValue {
                    value: self
                        .graph
                        .typed_view::<T>(value)
                        .map_err(crate::global::tensor_build_error)?,
                    capture_id: self.capture_id,
                    marker: PhantomData,
                });
            }
            Ok(outputs.map(|value| value.expect("every typed gradient was constructed")))
        }
        #[cfg(not(feature = "tensor"))]
        {
            let _ = (loss, targets);
            Err(PcuExecutionError::TensorExecutionUnavailable)
        }
    }

    /// Captures the derivative of an MSE loss with respect to an existing graph value.
    ///
    /// The current first-derivative profile is the existing F32/F64
    /// `Graph::backward_mse_for` contract: Add/Sub/Mul, `ReLU`, transpose-aware `MatMul`
    /// and SGD in the loss closure. Division, nested MSE and `ReLU` backward
    /// nodes reject; this is not a general higher-derivative contract.
    /// Derived nodes retain their forward node's numerical policy;
    /// this call does not overwrite it with the caller's current defaults.
    /// Only branches needed for the requested derivative are generated. Forward
    /// checked effects remain observable; unrequested derivatives cannot add faults.
    /// Repeated requests for the same loss/target reuse the captured result.
    ///
    /// # Errors
    /// Rejects foreign capture values, unsupported loss/operation/type, absent
    /// derivatives, shape failures or an unavailable tensor execution surface.
    #[doc(hidden)]
    #[cfg_attr(not(feature = "tensor"), allow(clippy::missing_const_for_fn))] // The disabled stub must retain the allocating cold transform's callable signature.
    pub fn gradient<T: PcuScalar>(
        &mut self,
        loss: PcuTensorGraphValue<T>,
        with_respect_to: PcuTensorGraphValue<T>,
    ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError> {
        #[cfg(feature = "tensor")]
        {
            self.validate_value(loss)?;
            self.validate_value(with_respect_to)?;
            let loss_id = loss.value.erase();
            let target = with_respect_to.value.erase();
            let target_index = self
                .graph
                .nodes()
                .position(|node| node.value == target)
                .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?;
            let cached = self
                .gradient_sets
                .iter()
                .find(|(root, _)| *root == loss_id)
                .and_then(|(_, gradients)| gradients.get(target_index))
                .copied()
                .flatten();
            let value = if let Some(value) = cached {
                value
            } else {
                let value = self
                    .graph
                    .backward_mse_for(loss_id, target)
                    .map_err(crate::global::tensor_build_error)?;
                if let Some((_, gradients)) = self
                    .gradient_sets
                    .iter_mut()
                    .find(|(root, _)| *root == loss_id)
                {
                    gradients.resize(gradients.len().max(target_index + 1), None);
                    gradients[target_index] = Some(value);
                } else {
                    let mut gradients = vec![None; target_index + 1];
                    gradients[target_index] = Some(value);
                    self.gradient_sets.push((loss_id, gradients));
                }
                value
            };
            let value = self
                .graph
                .typed_view::<T>(value)
                .map_err(crate::global::tensor_build_error)?;
            Ok(PcuTensorGraphValue {
                value,
                capture_id: self.capture_id,
                marker: PhantomData,
            })
        }
        #[cfg(not(feature = "tensor"))]
        {
            let _ = (loss, with_respect_to);
            Err(PcuExecutionError::TensorExecutionUnavailable)
        }
    }
}

#[cfg(all(test, feature = "tensor"))]
#[path = "tests/tests.rs"]
mod tests;
