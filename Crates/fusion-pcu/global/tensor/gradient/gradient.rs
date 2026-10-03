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
