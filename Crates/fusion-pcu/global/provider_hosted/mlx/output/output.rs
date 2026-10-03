//! Binding-specific output extents retained in the cold invocation cache.
use crate::PcuBindingRef;
use super::super::PcuExecutionError;

/// Full shapes of exclusive borrows. Ordered transport can read such a bank
/// without writing it; only the prepared kernel's actual writers publish.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(in crate::global::provider_hosted) struct MlxOutputLayout {
    outputs: [Option<(PcuBindingRef, usize)>; 4],
}

impl MlxOutputLayout {
    pub(in crate::global::provider_hosted) fn record(
        &mut self,
        binding: PcuBindingRef,
        count: usize,
    ) -> Result<(), PcuExecutionError> {
        if self
            .outputs
            .iter()
            .flatten()
            .any(|(target, _)| *target == binding)
        {
            return Err(super::map_mlx_error(
                crate::PcuHostDispatchError::Duplicate(binding),
            ));
        }
        let slot = self
            .outputs
            .iter_mut()
            .find(|output| output.is_none())
            .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?;
        *slot = Some((binding, count));
        Ok(())
    }

    pub(super) fn count(self, binding: PcuBindingRef) -> Option<usize> {
        self.outputs
            .into_iter()
            .flatten()
            .find_map(|(target, count)| (target == binding).then_some(count))
    }

    pub(super) fn validate(
        self,
        expected: [Option<(PcuBindingRef, usize)>; 2],
    ) -> Result<(), PcuExecutionError> {
        for (binding, _) in self.outputs.into_iter().flatten() {
            if !expected
                .iter()
                .flatten()
                .any(|(target, _)| *target == binding)
            {
                return Err(super::map_mlx_error(
                    crate::PcuHostDispatchError::Unexpected(binding),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
