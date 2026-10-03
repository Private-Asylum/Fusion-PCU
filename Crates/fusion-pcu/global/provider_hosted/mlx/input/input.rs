//! Full resident input extents, distinct from the kernel's minimum read spans.
#[rustfmt::skip]
use crate::{
    PcuBindingAccess,
    PcuBindingRef,
};
use super::super::PcuExecutionError;

/// Native MLX shapes are frozen during preparation. A host argument uses the IR
/// read span; an existing resident array retains its complete allocation shape.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(in crate::global::provider_hosted) struct MlxInputLayout {
    inputs: [Option<(PcuBindingRef, usize, PcuBindingAccess)>; 4],
}

impl MlxInputLayout {
    pub(in crate::global::provider_hosted) const fn new() -> Self {
        Self { inputs: [None; 4] }
    }
    pub(in crate::global::provider_hosted) fn record(
        &mut self,
        binding: PcuBindingRef,
        count: usize,
    ) -> Result<(), PcuExecutionError> {
        self.record_access(binding, count, PcuBindingAccess::ReadOnly)
    }

    /// An exclusive borrow can still supply old contents to ordered transport.
    /// Recording its shape does not grant permission to mutate native storage.
    pub(in crate::global::provider_hosted) fn record_mutable(
        &mut self,
        binding: PcuBindingRef,
        count: usize,
    ) -> Result<(), PcuExecutionError> {
        self.record_access(binding, count, PcuBindingAccess::ReadWrite)
    }

    fn record_access(
        &mut self,
        binding: PcuBindingRef,
        count: usize,
        access: PcuBindingAccess,
    ) -> Result<(), PcuExecutionError> {
        if self
            .inputs
            .iter()
            .flatten()
            .any(|(target, _, _)| *target == binding)
        {
            return Err(super::map_mlx_error(
                crate::PcuHostDispatchError::Duplicate(binding),
            ));
        }
        let slot = self
            .inputs
            .iter_mut()
            .find(|input| input.is_none())
            .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?;
        *slot = Some((binding, count, access));
        // Argument traversal order must not create another cold specialization.
        self.inputs
            .sort_unstable_by_key(|input| input.map(|(target, _, _)| (target.set, target.binding)));
        Ok(())
    }

    pub(super) fn extents(
        self,
        bindings: &[PcuBindingRef],
        minimum: [usize; 2],
    ) -> Result<[usize; 2], PcuExecutionError> {
        let mut extents = minimum;
        for (binding, count, access) in self.inputs.into_iter().flatten() {
            // Existing arithmetic families take readonly operands. Mutable shapes
            // belong to their separately prepared publication/prefix contracts.
            if access != PcuBindingAccess::ReadOnly {
                continue;
            }
            let slot = bindings
                .iter()
                .position(|target| *target == binding)
                .ok_or_else(|| {
                    super::map_mlx_error(crate::PcuHostDispatchError::Unexpected(binding))
                })?;
            let extent = extents
                .get_mut(slot)
                .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?;
            if count < *extent {
                return Err(super::map_mlx_error(
                    crate::PcuHostDispatchError::BufferTooSmall(binding),
                ));
            }
            *extent = count;
        }
        Ok(extents)
    }

    /// Projects only actual initial snapshots. A mutable bank stored before its
    /// first load needs no old contents and must not occupy a native input slot.
    pub(super) fn snapshot_extents<const N: usize>(
        self,
        bindings: &[PcuBindingRef],
        minimum: [usize; N],
    ) -> Result<[usize; N], PcuExecutionError> {
        if bindings.len() > N {
            return Err(PcuExecutionError::InvalidTensorSourcePlan);
        }
        let mut extents = minimum;
        for (binding, count, _) in self.inputs.into_iter().flatten() {
            if let Some(slot) = bindings.iter().position(|target| *target == binding) {
                if count < extents[slot] {
                    return Err(super::map_mlx_error(
                        crate::PcuHostDispatchError::BufferTooSmall(binding),
                    ));
                }
                extents[slot] = count;
            }
        }
        Ok(extents)
    }
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
