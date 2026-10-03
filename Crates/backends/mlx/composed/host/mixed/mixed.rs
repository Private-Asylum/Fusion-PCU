//! Actual initial snapshots only; all preflights precede any host staging or GPU work.
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuHostDispatchError,
    PcuScalarType,
};
#[rustfmt::skip]
use crate::{
    MlxEncodedArray,
    MlxError,
    MlxHostKernelError,
    MlxCheckedMapCompletion,
};
use super::MlxPreparedCheckedMapHostKernel;

/// One actual original-input snapshot. Unused declarations and store-before-load banks
/// do not occupy this slice. Read-write source bindings may supply immutable old snapshots.
#[derive(Clone, Copy)]
pub enum MlxCheckedMapInput<'a> {
    HostBytes {
        target: PcuBindingRef,
        scalar: PcuScalarType,
        bytes: &'a [u8],
    },
    Resident {
        target: PcuBindingRef,
        array: &'a MlxEncodedArray,
    },
}
impl MlxCheckedMapInput<'_> {
    const fn target(self) -> PcuBindingRef {
        match self {
            Self::HostBytes { target, .. } | Self::Resident { target, .. } => target,
        }
    }
}
impl MlxPreparedCheckedMapHostKernel {
    /// Completes private immutable output siblings from actual unique host/resident snapshots.
    /// Exact minimum resident shapes are frozen cold. Host inputs stage only their logical
    /// initial prefix; larger host tails remain valid. Larger resident capacity requires a
    /// separately qualified additive constructor and is not accepted by this candidate. No existing owner is mutated or published.
    /// # Errors
    /// Rejects missing/duplicate/unexpected roles, dtype/span/session or quarantine before
    /// staging; returns native or checked cleanup error without exposing incomplete outputs.
    pub fn execute_inputs(
        &mut self,
        inputs: &[MlxCheckedMapInput<'_>],
    ) -> Result<MlxCheckedMapCompletion, MlxHostKernelError> {
        self.may_have_written = false;
        self.session()
            .validate_access_available()
            .map_err(PcuHostDispatchError::Backend)?;
        let mut ordered = [None; 4];
        let width = usize::from(self.plan().value_type().scalar_type().bit_width()) / 8;
        for input in inputs {
            let target = input.target();
            let slot = self
                .kernel
                .input_bindings()
                .iter()
                .position(|binding| *binding == target)
                .ok_or(PcuHostDispatchError::Unexpected(target))?;
            if ordered[slot].replace(*input).is_some() {
                return Err(PcuHostDispatchError::Duplicate(target));
            }
            let count = self.prepared_input_element_counts()[slot];
            match *input {
                MlxCheckedMapInput::HostBytes { scalar, bytes, .. } => {
                    if scalar != self.plan().value_type().scalar_type() {
                        return Err(PcuHostDispatchError::TypeMismatch(target));
                    }
                    if bytes.len() < count * width {
                        return Err(PcuHostDispatchError::BufferTooSmall(target));
                    }
                }
                MlxCheckedMapInput::Resident { array, .. } => {
                    if array.scalar_type() != self.plan().value_type().scalar_type() {
                        return Err(PcuHostDispatchError::TypeMismatch(target));
                    }
                    if array.element_count() < count {
                        return Err(PcuHostDispatchError::BufferTooSmall(target));
                    }
                    if array.element_count() != self.prepared_input_element_counts()[slot] {
                        return Err(PcuHostDispatchError::Backend(MlxError::InvalidExtent));
                    }
                    if !array.same_session(self.session()) {
                        return Err(PcuHostDispatchError::Backend(MlxError::ForeignSession));
                    }
                    array
                        .validate_access_available()
                        .map_err(PcuHostDispatchError::Backend)?;
                }
            }
        }
        for (slot, &binding) in self.kernel.input_bindings().iter().enumerate() {
            if ordered[slot].is_none() {
                return Err(PcuHostDispatchError::Missing(binding));
            }
        }
        self.complete_inputs(&ordered)
    }
    fn complete_inputs(
        &self,
        ordered: &[Option<MlxCheckedMapInput<'_>>; 4],
    ) -> Result<MlxCheckedMapCompletion, MlxHostKernelError> {
        let mut staged: [Option<MlxEncodedArray>; 4] = std::array::from_fn(|_| None);
        let width = usize::from(self.plan().value_type().scalar_type().bit_width()) / 8;
        let count = self.kernel.input_bindings().len();
        for (slot, input) in ordered.iter().take(count).enumerate() {
            if let Some(MlxCheckedMapInput::HostBytes { scalar, bytes, .. }) = input {
                let extent = self.prepared_input_element_counts()[slot];
                staged[slot] = Some(
                    self.session()
                        .upload_transport_bytes(*scalar, extent, &bytes[..extent * width])
                        .map_err(PcuHostDispatchError::Backend)?,
                );
            }
        }
        let array = |slot: usize| match ordered[slot] {
            Some(MlxCheckedMapInput::HostBytes { .. }) => staged[slot].as_ref(),
            Some(MlxCheckedMapInput::Resident { array, .. }) => Some(array),
            None => None,
        };
        let first = array(0).ok_or(PcuHostDispatchError::Backend(MlxError::InvalidExtent))?;
        let mut arrays = [first; 4];
        for (slot, array) in arrays.iter_mut().take(count).enumerate() {
            *array = match ordered[slot] {
                Some(MlxCheckedMapInput::HostBytes { .. }) => staged[slot].as_ref(),
                Some(MlxCheckedMapInput::Resident { array, .. }) => Some(array),
                None => None,
            }
            .ok_or(PcuHostDispatchError::Backend(MlxError::InvalidExtent))?;
        }
        let completed = self
            .kernel
            .execute(&arrays[..count])
            .map_err(PcuHostDispatchError::Backend)?;
        for input in staged.into_iter().flatten() {
            input.release().map_err(PcuHostDispatchError::Backend)?;
        }
        Ok(completed)
    }
}
