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
    MlxTransportCompletion,
};
use super::MlxPreparedTransportHostKernel;

/// One actual original-input snapshot. Unused declarations and store-before-load banks
/// do not occupy this slice. Read-write source bindings may supply immutable old snapshots.
#[derive(Clone, Copy)]
pub enum MlxTransportInput<'a> {
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
impl MlxTransportInput<'_> {
    const fn target(self) -> PcuBindingRef {
        match self {
            Self::HostBytes { target, .. } | Self::Resident { target, .. } => target,
        }
    }
}
impl MlxPreparedTransportHostKernel {
    /// Completes private immutable output siblings from actual unique host/resident snapshots.
    /// Exact full resident shapes are frozen cold. Host inputs stage only their logical
    /// initial prefix; larger host tails remain valid. No existing owner is mutated or published.
    /// # Errors
    /// Rejects missing/duplicate/unexpected roles, dtype/span/session or quarantine before
    /// staging; returns native or checked cleanup error without exposing incomplete outputs.
    pub fn execute_inputs(
        &mut self,
        inputs: &[MlxTransportInput<'_>],
    ) -> Result<MlxTransportCompletion, MlxHostKernelError> {
        self.may_have_written = false;
        self.session()
            .validate_access_available()
            .map_err(PcuHostDispatchError::Backend)?;
        let mut ordered = [None; 4];
        let width = usize::from(self.plan().scalar_type().bit_width()) / 8;
        for input in inputs {
            let target = input.target();
            let slot = self
                .plan()
                .inputs()
                .iter()
                .position(|r| r.binding == target)
                .ok_or(PcuHostDispatchError::Unexpected(target))?;
            if ordered[slot].replace(*input).is_some() {
                return Err(PcuHostDispatchError::Duplicate(target));
            }
            let count = usize::try_from(self.plan().inputs()[slot].minimum_initial_read_elements)
                .map_err(|_| PcuHostDispatchError::Backend(MlxError::InvalidExtent))?;
            match *input {
                MlxTransportInput::HostBytes { scalar, bytes, .. } => {
                    if scalar != self.plan().scalar_type() {
                        return Err(PcuHostDispatchError::TypeMismatch(target));
                    }
                    if self.prepared_input_element_counts()[slot] != count {
                        return Err(PcuHostDispatchError::Backend(MlxError::InvalidExtent));
                    }
                    if bytes.len() < count * width {
                        return Err(PcuHostDispatchError::BufferTooSmall(target));
                    }
                }
                MlxTransportInput::Resident { array, .. } => {
                    if array.scalar_type() != self.plan().scalar_type() {
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
        for (slot, resource) in self.plan().inputs().iter().enumerate() {
            if ordered[slot].is_none() {
                return Err(PcuHostDispatchError::Missing(resource.binding));
            }
        }
        self.complete_inputs(&ordered)
    }
    fn complete_inputs(
        &self,
        ordered: &[Option<MlxTransportInput<'_>>; 4],
    ) -> Result<MlxTransportCompletion, MlxHostKernelError> {
        let mut staged: [Option<MlxEncodedArray>; 4] = std::array::from_fn(|_| None);
        let width = usize::from(self.plan().scalar_type().bit_width()) / 8;
        let count = self.plan().inputs().len();
        for (slot, input) in ordered.iter().take(count).enumerate() {
            if let Some(MlxTransportInput::HostBytes { scalar, bytes, .. }) = input {
                let extent =
                    usize::try_from(self.plan().inputs()[slot].minimum_initial_read_elements)
                        .unwrap();
                staged[slot] = Some(
                    self.session()
                        .upload_transport_bytes(*scalar, extent, &bytes[..extent * width])
                        .map_err(PcuHostDispatchError::Backend)?,
                );
            }
        }
        let array = |slot: usize| match ordered[slot] {
            Some(MlxTransportInput::HostBytes { .. }) => staged[slot].as_ref(),
            Some(MlxTransportInput::Resident { array, .. }) => Some(array),
            None => None,
        };
        let first = array(0).ok_or(PcuHostDispatchError::Backend(MlxError::InvalidExtent))?;
        let mut arrays = [first; 4];
        for (slot, array) in arrays.iter_mut().take(count).enumerate() {
            *array = match ordered[slot] {
                Some(MlxTransportInput::HostBytes { .. }) => staged[slot].as_ref(),
                Some(MlxTransportInput::Resident { array, .. }) => Some(array),
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
