//! Native mixed binders keep arithmetic outputs private until exact-byte commit.
#[path = "commit/commit.rs"]
pub(super) mod commit;
pub use commit::WriteState;
#[rustfmt::skip]
use super::{
    NativeResources,
    VulkanPreparedMap,
    PcuVulkanError,
    PcuExecutionFault,
    StatusPolicy,
    mem,
    ptr,
    vk,
    vk_try,
    submit_and_wait_measured,
    storage_buffer_write,
};
use crate::ffi::VulkanOwnedBuffer;

#[derive(Clone, Copy)]
pub enum Input<'a> {
    Host(&'a [u8]),
    Owned(&'a VulkanOwnedBuffer),
}
pub enum Output<'a> {
    Host(&'a mut [u8]),
    Owned(&'a mut VulkanOwnedBuffer),
}
impl Input<'_> {
    const fn byte_len(self) -> usize {
        match self {
            Self::Host(bytes) => bytes.len(),
            Self::Owned(owner) => owner.byte_len(),
        }
    }
}
impl Output<'_> {
    const fn byte_len(&self) -> usize {
        match self {
            Self::Host(bytes) => bytes.len(),
            Self::Owned(owner) => owner.byte_len(),
        }
    }
}
impl<const N: usize, const CHECKED: bool, const OUTPUTS: usize>
    VulkanPreparedMap<N, CHECKED, OUTPUTS>
{
    pub(super) fn enable_mixed(&mut self) -> Result<(), PcuVulkanError> {
        if self.commit.is_none() {
            self.commit = Some(commit::Commit::new(&self.device)?);
        }
        Ok(())
    }
    fn quarantine(&mut self, state: &mut WriteState) {
        state.completion_uncertain = true;
        self.device.poisoned.set(true);
        // NativeResources are raw owners. Removing them deliberately retains all mappings,
        // commands, pipeline and descriptor handles; leaked Rc keeps their full device alive.
        self.resources.take();
        mem::forget(std::rc::Rc::clone(&self.device));
    }
    pub(super) fn call_mixed(
        &mut self,
        inputs: &[Input<'_>],
        outputs: &mut [Output<'_>],
        state: &mut WriteState,
    ) -> Result<(), PcuVulkanError> {
        validate(self, inputs, outputs)?;
        let resources = self.resources.as_ref().ok_or(PcuVulkanError::Quarantined)?;
        for (index, input) in inputs.iter().enumerate() {
            if let Input::Host(bytes) = input {
                unsafe {
                    // SAFETY: Full preflight and terminal earlier work prove the live mapping
                    // spans this logical prefix; caller RAM never becomes a GPU pointer.
                    ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        resources.buffers[index].mapped,
                        self.lengths[index],
                    );
                }
            }
        }
        record(self, inputs)?;
        let resources = self.resources.as_ref().ok_or(PcuVulkanError::Quarantined)?;
        vk_try("reset mixed compute fence", unsafe {
            // SAFETY: Every prior compute or transfer is terminal; this fence is nonpending.
            vk_api_owner!(
                self.device,
                ResetFences,
                self.device.device.reset_fences(&[resources.fence])
            )
        })?;
        if let Err(error) =
            submit_and_wait_measured(&self.device, resources.command, resources.fence, None)
        {
            if matches!(error, PcuVulkanError::CompletionUnknown) {
                self.quarantine(state);
            }
            return Err(error);
        }
        let recovered = scan::<N, CHECKED>(resources, self.extent, self.status_policy)?;
        let mut regions: [Option<commit::Region<'_>>; 2] = [None, None];
        let mut count = 0;
        for (index, output) in outputs.iter().enumerate() {
            if let Output::Owned(owner) = output {
                regions[count] = Some(commit::Region {
                    source: resources.buffers[Self::OUTPUT + index].buffer,
                    target: owner,
                    bytes: self.lengths[Self::OUTPUT + index],
                });
                count += 1;
            }
        }
        let complete = match count {
            0 => Ok(()),
            1 => self
                .commit
                .as_mut()
                .ok_or(PcuVulkanError::InvalidArguments)?
                .execute(&[regions[0].take().expect("one resident output")], state),
            2 => self
                .commit
                .as_mut()
                .ok_or(PcuVulkanError::InvalidArguments)?
                .execute(
                    &[
                        regions[0].take().expect("first resident output"),
                        regions[1].take().expect("second resident output"),
                    ],
                    state,
                ),
            _ => unreachable!("cold mixed schema has at most two outputs"),
        };
        if let Err(error) = complete {
            if state.completion_uncertain {
                self.quarantine(state);
            }
            return Err(error);
        }
        // Host outputs publish only after every resident output transfer completed as well.
        let resources = self.resources.as_ref().ok_or(PcuVulkanError::Quarantined)?;
        for (index, output) in outputs.iter_mut().enumerate() {
            if let Output::Host(bytes) = output {
                state.may_have_written = true;
                unsafe {
                    // SAFETY: Complete status validation and terminal public transfer precede
                    // every initialized prefix; caller tails and native padding stay untouched.
                    ptr::copy_nonoverlapping(
                        resources.buffers[Self::OUTPUT + index].mapped,
                        bytes.as_mut_ptr(),
                        self.lengths[Self::OUTPUT + index],
                    );
                }
            }
        }
        recovered.map_or(Ok(()), |fault| Err(PcuVulkanError::Fault(fault)))
    }
}
fn validate<const N: usize, const CHECKED: bool, const OUTPUTS: usize>(
    plan: &VulkanPreparedMap<N, CHECKED, OUTPUTS>,
    inputs: &[Input<'_>],
    outputs: &[Output<'_>],
) -> Result<(), PcuVulkanError> {
    if plan.device.poisoned.get() {
        return Err(PcuVulkanError::Quarantined);
    }
    if plan.commit.is_none()
        || inputs.len() != VulkanPreparedMap::<N, CHECKED, OUTPUTS>::OUTPUT
        || outputs.len() != OUTPUTS
        || OUTPUTS > 2
        || inputs.len() > 2
    {
        return Err(PcuVulkanError::InvalidArguments);
    }
    for (input, required) in inputs.iter().zip(plan.lengths) {
        if input.byte_len() < required {
            return Err(PcuVulkanError::InvalidArguments);
        }
        if let Input::Owned(owner) = input {
            owner.validate_access_available()?;
            if !owner.same_session(&plan.device) {
                return Err(PcuVulkanError::InvalidArguments);
            }
        }
    }
    for (index, output) in outputs.iter().enumerate() {
        let required = plan.lengths[VulkanPreparedMap::<N, CHECKED, OUTPUTS>::OUTPUT + index];
        if output.byte_len() < required {
            return Err(PcuVulkanError::InvalidArguments);
        }
        if let Output::Owned(owner) = output {
            owner.validate_access_available()?;
            if !owner.same_session(&plan.device) || inputs.iter().any(|input|matches!(input,Input::Owned(source) if source.descriptor_info().buffer==owner.descriptor_info().buffer)) || outputs[..index].iter().any(|prior|matches!(prior,Output::Owned(target) if target.descriptor_info().buffer==owner.descriptor_info().buffer)) {return Err(PcuVulkanError::InvalidArguments);}
        }
    }
    Ok(())
}
fn scan<const N: usize, const CHECKED: bool>(
    resources: &NativeResources<N>,
    extent: usize,
    policy: StatusPolicy,
) -> Result<Option<PcuExecutionFault>, PcuVulkanError> {
    if !CHECKED {
        return Ok(None);
    }
    policy.scan(extent, |invocation| {
        // SAFETY: Each complete U32 writer and compute-to-host barrier precedes terminal
        // fence wait; cold geometry proves the whole initialized diagnostic span.
        Ok(unsafe {
            resources.buffers[N - 1]
                .mapped
                .add(invocation * 4)
                .cast::<u32>()
                .read_unaligned()
        })
    })
}
#[path = "record/record.rs"]
mod recording;
use recording::record;

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
