//! Explicit caller-owned SDK boundary accounting; no driver-internal allocator observations.
#[rustfmt::skip]
use fusion_pcu::insights::{InsightClock,InsightScopes,InsightStamp};
#[rustfmt::skip]
use std::{cell::RefCell,rc::Rc};

/// A clock for count-only ledgers. Vulkan API accounting never samples it.
#[derive(Clone, Copy, Debug, Default)]
pub struct PcuVulkanCountOnlyClock;
impl InsightClock for PcuVulkanCountOnlyClock {
    fn stamp(&mut self) -> InsightStamp {
        InsightStamp::default()
    }
}
/// Fixed caller-owned ledger; SDK counters start after the core point vocabulary.
pub type PcuVulkanApiInsights = InsightScopes<PcuVulkanCountOnlyClock, 96, 1>;

/// Provider-visible attempted SDK operations; allocator internals are outside this vocabulary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum PcuVulkanApiPoint {
    LoaderLoad = 32,
    InstanceVersion,
    CreateInstance,
    DestroyInstance,
    EnumeratePhysicalDevices,
    PhysicalDeviceProperties,
    QueueFamilyProperties,
    PhysicalDeviceFeatures,
    DeviceExtensions,
    CreateDevice,
    DestroyDevice,
    GetDeviceQueue,
    MemoryProperties,
    CreateBuffer,
    DestroyBuffer,
    BufferMemoryRequirements,
    BufferMemoryRequirements2,
    AllocateMemory,
    FreeMemory,
    BindBufferMemory,
    MapMemory,
    UnmapMemory,
    CreateShaderModule,
    DestroyShaderModule,
    CreateDescriptorSetLayout,
    DestroyDescriptorSetLayout,
    CreatePipelineLayout,
    DestroyPipelineLayout,
    CreateComputePipelines,
    DestroyPipeline,
    CreateDescriptorPool,
    DestroyDescriptorPool,
    AllocateDescriptorSets,
    UpdateDescriptorSets,
    CreateCommandPool,
    DestroyCommandPool,
    AllocateCommandBuffers,
    BeginCommandBuffer,
    EndCommandBuffer,
    BindPipeline,
    BindDescriptorSets,
    Dispatch,
    CopyBuffer,
    PipelineBarrier,
    PipelineBarrier2,
    CreateFence,
    DestroyFence,
    ResetFences,
    QueueSubmit,
    QueueSubmit2,
    WaitForFences,
    DeviceWaitIdle,
    PhysicalDeviceFeatures2,
    ResetCommandBuffer,
}
impl PcuVulkanApiPoint {
    /// Ledger slot for this exact SDK boundary.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
}
std::thread_local! {
    static ATTACHMENT: RefCell<Option<Rc<PcuVulkanApiInsights>>> = const { RefCell::new(None) };
}
pub struct Restore(Option<Rc<PcuVulkanApiInsights>>);
impl Drop for Restore {
    fn drop(&mut self) {
        ATTACHMENT.with(|slot| {
            slot.replace(self.0.take());
        });
    }
}
pub fn install(ledger: Option<Rc<PcuVulkanApiInsights>>) -> Restore {
    Restore(ATTACHMENT.with(|slot| slot.replace(ledger)))
}
pub fn count_retained(ledger: Option<&PcuVulkanApiInsights>, point: PcuVulkanApiPoint) {
    if let Some(ledger) = ledger {
        ledger.count(point.index(), 1);
    }
}

/// Attaches a ledger to Vulkan owners constructed by `operation` on this thread.
///
/// No discovery or clock sample occurs here. Nesting and unwinding restore the previous
/// attachment. Constructed owners retain their original ledger after this scope ends; cached
/// owners do not switch ledgers. Clear the facade invocation cache before attaching to ordinary
/// calls to remove invocation hits; separately retained device owners still keep their original
/// ledger. This attaches only newly constructed owners. Earlier discovery is not counted.
pub fn with_api_insights_scope<R>(
    ledger: Rc<PcuVulkanApiInsights>,
    operation: impl FnOnce() -> R,
) -> R {
    let guard = install(Some(ledger));
    let result = operation();
    drop(guard);
    result
}
pub fn attached() -> Option<Rc<PcuVulkanApiInsights>> {
    ATTACHMENT.with(|slot| slot.borrow().as_ref().map(Rc::clone))
}
pub fn count_attached(point: PcuVulkanApiPoint) {
    ATTACHMENT.with(|slot| {
        if let Some(ledger) = slot.borrow().as_ref() {
            ledger.count(point.index(), 1);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nested_and_unwinding_scopes_restore_the_original_attachment() {
        let outer = Rc::new(PcuVulkanApiInsights::new(PcuVulkanCountOnlyClock));
        let inner = Rc::new(PcuVulkanApiInsights::new(PcuVulkanCountOnlyClock));
        assert!(attached().is_none());
        with_api_insights_scope(Rc::clone(&outer), || {
            count_attached(PcuVulkanApiPoint::CreateBuffer);
            let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                with_api_insights_scope(Rc::clone(&inner), || {
                    count_attached(PcuVulkanApiPoint::AllocateMemory);
                    panic!("attachment unwind control");
                });
            }));
            assert!(failed.is_err());
            assert!(Rc::ptr_eq(&attached().unwrap(), &outer));
            count_attached(PcuVulkanApiPoint::CreateBuffer);
        });
        assert!(attached().is_none());
        assert_eq!(
            outer.records()[PcuVulkanApiPoint::CreateBuffer.index()].count,
            2
        );
        assert_eq!(
            inner.records()[PcuVulkanApiPoint::AllocateMemory.index()].count,
            1
        );
        assert!(outer.records().iter().all(|record| record.hits == 0));
    }
}
