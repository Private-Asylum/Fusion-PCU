//! Explicit benchmark-only attachment for actual handwritten SDK calls.
//! Owners retain the construction ledger. No clock, implicit attachment or warm Rc clone.
#[rustfmt::skip]
use std::{cell::RefCell,rc::Rc};
#[rustfmt::skip]
use fusion_pcu_vulkan::{PcuVulkanApiInsights,PcuVulkanApiPoint};
std::thread_local! {
    static ATTACHMENT: RefCell<Option<Rc<PcuVulkanApiInsights>>> = const { RefCell::new(None) };
}
#[allow(dead_code)] // Shared float harness does not attach this optional diagnostic.
struct Restore(Option<Rc<PcuVulkanApiInsights>>);
impl Drop for Restore {
    fn drop(&mut self) {
        ATTACHMENT.with(|slot| {
            slot.replace(self.0.take());
        });
    }
}
#[allow(dead_code)] // Shared float harness does not attach this optional diagnostic.
pub fn with_scope<T>(ledger: Rc<PcuVulkanApiInsights>, operation: impl FnOnce() -> T) -> T {
    let _restore = Restore(ATTACHMENT.with(|slot| slot.replace(Some(ledger))));
    operation()
}
pub fn current() -> Option<Rc<PcuVulkanApiInsights>> {
    ATTACHMENT.with(|slot| slot.borrow().clone())
}
pub fn count_current(point: PcuVulkanApiPoint) {
    ATTACHMENT.with(|slot| count(slot.borrow().as_deref(), point));
}
pub fn count(ledger: Option<&PcuVulkanApiInsights>, point: PcuVulkanApiPoint) {
    if let Some(ledger) = ledger {
        ledger.count(point.index(), 1);
    }
}
