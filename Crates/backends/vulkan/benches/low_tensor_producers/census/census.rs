//! Explicit caller-owned `CountOnly` ledgers; disabled builds omit this module.
use std::{cell::RefCell, rc::Rc};
#[rustfmt::skip]
use fusion_pcu_vulkan::{PcuVulkanApiInsights,PcuVulkanApiPoint as Point,PcuVulkanCountOnlyClock,with_api_insights_scope};
std::thread_local! {static ROW:RefCell<Option<(Rc<PcuVulkanApiInsights>,Rc<PcuVulkanApiInsights>)>>=const{RefCell::new(None)};}
struct Restore(Option<(Rc<PcuVulkanApiInsights>, Rc<PcuVulkanApiInsights>)>);
impl Drop for Restore {
    fn drop(&mut self) {
        ROW.with(|slot| {
            slot.replace(self.0.take());
        });
    }
}
fn counts(ledger: &PcuVulkanApiInsights) -> [u64; 96] {
    ledger.records().map(|record| {
        assert_eq!(
            (record.hits, record.inclusive_ticks, record.exclusive_ticks),
            (0, 0, 0)
        );
        record.count
    })
}
pub fn run(operation: impl FnOnce()) {
    let provider = Rc::new(PcuVulkanApiInsights::new(PcuVulkanCountOnlyClock));
    let native = Rc::new(PcuVulkanApiInsights::new(PcuVulkanCountOnlyClock));
    let _restore =
        Restore(ROW.with(|slot| slot.replace(Some((Rc::clone(&provider), Rc::clone(&native))))));
    with_api_insights_scope(Rc::clone(&provider), || {
        super::ffi::with_api_census_scope(Rc::clone(&native), operation);
    });
    println!(
        "vulkan-low-producer-api-total/provider {:?}/native {:?}",
        counts(&provider),
        counts(&native)
    );
}
pub fn warm(name: &str, route: u32, operation: impl FnOnce()) {
    ROW.with(|slot| {
        let row = slot.borrow();
        let (provider, native) = row.as_ref().unwrap();
        let before = [counts(provider), counts(native)];
        operation();
        let after = [counts(provider), counts(native)];
        let delta: [[u64; 96]; 2] = core::array::from_fn(|bank| {
            core::array::from_fn(|point| after[bank][point] - before[bank][point])
        });
        let mut expected = [[0u64; 96]; 2];
        if route == 2 {
            for point in [Point::ResetFences, Point::QueueSubmit, Point::WaitForFences] {
                expected[1][point.index()] = 64;
            }
        } else {
            for point in [
                Point::MemoryProperties,
                Point::CreateBuffer,
                Point::DestroyBuffer,
                Point::BufferMemoryRequirements2,
                Point::AllocateMemory,
                Point::FreeMemory,
                Point::BindBufferMemory,
                Point::MapMemory,
                Point::UnmapMemory,
                Point::CopyBuffer,
            ] {
                expected[0][point.index()] = 64;
            }
            for point in [
                Point::ResetCommandBuffer,
                Point::BeginCommandBuffer,
                Point::EndCommandBuffer,
                Point::ResetFences,
                Point::QueueSubmit2,
                Point::WaitForFences,
            ] {
                expected[0][point.index()] = 192;
            }
            for point in [
                Point::UpdateDescriptorSets,
                Point::BindPipeline,
                Point::BindDescriptorSets,
                Point::Dispatch,
            ] {
                expected[0][point.index()] = 128;
            }
            expected[0][Point::PipelineBarrier.index()] = 384;
        }
        assert_eq!(delta, expected, "exact provider-visible API vector {name}");
        println!(
            "vulkan-low-producer-api/{name}:64-changing-calls provider {:?} native {:?}",
            delta[0], delta[1]
        );
    });
}
