//! Explicit all-integer warm SDK accounting. Disabled builds omit this entire module.
#[rustfmt::skip]
use std::{cell::RefCell,rc::Rc};
#[rustfmt::skip]
use fusion_pcu_vulkan::{PcuVulkanApiInsights,PcuVulkanApiPoint as Point,PcuVulkanCountOnlyClock,with_api_insights_scope};
std::thread_local! {
    static ROW: RefCell<Option<(Rc<PcuVulkanApiInsights>,Rc<PcuVulkanApiInsights>)>> = const { RefCell::new(None) };
}
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
        "api_total provider {:?} native {:?}",
        counts(&provider),
        counts(&native)
    );
}
pub fn warm(name: &str, route: &str, operation: impl FnOnce()) {
    ROW.with(|slot| {
        let row = slot.borrow();
        let Some((provider, native)) = row.as_ref() else {
            return;
        };
        let before = [counts(provider), counts(native)];
        operation();
        let after = [counts(provider), counts(native)];
        let mut delta = [[0_u64; 96]; 2];
        for bank in 0..2 {
            for point in 0..96 {
                delta[bank][point] = after[bank][point] - before[bank][point];
            }
        }
        let active = usize::from(route == "native_glsl");
        for (bank, row) in delta.iter().enumerate() {
            for (point, count) in row.iter().enumerate() {
                if ![
                    Point::ResetFences.index(),
                    Point::QueueSubmit.index(),
                    Point::QueueSubmit2.index(),
                    Point::WaitForFences.index(),
                ]
                .contains(&point)
                {
                    assert_eq!(
                        *count, 0,
                        "unexpected warm SDK operation {name}/{route} bank{bank} point{point}"
                    );
                }
            }
            let calls = if bank == active { 64 } else { 0 };
            assert_eq!(row[Point::ResetFences.index()], calls);
            assert_eq!(row[Point::WaitForFences.index()], calls);
            assert_eq!(
                row[Point::QueueSubmit.index()] + row[Point::QueueSubmit2.index()],
                calls
            );
        }
        println!(
            "api_census {name}/{route}:64 changing calls provider {:?} native {:?}",
            delta[0], delta[1]
        );
    });
}
