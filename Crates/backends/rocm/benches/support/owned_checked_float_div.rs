//! Concrete checked division profiles share the canonical measurement boundaries.
#[macro_use]
#[path = "checked_float_common.rs"]
mod common;

mod binary32 {
    define_checked_float_benchmark!(f32, u32, FLOAT32, "owned_checked_f32_div", 0xf114, 0x3380_0000, 0x0080_0000, Div, /);
}
mod binary64 {
    define_checked_float_benchmark!(f64, u64, FLOAT64, "owned_checked_f64_div", 0xf118, 0x3ca0_0000_0000_0000, 0x0010_0000_0000_0000, Div, /);
}

pub fn run(criterion: &mut criterion::Criterion) -> Result<(), Box<dyn std::error::Error>> {
    binary32::run(criterion)?;
    binary64::run(criterion)
}
