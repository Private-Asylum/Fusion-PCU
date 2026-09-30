//! Binary32 and binary64 checked-add benchmark profiles.
#[macro_use]
#[path = "checked_float_common.rs"]
mod common;

mod binary32 {
    define_checked_float_benchmark!(
        f32,
        u32,
        FLOAT32,
        "owned_checked_f32_add",
        0xf104,
        0x3380_0000,
        0x0080_0000
    );
}

mod binary64 {
    define_checked_float_benchmark!(
        f64,
        u64,
        FLOAT64,
        "owned_checked_f64_add",
        0xf108,
        0x3ca0_0000_0000_0000,
        0x0010_0000_0000_0000
    );
}

pub fn run(criterion: &mut criterion::Criterion) -> Result<(), Box<dyn std::error::Error>> {
    binary32::run(criterion)?;
    binary64::run(criterion)
}
