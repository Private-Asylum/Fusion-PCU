//! Native product/carry boundary goldens; independent U128/I128 reference arithmetic.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/helper_ordered_integer_maps/oracle/oracle.rs"]
#[allow(dead_code)] // Full benchmark bank generation is separately qualified.
mod oracle;
#[path = "../../benches/helper_ordered_integer_maps/raw/raw.rs"]
#[allow(dead_code)] // This focused gate executes only the independent handwritten boundary.
mod raw;
#[path = "../../benches/strict_matmul/selection.rs"]
mod selection;
#[path = "../../benches/helper_ordered_integer_maps/source/source.rs"]
#[allow(dead_code)] // Genuine source admission remains a separate native certificate.
mod source;
#[rustfmt::skip]
use fusion_pcu::{
    PcuDispatchKernelIr,
    PcuFloatUnderflowPolicy,
    PcuImplementationRequirements,
    PcuNumericalMode,
    PcuRangePolicy,
};
use fusion_pcu_cuda::CudaOwnedDispatchBackend;
use oracle::Format;
const N: usize = 65;
fn verify<T: Format>(backend: &CudaOwnedDispatchBackend, ir: &PcuDispatchKernelIr<'_>) -> usize {
    let input = [T::from(1); N];
    let mut control = raw::Raw::handwritten::<T, N>(backend, ir, [&input, &input]);
    // Exact limits surrounding high-word carries, the signed magnitude edge,
    // and the largest admitted 2*x*x result. Expected arithmetic uses wider integers.
    let mut values = vec![
        0,
        1,
        2,
        17,
        0xffff,
        0x1_0000,
        0xffff_ffff,
        0x1_0000_0000,
        0x1_0000_0001,
        2_147_483_647,
        2_147_483_648,
        3_037_000_498,
        3_037_000_499,
        3_037_000_500,
        1_u64 << 62,
        (1_u64 << 63) - 1,
        1_u64 << 63,
        u64::MAX - 1,
        u64::MAX,
    ];
    if T::SIGNED {
        values.extend([
            0_u64.wrapping_sub(1),
            0_u64.wrapping_sub(2_147_483_647),
            0_u64.wrapping_sub(2_147_483_648),
            0_u64.wrapping_sub(3_037_000_499),
            (1_u64 << 63) + 1,
        ]);
    }
    let mut cases = 0;
    for value in values {
        let exceptional = [T::from(value); N];
        let (word, stage, output) =
            oracle::expected(&exceptional, ir.numerical_requirements.range_policy);
        assert_eq!(
            control.probe(&exceptional),
            word,
            "{} input {value:016x}",
            T::LABEL
        );
        if word == u64::MAX || word & (1_u64 << 63) != 0 {
            control.verify(&stage, &output);
        }
        // A healthy changing retry verifies that every status reset and retained RAM reuse works.
        let retry = [T::from(2); N];
        let (word, stage, output) =
            oracle::expected(&retry, ir.numerical_requirements.range_policy);
        assert_eq!(word, u64::MAX);
        assert_eq!(control.probe(&retry), word);
        control.verify(&stage, &output);
        cases += 1;
    }
    cases
}
fn main() {
    let (_, backend, _) = selection::selected_device();
    let mut cases = 0;
    for underflow in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    ] {
        for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
            let request = PcuImplementationRequirements {
                numerical_mode: PcuNumericalMode::Strict,
                float_underflow: underflow,
                range_policy: range,
                ..PcuImplementationRequirements::DEFAULT
            };
            macro_rules! width {
                ($module:ident,$ty:ty) => {{
                    let bindings = source::$module::direct_bindings();
                    let direct = source::$module::__direct_ir_with_float_underflow_policy::<N>(
                        &bindings, underflow, range, request,
                    )
                    .unwrap();
                    assert_eq!(direct.ir().numerical_requirements, request);
                    cases += verify::<$ty>(&backend, &direct.ir());
                    let grid = source::$module::__grid_ir_with_float_underflow_policy::<N>(
                        &bindings, underflow, range, request,
                    )
                    .unwrap();
                    grid.with_ir(|ir| {
                        assert_eq!(ir.numerical_requirements, request);
                        cases += verify::<$ty>(&backend, ir);
                    });
                }};
            }
            width!(single, u64);
            width!(double, i64);
        }
    }
    assert_eq!(cases, 516);
    println!("cuda handwritten product/carry goldens PASS516 probes and516 healthy retries");
}
