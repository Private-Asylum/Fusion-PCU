//! Independent exact tiny-product goldens for the retained handwritten checked helper.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/ordered_float_maps/oracle/oracle.rs"]
#[allow(dead_code)] // Shared representation constants; expected values below use raw integer bits.
mod oracle;
#[path = "../../benches/helper_ordered_float_maps/raw/raw.rs"]
#[allow(dead_code)] // Only the handwritten owned boundary is used by this focused golden gate.
mod raw;
#[path = "../../benches/strict_matmul/selection.rs"]
mod selection;
#[path = "../../benches/helper_ordered_float_maps/source/source.rs"]
#[allow(dead_code)]
// Ordinary source is qualified separately; this gate checks independent native bits.
mod source;
#[rustfmt::skip]
use fusion_pcu::{
    PcuDispatchKernelIr,
    PcuFloatUnderflowPolicy,
    PcuImplementationRequirements,
    PcuNumericalMode,
    PcuRangePolicy,
};
use fusion_pcu_rocm::RocmOwnedDispatchBackend;
use oracle::Format;
const N: usize = 65;
fn verify<T: Format>(backend: &RocmOwnedDispatchBackend, ir: &PcuDispatchKernelIr<'_>) {
    let input = [T::one(); N];
    let mut control = raw::Raw::handwritten::<T, N>(backend, ir, [&input, &input]);
    // 2*x*x is exactly half the smallest normal for these powers. Incrementing x's
    // significand by one gives half-normal+one subnormal unit plus less than half a unit.
    // Thus nearest-even rounds to half-normal+one; this second result is tiny-inexact.
    let tiny = match size_of::<T>() {
        4 => 0x1f80_0000,           // 2^-64
        8 => 0x1ff0_0000_0000_0000, // 2^-512
        _ => unreachable!("this handwritten control admits only F32/F64"),
    };
    for extra in [0_u64, 1] {
        let mut exceptional = input;
        exceptional[2] = T::from(tiny + extra);
        let fault = ir.numerical_requirements.float_underflow
            == PcuFloatUnderflowPolicy::RejectSubnormalResult
            || (extra != 0
                && ir.numerical_requirements.float_underflow
                    == PcuFloatUnderflowPolicy::IeeeAfterRounding);
        let recovered = fault && ir.numerical_requirements.range_policy == PcuRangePolicy::Clamp;
        let expected = if fault {
            (2 << 3) | 4 | if recovered { 1 << 63 } else { 0 }
        } else {
            u64::MAX
        };
        assert_eq!(control.probe(&exceptional), expected);
        if !fault || recovered {
            let mut stage = [T::from(T::ONE + T::MIN_NORMAL); N];
            let mut output = stage;
            stage[2] = T::from(tiny + extra + T::MIN_NORMAL);
            output[2] = T::from(T::MIN_NORMAL / 2 + extra);
            control.verify(&stage, &output);
        }
    }
}
fn main() {
    let (_, backend, _) = selection::selected_device();
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
                    verify::<$ty>(&backend, &direct.ir());
                    let grid = source::$module::__grid_ir_with_float_underflow_policy::<N>(
                        &bindings, underflow, range, request,
                    )
                    .unwrap();
                    grid.with_ir(|ir| {
                        assert_eq!(ir.numerical_requirements, request);
                        verify::<$ty>(&backend, ir);
                    });
                }};
            }
            width!(single, f32);
            width!(double, f64);
        }
    }
    println!("rocm handwritten exact/tiny-inexact nonzero subnormal goldens PASS24 cases");
}
