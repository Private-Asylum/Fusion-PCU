//! Ordinary literal composition: exact extrema, checked discarded effects and publication.
extern crate pcu_facade as fusion_pcu;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionFaultKind,
    PcuCompoundArithmeticPolicy,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuPrecisionPolicy,
    PcuNumericalOptions,
    PcuRangePolicy,
};

mod u8_map {
    use super::*;
    #[pcu(invocations = N)]
    pub fn direct<const N: usize>(input: &[u8], output: &mut [u8]) {
        let id = pcu::context::global_invocation_id();
        let first: u8 = input[id] + 255_u8;
        output[id] = first - 255_u8;
    }
    #[pcu(invocations = 3)]
    pub fn grid<const N: usize>(input: &[u8], output: &mut [u8]) {
        let mut id = pcu::context::global_invocation_id();
        let stride = pcu::context::invocation_count();
        while id < N {
            let first: u8 = input[id] + 0_u8;
            output[id] = first - 0_u8;
            id += stride;
        }
    }
    pub fn verify(range: PcuRangePolicy) {
        const N: usize = 65;
        let mut input = [0_u8; N];
        let mut output = [17_u8; N + 2];
        direct::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        assert_eq!(&output[N..], &[17_u8; 2]);
        grid::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        input[7] = 1;
        let before = output;
        let fault = direct::<N>(&input, &mut output)
            .unwrap_err()
            .arithmetic_fault()
            .unwrap();
        assert_eq!(
            (fault.invocation_id, fault.kind, fault.recovered),
            (
                7,
                PcuExecutionFaultKind::ArithmeticOverflow,
                range == PcuRangePolicy::Clamp
            )
        );
        if range == PcuRangePolicy::Reject {
            assert_eq!(output, before);
        } else {
            assert_eq!(&output[..N], &[0_u8; N]);
        }
        assert_eq!(&output[N..], &[17_u8; 2]);
        input[7] = 0;
        direct::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        // Grid processes changing positive data with the signed minimum literal.
        input.fill(1);
        grid::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        assert_eq!(&output[N..], &[17_u8; 2]);
    }
}
mod i8_map {
    use super::*;
    #[pcu(invocations = N)]
    pub fn direct<const N: usize>(input: &[i8], output: &mut [i8]) {
        let id = pcu::context::global_invocation_id();
        let first: i8 = input[id] + 127_i8;
        output[id] = first - 127_i8;
    }
    #[pcu(invocations = 3)]
    pub fn grid<const N: usize>(input: &[i8], output: &mut [i8]) {
        let mut id = pcu::context::global_invocation_id();
        let stride = pcu::context::invocation_count();
        while id < N {
            let first: i8 = input[id] + -128_i8;
            output[id] = first - -128_i8;
            id += stride;
        }
    }
    pub fn verify(range: PcuRangePolicy) {
        const N: usize = 65;
        let mut input = [0_i8; N];
        let mut output = [17_i8; N + 2];
        direct::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        assert_eq!(&output[N..], &[17_i8; 2]);
        grid::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        input[7] = 1;
        let before = output;
        let fault = direct::<N>(&input, &mut output)
            .unwrap_err()
            .arithmetic_fault()
            .unwrap();
        assert_eq!(
            (fault.invocation_id, fault.kind, fault.recovered),
            (
                7,
                PcuExecutionFaultKind::ArithmeticOverflow,
                range == PcuRangePolicy::Clamp
            )
        );
        if range == PcuRangePolicy::Reject {
            assert_eq!(output, before);
        } else {
            assert_eq!(&output[..N], &[0_i8; N]);
        }
        assert_eq!(&output[N..], &[17_i8; 2]);
        input[7] = 0;
        direct::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        // Grid processes changing positive data with the signed minimum literal.
        input.fill(1);
        grid::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        assert_eq!(&output[N..], &[17_i8; 2]);
    }
}
mod u16_map {
    use super::*;
    #[pcu(invocations = N)]
    pub fn direct<const N: usize>(input: &[u16], output: &mut [u16]) {
        let id = pcu::context::global_invocation_id();
        let first: u16 = input[id] + 65_535_u16;
        output[id] = first - 65_535_u16;
    }
    #[pcu(invocations = 3)]
    pub fn grid<const N: usize>(input: &[u16], output: &mut [u16]) {
        let mut id = pcu::context::global_invocation_id();
        let stride = pcu::context::invocation_count();
        while id < N {
            let first: u16 = input[id] + 0_u16;
            output[id] = first - 0_u16;
            id += stride;
        }
    }
    pub fn verify(range: PcuRangePolicy) {
        const N: usize = 65;
        let mut input = [0_u16; N];
        let mut output = [17_u16; N + 2];
        direct::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        assert_eq!(&output[N..], &[17_u16; 2]);
        grid::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        input[7] = 1;
        let before = output;
        let fault = direct::<N>(&input, &mut output)
            .unwrap_err()
            .arithmetic_fault()
            .unwrap();
        assert_eq!(
            (fault.invocation_id, fault.kind, fault.recovered),
            (
                7,
                PcuExecutionFaultKind::ArithmeticOverflow,
                range == PcuRangePolicy::Clamp
            )
        );
        if range == PcuRangePolicy::Reject {
            assert_eq!(output, before);
        } else {
            assert_eq!(&output[..N], &[0_u16; N]);
        }
        assert_eq!(&output[N..], &[17_u16; 2]);
        input[7] = 0;
        direct::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        // Grid processes changing positive data with the signed minimum literal.
        input.fill(1);
        grid::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        assert_eq!(&output[N..], &[17_u16; 2]);
    }
}
mod i16_map {
    use super::*;
    #[pcu(invocations = N)]
    pub fn direct<const N: usize>(input: &[i16], output: &mut [i16]) {
        let id = pcu::context::global_invocation_id();
        let first: i16 = input[id] + 32_767_i16;
        output[id] = first - 32_767_i16;
    }
    #[pcu(invocations = 3)]
    pub fn grid<const N: usize>(input: &[i16], output: &mut [i16]) {
        let mut id = pcu::context::global_invocation_id();
        let stride = pcu::context::invocation_count();
        while id < N {
            let first: i16 = input[id] + -32_768_i16;
            output[id] = first - -32_768_i16;
            id += stride;
        }
    }
    pub fn verify(range: PcuRangePolicy) {
        const N: usize = 65;
        let mut input = [0_i16; N];
        let mut output = [17_i16; N + 2];
        direct::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        assert_eq!(&output[N..], &[17_i16; 2]);
        grid::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        input[7] = 1;
        let before = output;
        let fault = direct::<N>(&input, &mut output)
            .unwrap_err()
            .arithmetic_fault()
            .unwrap();
        assert_eq!(
            (fault.invocation_id, fault.kind, fault.recovered),
            (
                7,
                PcuExecutionFaultKind::ArithmeticOverflow,
                range == PcuRangePolicy::Clamp
            )
        );
        if range == PcuRangePolicy::Reject {
            assert_eq!(output, before);
        } else {
            assert_eq!(&output[..N], &[0_i16; N]);
        }
        assert_eq!(&output[N..], &[17_i16; 2]);
        input[7] = 0;
        direct::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        // Grid processes changing positive data with the signed minimum literal.
        input.fill(1);
        grid::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        assert_eq!(&output[N..], &[17_i16; 2]);
    }
}
mod u32_map {
    use super::*;
    #[pcu(invocations = N)]
    pub fn direct<const N: usize>(input: &[u32], output: &mut [u32]) {
        let id = pcu::context::global_invocation_id();
        let first: u32 = input[id] + 4_294_967_295_u32;
        output[id] = first - 4_294_967_295_u32;
    }
    #[pcu(invocations = 3)]
    pub fn grid<const N: usize>(input: &[u32], output: &mut [u32]) {
        let mut id = pcu::context::global_invocation_id();
        let stride = pcu::context::invocation_count();
        while id < N {
            let first: u32 = input[id] + 0_u32;
            output[id] = first - 0_u32;
            id += stride;
        }
    }
    pub fn verify(range: PcuRangePolicy) {
        const N: usize = 65;
        let mut input = [0_u32; N];
        let mut output = [17_u32; N + 2];
        direct::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        assert_eq!(&output[N..], &[17_u32; 2]);
        grid::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        input[7] = 1;
        let before = output;
        let fault = direct::<N>(&input, &mut output)
            .unwrap_err()
            .arithmetic_fault()
            .unwrap();
        assert_eq!(
            (fault.invocation_id, fault.kind, fault.recovered),
            (
                7,
                PcuExecutionFaultKind::ArithmeticOverflow,
                range == PcuRangePolicy::Clamp
            )
        );
        if range == PcuRangePolicy::Reject {
            assert_eq!(output, before);
        } else {
            assert_eq!(&output[..N], &[0_u32; N]);
        }
        assert_eq!(&output[N..], &[17_u32; 2]);
        input[7] = 0;
        direct::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        // Grid processes changing positive data with the signed minimum literal.
        input.fill(1);
        grid::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        assert_eq!(&output[N..], &[17_u32; 2]);
    }
}
mod i32_map {
    use super::*;
    #[pcu(invocations = N)]
    pub fn direct<const N: usize>(input: &[i32], output: &mut [i32]) {
        let id = pcu::context::global_invocation_id();
        let first: i32 = input[id] + 2_147_483_647_i32;
        output[id] = first - 2_147_483_647_i32;
    }
    #[pcu(invocations = 3)]
    pub fn grid<const N: usize>(input: &[i32], output: &mut [i32]) {
        let mut id = pcu::context::global_invocation_id();
        let stride = pcu::context::invocation_count();
        while id < N {
            let first: i32 = input[id] + -2_147_483_648_i32;
            output[id] = first - -2_147_483_648_i32;
            id += stride;
        }
    }
    pub fn verify(range: PcuRangePolicy) {
        const N: usize = 65;
        let mut input = [0_i32; N];
        let mut output = [17_i32; N + 2];
        direct::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        assert_eq!(&output[N..], &[17_i32; 2]);
        grid::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        input[7] = 1;
        let before = output;
        let fault = direct::<N>(&input, &mut output)
            .unwrap_err()
            .arithmetic_fault()
            .unwrap();
        assert_eq!(
            (fault.invocation_id, fault.kind, fault.recovered),
            (
                7,
                PcuExecutionFaultKind::ArithmeticOverflow,
                range == PcuRangePolicy::Clamp
            )
        );
        if range == PcuRangePolicy::Reject {
            assert_eq!(output, before);
        } else {
            assert_eq!(&output[..N], &[0_i32; N]);
        }
        assert_eq!(&output[N..], &[17_i32; 2]);
        input[7] = 0;
        direct::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        // Grid processes changing positive data with the signed minimum literal.
        input.fill(1);
        grid::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        assert_eq!(&output[N..], &[17_i32; 2]);
    }
}
mod u64_map {
    use super::*;
    #[pcu(invocations = N)]
    pub fn direct<const N: usize>(input: &[u64], output: &mut [u64]) {
        let id = pcu::context::global_invocation_id();
        let first: u64 = input[id] + 18_446_744_073_709_551_615_u64;
        output[id] = first - 18_446_744_073_709_551_615_u64;
    }
    #[pcu(invocations = 3)]
    pub fn grid<const N: usize>(input: &[u64], output: &mut [u64]) {
        let mut id = pcu::context::global_invocation_id();
        let stride = pcu::context::invocation_count();
        while id < N {
            let first: u64 = input[id] + 0_u64;
            output[id] = first - 0_u64;
            id += stride;
        }
    }
    pub fn verify(range: PcuRangePolicy) {
        const N: usize = 65;
        let mut input = [0_u64; N];
        let mut output = [17_u64; N + 2];
        direct::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        assert_eq!(&output[N..], &[17_u64; 2]);
        grid::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        input[7] = 1;
        let before = output;
        let fault = direct::<N>(&input, &mut output)
            .unwrap_err()
            .arithmetic_fault()
            .unwrap();
        assert_eq!(
            (fault.invocation_id, fault.kind, fault.recovered),
            (
                7,
                PcuExecutionFaultKind::ArithmeticOverflow,
                range == PcuRangePolicy::Clamp
            )
        );
        if range == PcuRangePolicy::Reject {
            assert_eq!(output, before);
        } else {
            assert_eq!(&output[..N], &[0_u64; N]);
        }
        assert_eq!(&output[N..], &[17_u64; 2]);
        input[7] = 0;
        direct::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        // Grid processes changing positive data with the signed minimum literal.
        input.fill(1);
        grid::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        assert_eq!(&output[N..], &[17_u64; 2]);
    }
}
mod i64_map {
    use super::*;
    #[pcu(invocations = N)]
    pub fn direct<const N: usize>(input: &[i64], output: &mut [i64]) {
        let id = pcu::context::global_invocation_id();
        let first: i64 = input[id] + 9_223_372_036_854_775_807_i64;
        output[id] = first - 9_223_372_036_854_775_807_i64;
    }
    #[pcu(invocations = 3)]
    pub fn grid<const N: usize>(input: &[i64], output: &mut [i64]) {
        let mut id = pcu::context::global_invocation_id();
        let stride = pcu::context::invocation_count();
        while id < N {
            let first: i64 = input[id] + -9_223_372_036_854_775_808_i64;
            output[id] = first - -9_223_372_036_854_775_808_i64;
            id += stride;
        }
    }
    pub fn verify(range: PcuRangePolicy) {
        const N: usize = 65;
        let mut input = [0_i64; N];
        let mut output = [17_i64; N + 2];
        direct::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        assert_eq!(&output[N..], &[17_i64; 2]);
        grid::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        input[7] = 1;
        let before = output;
        let fault = direct::<N>(&input, &mut output)
            .unwrap_err()
            .arithmetic_fault()
            .unwrap();
        assert_eq!(
            (fault.invocation_id, fault.kind, fault.recovered),
            (
                7,
                PcuExecutionFaultKind::ArithmeticOverflow,
                range == PcuRangePolicy::Clamp
            )
        );
        if range == PcuRangePolicy::Reject {
            assert_eq!(output, before);
        } else {
            assert_eq!(&output[..N], &[0_i64; N]);
        }
        assert_eq!(&output[N..], &[17_i64; 2]);
        input[7] = 0;
        direct::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        // Grid processes changing positive data with the signed minimum literal.
        input.fill(1);
        grid::<N>(&input, &mut output).unwrap();
        assert_eq!(&output[..N], &input);
        assert_eq!(&output[N..], &[17_i64; 2]);
    }
}
fn verify_backend(backend: global::PcuBackendChoice) {
    for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound_arithmetic in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for float_underflow in [
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                ] {
                    for range_policy in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                        global::configure(global::PcuExecutionPolicy {
                            backend,
                            range_policy,
                            numerical_mode,
                            float_underflow,
                            numerical_options: PcuNumericalOptions {
                                compound_arithmetic,
                                precision,
                                ..Default::default()
                            },
                            ..Default::default()
                        })
                        .unwrap();
                        u8_map::verify(range_policy);
                        i8_map::verify(range_policy);
                        u16_map::verify(range_policy);
                        i16_map::verify(range_policy);
                        u32_map::verify(range_policy);
                        i32_map::verify(range_policy);
                        u64_map::verify(range_policy);
                        i64_map::verify(range_policy);
                        global::clear_thread_cache().unwrap();
                    }
                }
            }
        }
    }
}

#[test]
fn ordinary_integer_literal_frontend_reference() {
    verify_backend(global::PcuBackendChoice::Cpu);
}

#[test]
#[ignore = "actual CUDA device required: typed integer literals direct/grid all headers Reject/Clamp"]
fn ordinary_eight_width_integer_literals() {
    verify_backend(global::PcuBackendChoice::Cuda);
}
