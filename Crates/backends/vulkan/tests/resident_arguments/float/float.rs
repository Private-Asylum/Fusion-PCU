//! Literal IEEE encoding oracle for mixed publication across all six checked formats.
#[rustfmt::skip]
use super::{
    add,
    owned,
    same,
    Sample,
};
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuBf16Bits,
    PcuCheckedFloat,
    PcuCompoundArithmeticPolicy,
    PcuExecutionFaultKind,
    PcuF16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuRangePolicy,
};
trait Float: Sample + PcuCheckedFloat {
    const ONE: u64;
    const TWO: u64;
    const MAX: u64;
    const NAN: u64;
    fn raw(bits: u64) -> Self;
}
macro_rules! low {
    ($ty:ty,$bits:ty,$one:expr,$two:expr,$max:expr,$nan:expr) => {
        impl Float for $ty {
            const ONE: u64 = $one;
            const TWO: u64 = $two;
            const MAX: u64 = $max;
            const NAN: u64 = $nan;
            fn raw(bits: u64) -> Self {
                Self::from_bits(<$bits>::try_from(bits).unwrap())
            }
        }
    };
}
low!(PcuF16Bits, u16, 0x3c00, 0x4000, 0x7bff, 0x7e00);
low!(PcuBf16Bits, u16, 0x3f80, 0x4000, 0x7f7f, 0x7fc0);
low!(PcuF8E4M3FnBits, u8, 0x38, 0x40, 0x7e, 0x7f);
low!(PcuF8E5M2Bits, u8, 0x3c, 0x40, 0x7b, 0x7f);
impl Float for f32 {
    const ONE: u64 = 0x3f80_0000;
    const TWO: u64 = 0x4000_0000;
    const MAX: u64 = 0x7f7f_ffff;
    const NAN: u64 = 0x7fc0_0000;
    fn raw(bits: u64) -> Self {
        Self::from_bits(u32::try_from(bits).unwrap())
    }
}
impl Float for f64 {
    const ONE: u64 = 0x3ff0_0000_0000_0000;
    const TWO: u64 = 0x4000_0000_0000_0000;
    const MAX: u64 = 0x7fef_ffff_ffff_ffff;
    const NAN: u64 = 0x7ff8_0000_0000_0000;
    fn raw(bits: u64) -> Self {
        Self::from_bits(bits)
    }
}
fn format<T: Float>(mut policy: global::PcuExecutionPolicy) {
    let bank = [T::raw(T::ONE); 68];
    let one = [T::raw(T::ONE); 65];
    let max = [T::raw(T::MAX); 65];
    policy.range_policy = PcuRangePolicy::Reject;
    global::configure(policy).unwrap();
    let input = owned::identity(&one).unwrap();
    let max_input = owned::identity(&max).unwrap();
    let mut output = owned::identity(&bank).unwrap();
    let mut observed = bank;
    add(&input, &one, &mut output).unwrap();
    output.read_into(&mut observed).unwrap();
    same(&observed[..65], &[T::raw(T::TWO); 65]);
    same(&observed[65..], &bank[65..]);
    for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
        policy.range_policy = range;
        global::configure(policy).unwrap();
        let error = add(&max_input, &max, &mut output).unwrap_err();
        let fault = error.arithmetic_fault().unwrap();
        assert_eq!(fault.kind, PcuExecutionFaultKind::ArithmeticOverflow);
        assert_eq!(fault.invocation_id, 0);
        assert_eq!(fault.recovered, range == PcuRangePolicy::Clamp);
        output.read_into(&mut observed).unwrap();
        let prior = if range == PcuRangePolicy::Clamp {
            T::raw(T::MAX)
        } else {
            T::raw(T::TWO)
        };
        same(&observed[..65], &[prior; 65]);
        same(&observed[65..], &bank[65..]);
        let mut late_fatal = max;
        late_fatal[7] = T::raw(T::NAN);
        let error = add(&max_input, &late_fatal, &mut output).unwrap_err();
        let fault = error.arithmetic_fault().unwrap();
        assert!(!fault.recovered);
        assert_eq!(
            fault.invocation_id,
            if range == PcuRangePolicy::Clamp { 7 } else { 0 }
        );
        output.read_into(&mut observed).unwrap();
        same(&observed[..65], &[prior; 65]);
        same(&observed[65..], &bank[65..]);
        add(&input, &one, &mut output).unwrap();
        let result = add(&[T::raw(1); 65], &[T::raw(0); 65], &mut output);
        let value = if policy.float_underflow == PcuFloatUnderflowPolicy::RejectSubnormalResult {
            let fault = result.unwrap_err().arithmetic_fault().unwrap();
            assert_eq!(fault.kind, PcuExecutionFaultKind::ArithmeticUnderflow);
            assert_eq!(fault.invocation_id, 0);
            assert_eq!(fault.recovered, range == PcuRangePolicy::Clamp);
            if range == PcuRangePolicy::Clamp {
                T::raw(1)
            } else {
                T::raw(T::TWO)
            }
        } else {
            result.unwrap();
            T::raw(1)
        };
        output.read_into(&mut observed).unwrap();
        same(&observed[..65], &[value; 65]);
        same(&observed[65..], &bank[65..]);
        add(&input, &one, &mut output).unwrap();
    }
}
#[test]
#[ignore = "requires actual Vulkan GPU and exclusive correctness window"]
fn six_format_resident_binary_all_permissions_underflow_range_and_fatal_priority() {
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
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                ] {
                    let policy = global::PcuExecutionPolicy {
                        backend: global::PcuBackendChoice::Vulkan,
                        numerical_mode,
                        float_underflow,
                        numerical_options: PcuNumericalOptions {
                            compound_arithmetic,
                            precision,
                            ..Default::default()
                        },
                        ..Default::default()
                    };
                    format::<PcuF16Bits>(policy);
                    format::<PcuBf16Bits>(policy);
                    format::<PcuF8E4M3FnBits>(policy);
                    format::<PcuF8E5M2Bits>(policy);
                    format::<f32>(policy);
                    format::<f64>(policy);
                }
            }
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
