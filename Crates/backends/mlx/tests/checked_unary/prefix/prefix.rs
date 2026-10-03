//! Full resident shapes are independent of the logical unary read and fault domain.
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxCheckedUnaryPlan,
    MlxEncodedCompletion,
    MlxError,
    MlxRuntime,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuBf16Bits,
    PcuCheckedFloat,
    PcuCompoundArithmeticPolicy as Compound,
    PcuDispatchFloatUnaryOp as Op,
    PcuExecutionFault,
    PcuExecutionFaultKind as Kind,
    PcuF16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuFloatUnderflowPolicy as Policy,
    PcuImplementationRequirements,
    PcuNumericalMode as Mode,
    PcuPrecisionPolicy as Precision,
    PcuRangePolicy as Range,
    PcuReproducibility,
};
use super::graph;

fn permissions() -> impl Iterator<Item = (Mode, Compound, Precision)> {
    (0..8).map(|bits| {
        (
            if bits & 1 == 0 {
                Mode::Boundary
            } else {
                Mode::Strict
            },
            if bits & 2 == 0 {
                Compound::Checked
            } else {
                Compound::BackendDefined
            },
            if bits & 4 == 0 {
                Precision::Preserve
            } else {
                Precision::BackendOptimized
            },
        )
    })
}
const fn apply_permissions(
    requirements: &mut PcuImplementationRequirements,
    permissions: (Mode, Compound, Precision),
) {
    requirements.numerical_mode = permissions.0;
    requirements.numerical_options.compound_arithmetic = permissions.1;
    requirements.numerical_options.precision = permissions.2;
}

pub trait Sample: PcuCheckedFloat {
    const SIGN: u64;
    const MAX: u64;
    const FRACTION: u32;
    fn raw(bits: u64) -> Self;
    fn bits(self) -> u64;
}
macro_rules! small {
    ($ty:ty, $bits:ty, $sign:expr, $max:expr, $fraction:expr) => {
        impl Sample for $ty {
            const SIGN: u64 = $sign;
            const MAX: u64 = $max;
            const FRACTION: u32 = $fraction;
            fn raw(bits: u64) -> Self {
                Self::from_bits(<$bits>::try_from(bits).unwrap())
            }
            fn bits(self) -> u64 {
                u64::from(self.to_bits())
            }
        }
    };
}
small!(PcuF16Bits, u16, 0x8000, 0x7bff, 10);
small!(PcuBf16Bits, u16, 0x8000, 0x7f7f, 7);
small!(PcuF8E4M3FnBits, u8, 0x80, 0x7e, 3);
small!(PcuF8E5M2Bits, u8, 0x80, 0x7b, 2);
small!(f32, u32, 0x8000_0000, 0x7f7f_ffff, 23);
impl Sample for f64 {
    const SIGN: u64 = 0x8000_0000_0000_0000;
    const MAX: u64 = 0x7fef_ffff_ffff_ffff;
    const FRACTION: u32 = 52;
    fn raw(bits: u64) -> Self {
        Self::from_bits(bits)
    }
    fn bits(self) -> u64 {
        self.to_bits()
    }
}
macro_rules! six {
    ($function:ident) => {
        $function::<PcuF16Bits>();
        $function::<PcuBf16Bits>();
        $function::<PcuF8E4M3FnBits>();
        $function::<PcuF8E5M2Bits>();
        $function::<f32>();
        $function::<f64>();
    };
}
pub fn compare<T: Sample>(left: &[T], right: &[T]) {
    assert_eq!(left.len(), right.len());
    for (left, right) in left.iter().zip(right) {
        assert_eq!(left.bits(), right.bits());
    }
}
// No reference arithmetic, conversions or provider predicates: exact sign/finite bit selection.
pub fn reference<T: Sample>(
    input: &[T],
    op: Op,
    policy: Policy,
    range: Range,
    broadcast: bool,
) -> (Vec<T>, Option<PcuExecutionFault>) {
    let mut output = Vec::with_capacity(5);
    let mut fatal = None;
    let mut recovered = None;
    for lane in 0..5 {
        let bits = input[if broadcast { 0 } else { lane }].bits();
        let magnitude = bits & (T::SIGN - 1);
        let result = if op == Op::Neg {
            bits ^ T::SIGN
        } else if bits & T::SIGN == 0 && magnitude != 0 {
            bits
        } else {
            0
        };
        output.push(T::raw(result));
        let result_magnitude = result & (T::SIGN - 1);
        let kind = if magnitude > T::MAX {
            Some(Kind::InvalidFloatingOperand)
        } else if policy == Policy::RejectSubnormalResult
            && result_magnitude != 0
            && result_magnitude < (1 << T::FRACTION)
        {
            Some(Kind::ArithmeticUnderflow)
        } else {
            None
        };
        if let Some(kind) = kind {
            let fault = PcuExecutionFault {
                invocation_id: u64::try_from(lane).unwrap(),
                kind,
                recovered: kind == Kind::ArithmeticUnderflow && range == Range::Clamp,
            };
            if fault.recovered {
                recovered.get_or_insert(fault);
            } else {
                fatal.get_or_insert(fault);
            }
        }
    }
    (output, fatal.or(recovered))
}
fn verify<T: Sample>(
    actual: Result<MlxEncodedCompletion, MlxError>,
    expected: &(Vec<T>, Option<PcuExecutionFault>),
) {
    if let Some(fault) = expected.1.filter(|fault| !fault.recovered) {
        assert!(matches!(actual, Err(MlxError::Arithmetic(actual)) if actual == fault));
    } else {
        let (owner, recovered) = actual.unwrap().into_parts();
        assert_eq!(recovered, expected.1);
        let sentinel = T::raw(17);
        let mut output = vec![sentinel; 8];
        owner.read_into(&mut output).unwrap();
        compare(&output[..5], &expected.0);
        compare(&output[5..], &[sentinel; 3]);
    }
}
fn cold<T: Sample>() {
    for grid in [false, true] {
        for broadcast in [false, true] {
            graph::fixture_profile::<T, _>(
                5,
                Op::Relu,
                Policy::RejectSubnormalResult,
                Range::Clamp,
                grid,
                broadcast,
                |ir| {
                    let plan = MlxCheckedUnaryPlan::assess(ir).unwrap();
                    let minimum = if broadcast { 1 } else { 5 };
                    assert_eq!(plan.input_bindings().len(), 1);
                    assert_eq!(plan.input_element_count(), minimum);
                    assert_eq!(plan.assess_input_extents(&[minimum]).unwrap(), minimum);
                    assert_eq!(plan.assess_input_extents(&[11]).unwrap(), 11);
                    for permissions in permissions() {
                        let mut request = *ir;
                        apply_permissions(&mut request.numerical_requirements, permissions);
                        let exact = MlxCheckedUnaryPlan::assess(&request).unwrap();
                        assert_eq!(exact.requirements(), request.numerical_requirements);
                        request
                            .numerical_requirements
                            .numerical_options
                            .reproducibility = PcuReproducibility::PortableV1;
                        assert_eq!(
                            MlxCheckedUnaryPlan::assess(&request)
                                .unwrap()
                                .requirements(),
                            request.numerical_requirements
                        );
                    }
                    for extents in [
                        &[][..],
                        &[minimum - 1][..],
                        &[11, 11][..],
                        &[usize::MAX][..],
                    ] {
                        assert!(matches!(
                            plan.assess_input_extents(extents),
                            Err(MlxError::InvalidExtent)
                        ));
                    }
                },
            );
        }
    }
}
#[test]
fn six_format_unary_capacity_assessment_is_detached() {
    six!(cold);
}

fn bank<T: Sample>(phase: u64, broadcast: bool, full: usize) -> Vec<T> {
    let normal = 1 << T::FRACTION;
    let mut input = vec![T::raw(T::SIGN - 1); full]; // Faulting unvisited suffix.
    let prefix = [normal + phase, T::SIGN | normal, 0, T::SIGN, T::MAX];
    for (destination, bits) in input[..5].iter_mut().zip(prefix) {
        *destination = T::raw(bits);
    }
    if phase == 1 {
        input[0] = T::raw(1);
    }
    if broadcast && phase == 2 {
        input[0] = T::raw(T::SIGN | 1);
    }
    input
}
#[allow(clippy::too_many_lines)] // One retained tuple pairs native/IR terminal owners, faults, no-write preflights and post-drop lifetime.
fn profile<T: Sample, const PORTABLE: bool>(
    op: Op,
    policy: Policy,
    range: Range,
    grid: bool,
    broadcast: bool,
    permissions: (Mode, Compound, Precision),
) {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let foreign = runtime.open_gpu(0).unwrap();
    let full = if broadcast { 8 } else { 11 };
    let minimum = if broadcast { 1 } else { 5 };
    let mut prepared =
        graph::fixture_profile::<T, _>(5, op, policy, range, grid, broadcast, |ir| {
            let mut request = *ir;
            apply_permissions(&mut request.numerical_requirements, permissions);
            if PORTABLE {
                request
                    .numerical_requirements
                    .numerical_options
                    .reproducibility = PcuReproducibility::PortableV1;
            }
            let prepared =
                session.prepare_unary_host_kernel_with_input_extents(&request, &[full])?;
            assert_eq!(prepared.requirements(), request.numerical_requirements);
            Ok::<_, fusion_pcu_mlx::MlxHostKernelError>(prepared)
        })
        .unwrap();
    let mut native = session
        .prepare_checked_unary_control_with_input_extent(
            T::TYPE,
            op,
            policy,
            range,
            5,
            broadcast,
            full,
        )
        .unwrap();
    assert_eq!(prepared.prepared_input_element_count(), full);
    assert_eq!(
        prepared.input_byte_len(),
        minimum * usize::from(T::TYPE.bit_width()) / 8
    );
    for phase in 0..3 {
        let input = bank::<T>(phase, broadcast, full);
        let resident = session.upload_encoded(&input).unwrap();
        let expected = reference(&input, op, policy, range, broadcast);
        verify(prepared.execute_resident(&resident), &expected);
        verify(native.execute_resident(&resident), &expected);
        let wrong = session.upload_encoded(&input[..full - 1]).unwrap();
        assert!(matches!(
            prepared.execute_resident(&wrong),
            Err(MlxError::InvalidExtent)
        ));
        assert!(!prepared.last_call_may_have_written());
        let other = foreign.upload_encoded(&input).unwrap();
        assert!(matches!(
            prepared.execute_resident(&other),
            Err(MlxError::ForeignSession)
        ));
        assert!(!prepared.last_call_may_have_written());
        assert!(matches!(
            prepared.execute_encoded(&input[..minimum]),
            Err(MlxError::InvalidExtent)
        ));
        assert!(!prepared.last_call_may_have_written());
        let mut unchanged = vec![T::raw(17); full + 3];
        resident.read_into(&mut unchanged).unwrap();
        compare(&unchanged[..full], &input);
        compare(&unchanged[full..], &[T::raw(17); 3]);
    }
    let mut fatal = bank::<T>(0, broadcast, full);
    fatal[0] = T::raw(1);
    fatal[usize::from(!broadcast)] = T::raw(T::SIGN - 1);
    let resident = session.upload_encoded(&fatal).unwrap();
    let expected = reference(&fatal, op, policy, range, broadcast);
    verify(prepared.execute_resident(&resident), &expected);
    verify(native.execute_resident(&resident), &expected);
    let healthy = bank::<T>(0, broadcast, full);
    let resident = session.upload_encoded(&healthy).unwrap();
    let (output, notice) = prepared.execute_resident(&resident).unwrap().into_parts();
    let (sibling, sibling_notice) = native.execute_resident(&resident).unwrap().into_parts();
    assert!(notice.is_none() && sibling_notice.is_none());
    drop(prepared);
    drop(native);
    drop(resident);
    drop(session);
    let expected = reference(&healthy, op, policy, range, broadcast).0;
    let mut destination = vec![T::raw(17); 8];
    output.read_into(&mut destination).unwrap();
    compare(&destination[..5], &expected);
    drop(output);
    sibling.read_into(&mut destination).unwrap();
    compare(&destination[..5], &expected);
    compare(&destination[5..], &[T::raw(17); 3]);
}
fn qualify<T: Sample, const PORTABLE: bool>() {
    for op in [Op::Neg, Op::Relu] {
        for policy in [
            Policy::IeeeAfterRounding,
            Policy::RejectSubnormalResult,
            Policy::AllowGradualUnderflow,
        ] {
            for range in [Range::Reject, Range::Clamp] {
                for grid in [false, true] {
                    for broadcast in [false, true] {
                        for permissions in permissions() {
                            profile::<T, PORTABLE>(op, policy, range, grid, broadcast, permissions);
                        }
                    }
                }
            }
        }
    }
}
#[test]
#[ignore = "Requires actual pinned MLX GPU full-shape unary kernels and terminal encoded ownership."]
fn six_format_unary_full_capacity_resident_prefix_faults_and_lifetimes() {
    six!(qualify_normal);
}

fn qualify_normal<T: Sample>() {
    qualify::<T, false>();
}
fn qualify_portable<T: Sample>() {
    qualify::<T, true>();
}
#[test]
#[ignore = "Requires actual MLX Portable requested headers, full-capacity native kernels and terminal encoded ownership."]
fn six_format_portable_unary_full_capacity_resident_faults_and_lifetimes() {
    six!(qualify_portable);
}
