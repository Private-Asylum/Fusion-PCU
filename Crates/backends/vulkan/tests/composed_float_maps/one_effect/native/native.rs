//! Runtime overwritten effects, exact request identity and private fatal publication.
use super::super::offers::device;
#[rustfmt::skip]
use super::super::{
    global,
    observed,
    oracle::Format,
    PcuCompoundArithmeticPolicy,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuNumericalMode,
    PcuPrecisionPolicy,
    PcuVulkanBackend,
    PcuVulkanPreparedHost,
    Range,
    UF,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuCostBoundary,
    PcuExecutorId,
    PcuHostKernelBackend,
    PcuHostArgument,
    PcuBindingRef,
    PcuPreparedHostKernel,
    PcuImplementationRequirements,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuReproducibility,
    PcuFloatUnderflowPolicy as Uf,
};
use super::{
    __overwritten_add_ir_with_float_underflow_policy,
    __overwritten_division_ir_with_float_underflow_policy, overwritten_add_bindings,
    overwritten_division_bindings,
};

fn identity<T: Format>(
    backend: &PcuVulkanBackend,
    ordinal: u32,
    requirements: PcuImplementationRequirements,
) {
    let bindings = overwritten_add_bindings::<T>();
    __overwritten_add_ir_with_float_underflow_policy::<T, 7>(
        &bindings,
        requirements.float_underflow,
        requirements.range_policy,
        requirements,
    )
    .unwrap()
    .with_ir(|kernel| {
        assert_eq!(kernel.numerical_requirements, requirements);
        let request = PcuImplementationRequest {
            device: backend.device_identity().unwrap(),
            executor: PcuExecutorId(0),
            boundary: PcuCostBoundary::Host,
            operation: kernel,
            requirements: kernel.numerical_requirements,
        };
        let mut offers = [None];
        assert_eq!(
            backend
                .implementation_offers(&request, &mut offers)
                .unwrap(),
            1
        );
        let offer = offers[0].unwrap();
        assert_eq!(offer.implementation.local_id, 18944 + ordinal);
        assert_eq!(offer.implementation.revision, 1);
        assert_eq!(offer.implementation.device, request.device);
        assert_eq!(offer.implementation.executor, request.executor);
        assert_eq!(offer.requirements, request.requirements);
        assert!(matches!(
            backend.prepare_host_kernel(kernel).unwrap(),
            PcuVulkanPreparedHost::Composed(_)
        ));
        let mut mismatch = PcuImplementationRequest {
            device: request.device,
            executor: request.executor,
            boundary: request.boundary,
            operation: kernel,
            requirements: request.requirements,
        };
        mismatch.requirements.range_policy = if request.requirements.range_policy == Range::Reject {
            Range::Clamp
        } else {
            Range::Reject
        };
        assert_eq!(
            backend.implementation_offers(&mismatch, &mut []).unwrap(),
            0
        );
        let mut portable = *kernel;
        portable
            .numerical_requirements
            .numerical_options
            .reproducibility = PcuReproducibility::PortableV1;
        let request = PcuImplementationRequest {
            operation: &portable,
            requirements: portable.numerical_requirements,
            ..request
        };
        assert_eq!(backend.implementation_offers(&request, &mut []).unwrap(), 0);
        assert!(backend.prepare_host_kernel(&portable).is_err());
    });
}

fn width<T: Format>(
    backend: &PcuVulkanBackend,
    ordinal: u32,
    requirements: PcuImplementationRequirements,
) {
    let range = requirements.range_policy;
    let uf = requirements.float_underflow;
    identity::<T>(backend, ordinal, requirements);
    let sentinel = T::from(T::ONE + 1);
    let mut input = [T::from(T::ONE); 7];
    let mut output = [sentinel; 10];
    let bindings = overwritten_add_bindings::<T>();
    let mut add_plan = __overwritten_add_ir_with_float_underflow_policy::<T, 7>(
        &bindings,
        uf,
        range,
        requirements,
    )
    .unwrap()
    .with_ir(|kernel| {
        super::assert_policy(kernel, requirements);
        backend.prepare_host_kernel(kernel).unwrap()
    });
    let mut add = |input: &[T], output: &mut [T]| {
        add_plan.call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), output),
        ])
    };
    for phase in 0..3 {
        for (lane, value) in input.iter_mut().enumerate() {
            *value = T::from(
                T::ONE
                    | if (lane + phase).is_multiple_of(2) {
                        0
                    } else {
                        T::SIGN
                    },
            );
        }
        add(&input, &mut output).unwrap();
        assert_eq!(
            output[..7].iter().map(|x| x.bits()).collect::<Vec<_>>(),
            input.map(T::bits)
        );
        assert!(output[7..].iter().all(|x| x.bits() == sentinel.bits()));
    }
    input[2] = T::from(T::MAX);
    let before = output.map(T::bits);
    assert_eq!(
        observed(add(&input, &mut output)),
        Err(PcuExecutionFault {
            invocation_id: 2,
            kind: PcuExecutionFaultKind::ArithmeticOverflow,
            recovered: range == Range::Clamp
        })
    );
    if range == Range::Reject {
        assert_eq!(output.map(T::bits), before);
    } else {
        assert_eq!(output[2].bits(), T::MAX);
    }
    input[4] = T::from(T::SIGN - 1);
    let before = output.map(T::bits);
    let (lane, kind) = if range == Range::Reject {
        (2, PcuExecutionFaultKind::ArithmeticOverflow)
    } else {
        (4, PcuExecutionFaultKind::InvalidFloatingOperand)
    };
    assert_eq!(
        observed(add(&input, &mut output)),
        Err(PcuExecutionFault {
            invocation_id: lane,
            kind,
            recovered: false
        })
    );
    assert_eq!(output.map(T::bits), before);
    input.fill(T::from(T::ONE));
    add(&input, &mut output).unwrap();
    tiny::<T>(&mut add, &mut input, &mut output, uf, range);
    division(backend, requirements, &input, &mut output);
}

fn division<T: Format>(
    backend: &PcuVulkanBackend,
    requirements: PcuImplementationRequirements,
    input: &[T],
    output: &mut [T],
) {
    let bindings = overwritten_division_bindings::<T>();
    let mut div_plan = __overwritten_division_ir_with_float_underflow_policy::<T, 7>(
        &bindings,
        requirements.float_underflow,
        requirements.range_policy,
        requirements,
    )
    .unwrap()
    .with_ir(|kernel| {
        super::assert_policy(kernel, requirements);
        backend.prepare_host_kernel(kernel).unwrap()
    });
    let mut div = |input: &[T], divisor: &[T], output: &mut [T]| {
        div_plan.call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
            PcuHostArgument::read(PcuBindingRef::new(0, 1), divisor),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), output),
        ])
    };
    let mut divisor = [T::from(T::ONE); 7];
    divisor[2] = T::from(0);
    let before = output.iter().map(|value| value.bits()).collect::<Vec<_>>();
    assert_eq!(
        observed(div(input, &divisor, output)),
        Err(PcuExecutionFault {
            invocation_id: 2,
            kind: PcuExecutionFaultKind::DivideByZero,
            recovered: false
        })
    );
    assert_eq!(
        output.iter().map(|value| value.bits()).collect::<Vec<_>>(),
        before
    );
    divisor[2] = T::from(T::ONE);
    div(input, &divisor, output).unwrap();
    assert!(output[..7].iter().all(|x| x.bits() == T::ONE));
}

fn tiny<T: Format>(
    add: &mut impl FnMut(&[T], &mut [T]) -> Result<(), super::super::PcuVulkanError>,
    input: &mut [T],
    output: &mut [T],
    uf: Uf,
    range: Range,
) {
    input.fill(T::from(T::ONE));
    for zero in [0, T::SIGN] {
        input[0] = T::from(zero);
        add(input, output).unwrap();
        assert_eq!(output[0].bits(), zero);
    }
    input[0] = T::from(T::ONE);
    input[2] = T::from(1);
    let before = output.iter().map(|x| x.bits()).collect::<Vec<_>>();
    let expected = if uf == Uf::RejectSubnormalResult {
        Err(PcuExecutionFault {
            invocation_id: 2,
            kind: PcuExecutionFaultKind::ArithmeticUnderflow,
            recovered: range == Range::Clamp,
        })
    } else {
        Ok(())
    };
    assert_eq!(observed(add(input, output)), expected);
    if uf == Uf::RejectSubnormalResult && range == Range::Reject {
        assert_eq!(output.iter().map(|x| x.bits()).collect::<Vec<_>>(), before);
    } else {
        assert_eq!(output[2].bits(), 1);
    }
    input[4] = T::from(T::SIGN - 1);
    let before = output.iter().map(|x| x.bits()).collect::<Vec<_>>();
    let (lane, kind) = if uf == Uf::RejectSubnormalResult && range == Range::Reject {
        (2, PcuExecutionFaultKind::ArithmeticUnderflow)
    } else {
        (4, PcuExecutionFaultKind::InvalidFloatingOperand)
    };
    assert_eq!(
        observed(add(input, output)),
        Err(PcuExecutionFault {
            invocation_id: lane,
            kind,
            recovered: false
        })
    );
    assert_eq!(output.iter().map(|x| x.bits()).collect::<Vec<_>>(), before);
    input.fill(T::from(T::ONE));
    add(input, output).unwrap();
}

#[test]
#[ignore = "requires Vulkan hardware; new exact one-effect IDs and transaction"]
fn six_format_overwritten_effects_are_observable_and_atomic() {
    // Identity-bearing offers require the actual discovery-selected physical owner.
    let (backend, _) = device::selected();
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for uf in UF {
                    for range in [Range::Reject, Range::Clamp] {
                        let mut policy = global::PcuExecutionPolicy {
                            numerical_mode: mode,
                            float_underflow: uf,
                            range_policy: range,
                            ..global::PcuExecutionPolicy::default()
                        };
                        policy.numerical_options.compound_arithmetic = compound;
                        policy.numerical_options.precision = precision;
                        let requirements = PcuImplementationRequirements {
                            numerical_mode: mode,
                            float_underflow: uf,
                            range_policy: range,
                            numerical_options: policy.numerical_options,
                        };
                        global::configure(policy).unwrap();
                        width::<pcu_facade::PcuF16Bits>(&backend, 0, requirements);
                        width::<pcu_facade::PcuBf16Bits>(&backend, 1, requirements);
                        width::<pcu_facade::PcuF8E4M3FnBits>(&backend, 2, requirements);
                        width::<pcu_facade::PcuF8E5M2Bits>(&backend, 3, requirements);
                        width::<f32>(&backend, 4, requirements);
                        width::<f64>(&backend, 5, requirements);
                    }
                }
            }
        }
    }
    global::use_defaults().unwrap();
}
