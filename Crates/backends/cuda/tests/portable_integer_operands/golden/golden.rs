//! All three portable operations against independent full-width saturated goldens.
#[rustfmt::skip]
use super::{
    configure,
    equal,
    native,
    oracle,
    outcome,
    selection,
    source,
    Format,
    PcuBindingRef,
    PcuDispatchKernelIr,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuImplementationRequirements,
    PcuNumericalMode,
    PcuPreparedHostKernel,
    PcuRangePolicy,
    PcuReproducibility,
};
use fusion_pcu_cuda::CudaOwnedDispatchBackend;
const COUNT: usize = 129;
fn with_ir<T: Format>(
    operation: u32,
    requirements: PcuImplementationRequirements,
    mut f: impl FnMut(&PcuDispatchKernelIr<'_>),
) {
    macro_rules! build {
        ($bindings:ident,$builder:ident) => {{
            let bindings = source::$bindings::<T>();
            let builder = source::$builder::<T, COUNT>(
                &bindings,
                requirements.float_underflow,
                requirements.range_policy,
                requirements,
            )
            .unwrap();
            builder.with_ir(|ir| f(ir));
        }};
    }
    match operation {
        0 => build!(
            distinct_add_bindings,
            __distinct_add_ir_with_float_underflow_policy
        ),
        1 => build!(
            distinct_sub_bindings,
            __distinct_sub_ir_with_float_underflow_policy
        ),
        2 => build!(
            distinct_mul_bindings,
            __distinct_mul_ir_with_float_underflow_policy
        ),
        _ => unreachable!(),
    }
}
fn host<T: Format>(
    operation: u32,
    left: &[T],
    right: &[T],
    output: &mut [T],
) -> Result<(), super::PcuExecutionError> {
    match operation {
        0 => source::distinct_add::<T, COUNT>(left, right, output),
        1 => source::distinct_sub::<T, COUNT>(left, right, output),
        2 => source::distinct_mul::<T, COUNT>(left, right, output),
        _ => unreachable!(),
    }
}
fn check<T: Format>(
    result: Result<(), super::PcuExecutionError>,
    fault: Option<(usize, u32)>,
    clamp: bool,
    output: &[T],
    prior: &[T],
    expected: &[T],
) {
    if let Some((index, code)) = fault {
        outcome(result, u64::try_from(index).unwrap(), code, clamp);
        if !clamp {
            equal(output, prior);
            return;
        }
    } else {
        result.unwrap();
    }
    oracle::verify(expected, output);
}
fn cohort<T: Format>(
    backend: &CudaOwnedDispatchBackend,
    requirements: PcuImplementationRequirements,
) {
    configure(requirements);
    let clamp = requirements.range_policy == PcuRangePolicy::Clamp;
    for operation in 0..3 {
        let rows = oracle::rows::<T>(operation);
        assert!(rows.len() <= COUNT);
        with_ir::<T>(operation, requirements, |ir| {
            let descriptor = super::fusion_pcu::describe_portable_v1_integer_map(ir).unwrap();
            assert_eq!(
                descriptor.operands.input_element_counts(COUNT),
                [COUNT, COUNT]
            );
            let mut prepared = backend.prepare_host_kernel(ir).unwrap();
            let mut native = native::Native::new::<T, COUNT>(backend, ir, 1);
            for phase in [0usize, 17] {
                let batch = (0..COUNT)
                    .map(|i| rows[(i + phase) % rows.len()])
                    .collect::<Vec<_>>();
                let left = batch.iter().map(|row| row.0).collect::<Vec<_>>();
                let right = batch.iter().map(|row| row.1).collect::<Vec<_>>();
                let expected = batch.iter().map(|row| row.2).collect::<Vec<_>>();
                let fault = batch
                    .iter()
                    .enumerate()
                    .find(|(_, row)| row.3 != 0)
                    .map(|(i, row)| (i, row.3));
                let prior = vec![T::sentinel(); COUNT + 2];
                let mut output = prior.clone();
                check(
                    host(operation, &left, &right, &mut output),
                    fault,
                    clamp,
                    &output,
                    &prior,
                    &expected,
                );
                output.clone_from(&prior);
                let result = prepared.call(&mut [
                    PcuHostArgument::read(PcuBindingRef::new(0, 0), &left),
                    PcuHostArgument::read(PcuBindingRef::new(0, 1), &right),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output),
                ]);
                let result = result.map_err(|error| match error {
                    fusion_pcu_cuda::CudaHostKernelError::CheckedExecutionFault(fault) => {
                        super::PcuExecutionError::ArithmeticFault(fault)
                    }
                    other => panic!("{other:?}"),
                });
                check(result, fault, clamp, &output, &prior, &expected);
                native.upload(0, &left, &right);
                let word = fault.map_or(u64::MAX, |(i, code)| {
                    (u64::try_from(i).unwrap() << 3)
                        | u64::from(code)
                        | if clamp { 1 << 63 } else { 0 }
                });
                assert_eq!(native.submit(0), word);
                if clamp || fault.is_none() {
                    output.clone_from(&prior);
                    native.read(&mut output);
                    oracle::verify(&expected, &output);
                }
                let left = vec![T::small(3); COUNT];
                let right = vec![T::small(2); COUNT];
                let expected = vec![
                    T::small(match operation {
                        0 => 5,
                        1 => 1,
                        2 => 6,
                        _ => unreachable!(),
                    });
                    COUNT
                ];
                host(operation, &left, &right, &mut output).unwrap();
                oracle::verify(&expected, &output);
                prepared
                    .call(&mut [
                        PcuHostArgument::read(PcuBindingRef::new(0, 0), &left),
                        PcuHostArgument::read(PcuBindingRef::new(0, 1), &right),
                        PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output),
                    ])
                    .unwrap();
                oracle::verify(&expected, &output);
                native.host(&left, &right, &mut output);
                oracle::verify(&expected, &output);
            }
        });
    }
}
#[test]
#[ignore = "actual portable all14/all3 operation independent full-width golden source/prepared/native proof"]
fn portable_all_integer_operations_full_width_goldens() {
    let (_, backend, _) = selection::selected_device();
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
            let mut requirements = PcuImplementationRequirements::DEFAULT;
            requirements.numerical_mode = mode;
            requirements.range_policy = range;
            requirements.numerical_options.reproducibility = PcuReproducibility::PortableV1;
            all!(cohort, &backend, requirements);
        }
    }
}
