//! Genuine raw-carrier source through the distinct transport factory, no arithmetic fallback.

#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuScalar,
    PcuNumericalMode,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuBindingRef,
    PcuHostDispatchError,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuF128Bits,
    PcuF256Bits,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxRuntime,
    MlxHostKernelError,
};

#[pcu(invocations=N,crate_path=::pcu_facade)]
fn saved<T: PcuScalar, const N: usize>(
    input: &[T],
    seed: &T,
    ghost: &mut [T],
    stage: &mut [T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    stage[id] = input[id];
    let retained = stage[id];
    stage[id] = *seed;
    output[id] = retained;
}
#[pcu(invocations=3,crate_path=::pcu_facade)]
fn saved_grid<T: PcuScalar, const N: usize>(
    input: &[T],
    seed: &T,
    ghost: &mut [T],
    stage: &mut [T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        stage[id] = input[id];
        let retained = stage[id];
        stage[id] = *seed;
        output[id] = retained;
        id += stride;
    }
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
fn prior<T: PcuScalar, const N: usize>(input: &[T], stage: &mut [T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    let previous = stage[id];
    stage[id] = input[id];
    output[id] = previous;
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
fn simple<T: PcuScalar, const N: usize>(
    ghost: &mut [T],
    output: &mut [T],
    unused: &[T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id];
}
#[pcu(invocations=3,crate_path=::pcu_facade)]
fn simple_grid<T: PcuScalar, const N: usize>(input: &[T], ghost: &mut [T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = input[id];
        id += stride;
    }
}
trait Sample: PcuScalar {
    fn raw(lane: usize, phase: u8) -> Self;
}
macro_rules! samples {($($ty:ty=>$width:literal),+)=>{$(
    impl Sample for $ty {
        fn raw(lane:usize,phase:u8)->Self {
            let mut bytes=[0_u8;$width];
            for (index,byte) in bytes.iter_mut().enumerate() {
                *byte=match lane%7 {
                    0=>phase,
                    1=>0xff,
                    2=>if index==$width-1 {0x80} else {0},
                    3=>if index==0 {1} else {0},
                    _=>u8::try_from((lane*13+index*37)%256).unwrap().wrapping_add(phase),
                };
            }
            Self::decode_le(bytes)
        }
    }
)+};}
samples!(i8=>1,u8=>1,i16=>2,u16=>2,i32=>4,u32=>4,i64=>8,u64=>8,i128=>16,u128=>16,PcuI256=>32,PcuU256=>32,PcuI512=>64,PcuU512=>64,f32=>4,f64=>8,PcuF16Bits=>2,PcuBf16Bits=>2,PcuF8E4M3FnBits=>1,PcuF8E5M2Bits=>1,PcuF128Bits=>16,PcuF256Bits=>32);
fn bytes<T: PcuScalar>(values: &[T]) -> Vec<u8> {
    PcuHostArgument::read(PcuBindingRef::new(0, 0), values)
        .bytes()
        .to_vec()
}
fn verify<T: Sample, B: PcuHostKernelBackend<Error = MlxHostKernelError>>(backend: &B)
where
    B::Prepared: PcuPreparedHostKernel<Error = MlxHostKernelError>,
{
    const N: usize = 17;
    let mut source = saved_prepare::<T, N, _>(backend).unwrap();
    let mut grid = saved_grid_prepare::<T, N, _>(backend).unwrap();
    let mut incoming = prior_prepare::<T, N, _>(backend).unwrap();
    let mut copy = simple_prepare::<T, N, _>(backend).unwrap();
    let mut grid_copy = simple_grid_prepare::<T, N, _>(backend).unwrap();
    for phase in 0..3_u8 {
        let input: Vec<T> = (0..N + 7).map(|lane| T::raw(lane, phase)).collect();
        let seed = T::raw(1, phase);
        for grid_mode in [false, true] {
            let mut run =
                |input: &[T], seed: &T, ghost: &mut [T], stage: &mut [T], output: &mut [T]| {
                    if grid_mode {
                        grid(input, seed, ghost, stage, output)
                    } else {
                        source(input, seed, ghost, stage, output)
                    }
                };
            let mut stage = vec![T::raw(9, phase); N + 3];
            let mut output = vec![T::raw(11, phase); N + 5];
            let old_stage = stage.clone();
            let old_output = output.clone();
            run(&input, &seed, &mut [], &mut stage, &mut output).unwrap();
            let mut expected_stage = vec![seed; N];
            expected_stage.extend_from_slice(&old_stage[N..]);
            let mut expected_output = input[..N].to_vec();
            expected_output.extend_from_slice(&old_output[N..]);
            assert_eq!(bytes(&stage), bytes(&expected_stage));
            assert_eq!(bytes(&output), bytes(&expected_output));
            let mut short = vec![T::raw(9, phase); N - 1];
            let before = bytes(&stage);
            assert!(matches!(
                run(&input, &seed, &mut [], &mut stage, &mut short),
                Err(PcuHostDispatchError::BufferTooSmall(_))
            ));
            assert_eq!(bytes(&stage), before);
            run(&input, &seed, &mut [], &mut stage, &mut output).unwrap();
        }
        let mut stage: Vec<T> = (0..N + 3)
            .map(|lane| T::raw(lane, phase.wrapping_add(91)))
            .collect();
        let previous = stage.clone();
        let mut output = vec![T::raw(11, phase); N + 5];
        let old = output.clone();
        incoming(&input, &mut stage, &mut output).unwrap();
        let mut expected = input[..N].to_vec();
        expected.extend_from_slice(&previous[N..]);
        assert_eq!(bytes(&stage), bytes(&expected));
        let mut expected = previous[..N].to_vec();
        expected.extend_from_slice(&old[N..]);
        assert_eq!(bytes(&output), bytes(&expected));
        copy(&mut [], &mut output, &[], &input).unwrap();
        assert_eq!(
            &bytes(&output)[..N * T::HOST_SIZE],
            &bytes(&input)[..N * T::HOST_SIZE]
        );
        grid_copy(&input, &mut [], &mut output).unwrap();
        assert_eq!(
            &bytes(&output)[..N * T::HOST_SIZE],
            &bytes(&input)[..N * T::HOST_SIZE]
        );
        assert_eq!(
            &bytes(&output)[N * T::HOST_SIZE..],
            &bytes(&old)[N * T::HOST_SIZE..]
        );
    }
}
fn requests() -> Vec<pcu_facade::global::PcuExecutionPolicy> {
    let mut result = Vec::new();
    for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound_arithmetic in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for range_policy in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                    for float_underflow in [
                        PcuFloatUnderflowPolicy::IeeeAfterRounding,
                        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                        PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    ] {
                        let mut policy = pcu_facade::global::PcuExecutionPolicy {
                            backend: pcu_facade::global::PcuBackendChoice::Mlx,
                            numerical_mode,
                            range_policy,
                            float_underflow,
                            ..pcu_facade::global::PcuExecutionPolicy::default()
                        };
                        policy.numerical_options.compound_arithmetic = compound_arithmetic;
                        policy.numerical_options.precision = precision;
                        result.push(policy);
                    }
                }
            }
        }
    }
    result
}
#[test]
#[ignore = "Requires actual pinned MLX GPU runtime and all22 raw annotated transport source."]
fn actual_all_twenty_two_saved_ordered_transport_and_host_atomicity() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    qualify_backend(&session.transport_host_backend());
}

#[test]
#[ignore = "Requires genuine MLX static aggregate source dispatch and all22 transport carriers."]
fn actual_all_twenty_two_static_session_transport_dispatch_and_host_atomicity() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    qualify_backend(&session);
}

fn qualify_backend<B: PcuHostKernelBackend<Error = MlxHostKernelError>>(backend: &B)
where
    B::Prepared: PcuPreparedHostKernel<Error = MlxHostKernelError>,
{
    macro_rules! run {($($ty:ty),+)=>{$(verify::<$ty, B>(backend);)+};}
    for policy in requests() {
        pcu_facade::global::configure(policy).unwrap();
        run!(
            i8,
            u8,
            i16,
            u16,
            i32,
            u32,
            i64,
            u64,
            i128,
            u128,
            PcuI256,
            PcuU256,
            PcuI512,
            PcuU512,
            f32,
            f64,
            PcuF16Bits,
            PcuBf16Bits,
            PcuF8E4M3FnBits,
            PcuF8E5M2Bits,
            PcuF128Bits,
            PcuF256Bits
        );
    }
}
