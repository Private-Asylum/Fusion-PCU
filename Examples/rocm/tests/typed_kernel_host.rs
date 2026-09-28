//! Synchronous typed host-call regressions. Run explicitly on a selected `ROCm` device with:
//! `cargo test -p fusion-pcu-example-rocm --test typed_kernel_host -- --ignored --nocapture`

#[rustfmt::skip]
use fusion_pcu::{
    PcuMemoryPoolId,
    PcuDeviceBuffer,
    PcuScalar,
};
use fusion_pcu_macros::pcu;
#[rustfmt::skip]
use fusion_pcu_rocm::{
    RocmDeviceKernelError,
    RocmDiscovery,
    RocmHostKernelError,
    RocmMemoryResource,
    RocmOwnedDispatchBackend,
};

#[path = "../selection.rs"]
mod selection;

#[pcu(invocations = 250)]
fn transform<const N: usize>(input: &[f32], output: &mut [f32]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = input[id] + 3.0;
        id += stride;
    }
}

#[pcu(invocations = 250)]
fn increment_in_place<const N: usize>(values: &mut [f32]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        values[id] = values[id] + 1.0;
        id += stride;
    }
}

#[pcu(invocations = 250)]
fn identity<T: PcuScalar, const N: usize>(input: &[T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = input[id];
        id += stride;
    }
}

#[pcu(invocations = N)]
fn checked_div_rem<const N: usize>(
    left: &[u32],
    right: &[u32],
    quotient: &mut [u32],
    remainder: &mut [u32],
) {
    let id = pcu::context::global_invocation_id();
    let (q, r) = pcu::checked_div_rem(left[id], right[id]);
    quotient[id] = q;
    remainder[id] = r;
}

fn selected_session() -> (RocmOwnedDispatchBackend, String, PcuMemoryPoolId) {
    let discovery = RocmDiscovery::new();
    let candidates = selection::ranked_devices(
        &discovery,
        selection::preferred_device().expect("valid FUSION_ROCM_DEVICE"),
        false,
    )
    .expect("discover a selected ROCm candidate");
    let (session, candidate) =
        selection::open_ranked(&discovery, candidates, 256).expect("open selected ROCm device");
    (session, candidate.name, candidate.pool)
}

#[test]
#[ignore = "requires a visible ROCm device; run explicitly on ROCm hardware"]
fn typed_host_calls_reuse_fresh_borrows_and_preserve_unwritten_elements() {
    const N: usize = 2048;
    let (session, device, _) = selected_session();
    let mut run = transform_prepare::<N, _>(&session).expect("prepare typed transform");
    println!("typed host-call regressions on {device}");

    // Repeated calls use distinct host allocations and must observe their latest contents.
    for (pass, tail) in [7_usize, 31, 3].into_iter().enumerate() {
        let input: Vec<f32> = (0..N + tail)
            .map(|index| {
                f32::from(u16::try_from(index % 1024).expect("sample fits"))
                    + f32::from(u8::try_from(pass).expect("pass fits"))
            })
            .collect();
        let mut output = vec![-9.0_f32; N + tail];
        run(&input, &mut output).expect("typed host call");
        for (actual, source) in output[..N].iter().zip(&input) {
            assert_eq!(actual.to_bits(), (source + 3.0).to_bits());
        }
        assert!(
            output[N..]
                .iter()
                .all(|value| value.to_bits() == (-9.0_f32).to_bits())
        );
    }

    // The small case exercises a partial final wave; no invocation may write past the boundary.
    let mut short_grid =
        transform_prepare::<65, _>(&session).expect("prepare boundary specialization");
    let input = [2.0_f32; 65];
    let mut output = [0.0_f32; 66];
    short_grid(&input, &mut output).expect("65-element boundary call");
    assert!(
        output[..65]
            .iter()
            .all(|value| value.to_bits() == 5.0_f32.to_bits())
    );
    assert_eq!(output[65].to_bits(), 0.0_f32.to_bits());

    // Extent validation happens before launch, so a short input cannot alter the output.
    let too_short = [1.0_f32; N - 1];
    let mut unchanged = vec![42.0_f32; N];
    assert!(run(&too_short, &mut unchanged).is_err());
    assert!(
        unchanged
            .iter()
            .all(|value| value.to_bits() == 42.0_f32.to_bits())
    );

    let mut in_place =
        increment_in_place_prepare::<N, _>(&session).expect("prepare read/write kernel");
    let mut values = vec![4.0_f32; N];
    in_place(&mut values).expect("in-place typed call");
    assert!(
        values
            .iter()
            .all(|value| value.to_bits() == 5.0_f32.to_bits())
    );
}

#[test]
#[ignore = "requires a visible ROCm device; run explicitly on ROCm hardware"]
fn generic_typed_identity_supports_wide_and_half_transport() {
    #[rustfmt::skip]
    use fusion_pcu::{
        PcuBf16Bits,
        PcuF16Bits,
    };

    const N: usize = 65;
    let (session, device, _) = selected_session();
    println!("generic typed identity transport on {device}");

    let input64: [f64; N] =
        core::array::from_fn(|index| f64::from(u16::try_from(index).expect("fits")) + 0.5);
    let mut output64 = [0.0_f64; N];
    let mut copy64 = identity_prepare::<f64, N, _>(&session).expect("prepare f64 identity");
    copy64(&input64, &mut output64).expect("f64 identity call");
    assert_eq!(output64.map(f64::to_bits), input64.map(f64::to_bits));

    let input_u64: [u64; N] = core::array::from_fn(|index| {
        (u64::try_from(index).expect("fits") << 48) | 0x1234_5678_9abc
    });
    let mut output_u64 = [0_u64; N];
    let mut copy_u64 = identity_prepare::<u64, N, _>(&session).expect("prepare u64 identity");
    copy_u64(&input_u64, &mut output_u64).expect("u64 identity call");
    assert_eq!(output_u64, input_u64);

    let half_input: Vec<_> = (0..N)
        .map(|index| PcuF16Bits::from_bits([0x3c00_u16, 0xbc00, 0x7e01][index % 3]))
        .collect();
    let mut half_output = vec![PcuF16Bits::from_bits(0); N];
    let mut half_copy =
        identity_prepare::<PcuF16Bits, N, _>(&session).expect("prepare f16 bit identity");
    half_copy(&half_input, &mut half_output).expect("f16 bit identity call");
    assert_eq!(
        half_output
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        half_input
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    let input_bf16: Vec<_> = (0..N)
        .map(|index| PcuBf16Bits::from_bits([0x3f80_u16, 0xbf80, 0x7fc1][index % 3]))
        .collect();
    let mut output_bf16 = vec![PcuBf16Bits::from_bits(0); N];
    let mut copy_bf16 =
        identity_prepare::<PcuBf16Bits, N, _>(&session).expect("prepare bf16 bit identity");
    copy_bf16(&input_bf16, &mut output_bf16).expect("bf16 bit identity call");
    assert_eq!(
        output_bf16
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        input_bf16
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "requires a visible ROCm device; run explicitly on ROCm hardware"]
fn checked_fault_does_not_expose_host_outputs_and_prepared_calls_can_retry() {
    const N: usize = 4;
    let (session, device, pool) = selected_session();
    println!("checked host-fault propagation and retry on {device}");
    let mut host_call =
        checked_div_rem_prepare::<N, _>(&session).expect("prepare checked division kernel");
    let left = [20_u32, 19, 8, 7];
    let bad_right = [2_u32, 0, 3, 2];
    let mut quotient = [91_u32; N];
    let mut remainder = [73_u32; N];
    let error = match host_call(&left, &bad_right, &mut quotient, &mut remainder) {
        Ok(()) => panic!("zero divisor unexpectedly succeeded"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        RocmHostKernelError::CheckedExecutionFault(_)
    ));
    assert_eq!(quotient, [91; N], "faulting host outputs remain unchanged");
    assert_eq!(remainder, [73; N], "faulting host outputs remain unchanged");

    let right = [2_u32, 3, 4, 5];
    host_call(&left, &right, &mut quotient, &mut remainder)
        .expect("prepared host kernel remains retryable after checked fault");
    assert_eq!(quotient, [10, 6, 2, 1]);
    assert_eq!(remainder, [0, 1, 0, 2]);

    let mut device_call = checked_div_rem_prepare_device::<N, _>(&session)
        .expect("prepare checked resident division kernel");
    let device_left = session
        .upload_buffer(pool, &left)
        .expect("upload left operand");
    let mut device_right = session
        .upload_buffer(pool, &bad_right)
        .expect("upload bad divisor");
    let mut device_quotient = session
        .upload_buffer(pool, &[91_u32; N])
        .expect("upload quotient");
    let mut device_remainder = session
        .upload_buffer(pool, &[73_u32; N])
        .expect("upload remainder");
    let error = match device_call(
        &device_left,
        &device_right,
        &mut device_quotient,
        &mut device_remainder,
    ) {
        Ok(()) => panic!("resident zero divisor unexpectedly succeeded"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        RocmDeviceKernelError::CheckedExecutionFault(_)
    ));

    session
        .refresh_buffer(pool, &mut device_right, &right)
        .expect("refresh divisor for retry");
    device_call(
        &device_left,
        &device_right,
        &mut device_quotient,
        &mut device_remainder,
    )
    .expect("resident checked kernel remains retryable after checked fault");
    let mut quotient_readback = [0_u32; N];
    let mut remainder_readback = [0_u32; N];
    session
        .download_buffer(pool, &device_quotient, &mut quotient_readback)
        .expect("download resident quotient");
    session
        .download_buffer(pool, &device_remainder, &mut remainder_readback)
        .expect("download resident remainder");
    assert_eq!(quotient_readback, quotient);
    assert_eq!(remainder_readback, remainder);
}

fn assert_resident_values(
    session: &RocmOwnedDispatchBackend,
    pool: PcuMemoryPoolId,
    output: &PcuDeviceBuffer<f32, RocmMemoryResource>,
    expected: f32,
) {
    let mut values = vec![0.0_f32; output.len()];
    session
        .download_buffer(pool, output, &mut values)
        .expect("download resident output for conformance check");
    assert!(
        values[..65]
            .iter()
            .all(|value| value.to_bits() == expected.to_bits())
    );
    assert!(
        values[65..]
            .iter()
            .all(|value| value.to_bits() == (-11.0_f32).to_bits())
    );
}

#[test]
#[ignore = "requires a visible ROCm device; run explicitly on ROCm hardware"]
fn typed_device_calls_chain_refresh_and_reject_short_storage_before_mutation() {
    const N: usize = 65;
    let (session, device, pool) = selected_session();
    println!("typed resident chaining and validation on {device}");

    let first_input = vec![1.25_f32; N];
    let mut input = session
        .upload_buffer(pool, &first_input)
        .expect("upload typed input");
    let initial_output = vec![-11.0_f32; N + 5];
    let mut output = session
        .upload_buffer(pool, &initial_output)
        .expect("upload oversized mutable output");
    let mut transform =
        transform_prepare_device::<N, _>(&session).expect("prepare resident transform");
    let mut increment = increment_in_place_prepare_device::<N, _>(&session)
        .expect("prepare resident in-place increment");

    transform(&input, &mut output).expect("transform directly into the resident output");
    assert_resident_values(&session, pool, &output, 4.25);

    let refreshed_input = vec![7.25_f32; N];
    session
        .refresh_buffer(pool, &mut input, &refreshed_input)
        .expect("refresh resident input with new contents");
    transform(&input, &mut output).expect("reused transform observes refreshed contents");
    increment(&mut output).expect("chain in-place PCU work on the resident output");
    assert_resident_values(&session, pool, &output, 11.25);

    let short_host = vec![99.0_f32; N - 1];
    let short_physical = session
        .upload_buffer(pool, &short_host)
        .expect("allocate physically short input");
    let error = match transform(&short_physical, &mut output) {
        Ok(()) => panic!("short physical input unexpectedly succeeded"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        RocmDeviceKernelError::BufferTooSmall { .. }
    ));
    assert_resident_values(&session, pool, &output, 11.25);

    let declared_short = PcuDeviceBuffer::new(
        session
            .upload_buffer(pool, &refreshed_input)
            .expect("allocate physically full input")
            .into_resource(),
        N - 1,
    );
    let error = match transform(&declared_short, &mut output) {
        Ok(()) => panic!("short declared extent unexpectedly succeeded"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        RocmDeviceKernelError::BufferTooSmall { .. }
    ));
    assert_resident_values(&session, pool, &output, 11.25);

    let short_physical = PcuDeviceBuffer::new(short_physical.into_resource(), N);
    let error = match transform(&short_physical, &mut output) {
        Ok(()) => {
            panic!("short physical storage with a full declared extent unexpectedly succeeded")
        }
        Err(error) => error,
    };
    assert!(matches!(
        error,
        RocmDeviceKernelError::BufferTooSmall { .. }
    ));
    assert_resident_values(&session, pool, &output, 11.25);
}
