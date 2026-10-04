//! Public synchronous-copy readiness on a stream independent of legacy-default ordering.
#[rustfmt::skip]
use fusion_pcu_cuda::{
    CudaKernelArgument,
    CudaRuntime,
    CudaStreamMode,
    CudaStreamOptions,
    compile_cuda_source_for_device,
};

const PROBE: &str = r#"
extern "C" __global__ void copy_boundary_words(
    const unsigned int *input, unsigned int *output, unsigned long long words) {
    unsigned int lane = threadIdx.x;
    if (lane < 8) { output[lane] = input[lane]; }
    else if (lane < 16) { output[lane] = input[words - 16 + lane]; }
}
"#;

#[cfg(feature = "allocation-census")]
fn assert_copy_api(before: fusion_pcu_cuda::CudaApiCensus, host_to_device: bool) {
    let after = fusion_pcu_cuda::cuda_api_census();
    let mut expected = before;
    expected.runtime_driver_calls += 4;
    expected.device_selections += 2;
    expected.host_to_device_copies += u64::from(host_to_device);
    expected.device_to_device_copies += u64::from(!host_to_device);
    expected.stream_synchronizations += 1;
    // Instrumented wait duration is intentionally excluded from this API-count contract.
    expected.completion_wait_nanoseconds = after.completion_wait_nanoseconds;
    assert_eq!(after, expected);
}

#[test]
#[ignore = "requires native CUDA; public NonBlocking H2D/D2D changing-input readiness"]
fn completed_public_copies_are_ready_for_nonblocking_consumers() {
    let runtime = CudaRuntime::new(0).unwrap();
    // Compile and resolve before copies: cold preparation must not accidentally close a race.
    let image = compile_cuda_source_for_device(&runtime, PROBE).unwrap();
    let module = runtime.load_module(&image).unwrap();
    let kernel = module.function(c"copy_boundary_words").unwrap();
    let stream = runtime
        .create_stream_with_options(CudaStreamOptions {
            mode: CudaStreamMode::NonBlocking,
            priority: 0,
        })
        .unwrap();
    assert_eq!(stream.options().unwrap().mode, CudaStreamMode::NonBlocking);
    let output = runtime.allocate(64).unwrap();
    let mut completed = 0;
    for bytes in [16 * 1024 * 1024, 64 * 1024 * 1024, 256 * 1024 * 1024] {
        let mut host = vec![0_u8; bytes];
        let mut input = runtime.allocate(bytes).unwrap();
        let mut copied = runtime.allocate(bytes).unwrap();
        let words = u64::try_from(bytes / 4).unwrap().to_ne_bytes();
        for generation in 1_u8..=8 {
            host.fill(generation);
            #[cfg(feature = "allocation-census")]
            let before = fusion_pcu_cuda::cuda_api_census();
            input.copy_from(&host).unwrap();
            #[cfg(feature = "allocation-census")]
            assert_copy_api(before, true);
            for device_copy in [false, true] {
                let observed = if device_copy {
                    #[cfg(feature = "allocation-census")]
                    let before = fusion_pcu_cuda::cuda_api_census();
                    copied.copy_from_device(&input, bytes).unwrap();
                    #[cfg(feature = "allocation-census")]
                    assert_copy_api(before, false);
                    &copied
                } else {
                    &input
                };
                // SAFETY: input contains `words` aligned u32 values; the independent kernel
                // reads only its first/last eight and writes exactly 16 u32s to separate output.
                // Submission occurs only after the public copy returned. Buffer arguments keep
                // all allocations leased until this completion reaches a known terminal state.
                let mut completion = unsafe {
                    kernel.launch(
                        &stream,
                        [1, 1, 1],
                        [16, 1, 1],
                        0,
                        &[
                            CudaKernelArgument::Buffer(observed),
                            CudaKernelArgument::Buffer(&output),
                            CudaKernelArgument::Bytes(&words),
                        ],
                    )
                }
                .unwrap();
                completion.wait().unwrap();
                let mut actual = [0_u8; 64];
                #[cfg(feature = "allocation-census")]
                let before = fusion_pcu_cuda::cuda_api_census();
                output.copy_to(&mut actual).unwrap();
                #[cfg(feature = "allocation-census")]
                {
                    let after = fusion_pcu_cuda::cuda_api_census();
                    let mut expected = before;
                    expected.runtime_driver_calls += 2;
                    expected.device_selections += 1;
                    expected.device_to_host_copies += 1;
                    assert_eq!(after, expected);
                }
                assert_eq!(
                    actual, [generation; 64],
                    "bytes={bytes}, generation={generation}, device_copy={device_copy}"
                );
                completed += 1;
            }
        }
    }
    assert_eq!(completed, 48);
    println!("public-copy-nonblocking-readiness: 48 changing H2D/D2D boundary probes PASS");
}
