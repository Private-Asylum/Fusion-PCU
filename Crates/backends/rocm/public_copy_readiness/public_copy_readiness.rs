//! Independent nonblocking consumers observe public copy readiness before any cleanup wait.
use crate::*;
const SOURCE: &str = r#"
extern "C" __global__ void stamp(unsigned int* data, unsigned int words, unsigned int value) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < words) data[index] = value ^ (index * 0x9e3779b9U);
}
extern "C" __global__ void observe(const unsigned int* data, unsigned int words, unsigned int* result) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        result[0] = data[0];
        result[1] = data[words - 1];
    }
}
"#;
fn nonblocking(runtime: &HipRuntime) -> HipStreamHandle {
    HipStreamHandle {
        inner: Rc::new(StreamInner {
            runtime: runtime.clone(),
            raw: crate::ffi::create_nonblocking_stream_for_test(runtime).unwrap(),
        }),
    }
}
fn fill(
    kernel: &HipKernel,
    stream: &HipStreamHandle,
    buffer: &DeviceBuffer,
    words: u32,
    value: u32,
) {
    // SAFETY: The independent stamp ABI receives a full u32 allocation and copied scalar words.
    let mut completion = unsafe {
        kernel.launch(
            stream,
            [words.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            &[
                HipKernelArgument::Buffer(buffer),
                HipKernelArgument::Bytes(&words.to_ne_bytes()),
                HipKernelArgument::Bytes(&value.to_ne_bytes()),
            ],
        )
    }
    .unwrap();
    completion.wait().unwrap();
}
#[allow(clippy::too_many_lines)] // The observed consumer precedes the explicit cleanup boundary in one diagnostic.
fn witness(host_to_device: bool) {
    let runtime = HipRuntime::new(0).unwrap();
    // Compile, resolve both kernels, create the nonblocking stream and allocate before any observed copy.
    let image = crate::compile_hip_source_for_device(&runtime, SOURCE).unwrap();
    let module = runtime.load_module(&image).unwrap();
    let stamp = module.function(c"stamp").unwrap();
    let observe = module.function(c"observe").unwrap();
    let stream = nonblocking(&runtime);
    let probe = runtime.allocate(8).unwrap();
    for mib in [16_usize, 64, 256] {
        let bytes = mib * 1024 * 1024;
        let words = u32::try_from(bytes / 4).unwrap();
        let source = runtime.allocate(bytes).unwrap();
        let mut destination = runtime.allocate(bytes).unwrap();
        let mut host = if host_to_device {
            vec![0_u8; bytes]
        } else {
            Vec::new()
        };
        for iteration in 0..8_u32 {
            let value = 0x83a5_0000_u32 ^ u32::try_from(mib).unwrap() ^ iteration;
            fill(&stamp, &stream, &source, words, value);
            fill(&stamp, &stream, &destination, words, 0x1357_9bdf);
            #[cfg(feature = "allocation-census")]
            let before = crate::rocm_api_census();
            if host_to_device {
                for (index, chunk) in host.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    chunk.copy_from_slice(
                        &(value ^ u32::try_from(index).unwrap().wrapping_mul(0x9e37_79b9))
                            .to_ne_bytes(),
                    );
                }
                destination.copy_from(&host).unwrap();
            } else {
                destination.copy_from_device(&source, bytes).unwrap();
            }
            #[cfg(feature = "allocation-census")]
            {
                let after = crate::rocm_api_census();
                // Public D2D adds exactly one relevant queue wait; pageable HIP H2D
                // already completes at return. Neither success waits for the device.
                let waits = u64::from(!host_to_device);
                assert_eq!(after.runtime_calls - before.runtime_calls, 2 + 2 * waits);
                assert_eq!(
                    after.device_selections - before.device_selections,
                    1 + waits
                );
                assert_eq!(
                    after.stream_synchronizations - before.stream_synchronizations,
                    waits
                );
                assert_eq!(
                    after.device_synchronizations,
                    before.device_synchronizations
                );
                assert_eq!(
                    after.host_to_device_copies - before.host_to_device_copies,
                    u64::from(host_to_device)
                );
                assert_eq!(
                    after.device_to_device_copies - before.device_to_device_copies,
                    u64::from(!host_to_device)
                );
                assert_eq!(after.device_to_host_copies, before.device_to_host_copies);
                assert_eq!(after.allocations, before.allocations);
                assert_eq!(after.frees, before.frees);
                assert_eq!(after.kernel_launches, before.kernel_launches);
                assert_eq!(after.event_creates, before.event_creates);
                assert_eq!(after.event_records, before.event_records);
                assert_eq!(after.event_waits, before.event_waits);
                assert_eq!(after.event_destroys, before.event_destroys);
                assert_eq!(after.module_loads, before.module_loads);
                assert_eq!(after.symbol_resolutions, before.symbol_resolutions);
            }
            assert!(source.validate_access_available().is_ok());
            assert!(destination.validate_access_available().is_ok());
            // SAFETY: A distinct nonblocking stream reads a validated live u32 allocation and
            // writes exactly two words. No default-stream wait occurs before this observation.
            let mut completion = unsafe {
                observe.launch(
                    &stream,
                    [1, 1, 1],
                    [1, 1, 1],
                    0,
                    &[
                        HipKernelArgument::Buffer(&destination),
                        HipKernelArgument::Bytes(&words.to_ne_bytes()),
                        HipKernelArgument::Buffer(&probe),
                    ],
                )
            }
            .unwrap();
            assert!(destination.validate_access_available().is_err());
            assert!(probe.validate_access_available().is_err());
            completion.wait().unwrap();
            assert!(destination.validate_access_available().is_ok());
            assert!(probe.validate_access_available().is_ok());
            let mut encoded = [0_u8; 8];
            probe.copy_to(&mut encoded).unwrap();
            let actual = [
                u32::from_ne_bytes(encoded[..4].try_into().unwrap()),
                u32::from_ne_bytes(encoded[4..].try_into().unwrap()),
            ];
            let expected = [value, value ^ (words - 1).wrapping_mul(0x9e37_79b9)];
            println!(
                "rocm-public-copy-readiness/{}/{mib}MiB/{iteration}: expected={expected:x?} observed={actual:x?}",
                if host_to_device { "H2D" } else { "D2D" }
            );
            // SAFETY: Observation is already captured. Relevant null-stream cleanup prevents a
            // failed predecessor from releasing either endpoint while its old copy is pending.
            unsafe { crate::ffi::invoke_hipStreamSynchronize(&runtime, ptr::null_mut()) }.unwrap();
            assert_eq!(
                actual, expected,
                "public copy success must make all endpoints ready on a nonblocking consumer"
            );
        }
    }
}
#[test]
#[ignore = "requires HIP GPU; independent nonblocking consumer for public D2D readiness"]
fn rocm_gpu_public_d2d_is_ready_for_nonblocking_consumer() {
    witness(false);
}
#[test]
#[ignore = "requires HIP GPU; independently establishes pageable H2D readiness"]
fn rocm_gpu_public_pageable_h2d_is_ready_for_nonblocking_consumer() {
    witness(true);
}
