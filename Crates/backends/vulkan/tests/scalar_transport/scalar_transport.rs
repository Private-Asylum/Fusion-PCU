//! Actual raw-bit carrier transport with independent host byte oracle and private native ownership.
#[path = "device/device.rs"]
mod device;
#[path = "../../benches/scalar_transport/ffi/ffi.rs"]
#[allow(dead_code)]
mod ffi;
#[path = "sample/sample.rs"]
mod sample;
#[path = "source/source.rs"]
mod source;
use sample::{Sample, same};
#[rustfmt::skip]
use pcu_facade::{global,PcuI256,PcuU256,PcuI512,PcuU512,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,PcuF128Bits,PcuF256Bits,PcuScalar,PcuBindingRef,PcuHostArgument,PcuHostKernelBackend,PcuPreparedHostKernel};
use fusion_pcu_vulkan::{PcuVulkanBackend, PcuVulkanError, PcuVulkanPreparedHost};
macro_rules! carriers {($function:ident,$($argument:expr),*$(,)?)=>{{
    $function::<i8>($($argument),*);$function::<u8>($($argument),*);
    $function::<i16>($($argument),*);$function::<u16>($($argument),*);
    $function::<i32>($($argument),*);$function::<u32>($($argument),*);
    $function::<i64>($($argument),*);$function::<u64>($($argument),*);
    $function::<i128>($($argument),*);$function::<u128>($($argument),*);
    $function::<PcuI256>($($argument),*);$function::<PcuU256>($($argument),*);
    $function::<PcuI512>($($argument),*);$function::<PcuU512>($($argument),*);
    $function::<PcuF16Bits>($($argument),*);$function::<PcuBf16Bits>($($argument),*);
    $function::<PcuF8E4M3FnBits>($($argument),*);$function::<PcuF8E5M2Bits>($($argument),*);
    $function::<f32>($($argument),*);$function::<f64>($($argument),*);
    $function::<PcuF128Bits>($($argument),*);$function::<PcuF256Bits>($($argument),*);
}};}
fn call<T: PcuScalar>(
    graph: &mut PcuVulkanPreparedHost,
    input: &[T],
    output: &mut [T],
) -> Result<(), PcuVulkanError> {
    graph.call(&mut [
        PcuHostArgument::read_write(PcuBindingRef::new(0, 1), output),
        PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
    ])
}
fn raw<T: Sample>(backend: &PcuVulkanBackend, identity: pcu_facade::PcuStableDeviceIdentity) {
    const N: usize = 4096;
    let sentinel = T::pattern(17);
    let mut output = vec![sentinel; N + 3];
    let bindings = source::dense_bindings::<T>();
    let builder = source::dense_ir::<T, N>(&bindings).unwrap();
    let mut graph = backend.prepare_host_kernel(&builder.ir()).unwrap();
    let PcuVulkanPreparedHost::ScalarTransport(map) = &graph else {
        panic!("uniform transport owner required")
    };
    assert_eq!(map.profile().local_id(), Some(128 + T::TYPE as u32));
    assert_eq!(map.memory_realizations().unwrap().len(), 2);
    let mut native =
        ffi::NativeTransport::new(identity, u32::try_from(N).unwrap(), T::HOST_SIZE, false)
            .unwrap();
    let cases = if T::HOST_SIZE <= 2 {
        1usize << (T::HOST_SIZE * 8)
    } else {
        4096
    };
    for offset in (0..cases).step_by(N) {
        let input: Vec<T> = (0..N).map(|i| T::pattern((offset + i) % cases)).collect();
        call(&mut graph, &input, &mut output).unwrap();
        same(&output[..N], &input);
        same(&output[N..], &[sentinel; 3]);
        native
            .call(ffi::bytes(&input), ffi::bytes_mut(&mut output))
            .unwrap();
        same(&output[..N], &input);
        same(&output[N..], &[sentinel; 3]);
    }
    println!(
        "raw oracle {:?}: {cases} carrier patterns, dense provider + independently compiled native",
        T::TYPE
    );
}
#[test]
#[ignore = "requires actual Vulkan compute GPU and GLSL compiler"]
fn twenty_two_carrier_complete_narrow_and_full_width_raw_oracle() {
    let (backend, identity) = device::selected();
    carriers!(raw, &backend, identity);
}
fn source_profiles<T: Sample>(backend: &PcuVulkanBackend) {
    const N: usize = 65;
    let sentinel = T::pattern(17);
    let input: Vec<T> = (0..N).map(|i| T::pattern(i + 29)).collect();
    let mut output = [sentinel; N + 3];
    macro_rules! dense {
        ($entry:ident,$prepare:ident,$ir:ident,$bindings:ident) => {{
            let mut prepared = source::$prepare::<T, N, _>(backend).unwrap();
            let bindings = source::$bindings::<T>();
            let builder = source::$ir::<T, N>(&bindings).unwrap();
            let mut graph = backend.prepare_host_kernel(&builder.ir()).unwrap();
            prepared(&input, &mut output).unwrap();
            same(&output[..N], &input);
            same(&output[N..], &[sentinel; 3]);
            source::$entry::<T, N>(&input, &mut output).unwrap();
            same(&output[..N], &input);
            let before = output;
            assert!(call(&mut graph, &input[..N - 1], &mut output).is_err());
            same(&output, &before);
            assert!(call(&mut graph, &input, &mut output[..N - 1]).is_err());
            same(&output, &before);
            let wrong = [0u8; 1];
            assert!(
                graph
                    .call(&mut [
                        PcuHostArgument::read(PcuBindingRef::new(0, 9), &wrong),
                        PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output)
                    ])
                    .is_err()
            );
            same(&output, &before);
            call(&mut graph, &input, &mut output).unwrap();
            same(&output[..N], &input);
        }};
    }
    dense!(dense, dense_prepare, dense_ir, dense_bindings);
    dense!(
        dense_grid,
        dense_grid_prepare,
        dense_grid_ir,
        dense_grid_bindings
    );
    macro_rules! broadcast {
        ($entry:ident,$prepare:ident,$ir:ident,$bindings:ident) => {{
            let mut prepared = source::$prepare::<T, N, _>(backend).unwrap();
            let bindings = source::$bindings::<T>();
            let builder = source::$ir::<T, N>(&bindings).unwrap();
            let mut graph = backend.prepare_host_kernel(&builder.ir()).unwrap();
            let count = if T::HOST_SIZE == 1 {
                256
            } else {
                T::HOST_SIZE * 8 + 66
            };
            for seed in 0..count {
                let input = T::pattern(seed);
                prepared(&input, &mut output).unwrap();
                same(&output[..N], &[input; N]);
                same(&output[N..], &[sentinel; 3]);
                call(&mut graph, core::slice::from_ref(&input), &mut output).unwrap();
                same(&output[..N], &[input; N]);
                source::$entry::<T, N>(&input, &mut output).unwrap();
                same(&output[..N], &[input; N]);
            }
            let before = output;
            assert!(call::<T>(&mut graph, &[], &mut output).is_err());
            same(&output, &before);
            let input = T::pattern(3);
            call(&mut graph, core::slice::from_ref(&input), &mut output).unwrap();
            same(&output[..N], &[input; N]);
        }};
    }
    broadcast!(
        broadcast,
        broadcast_prepare,
        broadcast_ir,
        broadcast_bindings
    );
    broadcast!(
        broadcast_grid,
        broadcast_grid_prepare,
        broadcast_grid_ir,
        broadcast_grid_bindings
    );
}
#[test]
#[ignore = "requires actual Vulkan compute GPU"]
fn ordinary_prepared_graph_direct_grid_broadcast_transaction_retry_and_tails() {
    let (backend, _) = device::selected();
    for mode in [
        pcu_facade::PcuNumericalMode::Boundary,
        pcu_facade::PcuNumericalMode::Strict,
    ] {
        global::configure(global::PcuExecutionPolicy {
            backend: global::PcuBackendChoice::Vulkan,
            numerical_mode: mode,
            ..Default::default()
        })
        .unwrap();
        global::clear_thread_cache().unwrap();
        carriers!(source_profiles, &backend);
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
fn admission<T: Sample>(backend: &PcuVulkanBackend) {
    #[rustfmt::skip]
    use pcu_facade::{PcuCostBoundary,PcuExecutorId,PcuImplementationOffers,PcuImplementationRequest,PcuNumericalOptions,PcuReproducibility};
    let bindings = source::broadcast_bindings::<T>();
    let builder = source::broadcast_ir::<T, 65>(&bindings).unwrap();
    let mut kernel = builder.ir();
    let request = PcuImplementationRequest {
        device: backend.device_identity().unwrap(),
        executor: PcuExecutorId(0),
        operation: &kernel,
        requirements: kernel.numerical_requirements,
        boundary: PcuCostBoundary::Host,
    };
    let mut offers = [None];
    assert_eq!(
        backend
            .implementation_offers(&request, &mut offers)
            .unwrap(),
        1
    );
    assert_eq!(
        offers[0].unwrap().implementation.local_id,
        160 + T::TYPE as u32
    );
    kernel.numerical_requirements.numerical_options = PcuNumericalOptions {
        reproducibility: PcuReproducibility::PortableV1,
        ..Default::default()
    };
    let request = PcuImplementationRequest {
        operation: &kernel,
        requirements: kernel.numerical_requirements,
        device: backend.device_identity().unwrap(),
        executor: PcuExecutorId(0),
        boundary: PcuCostBoundary::Host,
    };
    assert_eq!(
        backend
            .implementation_offers(&request, &mut offers)
            .unwrap(),
        0
    );
    assert!(backend.prepare_host_kernel(&kernel).is_err());
}
#[test]
#[ignore = "requires physical discovery for exact cold offers"]
fn exact_carrier_ids_and_portable_exclusion() {
    let (backend, _) = device::selected();
    carriers!(admission, &backend);
}
