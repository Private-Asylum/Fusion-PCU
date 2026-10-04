//! Exact raw encodings and detached ownership; no wide arithmetic inference.
#[rustfmt::skip]
use super::{
    source::{
        self,
        RawSource as Raw,
    },
    ffi,
};
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuNumericalOptions,
    PcuNumericalMode,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuFloatUnderflowPolicy,
    PcuStableDeviceIdentity,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    Tensor,
};
#[rustfmt::skip]
use fusion_pcu_vulkan::{
    PcuVulkanBackend,
    PcuVulkanPreparedTensorGraph,
};
fn raw_vectors<T: Raw, const N: usize>(uniform: bool) -> Vec<u8> {
    let mut patterns = [
        vec![0_u8; T::ENCODED_SIZE],
        vec![0xff; T::ENCODED_SIZE],
        vec![0_u8; T::ENCODED_SIZE],
    ];
    patterns[0][T::ENCODED_SIZE - 1] = 0x80;
    patterns[2][0] = 7;
    (0..N)
        .flat_map(|index| {
            let pattern = if uniform {
                0
            } else if N == 6 && index >= 3 {
                5 - index
            } else {
                index % 3
            };
            patterns[pattern].clone()
        })
        .collect()
}

#[allow(clippy::chunks_exact_to_as_chunks)] // Closed carrier byte geometry is an associated constant.
fn vectors<T: Raw, const N: usize>(uniform: bool) -> Vec<T> {
    raw_vectors::<T, N>(uniform)
        .chunks_exact(T::ENCODED_SIZE)
        .map(T::from_bytes)
        .collect()
}

fn same<T: Raw>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert_eq!(a.encode_le().as_ref(), b.encode_le().as_ref());
    }
}
fn verify<T: Raw>(actual: &[T], expected: &[T]) {
    same(&actual[..expected.len()], expected);
    same(&actual[expected.len()..], &[T::SENTINEL; 2]);
}
fn source_owner<T: Raw, const N: usize>(uniform: bool) -> fusion_pcu::PcuTensor<T> {
    if uniform {
        T::uniform::<N>().unwrap()
    } else {
        T::literal::<N>().unwrap()
    }
}
#[allow(clippy::too_many_lines)] // Source and explicit/native owners share one precise raw profile and lifecycle.
fn case<T: Raw, const N: usize>(
    backend: &PcuVulkanBackend,
    identity: PcuStableDeviceIdentity,
    mode: PcuNumericalMode,
    options: PcuNumericalOptions,
    uf: PcuFloatUnderflowPolicy,
) {
    global::configure(global::PcuExecutionPolicy {
        backend: if std::env::var_os("PCU_RAW_INTEGER_VULKAN_AUTOMATIC").is_some() {
            global::PcuBackendChoice::Automatic
        } else {
            global::PcuBackendChoice::Vulkan
        },
        numerical_mode: mode,
        numerical_options: options,
        float_underflow: uf,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    T::capture::<N>(mode, options, uf);
    let shape = if N == 6 { vec![2, 3] } else { vec![N] };
    let mut portable = Graph::default();
    portable.set_numerical_options(fusion_pcu::PcuNumericalOptions {
        reproducibility: fusion_pcu::PcuReproducibility::PortableV1,
        ..options
    });
    let p = portable.constant_typed(Tensor::new(shape.clone(), vectors::<T, N>(false)).unwrap());
    assert!(matches!(
        PcuVulkanPreparedTensorGraph::<T>::assess(&portable, &[p.erase()]),
        Err(fusion_pcu_vulkan::PcuVulkanTensorError::Graph(
            fusion_pcu::dialect::tensor::TensorError::UnsupportedNumericalOptions { .. }
        ))
    ));
    let mut out = vec![T::SENTINEL; N + 2];
    for uniform in [false, true] {
        let expected = vectors::<T, N>(uniform);
        let mut graph = Graph::default();
        graph.set_numerical_mode(mode);
        graph.set_numerical_options(options);
        let output = if uniform {
            graph
                .uniform_typed(shape.clone(), expected[0])
                .unwrap()
                .erase()
        } else {
            graph
                .constant_typed(Tensor::new(shape.clone(), expected.clone()).unwrap())
                .erase()
        };
        let mut plan =
            PcuVulkanPreparedTensorGraph::<T>::prepare(backend, &graph, &[output]).unwrap();
        drop(graph);
        let mut native = ffi::NativeCopy::new(identity, N * T::ENCODED_SIZE).unwrap();
        // Native uploads and expected readbacks never roundtrip through a PCU codec.
        let bytes = raw_vectors::<T, N>(uniform);
        let mut value_expected = vectors::<T, N>(false);
        for (index, value) in value_expected.iter_mut().enumerate() {
            let pattern = if N == 6 && index >= 3 {
                5 - index
            } else {
                index % 3
            };
            let mut bytes = value.encode_le().as_ref().to_vec();
            bytes[0] = match pattern {
                0 => bytes[0] | 1,
                1 => 0xfe,
                _ => 8,
            };
            *value = T::from_bytes(&bytes);
        }
        let alternate_owner = T::alternate::<N>().unwrap();
        alternate_owner.read_into(&mut out).unwrap();
        verify(&out, &value_expected);
        let mut owner = source_owner::<T, N>(uniform);
        assert_eq!(owner.shape(), shape);
        owner.read_into(&mut out).unwrap();
        verify(&out, &expected);
        let mut short = vec![T::SENTINEL; N - 1];
        assert!(owner.read_into(&mut short).is_err());
        same(&short, &vec![T::SENTINEL; N - 1]);
        overwrite::<T, N>(&mut owner);
        let consumed = source::consume(owner).unwrap();
        consumed.read_into(&mut out).unwrap();
        verify(&out, &vec![T::SENTINEL; N]);
        let replay = source_owner::<T, N>(uniform);
        replay.read_into(&mut out).unwrap();
        verify(&out, &expected);
        global::clear_thread_cache().unwrap();
        consumed.read_into(&mut out).unwrap();
        verify(&out, &vec![T::SENTINEL; N]);
        drop(consumed);
        drop(replay);
        let graph_owner = plan.execute_owned(&[]).unwrap();
        assert!(
            plan.execute_owned(&[fusion_pcu_vulkan::PcuVulkanTensorInput::Host(&expected)])
                .is_err()
        );
        assert!(graph_owner.read_into(&mut short).is_err());
        same(&short, &vec![T::SENTINEL; N - 1]);
        let native_owner = native.copy_host(&bytes).unwrap();
        let mut native_bytes = vec![0; bytes.len()];
        graph_owner.read_into(&mut out).unwrap();
        verify(&out, &expected);
        native_read(&native_owner, &mut native_bytes, &bytes, &mut out[..N]);
        verify(&out, &expected);
        #[cfg(feature = "insights")]
        let (alternate_expected, alternate_bytes, mut alternate_plan) = {
            let alternate_expected = vectors::<T, N>(!uniform);
            let alternate_bytes = raw_vectors::<T, N>(!uniform);
            let mut graph = Graph::default();
            graph.set_numerical_mode(mode);
            graph.set_numerical_options(options);
            let alternate = if uniform {
                graph
                    .constant_typed(Tensor::new(shape.clone(), alternate_expected.clone()).unwrap())
                    .erase()
            } else {
                graph
                    .uniform_typed(shape.clone(), alternate_expected[0])
                    .unwrap()
                    .erase()
            };
            let alternate_plan =
                PcuVulkanPreparedTensorGraph::<T>::prepare(backend, &graph, &[alternate]).unwrap();
            source_owner::<T, N>(uniform).read_into(&mut out).unwrap();
            source_owner::<T, N>(!uniform).read_into(&mut out).unwrap();
            (alternate_expected, alternate_bytes, alternate_plan)
        };
        for route in 0..3 {
            #[cfg(not(feature = "insights"))]
            for _ in 0..4 {
                match route {
                    0 => source_owner::<T, N>(uniform).read_into(&mut out).unwrap(),
                    1 => plan
                        .execute_owned(&[])
                        .unwrap()
                        .read_into(&mut out)
                        .unwrap(),
                    _ => {
                        let owner = native.copy_host(&bytes).unwrap();
                        native_read(&owner, &mut native_bytes, &bytes, &mut out[..N]);
                    }
                }
                verify(&out, &expected);
            }
            #[cfg(feature = "insights")]
            {
                let name = format!(
                    "{mode:?}/{:?}/{:?}/{uf:?}/{}/{N}/uniform={uniform}/{route}",
                    options.compound_arithmetic,
                    options.precision,
                    T::LABEL
                );
                super::census::warm(&name, route, || {
                    let heap = super::heap::count_heap(|| {
                        for call in 0..64 {
                            let alternate = call & 1 != 0;
                            match route {
                                0 => source_owner::<T, N>(uniform ^ alternate)
                                    .read_into(&mut out)
                                    .unwrap(),
                                1 => if alternate {
                                    &mut alternate_plan
                                } else {
                                    &mut plan
                                }
                                .execute_owned(&[])
                                .unwrap()
                                .read_into(&mut out)
                                .unwrap(),
                                _ => {
                                    let owner = native
                                        .copy_host(if alternate {
                                            &alternate_bytes
                                        } else {
                                            &bytes
                                        })
                                        .unwrap();
                                    native_read(
                                        &owner,
                                        &mut native_bytes,
                                        if alternate { &alternate_bytes } else { &bytes },
                                        &mut out[..N],
                                    );
                                }
                            }
                            verify(
                                &out,
                                if alternate {
                                    &alternate_expected
                                } else {
                                    &expected
                                },
                            );
                        }
                    });
                    if route == 2 {
                        assert_eq!(
                            (heap.allocations, heap.reallocations, heap.frees),
                            (0, 0, 0)
                        );
                    }
                    println!("vulkan-raw-integer-producer-heap/{name}:64-changing-calls {heap:?}");
                });
            }
            println!(
                "vulkan-raw-integer-producer/{mode:?}/{:?}/{:?}/{uf:?}/{}/{N}/uniform={uniform}/{route}: exact bits/tails/ownership PASS",
                options.compound_arithmetic,
                options.precision,
                T::LABEL
            );
        }
        drop(native);
        native_read(&native_owner, &mut native_bytes, &bytes, &mut out[..N]);
        verify(&out, &expected);
        alternate_owner.read_into(&mut out).unwrap();
        verify(&out, &value_expected);
        drop(plan);
        graph_owner.read_into(&mut out).unwrap();
        verify(&out, &expected);
        println!(
            "vulkan-raw-integer-producer-lifetime/{mode:?}/{:?}/{:?}/{uf:?}/{}/{N}/uniform={uniform}: noargument/rank2/mutation/consume/replay/cacheclear/drop PASS",
            options.compound_arithmetic,
            options.precision,
            T::LABEL
        );
    }
}
#[allow(clippy::chunks_exact_to_as_chunks)] // Encoded size is an associated value in the closed generic fixture.
fn native_read<T: Raw>(owner: &ffi::Owner, bytes: &mut [u8], expected: &[u8], out: &mut [T]) {
    owner.read(bytes);
    assert_eq!(bytes, expected);
    for (value, chunk) in out.iter_mut().zip(bytes.chunks_exact(T::ENCODED_SIZE)) {
        *value = T::from_bytes(chunk);
    }
}
#[cfg(test)]
fn cpu_case<T: Raw, const N: usize>(
    mode: PcuNumericalMode,
    options: PcuNumericalOptions,
    uf: PcuFloatUnderflowPolicy,
) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        numerical_mode: mode,
        numerical_options: options,
        float_underflow: uf,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    T::capture::<N>(mode, options, uf);
    for uniform in [false, true] {
        let expected = vectors::<T, N>(uniform);
        let mut out = vec![T::SENTINEL; N + 2];
        let mut value_expected = vectors::<T, N>(false);
        for (index, value) in value_expected.iter_mut().enumerate() {
            let pattern = if N == 6 && index >= 3 {
                5 - index
            } else {
                index % 3
            };
            let mut bytes = value.encode_le().as_ref().to_vec();
            bytes[0] = match pattern {
                0 => bytes[0] | 1,
                1 => 0xfe,
                _ => 8,
            };
            *value = T::from_bytes(&bytes);
        }
        let alternate_owner = T::alternate::<N>().unwrap();
        alternate_owner.read_into(&mut out).unwrap();
        verify(&out, &value_expected);
        let mut owner = source_owner::<T, N>(uniform);
        owner.read_into(&mut out).unwrap();
        verify(&out, &expected);
        overwrite::<T, N>(&mut owner);
        let consumed = source::consume(owner).unwrap();
        consumed.read_into(&mut out).unwrap();
        verify(&out, &vec![T::SENTINEL; N]);
        source_owner::<T, N>(uniform).read_into(&mut out).unwrap();
        verify(&out, &expected);
        global::clear_thread_cache().unwrap();
        consumed.read_into(&mut out).unwrap();
        verify(&out, &vec![T::SENTINEL; N]);
        println!(
            "vulkan-raw-integer-cpu-reference/{mode:?}/{:?}/{:?}/{uf:?}/{}/{N}/uniform={uniform}: exact capture/zeroargs/rank2/ownership PASS",
            options.compound_arithmetic,
            options.precision,
            T::LABEL
        );
    }
}
fn dispatch<T: Raw, const N: usize>(
    selected: Option<&(PcuVulkanBackend, PcuStableDeviceIdentity)>,
    mode: PcuNumericalMode,
    options: PcuNumericalOptions,
    uf: PcuFloatUnderflowPolicy,
) {
    if let Some((backend, identity)) = selected {
        case::<T, N>(backend, *identity, mode, options, uf);
        return;
    }
    #[cfg(test)]
    cpu_case::<T, N>(mode, options, uf);
}
pub fn run() {
    let cpu = cfg!(test) && std::env::var_os("PCU_RAW_INTEGER_CPU_REFERENCE").is_some();
    let selected = if cpu {
        None
    } else {
        Some(super::device::selected())
    };
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound_arithmetic in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for uf in [
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                ] {
                    let options = PcuNumericalOptions {
                        compound_arithmetic,
                        precision,
                        ..Default::default()
                    };
                    macro_rules! width {
                        ($ty:ty) => {
                            dispatch::<$ty, 6>(selected.as_ref(), mode, options, uf);
                            dispatch::<$ty, 65>(selected.as_ref(), mode, options, uf);
                            dispatch::<$ty, 4096>(selected.as_ref(), mode, options, uf);
                        };
                    }
                    width!(u8);
                    width!(i8);
                    width!(u16);
                    width!(i16);
                    width!(u32);
                    width!(i32);
                    width!(u64);
                    width!(i64);
                    width!(u128);
                    width!(i128);
                    width!(fusion_pcu::PcuU256);
                    width!(fusion_pcu::PcuI256);
                    width!(fusion_pcu::PcuU512);
                    width!(fusion_pcu::PcuI512);
                }
            }
        }
    }
}

fn overwrite<T: Raw, const N: usize>(owner: &mut fusion_pcu::PcuTensor<T>) {
    if N == 6 {
        source::overwrite_matrix(&[[T::SENTINEL; 3]; 2], owner).unwrap();
    } else {
        source::overwrite::<T, N>(&[T::SENTINEL], owner).unwrap();
    }
}
