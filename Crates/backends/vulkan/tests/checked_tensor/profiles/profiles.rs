//! Actual source and native-owner graphs, independent midpoint/base256/bigint value corpora.
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuScalar,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuNumericalOptions,
    PcuReproducibility,
    PcuDispatchIntegerBinaryOp,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
#[rustfmt::skip]
use pcu_facade::dialect::tensor::{
    Graph,
    TensorElement,
};
#[rustfmt::skip]
use fusion_pcu_vulkan::{
    PcuVulkanBackend,
    PcuVulkanPreparedTensorGraph,
    PcuVulkanTensorInput,
};
use super::{bits, device, integer_oracle, low_oracle, sample::Sample, source, wide_oracle};

const POLICIES: [PcuFloatUnderflowPolicy; 3] = [
    PcuFloatUnderflowPolicy::IeeeAfterRounding,
    PcuFloatUnderflowPolicy::RejectSubnormalResult,
    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
];

pub fn graph<T: PcuScalar>(
    n: usize,
    op: u32,
    policy: PcuFloatUnderflowPolicy,
) -> (Graph, pcu_facade::dialect::tensor::ValueId) {
    configured_graph::<T>(
        n,
        op,
        policy,
        PcuNumericalMode::Boundary,
        PcuNumericalOptions::default(),
    )
}

fn configured_graph<T: PcuScalar>(
    n: usize,
    op: u32,
    policy: PcuFloatUnderflowPolicy,
    mode: PcuNumericalMode,
    options: PcuNumericalOptions,
) -> (Graph, pcu_facade::dialect::tensor::ValueId) {
    let mut graph = Graph::default();
    graph.set_numerical_mode(mode);
    graph.set_numerical_options(options);
    let a = graph.input([n], T::TYPE).unwrap();
    let b = graph.input([n], T::TYPE).unwrap();
    let output = match op {
        0 => graph.add(a, b),
        1 => graph.sub(a, b),
        2 => graph.mul(a, b),
        3 => graph.div(a, b),
        4 => graph.relu(a),
        _ => unreachable!(),
    }
    .unwrap();
    if matches!(
        T::TYPE,
        pcu_facade::PcuScalarType::F16
            | pcu_facade::PcuScalarType::BF16
            | pcu_facade::PcuScalarType::F8E4M3FN
            | pcu_facade::PcuScalarType::F8E5M2
            | pcu_facade::PcuScalarType::F32
            | pcu_facade::PcuScalarType::F64
    ) {
        graph
            .set_value_float_underflow_policy(output, policy)
            .unwrap();
    }
    (graph, output)
}

fn batch<T: TensorElement + PcuScalar>(
    backend: &PcuVulkanBackend,
    op: u32,
    policy: PcuFloatUnderflowPolicy,
    a: &[T],
    b: &[T],
    want: &[T],
    sentinel: T,
) {
    let (graph, output) = graph::<T>(a.len(), op, policy);
    let mut plan = PcuVulkanPreparedTensorGraph::<T>::prepare(backend, &graph, &[output]).unwrap();
    drop(graph);
    let left = backend.upload_owned(a).unwrap();
    let right = backend.upload_owned(b).unwrap();
    let mut observed = vec![sentinel; a.len() + 3];
    for borrowed in [false, true] {
        let args = if borrowed {
            [
                PcuVulkanTensorInput::Owned(&left),
                PcuVulkanTensorInput::Owned(&right),
            ]
        } else {
            [PcuVulkanTensorInput::Host(a), PcuVulkanTensorInput::Host(b)]
        };
        let result = plan
            .execute_owned(&args[..if op == 4 { 1 } else { 2 }])
            .unwrap();
        result.read_into(&mut observed).unwrap();
        bits::same(&observed[..a.len()], want);
        bits::same(&observed[a.len()..], &[sentinel; 3]);
        let source_a = source::retain(a).unwrap();
        let source_b = source::retain(b).unwrap();
        let result = if borrowed {
            source::call(op, &source_a, &source_b).unwrap()
        } else {
            source::call(op, a, b).unwrap()
        };
        result.read_into(&mut observed).unwrap();
        bits::same(&observed[..a.len()], want);
        bits::same(&observed[a.len()..], &[sentinel; 3]);
    }
}

fn numeric<T: Sample>(backend: &PcuVulkanBackend) {
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for policy in POLICIES {
                    global::configure(global::PcuExecutionPolicy {
                        backend: global::PcuBackendChoice::Vulkan,
                        numerical_mode: mode,
                        numerical_options: PcuNumericalOptions {
                            compound_arithmetic: compound,
                            precision,
                            reproducibility: PcuReproducibility::Unspecified,
                        },
                        float_underflow: policy,
                        ..Default::default()
                    })
                    .unwrap();
                    for op in 0..if T::FLOAT { 5 } else { 3 } {
                        let a = [T::small(3); 65];
                        let b = [T::small(1); 65];
                        let expected = [T::small(match op {
                            0 => 4,
                            1 => 2,
                            2..=4 => 3,
                            _ => unreachable!(),
                        }); 65];
                        let (graph, output) = configured_graph::<T>(
                            65,
                            op,
                            policy,
                            mode,
                            PcuNumericalOptions {
                                compound_arithmetic: compound,
                                precision,
                                reproducibility: PcuReproducibility::Unspecified,
                            },
                        );
                        let owner = source::call(op, &a, &b).unwrap();
                        let mut observed = [T::small(11); 68];
                        owner.read_into(&mut observed).unwrap();
                        bits::same(&observed[..65], &expected);
                        bits::same(&observed[65..], &[T::small(11); 3]);
                        let mut native =
                            PcuVulkanPreparedTensorGraph::<T>::prepare(backend, &graph, &[output])
                                .unwrap();
                        assert_eq!(native.input_bindings().len(), if op == 4 { 1 } else { 2 });
                        let args = [
                            PcuVulkanTensorInput::Host(&a),
                            PcuVulkanTensorInput::Host(&b),
                        ];
                        native
                            .execute_owned(&args[..if op == 4 { 1 } else { 2 }])
                            .unwrap()
                            .read_into(&mut observed)
                            .unwrap();
                        bits::same(&observed[..65], &expected);
                        bits::same(&observed[65..], &[T::small(11); 3]);
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires actual Vulkan GPU; complete twenty-format pointwise source permissions"]
fn twenty_formats_pointwise_source_permissions() {
    let (backend, _) = device::selected();
    macro_rules! types {($($ty:ty),+)=>{$(numeric::<$ty>(&backend);)+};}
    types!(
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
        PcuF16Bits,
        PcuBf16Bits,
        PcuF8E4M3FnBits,
        PcuF8E5M2Bits,
        f32,
        f64
    );
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

fn low<T: low_oracle::Format + TensorElement>(backend: &PcuVulkanBackend) {
    for policy in POLICIES {
        global::configure(global::PcuExecutionPolicy {
            backend: global::PcuBackendChoice::Vulkan,
            float_underflow: policy,
            ..Default::default()
        })
        .unwrap();
        for op in 0..5 {
            for phase in [1, 17, 51] {
                let (a, b, want) = low_oracle::inputs::<T>(257, phase, op, policy);
                batch(backend, op, policy, &a, &b, &want, T::sentinel());
            }
        }
    }
}
#[test]
#[ignore = "requires actual Vulkan GPU; independent low-format midpoint result corpus"]
fn low_four_independent_midpoint_corpus() {
    let (backend, _) = device::selected();
    low::<PcuF16Bits>(&backend);
    low::<PcuBf16Bits>(&backend);
    low::<PcuF8E4M3FnBits>(&backend);
    low::<PcuF8E5M2Bits>(&backend);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

fn wide<T: wide_oracle::Format + TensorElement>(backend: &PcuVulkanBackend) {
    for op in 0..3 {
        for phase in [0, 17, 51] {
            let (a, b, want) = wide_oracle::inputs::<T>(65, phase, op);
            batch(backend, op, POLICIES[0], &a, &b, &want, T::sentinel());
        }
    }
}
#[test]
#[ignore = "requires actual Vulkan GPU; independently generated full-bit bigint tensor rows"]
fn six_wide_bigint_bytes_and_native_owners() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })
    .unwrap();
    let (backend, _) = device::selected();
    wide::<i128>(&backend);
    wide::<u128>(&backend);
    wide::<PcuI256>(&backend);
    wide::<PcuU256>(&backend);
    wide::<PcuI512>(&backend);
    wide::<PcuU512>(&backend);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

fn integer<T: integer_oracle::Wide + TensorElement>(backend: &PcuVulkanBackend) {
    let mut state = 0x729b_d813_15c4_a0f7_u64;
    for (op, operation) in [
        PcuDispatchIntegerBinaryOp::Add,
        PcuDispatchIntegerBinaryOp::Sub,
        PcuDispatchIntegerBinaryOp::Mul,
    ]
    .into_iter()
    .enumerate()
    {
        let (mut a, mut b, mut want) = (Vec::new(), Vec::new(), Vec::new());
        while a.len() < 65 {
            let pair: [T; 2] = std::array::from_fn(|_| {
                let mut bytes = [0; 64];
                let count = if op == 2 {
                    (T::BYTES / 2).max(1)
                } else {
                    T::BYTES
                };
                for byte in &mut bytes[..count] {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    *byte = state.to_le_bytes()[0];
                }
                if op == 2 {
                    bytes[count - 1] &= 0x3f;
                }
                T::from_bytes(bytes)
            });
            if let Ok(expected) = integer_oracle::evaluate(pair[0], pair[1], operation) {
                a.push(pair[0]);
                b.push(pair[1]);
                want.push(expected);
            }
        }
        batch(
            backend,
            u32::try_from(op).unwrap(),
            POLICIES[0],
            &a,
            &b,
            &want,
            integer_oracle::small(11),
        );
    }
}
#[test]
#[ignore = "requires actual Vulkan GPU; independent base256 raw integer tensor operands"]
fn fourteen_integer_independent_base256_corpus() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })
    .unwrap();
    let (backend, _) = device::selected();
    macro_rules! types {($($ty:ty),+)=>{$(integer::<$ty>(&backend);)+};}
    types!(
        i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, PcuI256, PcuU256, PcuI512, PcuU512
    );
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
