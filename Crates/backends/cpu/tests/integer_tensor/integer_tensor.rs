//! Independent base-256 tensor arithmetic, ownership and whole-call publication.
#[path = "../prepared_tensor/allocation/allocation.rs"]
mod allocation;
#[path = "../wide_integer/oracle/oracle.rs"]
mod oracle;
#[path = "source/source.rs"]
mod source;
use oracle::Wide;
#[rustfmt::skip]
use pcu_facade::{global,PcuDispatchIntegerBinaryOp,PcuScalar,PcuNumericalMode,PcuNumericalOptions,PcuReproducibility,PcuCompoundArithmeticPolicy,PcuPrecisionPolicy,PcuI256,PcuU256,PcuI512,PcuU512};
#[rustfmt::skip]
use pcu_facade::dialect::tensor::{Graph,Tensor,TensorElement,TensorError};
use fusion_pcu_cpu::PcuCpuPreparedIntegerTensorGraph;
const OPS: [PcuDispatchIntegerBinaryOp; 3] = [
    PcuDispatchIntegerBinaryOp::Add,
    PcuDispatchIntegerBinaryOp::Sub,
    PcuDispatchIntegerBinaryOp::Mul,
];
fn source_call<T: Wide>(
    op: PcuDispatchIntegerBinaryOp,
    a: &[T],
    b: &[T],
) -> Result<pcu_facade::PcuTensor<T>, global::PcuExecutionError> {
    match op {
        PcuDispatchIntegerBinaryOp::Add => source::add(a, b),
        PcuDispatchIntegerBinaryOp::Sub => source::sub(a, b),
        PcuDispatchIntegerBinaryOp::Mul => source::mul(a, b),
    }
}
#[allow(clippy::too_many_lines)] // One detached planner and ordinary owner are compared against every independent full-bit pair.
fn sweep<T: Wide + TensorElement>() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    let zero = oracle::small::<T>(0);
    let one = oracle::small::<T>(1);
    let sentinel = oracle::small::<T>(77);
    let edges = [
        zero,
        one,
        oracle::small(2),
        oracle::minimum(),
        oracle::maximum(),
    ];
    let mut state = 0x99d4_5bef_6384_23d1u64;
    let count = if T::BYTES == 1 { 65536 } else { 2048 };
    for op in OPS {
        let mut graph = Graph::default();
        let a = graph.input([1], T::TYPE).unwrap();
        let b = graph.input([1], T::TYPE).unwrap();
        let output = match op {
            PcuDispatchIntegerBinaryOp::Add => graph.add(a, b),
            PcuDispatchIntegerBinaryOp::Sub => graph.sub(a, b),
            PcuDispatchIntegerBinaryOp::Mul => graph.mul(a, b),
        }
        .unwrap();
        let mut plan = PcuCpuPreparedIntegerTensorGraph::<T>::prepare(&graph, &[output]).unwrap();
        drop(graph);
        for row in 0..count {
            let mut a = [0; 64];
            let mut b = [0; 64];
            let (left, right) = if T::BYTES == 1 {
                a[0] = u8::try_from(row / 256).unwrap();
                b[0] = u8::try_from(row % 256).unwrap();
                (T::from_bytes(a), T::from_bytes(b))
            } else if row < 25 {
                (edges[row / 5], edges[row % 5])
            } else {
                for bytes in [&mut a, &mut b] {
                    for byte in &mut bytes[..T::BYTES] {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        *byte = state.to_le_bytes()[0];
                    }
                }
                (T::from_bytes(a), T::from_bytes(b))
            };
            let expected = oracle::evaluate(left, right, op);
            let mut observed = [sentinel; 3];
            let result = plan.call(&[&[left], &[right]], &mut [&mut observed]);
            match expected {
                Ok(value) => {
                    result.unwrap();
                    assert_eq!(observed, [value, sentinel, sentinel]);
                    let owner = source_call(op, &[left], &[right]).unwrap();
                    observed.fill(sentinel);
                    owner.read_into(&mut observed).unwrap();
                    assert_eq!(observed, [value, sentinel, sentinel]);
                }
                Err(kind) => {
                    assert!(
                        matches!(result,Err(TensorError::ArithmeticFault{value,element_index:0,kind:k})if value==output&&k==kind)
                    );
                    assert_eq!(observed, [sentinel; 3]);
                    assert!(plan.output(0).is_none());
                    assert!(source_call(op, &[left], &[right]).is_err());
                }
            }
        }
        let mut observed = [sentinel; 3];
        assert!(plan.call(&[&[], &[one]], &mut [&mut observed]).is_err());
        assert_eq!(observed, [sentinel; 3]);
        assert_eq!(
            allocation::count(|| {
                for _ in 0..64 {
                    plan.call(&[&[one], &[one]], &mut [&mut observed]).unwrap();
                }
            }),
            0
        );
        assert_eq!(observed[1..], [sentinel; 2]);
    }
    let input = [one; 7];
    let mut observed = [sentinel; 10];
    let owner = source::identity(&input).unwrap();
    let sibling = source::identity(&owner).unwrap();
    source::add(&owner, &input)
        .unwrap()
        .read_into(&mut observed)
        .unwrap();
    assert_eq!(
        allocation::count(|| {
            let output = source::add(&owner, &input).unwrap();
            output.read_into(&mut observed).unwrap();
            drop(output);
        }),
        1
    );
    sibling.read_into(&mut observed).unwrap();
    assert_eq!(&observed[..7], &input);
    source::consume(owner)
        .unwrap()
        .read_into(&mut observed)
        .unwrap();
    source::permitted(&input, &input).unwrap();
    assert!(source::portable(&input, &input).is_err());
    assert_eq!(observed[7..], [sentinel; 3]);
    println!("{:?} independent tensor pairs {}", T::TYPE, count * 3);
}
macro_rules! widths{($($name:ident:$ty:ty),+)=>{$(#[test]fn $name(){sweep::<$ty>();})+};}
widths!(i8_exact:i8,u8_exact:u8,i16_exact:i16,u16_exact:u16,i32_exact:i32,u32_exact:u32,i64_exact:i64,u64_exact:u64,i128_exact:i128,u128_exact:u128,i256_exact:PcuI256,u256_exact:PcuU256,i512_exact:PcuI512,u512_exact:PcuU512);
#[test]
fn cold_permissions_constants_repeated_operands_detached_graph_and_publication() {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let x = graph.input([3], i32::TYPE).unwrap();
    let constant = graph.constant_typed(Tensor::new([3], vec![2i32; 3]).unwrap());
    let uniform = graph.uniform_typed([3], 3i32).unwrap();
    let added = graph.add(x, constant.erase()).unwrap();
    let multiplied = graph.mul(added, uniform.erase()).unwrap();
    let repeated = graph.sub(multiplied, multiplied).unwrap();
    let mut plan =
        PcuCpuPreparedIntegerTensorGraph::<i32>::prepare(&graph, &[added, repeated]).unwrap();
    drop(graph);
    let mut first = [77; 5];
    let mut second = [77; 5];
    plan.call(&[&[1, 2, 3]], &mut [&mut first, &mut second])
        .unwrap();
    assert_eq!(first, [3, 4, 5, 77, 77]);
    assert_eq!(second, [0, 0, 0, 77, 77]);
    let before = (first, second);
    assert!(
        plan.call(&[&[1, i32::MAX, 3]], &mut [&mut first, &mut second])
            .is_err()
    );
    assert_eq!((first, second), before);
    assert!(plan.output(0).is_none());
    plan.call(&[&[1, 2, 3]], &mut [&mut first, &mut second])
        .unwrap();
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for repro in [
                    PcuReproducibility::Unspecified,
                    PcuReproducibility::PortableV1,
                ] {
                    let mut g = Graph::default();
                    g.set_numerical_mode(mode);
                    g.set_numerical_options(PcuNumericalOptions {
                        compound_arithmetic: compound,
                        precision,
                        reproducibility: repro,
                    });
                    let a = g.input([1], i32::TYPE).unwrap();
                    let b = g.add(a, a).unwrap();
                    assert_eq!(
                        PcuCpuPreparedIntegerTensorGraph::<i32>::prepare(&g, &[b]).is_ok(),
                        repro == PcuReproducibility::Unspecified
                    );
                }
            }
        }
    }
}

fn golden<T: Wide + TensorElement>(label: &str) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    let mut count = 0;
    for (code, op) in OPS.into_iter().enumerate() {
        let mut graph = Graph::default();
        let a = graph.input([1], T::TYPE).unwrap();
        let b = graph.input([1], T::TYPE).unwrap();
        let value = match op {
            PcuDispatchIntegerBinaryOp::Add => graph.add(a, b),
            PcuDispatchIntegerBinaryOp::Sub => graph.sub(a, b),
            PcuDispatchIntegerBinaryOp::Mul => graph.mul(a, b),
        }
        .unwrap();
        let mut plan = PcuCpuPreparedIntegerTensorGraph::<T>::prepare(&graph, &[value]).unwrap();
        for line in include_str!("oracle/vectors.txt")
            .lines()
            .filter(|line| !line.starts_with('#'))
        {
            let fields: Vec<_> = line.split_whitespace().collect();
            if fields[0] != label || fields[1].parse::<usize>().unwrap() != code {
                continue;
            }
            let a = decode::<T>(fields[2]);
            let b = decode::<T>(fields[3]);
            let expected = decode::<T>(fields[4]);
            let status = fields[5].parse::<u32>().unwrap();
            let mut out = [oracle::small(77); 3];
            let result = plan.call(&[&[a], &[b]], &mut [&mut out]);
            if status == 0 {
                result.unwrap();
                assert_eq!(out[0], expected);
                source_call(op, &[a], &[b])
                    .unwrap()
                    .read_into(&mut out)
                    .unwrap();
                assert_eq!(out[0], expected);
            } else {
                let kind = match status {
                    4 => pcu_facade::PcuExecutionFaultKind::ArithmeticUnderflow,
                    3 => pcu_facade::PcuExecutionFaultKind::ArithmeticOverflow,
                    _ => panic!("invalid golden status"),
                };
                assert!(
                    matches!(result,Err(TensorError::ArithmeticFault {element_index:0,kind:k,..})if k==kind)
                );
                assert_eq!(out, [oracle::small(77); 3]);
                assert!(
                    matches!(source_call(op,&[a],&[b]),Err(global::PcuExecutionError::TensorBuild(TensorError::ArithmeticFault {element_index:0,kind:k,..}))if k==kind)
                );
            }
            assert_eq!(out[1..], [oracle::small(77); 2]);
            count += 1;
        }
    }
    assert_eq!(count, 195 + if T::SIGNED { 12 } else { 6 }); // 65 valid rows per operation plus independent signed/unsigned range boundaries.
}
#[test]
fn six_wide_arbitrary_precision_golden_bytes() {
    golden::<i128>("i128");
    golden::<u128>("u128");
    golden::<PcuI256>("i256");
    golden::<PcuU256>("u256");
    golden::<PcuI512>("i512");
    golden::<PcuU512>("u512");
}

fn decode<T: Wide>(hex: &str) -> T {
    let mut bits = [0; 64];
    assert_eq!(hex.len(), T::BYTES * 2);
    for (i, pair) in hex.as_bytes().as_chunks::<2>().0.iter().enumerate() {
        bits[i] = u8::from_str_radix(core::str::from_utf8(pair).unwrap(), 16).unwrap();
    }
    T::from_bytes(bits)
}
