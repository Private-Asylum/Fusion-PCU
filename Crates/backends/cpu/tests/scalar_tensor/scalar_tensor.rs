//! All22 leaf-only carrier plans; arbitrary bits, detached constants and preflight transaction.
extern crate pcu_facade as fusion_pcu;
#[path = "../prepared_tensor/allocation/allocation.rs"]
mod allocation;
#[path = "../scalar_broadcast/sample/sample.rs"]
mod sample;
#[path = "source/source.rs"]
mod source;
#[rustfmt::skip]
use fusion_pcu::{PcuI256,PcuU256,PcuI512,PcuU512,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,PcuF128Bits,PcuF256Bits,PcuScalar,PcuNumericalOptions,PcuReproducibility};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{Graph,Tensor,TensorElement,TensorError};
use fusion_pcu_cpu::PcuCpuPreparedScalarTensorGraph;
use sample::{Sample, same};
fn check<T: Sample + TensorElement>() {
    const N: usize = 65;
    let mut graph = Graph::default();
    let input = graph.input([N], T::TYPE).unwrap();
    let constant: Vec<T> = (0..N).map(T::pattern).collect();
    let constant_value = graph.constant_typed(Tensor::new([N], constant.clone()).unwrap());
    let uniform = T::pattern(1);
    let uniform_value = graph.uniform_typed([N], uniform).unwrap();
    let mut plan = PcuCpuPreparedScalarTensorGraph::<T>::prepare(
        &graph,
        &[input, constant_value.erase(), uniform_value.erase()],
    )
    .unwrap();
    drop(graph);
    assert_eq!(plan.scratch_slot_count(), 0);
    let sentinel = T::pattern(17);
    let mut first = [sentinel; 68];
    let mut second = [sentinel; 68];
    let mut third = [sentinel; 68];
    for seed in 0..64 {
        let values: Vec<T> = (0..N).map(|i| T::pattern(i + seed + 29)).collect();
        assert_eq!(
            allocation::count(|| {
                plan.call(&[&values], &mut [&mut first, &mut second, &mut third])
                    .unwrap();
            }),
            0
        );
        same(&first[..N], &values);
        same(&second[..N], &constant);
        same(&third[..N], &[uniform; N]);
        same(&first[N..], &[sentinel; 3]);
        same(&second[N..], &[sentinel; 3]);
        same(&third[N..], &[sentinel; 3]);
    }
    let before = (first, second, third);
    assert!(
        plan.call(&[&[]], &mut [&mut first, &mut second, &mut third])
            .is_err()
    );
    same(&first, &before.0);
    same(&second, &before.1);
    same(&third, &before.2);
    assert!(plan.output(0).is_none());
    let values = [T::pattern(1); N];
    assert!(
        plan.call(&[&values], &mut [&mut first, &mut second, &mut third[..64]])
            .is_err()
    );
    same(&first, &before.0);
    same(&second, &before.1);
    same(&third, &before.2);
    plan.call(&[&values], &mut [&mut first, &mut second, &mut third])
        .unwrap();
    same(&first[..N], &values);
    let mut graph = Graph::default();
    graph.set_numerical_options(PcuNumericalOptions {
        reproducibility: PcuReproducibility::PortableV1,
        ..Default::default()
    });
    let input = graph.input([N], T::TYPE).unwrap();
    assert!(matches!(
        PcuCpuPreparedScalarTensorGraph::<T>::prepare(&graph, &[input]),
        Err(TensorError::UnsupportedNumericalOptions { .. })
    ));
}
macro_rules! carriers{($($name:ident:$ty:ty),+)=>{$(#[test]fn $name(){check::<$ty>();})+};}
carriers!(i8_leaf:i8,u8_leaf:u8,i16_leaf:i16,u16_leaf:u16,i32_leaf:i32,u32_leaf:u32,i64_leaf:i64,u64_leaf:u64,i128_leaf:i128,u128_leaf:u128,i256_leaf:PcuI256,u256_leaf:PcuU256,i512_leaf:PcuI512,u512_leaf:PcuU512,f16_leaf:PcuF16Bits,bf16_leaf:PcuBf16Bits,e4_leaf:PcuF8E4M3FnBits,e5_leaf:PcuF8E5M2Bits,f32_leaf:f32,f64_leaf:f64,f128_leaf:PcuF128Bits,f256_leaf:PcuF256Bits);
#[test]
fn leaf_only_table_rejects_arithmetic_without_mutating_existing_plans() {
    let mut graph = Graph::default();
    let left = graph.input([3], i32::TYPE).unwrap();
    let right = graph.input([3], i32::TYPE).unwrap();
    let out = graph.add(left, right).unwrap();
    assert!(PcuCpuPreparedScalarTensorGraph::<i32>::prepare(&graph, &[out]).is_err());
}

fn source_carrier<T: Sample>() {
    const N: usize = 256;
    let patterns = if T::HOST_SIZE <= 2 {
        1_usize << (T::HOST_SIZE * 8)
    } else {
        4096
    };
    let sentinel = T::pattern(17);
    let mut observed = [sentinel; N + 3];
    for seed in (0..patterns).step_by(N) {
        let mut input: [T; N] = std::array::from_fn(|i| T::pattern(seed + i));
        let expected = input;
        let owner = source::identity(&input).unwrap();
        input.fill(sentinel);
        owner.read_into(&mut observed).unwrap();
        same(&observed[..N], &expected);
        same(&observed[N..], &[sentinel; 3]);
        let sibling = source::identity(&owner).unwrap();
        let moved = source::consume(owner).unwrap();
        drop(sibling);
        moved.read_into(&mut observed).unwrap();
        same(&observed[..N], &expected);
        let mut short = [sentinel; N - 1];
        assert!(moved.read_into(&mut short).is_err());
        same(&short, &[sentinel; N - 1]);
    }
    println!(
        "ordinary carrier {:?}: {patterns} generated raw patterns, host overwrite, borrowed/consumed owners, tails and short rollback",
        T::TYPE
    );
}

#[test]
fn ordinary_all_carriers_preserve_full_bits_and_owner_lifetimes() {
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    macro_rules! formats {($($ty:ty),+)=>{$(source_carrier::<$ty>();)+};}
    formats!(
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
        f64,
        PcuF128Bits,
        PcuF256Bits
    );
    pcu_facade::global::clear_thread_cache().unwrap();
    pcu_facade::global::use_defaults().unwrap();
}
