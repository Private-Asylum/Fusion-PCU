//! Actual annotated checked integer host calls beside matched transactional native peers.

use core::fmt::Debug;
use std::hint::black_box;
#[rustfmt::skip]
use criterion::{
    Criterion,
    Throughput,
    criterion_group,
    criterion_main,
};
#[rustfmt::skip]
use fusion_pcu_cpu::{
    PcuCpuCheckedInteger,
    PcuCpuCheckedIntegerError,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuCheckedInteger,
    PcuDispatchIntegerBinaryOp,
    PcuExecutionFault,
};

#[path = "../../tests/checked_integer/source/source.rs"]
mod source;

const COUNT: usize = 4096;

fn evaluate<T: PcuCheckedInteger>(
    op: PcuDispatchIntegerBinaryOp,
    lhs: T,
    rhs: T,
) -> Result<T, pcu_facade::PcuExecutionFaultKind> {
    match op {
        PcuDispatchIntegerBinaryOp::Add => lhs.pcu_checked_add(rhs),
        PcuDispatchIntegerBinaryOp::Sub => lhs.pcu_checked_sub(rhs),
        PcuDispatchIntegerBinaryOp::Mul => lhs.pcu_checked_mul(rhs),
    }
}

fn native<T: PcuCheckedInteger>(
    op: PcuDispatchIntegerBinaryOp,
    lhs: &[T],
    rhs: &[T],
    output: &mut [T],
) -> Result<(), PcuCpuCheckedIntegerError> {
    if lhs.len() < COUNT || rhs.len() < COUNT || output.len() < COUNT {
        return Err(PcuCpuCheckedIntegerError::InvalidArguments);
    }
    for index in 0..COUNT {
        evaluate(op, lhs[index], rhs[index]).map_err(|kind| {
            PcuCpuCheckedIntegerError::Fault(PcuExecutionFault {
                recovered: false,
                kind,
                invocation_id: u64::try_from(index).unwrap(),
            })
        })?;
    }
    for index in 0..COUNT {
        output[index] = evaluate(op, lhs[index], rhs[index]).expect("checked preflight succeeded");
    }
    Ok(())
}

fn compare<T: PcuCheckedInteger + Debug + PartialEq>(
    criterion: &mut Criterion,
    name: &str,
    op: PcuDispatchIntegerBinaryOp,
    mut call: impl FnMut(&[T], &[T], &mut [T]) -> Result<(), PcuCpuCheckedIntegerError>,
    values: [T; 6],
) {
    let [zero, one, two, three, minimum, maximum] = values;
    let mut lhs = std::vec![three; COUNT];
    let rhs = std::vec![one; COUNT];
    let mut output = std::vec![zero; COUNT + 2];
    let mut expected = std::vec![zero; COUNT + 2];
    for index in 0..COUNT {
        expected[index] = evaluate(op, lhs[index], rhs[index]).unwrap();
    }
    // Complete common-oracle equality and host-tail preservation are checked outside timing.
    call(&lhs, &rhs, &mut output).unwrap();
    assert_eq!(output, expected);
    output.fill(zero);
    native(op, &lhs, &rhs, &mut output).unwrap();
    assert_eq!(output, expected);
    let mut fault_lhs = lhs.clone();
    let mut fault_rhs = rhs.clone();
    fault_lhs[5] = if op == PcuDispatchIntegerBinaryOp::Sub {
        minimum
    } else {
        maximum
    };
    fault_rhs[5] = if op == PcuDispatchIntegerBinaryOp::Mul {
        maximum
    } else {
        one
    };
    output.fill(three);
    let native_fault = native(op, &fault_lhs, &fault_rhs, &mut output);
    assert!(native_fault.is_err());
    assert_eq!(output, std::vec![three; COUNT + 2]);
    assert_eq!(call(&fault_lhs, &fault_rhs, &mut output), native_fault);
    assert_eq!(output, std::vec![three; COUNT + 2]);
    call(&lhs, &rhs, &mut output).unwrap();
    expected[COUNT..].fill(three);
    assert_eq!(output, expected);
    let mut group = criterion.benchmark_group(format!("cpu_checked_integer/{name}/{op:?}"));
    group.throughput(Throughput::Elements(COUNT as u64));
    group.bench_function("source_host", |bencher| {
        bencher.iter(|| {
            lhs[0] = if lhs[0] == one { two } else { one };
            call(black_box(&lhs), black_box(&rhs), black_box(&mut output)).unwrap();
            black_box(&output);
        });
    });
    group.bench_function("native_checked", |bencher| {
        bencher.iter(|| {
            lhs[0] = if lhs[0] == one { two } else { one };
            native(op, black_box(&lhs), black_box(&rhs), black_box(&mut output)).unwrap();
            black_box(&output);
        });
    });
    group.finish();
}

fn benchmarks(criterion: &mut Criterion) {
    macro_rules! width {
        ($module:ident, $ty:ty) => {{
            let backend = PcuCpuCheckedInteger::<$ty>::new();
            compare(
                criterion,
                stringify!($ty),
                PcuDispatchIntegerBinaryOp::Add,
                source::$module::add_prepare::<COUNT, _>(&backend).unwrap(),
                [0, 1, 2, 3, <$ty>::MIN, <$ty>::MAX],
            );
            compare(
                criterion,
                stringify!($ty),
                PcuDispatchIntegerBinaryOp::Sub,
                source::$module::sub_prepare::<COUNT, _>(&backend).unwrap(),
                [0, 1, 2, 3, <$ty>::MIN, <$ty>::MAX],
            );
            compare(
                criterion,
                stringify!($ty),
                PcuDispatchIntegerBinaryOp::Mul,
                source::$module::mul_prepare::<COUNT, _>(&backend).unwrap(),
                [0, 1, 2, 3, <$ty>::MIN, <$ty>::MAX],
            );
        }};
    }
    width!(i8_source, i8);
    width!(u8_source, u8);
    width!(i16_source, i16);
    width!(u16_source, u16);
    width!(i32_source, i32);
    width!(u32_source, u32);
    width!(i64_source, i64);
    width!(u64_source, u64);
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
