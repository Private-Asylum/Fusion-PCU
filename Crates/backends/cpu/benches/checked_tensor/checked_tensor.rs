//! Genuine owned source / explicit graph diagnostic / independent direct checked CPU pairs.
extern crate pcu_facade as fusion_pcu;
#[rustfmt::skip]
use std::{
    hint::black_box,
    rc::Rc,
};
#[rustfmt::skip]
use criterion::{
    Criterion,
    Throughput,
    criterion_group,
    criterion_main,
};
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuNumericalMode,
    PcuScalar,
    dialect::tensor::Graph,
};
use fusion_pcu_cpu::PcuCpuPreparedTensorGraph;
#[path = "native/native.rs"]
mod native;
#[path = "source/source.rs"]
#[allow(dead_code)] // Admission policy variants are asserted by the source integration test.
mod source;
fn loss_cases(criterion: &mut Criterion) {
    macro_rules! loss {
        ($ty:ty, $count:expr) => {{
            let inputs: [Box<[$ty; $count]>; 2] = core::array::from_fn(|bank| {
                vec![if bank == 0 { 1.25 as $ty } else { 2.5 as $ty }; $count]
                    .into_boxed_slice()
                    .try_into()
                    .unwrap()
            });
            let targets: [Box<[$ty; $count]>; 2] = core::array::from_fn(|bank| {
                vec![if bank == 0 { 0.25 as $ty } else { 0.5 as $ty }; $count]
                    .into_boxed_slice()
                    .try_into()
                    .unwrap()
            });
            let mut graph = Graph::default();
            graph.set_numerical_mode(PcuNumericalMode::Strict);
            let left = graph.input([$count], <$ty>::TYPE).unwrap();
            let right = graph.input([$count], <$ty>::TYPE).unwrap();
            let output = graph.mean_squared_error(left, right).unwrap();
            let mut prepared =
                PcuCpuPreparedTensorGraph::<$ty>::prepare(&graph, &[output]).unwrap();
            let shape: Rc<[usize]> = Rc::from([]);
            for bank in 0..2 {
                let expected = native::loss(&*inputs[bank], &*targets[bank], &shape).unwrap();
                let mut observed = [0.0 as $ty];
                source::loss(&*inputs[bank], &*targets[bank])
                    .unwrap()
                    .read_into(&mut observed)
                    .unwrap();
                assert_eq!(observed[0].to_bits(), expected.data()[0].to_bits());
                assert_eq!(
                    prepared
                        .execute_owned(&[&*inputs[bank], &*targets[bank]])
                        .unwrap()
                        .data()[0]
                        .to_bits(),
                    expected.data()[0].to_bits()
                );
            }
            let mut group = criterion.benchmark_group(format!(
                "checked_tensor/loss/{}/{}",
                stringify!($ty),
                $count
            ));
            group.throughput(Throughput::Elements($count));
            for route in [
                "source_owned_read",
                "explicit_graph_owned_read",
                "native_owned_read",
            ] {
                let mut bank = 0;
                let mut observed = [0.0 as $ty];
                group.bench_function(route, |bench| {
                    bench.iter(|| {
                        bank ^= 1;
                        let prediction = black_box(&*inputs[bank]);
                        let target = black_box(&*targets[bank]);
                        match route {
                            "source_owned_read" => source::loss(prediction, target)
                                .unwrap()
                                .read_into(&mut observed)
                                .unwrap(),
                            "explicit_graph_owned_read" => {
                                prepared.execute(&[prediction, target]).unwrap();
                                let output = native::HostOutput::new(
                                    prepared.output(0).unwrap().to_vec(),
                                    &shape,
                                );
                                observed.copy_from_slice(output.data());
                            }
                            _ => {
                                let output = native::loss(prediction, target, &shape).unwrap();
                                observed.copy_from_slice(output.data());
                            }
                        }
                        black_box(&observed);
                    })
                });
                assert_eq!(
                    observed[0].to_bits(),
                    native::loss(&*inputs[bank], &*targets[bank], &shape)
                        .unwrap()
                        .data()[0]
                        .to_bits()
                );
            }
            group.finish();
        }};
    }
    loss!(f32, 16);
    loss!(f32, 4096);
    loss!(f64, 16);
    loss!(f64, 4096);
}

#[allow(clippy::too_many_lines)] // Keeps matched source/graph/native training boundaries together.
fn training_cases(criterion: &mut Criterion) {
    macro_rules! training {
        ($ty:ty) => {{
            let x = [[1.0 as $ty, 2.0], [3.0, 4.0]];
            let xt = [[1.0, 3.0], [2.0, 4.0]];
            let weights = [[[0.5], [0.25]], [[0.75], [0.125]]];
            let targets = [[[0.0], [1.0]], [[0.5], [0.75]]];
            let mut graph = Graph::default();
            graph.set_numerical_mode(PcuNumericalMode::Strict);
            let input = graph.input([2, 2], <$ty>::TYPE).unwrap();
            let transpose = graph.input([2, 2], <$ty>::TYPE).unwrap();
            let w = graph.input([2, 1], <$ty>::TYPE).unwrap();
            let y = graph.input([2, 1], <$ty>::TYPE).unwrap();
            let activation = graph.matmul(input, w).unwrap();
            let prediction = graph.relu(activation).unwrap();
            let _loss = graph.mean_squared_error(prediction, y).unwrap();
            let difference = graph.sub(prediction, y).unwrap();
            let derivative = graph.relu_backward(activation, difference).unwrap();
            let gradient = graph.matmul(transpose, derivative).unwrap();
            let output = graph.sgd_update(w, gradient, 0.5).unwrap();
            let mut prepared =
                PcuCpuPreparedTensorGraph::<$ty>::prepare(&graph, &[output]).unwrap();
            let flat_x = x.concat();
            let flat_xt = xt.concat();
            let flat_w = [weights[0].concat(), weights[1].concat()];
            let flat_y = [targets[0].concat(), targets[1].concat()];
            let shape: Rc<[usize]> = Rc::from([2, 1]);
            for bank in 0..2 {
                let expected =
                    native::training(&x, &xt, &weights[bank], &targets[bank], &shape).unwrap();
                let mut observed = [0.0 as $ty; 2];
                source::training(&x, &xt, &weights[bank], &targets[bank])
                    .unwrap()
                    .read_into(&mut observed)
                    .unwrap();
                assert_eq!(
                    observed.map(<$ty>::to_bits),
                    [expected.data()[0], expected.data()[1]].map(<$ty>::to_bits)
                );
                let actual = prepared
                    .execute_owned(&[&flat_x, &flat_xt, &flat_w[bank], &flat_y[bank]])
                    .unwrap();
                assert_eq!(
                    [actual.data()[0], actual.data()[1]].map(<$ty>::to_bits),
                    [expected.data()[0], expected.data()[1]].map(<$ty>::to_bits)
                );
            }
            let mut group =
                criterion.benchmark_group(format!("checked_tensor/training/{}", stringify!($ty)));
            for route in [
                "source_owned_read",
                "explicit_graph_owned_read",
                "native_owned_read",
            ] {
                let mut bank = 0;
                let mut observed = [0.0 as $ty; 2];
                group.bench_function(route, |bench| {
                    bench.iter(|| {
                        bank ^= 1;
                        match route {
                            "source_owned_read" => source::training(
                                black_box(&x),
                                black_box(&xt),
                                black_box(&weights[bank]),
                                black_box(&targets[bank]),
                            )
                            .unwrap()
                            .read_into(&mut observed)
                            .unwrap(),
                            "explicit_graph_owned_read" => {
                                prepared
                                    .execute(black_box(&[
                                        &flat_x,
                                        &flat_xt,
                                        &flat_w[bank],
                                        &flat_y[bank],
                                    ]))
                                    .unwrap();
                                let output = native::HostOutput::new(
                                    prepared.output(0).unwrap().to_vec(),
                                    &shape,
                                );
                                observed.copy_from_slice(output.data());
                            }
                            _ => {
                                let output = native::training(
                                    black_box(&x),
                                    black_box(&xt),
                                    black_box(&weights[bank]),
                                    black_box(&targets[bank]),
                                    &shape,
                                )
                                .unwrap();
                                observed.copy_from_slice(output.data());
                            }
                        }
                        black_box(&observed);
                    })
                });
                let expected =
                    native::training(&x, &xt, &weights[bank], &targets[bank], &shape).unwrap();
                assert_eq!(
                    observed.map(<$ty>::to_bits),
                    [expected.data()[0], expected.data()[1]].map(<$ty>::to_bits)
                );
            }
            group.finish();
        }};
    }
    training!(f32);
    training!(f64);
}
fn benchmarks(criterion: &mut Criterion) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..global::PcuExecutionPolicy::default()
    })
    .unwrap();
    loss_cases(criterion);
    training_cases(criterion);
}
criterion_group!(benches, benchmarks);
criterion_main!(benches);
