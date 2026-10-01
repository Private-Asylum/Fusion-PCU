//! Untimed-oracle, balanced-order full-host diagnostic. Raw samples are not Criterion CIs.

#[rustfmt::skip]
use super::{
    HostBank,
    Prepared,
    prepare,
    verify,
};
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxSession,
};
#[rustfmt::skip]
use std::{
    hint::black_box,
    time::Instant,
};

const BATCH: usize = 20;
const ROUNDS: usize = 60;

fn batch<const SIZE: usize, const HOST: bool>(
    plan: &Prepared,
    session: &MlxSession,
    hosts: &[HostBank],
    arrays: &[(fusion_pcu_mlx::MlxArray, fusion_pcu_mlx::MlxArray)],
    mut index: usize,
    host: &mut [f32],
) {
    for _ in 0..BATCH {
        index = (index + 1) % hosts.len();
        if HOST {
            let a = session.upload_f32([SIZE, SIZE], &hosts[index].0).unwrap();
            let b = session.upload_f32([SIZE, SIZE], &hosts[index].1).unwrap();
            let output = plan.execute(session, &a, &b);
            output.read_into_f32(host).unwrap();
            black_box(&*host);
        } else {
            drop(black_box(plan.execute(
                session,
                &arrays[index].0,
                &arrays[index].1,
            )));
        }
    }
}

fn size<const SIZE: usize, const HOST: bool>(session: &MlxSession) {
    let plans: Vec<_> = (0..3)
        .map(|route| prepare::<SIZE>(session, route))
        .collect();
    let hosts: Vec<_> = [1.0_f32, 2.0, 3.0]
        .into_iter()
        .map(|phase| {
            (
                vec![phase; SIZE * SIZE],
                vec![4.0 - phase; SIZE * SIZE],
                phase * (4.0 - phase),
            )
        })
        .collect();
    let arrays: Vec<_> = hosts
        .iter()
        .map(|(a, b, _)| {
            (
                session.upload_f32([SIZE, SIZE], a).unwrap(),
                session.upload_f32([SIZE, SIZE], b).unwrap(),
            )
        })
        .collect();
    verify::<SIZE>(&plans, &hosts, &arrays, session);
    let mut host = vec![0.0; SIZE * SIZE];
    for warmup in 0..6 {
        for position in 0..3 {
            let route = (warmup + position) % 3;
            batch::<SIZE, HOST>(&plans[route], session, &hosts, &arrays, warmup, &mut host);
        }
    }

    // Each route sees the same changing-input sequence within each paired round. The six
    // permutations balance both position and predecessor; formatting stays outside timing.
    let orders = [
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
        [1, 0, 2],
        [0, 2, 1],
    ];
    let started = Instant::now();
    let mut samples = Vec::with_capacity(ROUNDS * 3);
    for round in 0..ROUNDS {
        for (position, route) in orders[round % orders.len()].into_iter().enumerate() {
            let sample = Instant::now();
            batch::<SIZE, HOST>(&plans[route], session, &hosts, &arrays, round, &mut host);
            samples.push((round, position, route, sample.elapsed().as_nanos()));
        }
    }
    let elapsed = started.elapsed().as_nanos();
    let boundary = if HOST {
        "full_host"
    } else {
        "preuploaded_terminal_drop"
    };
    verify::<SIZE>(&plans, &hosts, &arrays, session);
    eprintln!(
        "MLX_PAIRED_METADATA n={SIZE} boundary={boundary} rounds={ROUNDS} batch={BATCH} timed_calls={} loop_ns={elapsed} orders=all6 balanced_positions_and_predecessors",
        ROUNDS * 3 * BATCH
    );
    for (round, position, route, nanoseconds) in samples {
        eprintln!(
            "MLX_PAIRED_SAMPLE n={SIZE} boundary={boundary} round={round} position={position} route={route} batch_ns={nanoseconds}"
        );
    }
}

pub(super) fn run(session: &MlxSession) {
    size::<32, false>(session);
    size::<32, true>(session);
    size::<128, false>(session);
    size::<128, true>(session);
}
