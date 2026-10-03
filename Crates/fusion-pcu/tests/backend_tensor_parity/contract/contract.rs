//! The oracle uses exactly representable constants, independent of provider reduction code.
use std::sync::Mutex;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuScalar,
};
use super::source;

pub static POLICY_LOCK: Mutex<()> = Mutex::new(());

fn bits<T: PcuScalar>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.encode_le().as_ref(), expected.encode_le().as_ref());
    }
}

macro_rules! verify_format {
    ($scalar:ty, $loss:ident, $train:ident, $retain:ident, $sum:ident, $consume:ident) => {{
        let input: [[$scalar; 2]; 2] = [[1.0, 0.0], [0.0, 1.0]];
        let target: [[$scalar; 1]; 2] = [[1.0], [0.0]];
        for (weights, expected_loss, expected_weights) in [
            ([[2.0], [-1.0]], 0.5, [1.5, -1.0]),
            ([[3.0], [-1.0]], 2.0, [2.0, -1.0]),
            ([[4.0], [-1.0]], 4.5, [2.5, -1.0]),
        ] {
            let before = source::$loss(&input, &weights, &target).unwrap();
            let updated = source::$train(&input, &input, &weights, &target).unwrap();
            let mut loss = [0.0; 1];
            let mut host_weights = [0.0; 2];
            before.read_into(&mut loss).unwrap();
            updated.read_into(&mut host_weights).unwrap();
            bits(&loss, &[expected_loss]);
            bits(&host_weights, &expected_weights);

            // Returned values escape the captured graph. Subsequent preparations must not
            // recycle backing while the old Rust owner still retains semantic access.
            let changed = source::$train(&input, &input, &[[5.0], [-1.0]], &target).unwrap();
            let mut changed_host = [0.0; 2];
            changed.read_into(&mut changed_host).unwrap();
            bits(&changed_host, &[3.0, -1.0]);
            updated.read_into(&mut host_weights).unwrap();
            bits(&host_weights, &expected_weights);
            before.read_into(&mut loss).unwrap();
            bits(&loss, &[expected_loss]);

            let branch = source::$retain(&host_weights).unwrap();
            let added = source::$sum(&branch, &[1.0, 0.5]).unwrap();
            let consumed = source::$consume(added).unwrap();
            consumed.read_into(&mut host_weights).unwrap();
            bits(&host_weights, &[expected_weights[0] + 1.0, 0.0]);
            branch.read_into(&mut host_weights).unwrap();
            bits(&host_weights, &expected_weights);
            // Dropping escaped owners releases semantic ownership. The backend may retain
            // quiescent physical storage in its prepared bank; this is not an early-free API.
            drop(consumed);
            drop(branch);
            drop(updated);
            drop(changed);
            drop(before);
        }
    }};
}

pub fn verify(backend: global::PcuBackendChoice) {
    let _guard = POLICY_LOCK.lock().unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    verify_format!(f32, loss_f32, train_f32, retain_f32, sum_f32, consume_f32);
    verify_format!(f64, loss_f64, train_f64, retain_f64, sum_f64, consume_f64);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
