//! Ordinary annotated matrix calls retain opaque results and stage RAM automatically.
#[path = "../../tests/opaque/source.rs"]
#[allow(dead_code)] // Contract rejection fixtures are covered by the integration test.
mod source;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    global::{PcuBackendChoice, PcuExecutionPolicy},
    PcuExecutionError,
};

fn main() -> Result<(), PcuExecutionError> {
    global::configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Mlx,
        ..PcuExecutionPolicy::default()
    })?;
    let left = [[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let right = [[7.0_f32, 8.0], [9.0, 10.0], [11.0, 12.0]];
    let identity = [[1.0_f32, 0.0], [0.0, 1.0]];
    let result = source::matrix(&left, &right)?;
    let result = source::matrix::<2, 2, 2>(&result, &identity)?;
    let result = source::consume_matrix(result, &identity)?;
    global::clear_thread_cache()?;
    let mut host = [[0.0_f32; 2]; 2];
    result.read_into(host.as_flattened_mut())?;
    assert_eq!(
        host.map(|row| row.map(f32::to_bits)),
        [[58.0_f32, 64.0], [139.0, 154.0]].map(|row| row.map(f32::to_bits))
    );
    println!("Matrix retained through borrowed and consumed calls: {host:?}");
    Ok(())
}
