//! Genuine checked F32 source backed by Metal integer synthesis.
#[path = "../../benches/checked_f32/source.rs"]
mod source;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Metal,
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })?;
    let mut output = [91.0_f32; 5];
    source::add::<3>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &mut output)?;
    source::sub::<3>(&[5.0, 7.0, 9.0], &[1.0, 2.0, 3.0], &mut output)?;
    source::mul::<3>(&[2.0, 3.0, 4.0], &[4.0, 5.0, 6.0], &mut output)?;
    source::div::<3>(&[8.0, 15.0, 24.0], &[4.0, 5.0, 6.0], &mut output)?;
    assert_eq!(
        output.map(f32::to_bits),
        [2.0_f32, 3.0, 4.0, 91.0, 91.0].map(f32::to_bits)
    );
    println!("Checked F32 source with integer GPU arithmetic: {output:?}");
    Ok(())
}
