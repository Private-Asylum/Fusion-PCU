//! Authentic source-owned wide raw encoding through an explicit retained MLX integer carrier.
#[path = "source/source.rs"]
mod source;
#[rustfmt::skip]
use pcu_facade::{PcuScalar,PcuU512,PcuImplementationRequirements};
use std::sync::Arc;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = fusion_pcu_mlx::MlxRuntime::load_default()?;
    let session = runtime.open_gpu(0)?;
    let requirements = PcuImplementationRequirements::default();
    let capture = pcu_facade::global::__pcu_capture_tensor_program(
        [pcu_facade::global::PcuSourceShape::Slice { length: 3 }],
        requirements.float_underflow,
        requirements.numerical_mode,
        requirements.numerical_options,
        source::retain::__pcu_capture_entry::<PcuU512>,
    )
    .map_err(|error| std::io::Error::other(format!("{error:?}")))?;
    let prepared = session.prepare_checked_program(Arc::clone(capture.program()), requirements)?;
    let input = [
        PcuU512::decode_le([0xff; 64]),
        PcuU512::decode_le([0; 64]),
        PcuU512::decode_le([0x80; 64]),
    ];
    let owner = prepared.execute_host(&input)?;
    drop(prepared);
    drop(session);
    drop(runtime);
    let sentinel = PcuU512::decode_le([0x37; 64]);
    let mut stack = [sentinel; 5];
    owner.read_into(&mut stack)?;
    assert_eq!(&stack[..3], &input);
    assert_eq!(&stack[3..], &[sentinel; 2]);
    println!("MLX retained three exact U512 encodings after dropping preparation and runtime.");
    Ok(())
}
