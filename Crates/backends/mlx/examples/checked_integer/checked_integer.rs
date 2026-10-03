//! Prepared genuine source integer Clamp publishes a useful terminal prefix with observable fault.
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxRuntime,
    MlxError,
};
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuU512,
    PcuCheckedInteger,
    PcuExecutionFaultKind,
    PcuHostDispatchError,
};
#[pcu(invocations=3,flag(strict),flag(clamp_range),crate_path=::pcu_facade)]
fn add<T: PcuCheckedInteger>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
const fn small(value: u64) -> PcuU512 {
    let mut limbs = [0; 8];
    limbs[0] = value;
    PcuU512::from_limbs_le(limbs)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let session = MlxRuntime::load_default()?.open_gpu(0)?;
    let backend = session.checked_integer_backend();
    let mut prepared = add_prepare::<PcuU512, _>(&backend)
        .map_err(|error| std::io::Error::other(format!("{error:?}")))?;
    let mut output = [small(7); 5];
    let result = prepared(
        &[small(0), PcuU512::MAX, small(3)],
        &[small(1), small(1), small(4)],
        &mut output,
    );
    assert!(
        matches!(result,Err(PcuHostDispatchError::Backend(MlxError::Arithmetic(fault))) if fault.recovered&&fault.invocation_id==1&&fault.kind==PcuExecutionFaultKind::ArithmeticOverflow)
    );
    assert_eq!(
        output,
        [small(1), PcuU512::MAX, small(7), small(7), small(7)]
    );
    println!(
        "MLX source U512 completed Clamp output, recovered overflow at lane1, caller tail preserved"
    );
    Ok(())
}
