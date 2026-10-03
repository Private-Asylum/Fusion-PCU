//! Complete Rust borrow schema, original snapshots and terminal joint publication.
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxPreparedEncodedPrefix,
    MlxPreparedTransportHostKernel,
};
#[rustfmt::skip]
use crate::{
    global::arguments::{
        PcuCallArgument,
        PcuCallArgumentKind,
    },
    PcuExecutionError,
    PcuHostArgument,
    PcuPreparedHostKernel,
};
use super::map_mlx_error;
#[path = "preflight/preflight.rs"]
mod preflight;
#[path = "publication/publication.rs"]
mod publication;
#[path = "snapshots/snapshots.rs"]
mod snapshots;

pub(super) fn call<const N: usize>(
    kernel: &mut MlxPreparedTransportHostKernel,
    prefixes: &[Option<MlxPreparedEncodedPrefix>; 2],
    readback: &mut [Vec<u8>; 2],
    arguments: [PcuCallArgument<'_>; N],
) -> Result<(), PcuExecutionError> {
    let kinds = arguments.map(|argument| argument.into_parts().1);
    if kinds
        .iter()
        .all(|kind| matches!(kind, PcuCallArgumentKind::Host(_)))
    {
        let mut slots: [Option<PcuHostArgument<'_>>; N] = core::array::from_fn(|_| None);
        for (slot, kind) in slots.iter_mut().zip(kinds) {
            let PcuCallArgumentKind::Host(host) = kind else {
                unreachable!("all complete arguments were host declarations");
            };
            *slot = Some(host);
        }
        let mut host = slots.map(|slot| slot.expect("all complete host arguments were collected"));
        return kernel.call(&mut host).map_err(map_mlx_error);
    }
    let mut kinds = kinds;
    preflight::validate(kernel, prefixes, readback, &kinds)?;
    // Snapshot borrows end after terminal execution. An exclusive input is still
    // its original immutable backing until both output siblings are ready.
    let completed = snapshots::execute(kernel, &kinds)?;
    publication::publish(kernel, prefixes, readback, &mut kinds, completed)
}
