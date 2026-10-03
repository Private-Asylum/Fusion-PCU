//! Benchmark-only independent native limb kernels; no PCU IR or scalar-map lowering.
#[rustfmt::skip]
use crate::{
    ffi,
    MlxEncodedArray,
    MlxError,
    MlxSession,
};
use fusion_pcu::PcuScalarType;

/// Independent bounded raw-bit workloads, in the documented native input order.
#[derive(Clone, Copy, Debug)]
pub enum MlxNativeTransportWorkload {
    /// Input prefix and one scalar seed; returns repeated seed and saved input prefix.
    SavedInput,
    /// Input prefix, scalar seed and original stage; returns repeated seed and old stage.
    PriorStage,
    /// Input prefix, scalar seed, original stage and original output; swaps the two banks.
    SwapBanks,
}
impl MlxNativeTransportWorkload {
    #[must_use]
    pub const fn input_count(self) -> usize {
        match self {
            Self::SavedInput => 2,
            Self::PriorStage => 3,
            Self::SwapBanks => 4,
        }
    }
}
/// Authentic retained native custom primitive, with independently handwritten kernel logic.
/// Its terminal ownership protocol is shared with the contained native transport ABI.
pub struct MlxNativeTransportControl {
    session: MlxSession,
    native: ffi::Transport,
}
impl MlxSession {
    /// Prepares a benchmark-only native raw-bit control and completes a full-shape cold prime.
    /// This does not assess PCU source, generate a PCU transport plan or advertise capability.
    /// Inputs are dense arrays in workload order; only slot one is a scalar read. Full cold
    /// synthetic input uploads cost `sum(extents) * scalar_width` host/device bytes.
    /// # Errors
    /// Rejects unsupported carriers, arity, short/overflowing shapes or native failure.
    pub fn prepare_native_transport_control(
        &self,
        scalar: PcuScalarType,
        element_count: usize,
        workload: MlxNativeTransportWorkload,
        extents: &[usize],
    ) -> Result<MlxNativeTransportControl, MlxError> {
        let native = self.prepare_native_transport_control_internal(
            scalar,
            element_count,
            workload,
            extents,
        )?;
        let prepared = MlxNativeTransportControl {
            session: self.clone(),
            native,
        };
        let mut inputs: [Option<MlxEncodedArray>; 4] = std::array::from_fn(|_| None);
        for (slot, &count) in extents.iter().enumerate() {
            let width = usize::from(scalar.bit_width()) / 8;
            let bytes = count.checked_mul(width).ok_or(MlxError::InvalidExtent)?;
            inputs[slot] = Some(self.upload_transport_bytes(scalar, count, &vec![0; bytes])?);
        }
        let first = inputs[0].as_ref().ok_or(MlxError::InvalidExtent)?;
        let mut borrowed = [first; 4];
        for (slot, input) in inputs.iter().enumerate() {
            if let Some(input) = input {
                borrowed[slot] = input;
            }
        }
        let (stage, output) = prepared.execute(&borrowed[..extents.len()])?;
        stage.release()?;
        output.release()?;
        for input in inputs.into_iter().flatten() {
            input.release()?;
        }
        Ok(prepared)
    }
}
impl MlxNativeTransportControl {
    #[must_use]
    pub fn prepared_input_element_counts(&self) -> &[usize] {
        self.native.input_element_counts()
    }
    /// Replays exact frozen native shapes and returns two terminal private immutable outputs.
    /// Caller publication, prefix merge and checked release remain outside this raw control.
    /// # Errors
    /// Rejects arity/type/session/shape before work; unknown completion quarantines the session
    /// and retains actual pending owners. This does not observe SDK heap or internal JIT cost.
    pub fn execute(
        &self,
        inputs: &[&MlxEncodedArray],
    ) -> Result<(MlxEncodedArray, MlxEncodedArray), MlxError> {
        let first = inputs.first().ok_or(MlxError::InvalidExtent)?.native();
        if inputs.len() > 4 {
            return Err(MlxError::InvalidExtent);
        }
        let mut native = [first; 4];
        for (slot, input) in inputs.iter().enumerate() {
            native[slot] = input.native();
        }
        let [Some(stage), Some(output)] = self.native.execute(&native[..inputs.len()])? else {
            return Err(MlxError::Abi(
                "native transport control requires two writers".into(),
            ));
        };
        Ok((
            self.session.wrap_transport_output(stage),
            self.session.wrap_transport_output(output),
        ))
    }
}
