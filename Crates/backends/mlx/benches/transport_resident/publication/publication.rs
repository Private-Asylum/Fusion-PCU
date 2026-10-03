//! Two terminal siblings, both cold prefix plans and infallible host/owner commit.
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxEncodedArray,
    MlxPreparedEncodedPrefix,
    MlxSession,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuBindingRef,
    PcuHostArgument,
    PcuScalar,
};
pub struct Publication<T> {
    pub host: [Vec<T>; 2],
    pub owners: [MlxEncodedArray; 2],
    prefixes: [MlxPreparedEncodedPrefix; 2],
    scratch: [Vec<u8>; 2],
}
impl<T: PcuScalar> Publication<T> {
    pub fn new<const N: usize>(session: &MlxSession, sentinel: T) -> Self {
        let host = std::array::from_fn(|_| vec![sentinel; N + 5]);
        let owners = std::array::from_fn(|_| session.upload_encoded(&host[0]).unwrap());
        let prefixes =
            std::array::from_fn(|_| session.prepare_encoded_prefix(T::TYPE, N, N + 5).unwrap());
        Self {
            host,
            owners,
            prefixes,
            scratch: std::array::from_fn(|_| vec![0; N * T::HOST_SIZE]),
        }
    }
    pub fn reset<const N: usize>(&mut self, session: &MlxSession, boundary: usize, sentinel: T) {
        let count = if boundary == 1 { N } else { N + 5 };
        self.owners =
            std::array::from_fn(|_| session.upload_encoded(&vec![sentinel; count]).unwrap());
        for host in &mut self.host {
            host.fill(sentinel);
        }
    }
    pub fn commit(&mut self, outputs: (MlxEncodedArray, MlxEncodedArray), boundary: usize) {
        let (stage, output) = outputs;
        match boundary {
            0 => {
                stage.read_bytes_into(&mut self.scratch[0]).unwrap();
                output.read_bytes_into(&mut self.scratch[1]).unwrap();
                stage.release().unwrap();
                output.release().unwrap();
                for slot in 0..2 {
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 0), &mut self.host[slot])
                        .bytes_mut()
                        .unwrap()[..self.scratch[slot].len()]
                        .copy_from_slice(&self.scratch[slot]);
                }
            }
            1 => self.owners = [stage, output],
            2 => {
                let merged_stage = self.prefixes[0].execute(&stage, &self.owners[0]).unwrap();
                let merged_output = self.prefixes[1].execute(&output, &self.owners[1]).unwrap();
                stage.release().unwrap();
                output.release().unwrap();
                self.owners = [merged_stage, merged_output];
            }
            _ => unreachable!("three explicit physical boundaries"),
        }
    }
}
