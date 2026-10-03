//! Independent Rust checked arithmetic with a cold-retained transactional output bank.
#[rustfmt::skip]
use super::{
    Error,
    fault,
    PcuExecutionFaultKind,
};

pub struct Helper {
    output: Vec<u64>,
}

impl Helper {
    pub fn new(extent: usize) -> Self {
        Self {
            output: vec![0; extent],
        }
    }

    #[allow(clippy::trivially_copy_pass_by_ref)] // Match the annotated borrowed scalar boundary.
    pub fn call(&mut self, input: &[u64], seed: &u64, output: &mut [u64]) -> Result<(), Error> {
        let extent = self.output.len();
        if input.len() < extent || output.len() < extent {
            return Err(Error::Schema);
        }
        for (index, &original) in input[..extent].iter().enumerate() {
            let shifted = original
                .checked_add(*seed)
                .ok_or_else(|| fault(index, PcuExecutionFaultKind::ArithmeticOverflow))?;
            let scaled = shifted
                .checked_mul(original)
                .ok_or_else(|| fault(index, PcuExecutionFaultKind::ArithmeticOverflow))?;
            self.output[index] = scaled
                .checked_sub(original)
                .ok_or_else(|| fault(index, PcuExecutionFaultKind::ArithmeticUnderflow))?;
        }
        output[..extent].copy_from_slice(&self.output);
        Ok(())
    }
}
