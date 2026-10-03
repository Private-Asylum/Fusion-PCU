//! Cold-retained native working banks, checked per operation and transactional publication.
use super::Error;
use fusion_pcu::PcuCheckedFloat;

pub struct Composed {
    output: Vec<f32>,
    stage: Option<Vec<f32>>,
}

impl Composed {
    pub fn new(extent: usize, ordered: bool) -> Self {
        Self {
            output: vec![0.0; extent],
            stage: ordered.then(|| vec![0.0; extent]),
        }
    }

    pub fn call(
        &mut self,
        input: &[f32],
        stage: &mut [f32],
        output: &mut [f32],
    ) -> Result<(), Error> {
        let extent = self.output.len();
        if input.len() < extent
            || output.len() < extent
            || (self.stage.is_some() && stage.len() < extent)
        {
            return Err(Error::Schema);
        }
        for (index, value) in input[..extent].iter().copied().enumerate() {
            let doubled = value
                .pcu_checked_add(value)
                .map_err(|kind| super::fault(index, kind))?;
            if let Some(working) = &mut self.stage {
                working[index] = doubled;
            }
            let updated = self
                .stage
                .as_ref()
                .map_or(doubled, |working| working[index]);
            self.output[index] = updated
                .pcu_checked_mul(value)
                .map_err(|kind| super::fault(index, kind))?;
        }
        if let Some(working) = &self.stage {
            stage[..extent].copy_from_slice(working);
        }
        output[..extent].copy_from_slice(&self.output);
        Ok(())
    }
}
