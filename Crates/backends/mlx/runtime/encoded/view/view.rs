//! Cached exact F32 typed descriptors; cache edges never point back to their source owner.
#[rustfmt::skip]
use std::{rc::Rc,cell::{Cell,RefCell}};
use fusion_pcu::PcuScalarType;
use crate::MlxError;
#[rustfmt::skip]
use super::{MlxEncodedArray,super::{MlxArray,Array,MlxArrayResidency}};
impl MlxArray {
    /// Retains an official same-width F32-to-UInt32 bit view on this exact session.
    /// The first call constructs and completes view/reshape descriptors; later calls share
    /// the cached immutable descriptor. No host readback or numerical conversion occurs.
    ///
    /// # Errors
    /// Returns quarantine, invalid native storage, descriptor or terminal failure.
    pub fn encoded_f32_view(&self) -> Result<MlxEncodedArray, MlxError> {
        self.validate_access_available()?;
        if let Some(native) = self.array.encoded_view.borrow().as_ref() {
            return Ok(MlxEncodedArray {
                session: self.session.clone(),
                native: Rc::clone(native),
            });
        }
        let owner = self
            .session
            .wrap_encoded(self.array.native.encoded_f32_view()?);
        *self.array.encoded_view.borrow_mut() = Some(Rc::clone(&owner.native));
        Ok(owner)
    }
}
impl MlxEncodedArray {
    /// Retains an exact logical F32-to-native F32 matrix view on this same session.
    /// Each distinct positive matrix shape is cached after terminal completion. All views
    /// preserve logical bits and immutable backing; no host readback or conversion occurs.
    ///
    /// # Errors
    /// Rejects every non-F32 logical tag, mismatched/overflowing shape and quarantined session.
    /// Returns native descriptor or uncertain completion errors without publishing a view.
    pub fn native_f32_view(&self, shape: [usize; 2]) -> Result<MlxArray, MlxError> {
        if self.scalar_type() != PcuScalarType::F32 {
            return Err(MlxError::UnsupportedScalar(self.scalar_type()));
        }
        if shape.contains(&0) || shape[0].checked_mul(shape[1]) != Some(self.element_count()) {
            return Err(MlxError::InvalidExtent);
        }
        self.validate_access_available()?;
        if let Some((_, array)) = self
            .native
            .views
            .borrow()
            .iter()
            .find(|(cached, _)| *cached == shape)
        {
            return Ok(MlxArray {
                session: self.session.clone(),
                array: Rc::clone(array),
            });
        }
        let native = self.native.native_f32_view(shape)?;
        let array = Rc::new(Array {
            native,
            residency: Cell::new(MlxArrayResidency::GpuEvaluated),
            encoded_view: RefCell::new(None),
        });
        self.native
            .views
            .borrow_mut()
            .push((shape, Rc::clone(&array)));
        Ok(MlxArray {
            session: self.session.clone(),
            array,
        })
    }
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
