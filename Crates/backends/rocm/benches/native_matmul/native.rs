//! Direct typed rocBLAS control; fresh output, no numerical checker or fault status.
#[rustfmt::skip]
use std::{
    error::Error,
    mem::size_of,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuHostArgument,
    PcuScalarType,
};
#[rustfmt::skip]
use fusion_pcu_rocm::{
    DeviceBuffer,
    HipRuntime,
    HipCompletionBatch,
    HipStreamHandle,
    Rocblas,
};
use super::oracle::Scalar;
pub struct Native {
    runtime: HipRuntime,
    stream: HipStreamHandle,
    blas: Rocblas,
    left: [DeviceBuffer; 2],
    right: [DeviceBuffer; 2],
    output_bytes: usize,
}
impl Native {
    pub fn new<T: Scalar, const R: usize, const K: usize, const C: usize>(
        runtime: &HipRuntime,
    ) -> Result<Self, Box<dyn Error>> {
        let stream = runtime.create_stream()?;
        let mut blas = Rocblas::new(runtime)?;
        blas.bind_stream(&stream)?;
        Ok(Self {
            runtime: runtime.clone(),
            stream,
            blas,
            left: [
                runtime.allocate(R * K * size_of::<T>())?,
                runtime.allocate(R * K * size_of::<T>())?,
            ],
            right: [
                runtime.allocate(K * C * size_of::<T>())?,
                runtime.allocate(K * C * size_of::<T>())?,
            ],
            output_bytes: R * C * size_of::<T>(),
        })
    }
    pub fn upload<T: Scalar>(
        &mut self,
        bank: usize,
        left: &[T],
        right: &[T],
    ) -> Result<(), Box<dyn Error>> {
        self.left[bank].copy_from(PcuHostArgument::read(PcuBindingRef::new(0, 0), left).bytes())?;
        self.right[bank]
            .copy_from(PcuHostArgument::read(PcuBindingRef::new(0, 1), right).bytes())?;
        Ok(())
    }
    pub fn submit<T: Scalar, const R: usize, const K: usize, const C: usize>(
        &self,
        bank: usize,
    ) -> Result<DeviceBuffer, Box<dyn Error>> {
        let output = self.runtime.allocate(self.output_bytes)?;
        // Row-major AB is column-major B^T A^T. Typed batch wrappers validate
        // widths/extents and retain all operands through terminal device completion.
        let mut batch = HipCompletionBatch::new(&self.stream);
        match T::TYPE {
            PcuScalarType::F32 => self.blas.sgemm_into_batch(
                &mut batch,
                false,
                false,
                C,
                R,
                K,
                1.0,
                &self.right[bank],
                C,
                &self.left[bank],
                K,
                0.0,
                &output,
                C,
            )?,
            PcuScalarType::F64 => self.blas.dgemm_into_batch(
                &mut batch,
                false,
                false,
                C,
                R,
                K,
                1.0,
                &self.right[bank],
                C,
                &self.left[bank],
                K,
                0.0,
                &output,
                C,
            )?,
            _ => return Err("unsupported native benchmark scalar".into()),
        }
        let mut completion = batch.finish()?;
        completion.wait()?;
        Ok(output)
    }
    pub fn read<T: Scalar>(
        output: &DeviceBuffer,
        observed: &mut [T],
    ) -> Result<(), Box<dyn Error>> {
        output.copy_to(
            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), observed)
                .bytes_mut()
                .unwrap(),
        )?;
        Ok(())
    }
}
