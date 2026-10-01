//! Independent cold Lt plan with the same typed profile, retained inputs, fresh output and terminal stream-event completion.
#[rustfmt::skip]
use std::{
    error::Error,
    mem::size_of,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuHostArgument,
    PcuPrecisionPolicy,
};
#[rustfmt::skip]
use fusion_pcu_cuda::{
    CublasLtMatmulPlan,
    CublasLtMatmulDescriptor,
    CublasEnvironmentSnapshot,
    CublasNumericalConfig,
    CudaCompletionBatch,
    CudaRuntime,
    CudaStreamHandle,
    DeviceBuffer,
};
use super::oracle::Scalar;

pub struct NativeLt {
    blas: CublasLtMatmulPlan,
    stream: CudaStreamHandle,
    runtime: CudaRuntime,
    left: DeviceBuffer,
    right: DeviceBuffer,
}

impl NativeLt {
    pub fn new<T: Scalar, const R: usize, const K: usize, const C: usize>(
        runtime: &CudaRuntime,
        precision: PcuPrecisionPolicy,
    ) -> Result<Self, Box<dyn Error>> {
        let config =
            CublasNumericalConfig::new(T::TYPE, precision, CublasEnvironmentSnapshot::capture())?;
        let stream = runtime.create_stream()?;
        let descriptor = CublasLtMatmulDescriptor::new([R, K], [K, C], [false; 2], config)?;
        let blas = CublasLtMatmulPlan::prepare(runtime, &stream, descriptor)?;
        eprintln!("independent native Lt control {:?}", blas.identity());
        Ok(Self {
            blas,
            stream,
            runtime: runtime.clone(),
            left: runtime.allocate(R * K * size_of::<T>())?,
            right: runtime.allocate(K * C * size_of::<T>())?,
        })
    }

    pub fn host<T: Scalar, const R: usize, const K: usize, const C: usize>(
        &mut self,
        left: &[[T; K]; R],
        right: &[[T; C]; K],
        observed: &mut [T],
    ) -> Result<(), Box<dyn Error>> {
        let output = self.runtime.allocate(R * C * size_of::<T>())?;
        self.left.copy_from(
            PcuHostArgument::read(PcuBindingRef::new(0, 0), left.as_flattened()).bytes(),
        )?;
        self.right.copy_from(
            PcuHostArgument::read(PcuBindingRef::new(0, 1), right.as_flattened()).bytes(),
        )?;
        let mut batch = CudaCompletionBatch::new(&self.stream);
        self.blas
            .submit_into_batch(&mut batch, &self.left, &self.right, &output)?;
        batch.finish()?.wait()?;
        drop(batch);
        output.copy_to(
            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), observed)
                .bytes_mut()
                .unwrap(),
        )?;
        Ok(())
    }
}
