//! Same typed cuBLAS profile, retained inputs, fresh output and terminal stream-event completion.
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
    Cublas,
    CublasEnvironmentSnapshot,
    CublasNumericalConfig,
    CudaCompletionBatch,
    CudaRuntime,
    CudaStreamHandle,
    DeviceBuffer,
};
use super::oracle::Scalar;

pub struct Native {
    blas: Cublas,
    stream: CudaStreamHandle,
    runtime: CudaRuntime,
    left: DeviceBuffer,
    right: DeviceBuffer,
}

impl Native {
    pub fn new<T: Scalar, const R: usize, const K: usize, const C: usize>(
        runtime: &CudaRuntime,
        precision: PcuPrecisionPolicy,
    ) -> Result<Self, Box<dyn Error>> {
        let config =
            CublasNumericalConfig::new(T::TYPE, precision, CublasEnvironmentSnapshot::capture())?;
        let mut blas = Cublas::new_with_numerical_config(runtime, config)?;
        let stream = runtime.create_stream()?;
        blas.bind_stream(&stream)?;
        let (config, observed) = blas.numerical_config().unwrap();
        eprintln!(
            "native admitted precision={:?}, scalar={:?}, observed={observed:?}, environment={:?}; no checked numerical/reproducibility claim",
            config.precision(),
            config.scalar_type(),
            config.environment()
        );
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
        match T::TYPE {
            fusion_pcu::PcuScalarType::F32 => self.blas.sgemm_into_batch(
                &mut batch,
                false,
                false,
                C,
                R,
                K,
                1.0,
                &self.right,
                C,
                &self.left,
                K,
                0.0,
                &output,
                C,
            )?,
            fusion_pcu::PcuScalarType::F64 => self.blas.dgemm_into_batch(
                &mut batch,
                false,
                false,
                C,
                R,
                K,
                1.0,
                &self.right,
                C,
                &self.left,
                K,
                0.0,
                &output,
                C,
            )?,
            _ => return Err("unsupported native benchmark scalar".into()),
        }
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
