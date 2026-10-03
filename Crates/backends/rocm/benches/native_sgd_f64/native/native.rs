//! Independently launched identical native F64 update with matching fresh output/event boundary.
#[path = "ffi/ffi.rs"]
mod ffi;
#[rustfmt::skip]
use std::{error::Error, mem::size_of, time::{Duration, Instant}};
#[rustfmt::skip]
use fusion_pcu::{PcuBindingRef, PcuHostArgument, PcuPrecisionPolicy, PcuScalarType};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{Graph, ValueId};
#[rustfmt::skip]
use fusion_pcu_rocm::{HipRuntime, HipKernel, HipStreamHandle, DeviceBuffer, compile_hip_source_for_device, lower_native_sgd_to_hip_source};
use super::oracle::Scalar;
pub struct Native {
    runtime: HipRuntime,
    kernel: HipKernel,
    stream: HipStreamHandle,
    weights: [DeviceBuffer; 2],
    gradient: [DeviceBuffer; 2],
    output_bytes: usize,
    count: u32,
    rate: f64,
}
impl Native {
    pub fn new<T: Scalar, const N: usize>(
        runtime: &HipRuntime,
        graph: &Graph,
        output: ValueId,
    ) -> Result<Self, Box<dyn Error>> {
        assert_eq!(T::TYPE, PcuScalarType::F64);
        let node = graph.node(output)?;
        let fusion_pcu::dialect::tensor::OpDescriptor::SgdUpdate { learning_rate, .. } = node.op
        else {
            unreachable!()
        };
        let source = lower_native_sgd_to_hip_source(graph, output)?;
        let image = compile_hip_source_for_device(runtime, &source)?;
        let module = runtime.load_module(&image)?;
        Ok(Self {
            runtime: runtime.clone(),
            kernel: module.function(
                if node.numerical_options.precision == PcuPrecisionPolicy::BackendOptimized {
                    c"tensor_sgd_update_contracted"
                } else {
                    c"tensor_sgd_update"
                },
            )?,
            stream: runtime.create_stream()?,
            weights: [
                runtime.allocate(N * size_of::<T>())?,
                runtime.allocate(N * size_of::<T>())?,
            ],
            gradient: [
                runtime.allocate(N * size_of::<T>())?,
                runtime.allocate(N * size_of::<T>())?,
            ],
            output_bytes: N * size_of::<T>(),
            count: u32::try_from(N)?,
            rate: f64::from(learning_rate),
        })
    }
    pub fn upload<T: Scalar>(
        &mut self,
        bank: usize,
        weights: &[T],
        gradient: &[T],
    ) -> Result<(), Box<dyn Error>> {
        self.weights[bank]
            .copy_from(PcuHostArgument::read(PcuBindingRef::new(0, 0), weights).bytes())?;
        self.gradient[bank]
            .copy_from(PcuHostArgument::read(PcuBindingRef::new(0, 1), gradient).bytes())?;
        Ok(())
    }
    fn launch(
        &self,
        bank: usize,
        output: &DeviceBuffer,
    ) -> Result<fusion_pcu_rocm::HipCompletion, Box<dyn Error>> {
        ffi::launch(
            &self.kernel,
            &self.stream,
            self.count,
            self.rate,
            [&self.weights[bank], &self.gradient[bank], output],
        )
        .map_err(Into::into)
    }
    pub fn submit(&self, bank: usize) -> Result<DeviceBuffer, Box<dyn Error>> {
        let output = self.runtime.allocate(self.output_bytes)?;
        self.launch(bank, &output)?.wait()?;
        Ok(output)
    }
    pub fn phases(&self) -> Result<(Duration, Duration), Box<dyn Error>> {
        let start = Instant::now();
        let output = self.runtime.allocate(self.output_bytes)?;
        let mut completion = self.launch(0, &output)?;
        let submission = start.elapsed();
        let start = Instant::now();
        completion.wait()?;
        drop(output);
        Ok((submission, start.elapsed()))
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
