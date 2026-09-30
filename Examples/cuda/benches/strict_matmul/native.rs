//! Native launch of the public generated ordered checker, with terminal status ownership.
#[rustfmt::skip]
use std::{
    error::Error,
    mem::size_of,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuHostArgument,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    ValueId,
};
#[rustfmt::skip]
use fusion_pcu_cuda::{
    CudaRuntime,
    CudaKernel,
    CudaStreamHandle,
    CudaKernelArgument,
    DeviceBuffer,
    Cublas,
    compile_cuda_source_for_device,
    lower_strict_matmul_to_cuda_source,
};
use super::oracle::Scalar;

pub struct Native {
    runtime: CudaRuntime,
    output_bytes: usize,
    kernel: CudaKernel,
    stream: CudaStreamHandle,
    left: [DeviceBuffer; 2],
    right: [DeviceBuffer; 2],
    output: DeviceBuffer,
    status: DeviceBuffer,
    grid: u32,
    sentinel: bool,
    blas: Cublas,
}

impl Native {
    pub fn new<T: Scalar, const R: usize, const K: usize, const C: usize>(
        runtime: &CudaRuntime,
        graph: &Graph,
        output: ValueId,
    ) -> Result<Self, Box<dyn Error>> {
        let source = lower_strict_matmul_to_cuda_source(graph, output)?;
        let image = compile_cuda_source_for_device(runtime, &source)?;
        let module = runtime.load_module(&image)?;
        let kernel = module.function(c"fusion_kernel")?;
        let stream = runtime.create_stream()?;
        let mut blas = Cublas::new(runtime)?;
        blas.bind_stream(&stream)?;
        Ok(Self {
            runtime: runtime.clone(),
            output_bytes: R * C * size_of::<T>(),
            kernel,
            stream,
            left: [
                runtime.allocate(R * K * size_of::<T>())?,
                runtime.allocate(R * K * size_of::<T>())?,
            ],
            right: [
                runtime.allocate(K * C * size_of::<T>())?,
                runtime.allocate(K * C * size_of::<T>())?,
            ],
            output: runtime.allocate(R * C * size_of::<T>())?,
            status: runtime.allocate(size_of::<u64>())?,
            grid: u32::try_from(R * C)?.div_ceil(256),
            sentinel: false,
            blas,
        })
    }

    pub fn upload<T: Scalar>(&mut self, left: &[T], right: &[T]) -> Result<(), Box<dyn Error>> {
        self.upload_bank(0, left, right)
    }

    pub fn upload_bank<T: Scalar>(
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

    #[allow(unsafe_code)] // Launch the exact public strict checker with its validated dense ABI.
    pub fn submit(&mut self) -> Result<u64, Box<dyn Error>> {
        self.submit_bank(0)
    }

    #[allow(unsafe_code)] // Same validated ABI as submit; only the preloaded input bank changes.
    pub fn submit_bank(&mut self, bank: usize) -> Result<u64, Box<dyn Error>> {
        if !std::mem::take(&mut self.sentinel) {
            self.status.copy_from(&u64::MAX.to_le_bytes())?;
        }
        let arguments = [
            CudaKernelArgument::Buffer(&self.left[bank]),
            CudaKernelArgument::Buffer(&self.right[bank]),
            CudaKernelArgument::Buffer(&self.output),
            CudaKernelArgument::Buffer(&self.status),
        ];
        // SAFETY: lowering validates dense F32/F64 strict MatMul. All four allocations cover
        // exactly its operands/output/status, grid covers every output cell, and terminal wait
        // retains buffers, stream and kernel until device access has ended.
        let mut completion = unsafe {
            self.kernel
                .launch(&self.stream, [self.grid, 1, 1], [256, 1, 1], 0, &arguments)
        }?;
        completion.wait()?;
        let mut word = [0_u8; 8];
        self.status.copy_to(&mut word)?;
        let fault = u64::from_le_bytes(word);
        self.sentinel = fault == u64::MAX;
        Ok(fault)
    }

    /// Fresh output and status match the checked owned scheduler's per-call physical work.
    #[allow(unsafe_code)] // Identical generated ABI and extents, with freshly allocated destinations.
    pub fn submit_fresh(&self, bank: usize) -> Result<(DeviceBuffer, u64), Box<dyn Error>> {
        let output = self.runtime.allocate(self.output_bytes)?;
        let mut status = self.runtime.allocate(size_of::<u64>())?;
        status.copy_from(&u64::MAX.to_le_bytes())?;
        let arguments = [
            CudaKernelArgument::Buffer(&self.left[bank]),
            CudaKernelArgument::Buffer(&self.right[bank]),
            CudaKernelArgument::Buffer(&output),
            CudaKernelArgument::Buffer(&status),
        ];
        // SAFETY: same validated dense checker and four-buffer ABI as submit_bank; fresh output
        // covers every cell and private status covers one u64. Terminal wait precedes both drops.
        let mut completion = unsafe {
            self.kernel
                .launch(&self.stream, [self.grid, 1, 1], [256, 1, 1], 0, &arguments)
        }?;
        completion.wait()?;
        let mut word = [0_u8; 8];
        status.copy_to(&mut word)?;
        Ok((output, u64::from_le_bytes(word)))
    }

    pub fn read_fresh<T: Scalar>(
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

    /// Vendor throughput control: has a different, unchecked compound numerical contract.
    pub fn vendor<T: Scalar, const R: usize, const K: usize, const C: usize>(
        &self,
        bank: usize,
    ) -> Result<(), Box<dyn Error>> {
        match T::TYPE {
            fusion_pcu::PcuScalarType::F32 => self.blas.sgemm(
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
                &self.output,
                C,
            )?,
            fusion_pcu::PcuScalarType::F64 => self.blas.dgemm(
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
                &self.output,
                C,
            )?,
            _ => return Err("unsupported benchmark scalar".into()),
        }
        Ok(())
    }

    pub fn read<T: Scalar>(&self, output: &mut [T]) -> Result<(), Box<dyn Error>> {
        self.output.copy_to(
            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), output)
                .bytes_mut()
                .unwrap(),
        )?;
        Ok(())
    }
}
