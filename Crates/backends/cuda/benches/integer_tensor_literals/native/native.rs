//! Handwritten checked integer pipeline with independent raw SDK ownership.
#[path = "ffi/ffi.rs"]
mod ffi;
#[cfg(feature = "allocation-census")]
pub use ffi::Api;
#[rustfmt::skip]
use fusion_pcu::{PcuHostArgument,PcuBindingRef};
#[rustfmt::skip]
use fusion_pcu_cuda::{CudaRuntime,compile_cuda_source_for_device};
use super::oracle::Format;
pub struct Native {
    owner: ffi::Owner,
}
macro_rules! diagnostic_call {
    ($method:ident) => {
        #[allow(dead_code)] // Shared producer benches do not exercise the physical-work ladder.
        pub fn $method<T: Format>(
            &mut self,
            input: &[T],
            constant: &[T],
            uniform: &[T],
            output: &mut [T],
        ) -> Result<Option<(u32, u32)>, ()> {
            let input = PcuHostArgument::read(PcuBindingRef::new(0, 0), input);
            let constant = PcuHostArgument::read(PcuBindingRef::new(0, 1), constant);
            let uniform = PcuHostArgument::read(PcuBindingRef::new(0, 2), uniform);
            let mut output = PcuHostArgument::read_write(PcuBindingRef::new(0, 3), output);
            self.owner.$method(
                input.bytes(),
                constant.bytes(),
                uniform.bytes(),
                output.bytes_mut().unwrap(),
            )
        }
    };
}
impl Native {
    diagnostic_call!(call_fresh_output);
    diagnostic_call!(call_split_completion);
    diagnostic_call!(call_fresh_split_completion);
    pub fn new<T: Format, const N: usize>(device: u32) -> Self {
        const { assert!(cfg!(target_endian = "little")) };
        let runtime = CudaRuntime::new(device).unwrap();
        let code = format!(
            "#define BITS {}\n#define SIGNED {}\n{}\n{}",
            T::ENCODED_SIZE * 8,
            u8::from(T::SIGNED),
            include_str!("../../composed_integer/native/arithmetic.cu"),
            include_str!("pipeline.cu")
        );
        let image = compile_cuda_source_for_device(&runtime, &code).unwrap();
        Self {
            owner: ffi::Owner::new(
                runtime,
                &image,
                N * T::ENCODED_SIZE,
                u32::try_from(N).unwrap(),
                u32::try_from(N).unwrap(),
                false,
                false,
                [u32::try_from(N).unwrap().div_ceil(256), 1, 1],
                [256, 1, 1],
            ),
        }
    }
    pub fn call<T: Format>(
        &mut self,
        input: &[T],
        constant: &[T],
        uniform: &[T],
        output: &mut [T],
    ) -> Result<Option<(u32, u32)>, ()> {
        let input = PcuHostArgument::read(PcuBindingRef::new(0, 0), input);
        let constant = PcuHostArgument::read(PcuBindingRef::new(0, 1), constant);
        let uniform = PcuHostArgument::read(PcuBindingRef::new(0, 2), uniform);
        let mut output = PcuHostArgument::read_write(PcuBindingRef::new(0, 3), output);
        self.owner.call(
            input.bytes(),
            constant.bytes(),
            uniform.bytes(),
            output.bytes_mut().unwrap(),
        )
    }
    #[cfg(feature = "allocation-census")]
    pub fn counter(&self) -> std::rc::Rc<std::cell::Cell<Api>> {
        self.owner.counter()
    }
}
