//! Independent arithmetic source and raw SDK owner; no PCU arithmetic lowering is imported.
#[path = "ffi/ffi.rs"]
mod ffi;
#[cfg(feature = "allocation-census")]
pub use ffi::Api;
#[rustfmt::skip]
use fusion_pcu::{PcuHostArgument,PcuBindingRef,PcuRangePolicy};
#[rustfmt::skip]
use fusion_pcu_cuda::{CudaRuntime,compile_cuda_source_for_device};
use super::oracle::Format;
pub struct Native {
    owner: ffi::Owner,
}
impl Native {
    pub fn new<T: Format, const N: usize>(
        device: u32,
        grid: bool,
        range: PcuRangePolicy,
        dead: bool,
        launch: ([u32; 3], [u32; 3]),
    ) -> Self {
        const { assert!(cfg!(target_endian = "little")) };
        let runtime = CudaRuntime::new(device).unwrap();
        let code = format!(
            "#define BITS {}\n#define SIGNED {}\n{}",
            T::ENCODED_SIZE * 8,
            u8::from(T::SIGNED),
            include_str!("arithmetic.cu")
        );
        let image = compile_cuda_source_for_device(&runtime, &code).unwrap();
        Self {
            owner: ffi::Owner::new(
                runtime,
                &image,
                N * T::ENCODED_SIZE,
                u32::try_from(N).unwrap(),
                if grid { 3 } else { u32::try_from(N).unwrap() },
                range == PcuRangePolicy::Clamp,
                dead,
                launch.0,
                launch.1,
            ),
        }
    }
    pub fn call<T: Format>(
        &mut self,
        input: &[T],
        seed: &T,
        stage: Option<&mut [T]>,
        output: &mut [T],
    ) -> Result<Option<(u32, u32)>, ()> {
        let input = PcuHostArgument::read(PcuBindingRef::new(0, 0), input);
        let seed = PcuHostArgument::read_scalar(PcuBindingRef::new(0, 1), seed);
        let mut stage = stage.map(|v| PcuHostArgument::read_write(PcuBindingRef::new(0, 2), v));
        let mut output = PcuHostArgument::read_write(PcuBindingRef::new(0, 3), output);
        self.owner.call(
            input.bytes(),
            seed.bytes(),
            stage.as_mut().map(|v| v.bytes_mut().unwrap()),
            output.bytes_mut().unwrap(),
        )
    }
    #[cfg(feature = "allocation-census")]
    pub fn counter(&self) -> std::rc::Rc<std::cell::Cell<Api>> {
        self.owner.counter()
    }
    pub fn known_terminal_retirement_witness(self) {
        self.owner.known_terminal_retirement_witness();
    }
}
