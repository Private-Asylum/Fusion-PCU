//! Handwritten checked integer pipeline with independent raw SDK ownership.
#[path = "ffi/ffi.rs"]
mod ffi;
#[cfg(feature = "allocation-census")]
pub use ffi::Api;
#[rustfmt::skip]
use fusion_pcu::{PcuHostArgument,PcuBindingRef};
#[rustfmt::skip]
use fusion_pcu_rocm::{HipRuntime,compile_hip_source_for_device};
use super::oracle::Format;
pub struct Native {
    owner: ffi::Owner,
}
impl Native {
    pub fn new<T: Format, const N: usize>(device: u32) -> Self {
        const { assert!(cfg!(target_endian = "little")) };
        let runtime = HipRuntime::new(device).unwrap();
        let code = format!(
            "#define BITS {}\n#define SIGNED {}\n{}\n{}",
            T::ENCODED_SIZE * 8,
            u8::from(T::SIGNED),
            include_str!("arithmetic.hip"),
            include_str!("pipeline.hip")
        );
        let image = compile_hip_source_for_device(&runtime, &code).unwrap();
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
    /// Independent native diagnostic with compile-time physical-work choices.
    #[allow(dead_code)] // Shared by producer benches that do not exercise this diagnostic ladder.
    pub fn call_physical_work<T: Format, const FRESH: bool, const SPLIT: bool>(
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
        self.owner.call_physical_work::<FRESH, SPLIT>(
            input.bytes(),
            constant.bytes(),
            uniform.bytes(),
            output.bytes_mut().unwrap(),
        )
    }
    /// Independent physical-work diagnostic; primary native path is unchanged.
    #[allow(dead_code)] // Shared by producer benches; only the literal witness uses this variant.
    pub fn call_fresh_output<T: Format>(
        &mut self,
        input: &[T],
        constant: &[T],
        uniform: &[T],
        output: &mut [T],
    ) -> Result<Option<(u32, u32)>, ()> {
        self.call_physical_work::<T, true, false>(input, constant, uniform, output)
    }
    /// Independent physical-work diagnostic; primary native path is unchanged.
    #[allow(dead_code)] // Shared by producer benches; only the literal witness uses this variant.
    pub fn call_split_completion<T: Format>(
        &mut self,
        input: &[T],
        constant: &[T],
        uniform: &[T],
        output: &mut [T],
    ) -> Result<Option<(u32, u32)>, ()> {
        self.call_physical_work::<T, false, true>(input, constant, uniform, output)
    }
    /// Independent physical-work diagnostic; primary native path is unchanged.
    #[allow(dead_code)] // Shared by producer benches; only the literal witness uses this variant.
    pub fn call_fresh_split_completion<T: Format>(
        &mut self,
        input: &[T],
        constant: &[T],
        uniform: &[T],
        output: &mut [T],
    ) -> Result<Option<(u32, u32)>, ()> {
        self.call_physical_work::<T, true, true>(input, constant, uniform, output)
    }
    #[cfg(feature = "allocation-census")]
    pub fn counter(&self) -> std::rc::Rc<std::cell::Cell<Api>> {
        self.owner.counter()
    }
}
