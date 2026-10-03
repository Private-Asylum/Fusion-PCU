//! Cold scalar dispatch and exact sealed Rust type bridges, without boxed warm executors.
#[rustfmt::skip]
use core::{
    any::TypeId,
    mem::ManuallyDrop,
    ptr,
    slice,
};
use std::rc::Rc;
#[rustfmt::skip]
use fusion_pcu_core::{
    PcuScalar,
    PcuScalarType,
    dialect::tensor::{TensorElement, TensorOwnedSelectedProgram},
};
#[rustfmt::skip]
use fusion_pcu_vulkan::{
    PcuVulkanBackend,
    PcuVulkanOwnedBuffer,
    PcuVulkanPreparedTensorGraph,
    PcuVulkanTensorInput,
};
#[rustfmt::skip]
use crate::global::{
    arguments::TensorBacking,
    tensor::{PcuTensorInput, TensorInputKind},
    PcuExecutionError,
};

macro_rules! types {
    ($($variant:ident: $scalar:ident => $ty:ty;)+) => {
        pub(super) enum Typed { $( $variant(PcuVulkanPreparedTensorGraph<$ty>), )+ }
        impl Typed {
            pub(super) fn prepare(root: &PcuVulkanBackend, program: &TensorOwnedSelectedProgram, scalar: PcuScalarType) -> Result<Self, PcuExecutionError> {
                match scalar {
                    $(PcuScalarType::$scalar => PcuVulkanPreparedTensorGraph::<$ty>::prepare(root, program.graph(), program.output_values()).map(Self::$variant),)+
                    _ => return Err(PcuExecutionError::InvalidTensorSourcePlan),
                }.map_err(super::map_error)
            }
            pub(super) fn assess(program: &TensorOwnedSelectedProgram, scalar: PcuScalarType) -> Result<(), PcuExecutionError> {
                match scalar {
                    $(PcuScalarType::$scalar => PcuVulkanPreparedTensorGraph::<$ty>::assess(program.graph(), program.output_values()),)+
                    _ => return Err(PcuExecutionError::InvalidTensorSourcePlan),
                }.map_err(super::map_error)
            }
            pub(super) fn shape(&self) -> Rc<[usize]> {
                match self { $(Self::$variant(plan) => Rc::clone(plan.output_shape()),)+ }
            }
            pub(super) fn execute<T: PcuScalar, const N: usize>(&mut self, inputs: &[PcuTensorInput<'_, T>; N], indices: &[usize]) -> Result<PcuVulkanOwnedBuffer<T>, PcuExecutionError> {
                match self { $(Self::$variant(plan) => execute(plan, inputs, indices),)+ }
            }
        }
    };
}
types! {
    U8: U8 => u8; I8: I8 => i8; U16: U16 => u16; I16: I16 => i16;
    U32: U32 => u32; I32: I32 => i32; U64: U64 => u64; I64: I64 => i64;
    U128: U128 => u128; I128: I128 => i128;
    U256: U256 => fusion_pcu_core::PcuU256; I256: I256 => fusion_pcu_core::PcuI256;
    U512: U512 => fusion_pcu_core::PcuU512; I512: I512 => fusion_pcu_core::PcuI512;
    F16: F16 => fusion_pcu_core::PcuF16Bits; Bf16: BF16 => fusion_pcu_core::PcuBf16Bits;
    E4M3: F8E4M3FN => fusion_pcu_core::PcuF8E4M3FnBits; E5M2: F8E5M2 => fusion_pcu_core::PcuF8E5M2Bits;
    F32: F32 => f32; F64: F64 => f64;
    F128: F128 => fusion_pcu_core::PcuF128Bits; F256: F256 => fusion_pcu_core::PcuF256Bits;
}

fn execute<T: PcuScalar, U: TensorElement + PcuScalar, const N: usize>(
    plan: &mut PcuVulkanPreparedTensorGraph<U>,
    inputs: &[PcuTensorInput<'_, T>; N],
    indices: &[usize],
) -> Result<PcuVulkanOwnedBuffer<T>, PcuExecutionError> {
    if TypeId::of::<T>() != TypeId::of::<U>() || indices.len() > N {
        return Err(PcuExecutionError::InvalidTensorSourcePlan);
    }
    let mut mapped: [PcuVulkanTensorInput<'_, U>; N] =
        core::array::from_fn(|_| PcuVulkanTensorInput::Host(&[]));
    for (binding, index) in indices.iter().enumerate() {
        let input = inputs
            .get(*index)
            .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?;
        mapped[binding] = bind_input(input)?;
    }
    let output = plan
        .execute_owned(&mapped[..indices.len()])
        .map_err(super::map_error)?;
    let output = ManuallyDrop::new(output);
    // SAFETY: Exact T=U equality proves the same owned Rust type. Read moves it once, while
    // ManuallyDrop suppresses destruction of the old binding; no raw native handle is forged.
    Ok(unsafe { ptr::read(ptr::from_ref(&*output).cast::<PcuVulkanOwnedBuffer<T>>()) })
}

fn bind_input<'a, T: PcuScalar, U: PcuScalar>(
    input: &'a PcuTensorInput<'_, T>,
) -> Result<PcuVulkanTensorInput<'a, U>, PcuExecutionError> {
    if TypeId::of::<T>() != TypeId::of::<U>() {
        return Err(PcuExecutionError::InvalidTensorSourcePlan);
    }
    match input.kind {
        TensorInputKind::Host(values) => {
            // SAFETY: Exact T=U proves identical layout/validity. Immutable storage remains
            // borrowed through terminal execution and every owner conversion precedes submission.
            let values =
                unsafe { slice::from_raw_parts(values.as_ptr().cast::<U>(), values.len()) };
            Ok(PcuVulkanTensorInput::Host(values))
        }
        TensorInputKind::Resident(owner) => match &owner.backing {
            TensorBacking::Vulkan { buffer, .. } => {
                // SAFETY: Exact T=U proves this is the identical complete Rust owner type.
                let buffer = unsafe { &*ptr::from_ref(buffer).cast::<PcuVulkanOwnedBuffer<U>>() };
                Ok(PcuVulkanTensorInput::Owned(buffer))
            }
            #[cfg(any(
                feature = "cpu",
                feature = "mlx",
                feature = "rocm",
                feature = "cuda",
                feature = "metal"
            ))]
            _ => Err(PcuExecutionError::Argument(
                crate::global::PcuArgumentError::UnsupportedResidentBorrow,
            )),
        },
    }
}
