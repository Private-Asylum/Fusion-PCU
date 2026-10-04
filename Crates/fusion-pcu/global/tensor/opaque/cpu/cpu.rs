//! Static CPU tensor adapters; ordinary host owners never become device pointers.
#[rustfmt::skip]
use core::{
    any::TypeId,
    slice,
};
#[rustfmt::skip]
use fusion_pcu_core::{
    PcuCheckedFloat,
    PcuCheckedInteger,
    PcuScalar,
    PcuScalarType,
    dialect::tensor::{
        TensorElement,
        TensorError,
        TensorOwnedSelectedProgram,
    },
};
#[rustfmt::skip]
use fusion_pcu_cpu::{
    PcuCpuPreparedIntegerTensorGraph,
    PcuCpuPreparedScalarTensorGraph,
    PcuCpuPreparedTensorGraph,
    PcuCpuTensorBinding,
};
use crate::PcuExecutionError;

// This trait describes the shared retained workspace behavior. The concrete enum selects
// statically; neither a boxed executor nor a trait-object call is introduced on submission.
trait CpuGraph<T: PcuScalar> {
    fn execute(&mut self, inputs: &[&[T]]) -> Result<(), TensorError>;
    fn output(&self, index: usize) -> Option<&[T]>;
    fn output_bindings(&self) -> &[PcuCpuTensorBinding];
}
impl<T: PcuCheckedFloat + TensorElement> CpuGraph<T> for PcuCpuPreparedTensorGraph<T> {
    fn execute(&mut self, inputs: &[&[T]]) -> Result<(), TensorError> {
        Self::execute(self, inputs)
    }
    fn output(&self, index: usize) -> Option<&[T]> {
        Self::output(self, index)
    }
    fn output_bindings(&self) -> &[PcuCpuTensorBinding] {
        Self::output_bindings(self)
    }
}
impl<T: PcuCheckedInteger + TensorElement> CpuGraph<T> for PcuCpuPreparedIntegerTensorGraph<T> {
    fn execute(&mut self, inputs: &[&[T]]) -> Result<(), TensorError> {
        Self::execute(self, inputs)
    }
    fn output(&self, index: usize) -> Option<&[T]> {
        Self::output(self, index)
    }
    fn output_bindings(&self) -> &[PcuCpuTensorBinding] {
        Self::output_bindings(self)
    }
}
impl<T: PcuScalar + TensorElement> CpuGraph<T> for PcuCpuPreparedScalarTensorGraph<T> {
    fn execute(&mut self, inputs: &[&[T]]) -> Result<(), TensorError> {
        Self::execute(self, inputs)
    }
    fn output(&self, index: usize) -> Option<&[T]> {
        Self::output(self, index)
    }
    fn output_bindings(&self) -> &[PcuCpuTensorBinding] {
        Self::output_bindings(self)
    }
}

macro_rules! prepared_types {
    ($( $variant:ident: $scalar:ident => $element:ty, $plan:ident; )+) => {
        pub(super) enum CpuPrepared { $( $variant($plan<$element>), )+ }
        impl CpuPrepared {
            pub(super) fn prepare(program: &TensorOwnedSelectedProgram, scalar: PcuScalarType) -> Result<Self, PcuExecutionError> {
                if program.output_values().len() != 1 {
                    return Err(PcuExecutionError::TensorBuild(TensorError::DataLength { expected: 1, actual: program.output_values().len() }));
                }
                Self::prepare_outputs(program, scalar)
            }
            pub(super) fn prepare_outputs(program: &TensorOwnedSelectedProgram, scalar: PcuScalarType) -> Result<Self, PcuExecutionError> {
                if program.output_values().is_empty() {
                    return Err(PcuExecutionError::TensorBuild(TensorError::DataLength {
                        expected: 1,
                        actual: program.output_values().len(),
                    }));
                }
                match scalar {
                    $( PcuScalarType::$scalar => $plan::<$element>::prepare(program.graph(), program.output_values()).map(Self::$variant), )+
                    scalar_type => Err(TensorError::UnsupportedScalarType {
                        value: program.output_values()[0], scalar_type,
                    }),
                }.map_err(PcuExecutionError::TensorBuild)
            }
            pub(super) fn output_shapes(&self) -> Vec<std::rc::Rc<[usize]>> {
                match self { $( Self::$variant(prepared) => prepared.output_bindings().iter().map(|binding| std::rc::Rc::from(binding.shape.as_slice())).collect(), )+ }
            }
            pub(super) fn execute_outputs<T: PcuScalar, const M: usize>(&mut self, inputs: &[&[T]]) -> Result<[Vec<T>; M], PcuExecutionError> {
                match self { $( Self::$variant(prepared) => execute_outputs(prepared, inputs), )+ }
            }
            pub(super) fn output_shape(&self) -> &[usize] {
                match self { $( Self::$variant(prepared) => &prepared.output_bindings()[0].shape, )+ }
            }
            pub(super) fn execute<T: PcuScalar>(&mut self, inputs: &[&[T]]) -> Result<Vec<T>, PcuExecutionError> {
                match self { $( Self::$variant(prepared) => execute(prepared, inputs), )+ }
            }
        }
    };
}
prepared_types! {
    F16: F16 => fusion_pcu_core::PcuF16Bits, PcuCpuPreparedTensorGraph;
    Bf16: BF16 => fusion_pcu_core::PcuBf16Bits, PcuCpuPreparedTensorGraph;
    F8E4M3Fn: F8E4M3FN => fusion_pcu_core::PcuF8E4M3FnBits, PcuCpuPreparedTensorGraph;
    F8E5M2: F8E5M2 => fusion_pcu_core::PcuF8E5M2Bits, PcuCpuPreparedTensorGraph;
    F32: F32 => f32, PcuCpuPreparedTensorGraph;
    F64: F64 => f64, PcuCpuPreparedTensorGraph;
    F128: F128 => fusion_pcu_core::PcuF128Bits, PcuCpuPreparedScalarTensorGraph;
    F256: F256 => fusion_pcu_core::PcuF256Bits, PcuCpuPreparedScalarTensorGraph;
    I8: I8 => i8, PcuCpuPreparedIntegerTensorGraph;
    U8: U8 => u8, PcuCpuPreparedIntegerTensorGraph;
    I16: I16 => i16, PcuCpuPreparedIntegerTensorGraph;
    U16: U16 => u16, PcuCpuPreparedIntegerTensorGraph;
    I32: I32 => i32, PcuCpuPreparedIntegerTensorGraph;
    U32: U32 => u32, PcuCpuPreparedIntegerTensorGraph;
    I64: I64 => i64, PcuCpuPreparedIntegerTensorGraph;
    U64: U64 => u64, PcuCpuPreparedIntegerTensorGraph;
    I128: I128 => i128, PcuCpuPreparedIntegerTensorGraph;
    U128: U128 => u128, PcuCpuPreparedIntegerTensorGraph;
    I256: I256 => fusion_pcu_core::PcuI256, PcuCpuPreparedIntegerTensorGraph;
    U256: U256 => fusion_pcu_core::PcuU256, PcuCpuPreparedIntegerTensorGraph;
    I512: I512 => fusion_pcu_core::PcuI512, PcuCpuPreparedIntegerTensorGraph;
    U512: U512 => fusion_pcu_core::PcuU512, PcuCpuPreparedIntegerTensorGraph;
}

fn execute<T: PcuScalar, U: PcuScalar>(
    prepared: &mut impl CpuGraph<U>,
    inputs: &[&[T]],
) -> Result<Vec<T>, PcuExecutionError> {
    if TypeId::of::<T>() != TypeId::of::<U>() {
        return Err(PcuExecutionError::TensorBuild(
            TensorError::ScalarTypeMismatch {
                value: prepared.output_bindings()[0].value,
                expected: U::TYPE,
                actual: T::TYPE,
            },
        ));
    }
    // SAFETY: TypeId equality proves T and U are the identical Rust type, including layout,
    // alignment and validity. Therefore &[T] and &[U], and the containing reference slices,
    // are identical types. The returned borrow cannot outlive inputs, and nothing is mutated.
    let inputs = unsafe { slice::from_raw_parts(inputs.as_ptr().cast::<&[U]>(), inputs.len()) };
    prepared
        .execute(inputs)
        .map_err(PcuExecutionError::TensorBuild)?;
    let binding = &prepared.output_bindings()[0];
    let output =
        prepared
            .output(0)
            .ok_or(PcuExecutionError::TensorBuild(TensorError::UnknownValue(
                binding.value,
            )))?;
    // SAFETY: The same exact TypeId check proves U=T. Copy to fresh owned host memory before
    // the retained workspace is reused; no native address or uninitialized representation exists.
    let output = unsafe { slice::from_raw_parts(output.as_ptr().cast::<T>(), output.len()) };
    Ok(output.to_vec())
}

fn execute_outputs<T: PcuScalar, U: PcuScalar, const M: usize>(
    prepared: &mut impl CpuGraph<U>,
    inputs: &[&[T]],
) -> Result<[Vec<T>; M], PcuExecutionError> {
    if prepared.output_bindings().len() != M {
        return Err(PcuExecutionError::TensorBuild(TensorError::DataLength {
            expected: M,
            actual: prepared.output_bindings().len(),
        }));
    }
    if TypeId::of::<T>() != TypeId::of::<U>() {
        return Err(PcuExecutionError::TensorBuild(
            TensorError::ScalarTypeMismatch {
                value: prepared.output_bindings()[0].value,
                expected: U::TYPE,
                actual: T::TYPE,
            },
        ));
    }
    // SAFETY: Exact TypeId equality proves identical scalar types and reference layouts.
    let inputs = unsafe { slice::from_raw_parts(inputs.as_ptr().cast::<&[U]>(), inputs.len()) };
    prepared
        .execute(inputs)
        .map_err(PcuExecutionError::TensorBuild)?;
    let mut outputs: [Option<Vec<T>>; M] = core::array::from_fn(|_| None);
    for (index, output) in outputs.iter_mut().enumerate() {
        let values = prepared.output(index).ok_or_else(|| {
            PcuExecutionError::TensorBuild(TensorError::UnknownValue(
                prepared.output_bindings()[index].value,
            ))
        })?;
        // SAFETY: The exact TypeId check above proves U and T have identical validity and layout.
        let values = unsafe { slice::from_raw_parts(values.as_ptr().cast::<T>(), values.len()) };
        *output = Some(values.to_vec());
    }
    Ok(outputs.map(|output| output.expect("every output copied after successful execution")))
}
