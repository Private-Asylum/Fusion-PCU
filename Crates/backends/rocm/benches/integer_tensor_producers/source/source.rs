//! Genuine immutable Constant/Uniform source; Rust evaluates every payload at compile time.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuScalar,
    PcuTensor,
    PcuExecutionError,
    PcuU256,
    PcuI256,
    PcuU512,
    PcuI512,
};
use super::oracle::Format;
pub trait ProducerSource: Format {
    fn capture<const N: usize>(
        mode: pcu_facade::PcuNumericalMode,
        options: pcu_facade::PcuNumericalOptions,
    ) -> Result<pcu_facade::global::PcuCapturedTensorProgram, PcuExecutionError>;
    fn pipeline<const N: usize>(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError>;
    fn literal<const N: usize>(unused: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError>;
    fn uniform(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError>;
    fn unused_owner<const N: usize>(
        unused: &PcuTensor<Self>,
    ) -> Result<PcuTensor<Self>, PcuExecutionError>;
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)] // low is independently bounded to1..=3.
const fn payload_u8<const N: usize>() -> [u8; N] {
    let mut values = [0_u8; N];
    let mut index = 0;
    while index < N {
        let low = index % 3 + 1;
        values[index] = (1_u8 << 3) | (low as u8);
        index += 1;
    }
    values
}
#[pcu(crate_path=::pcu_facade)]
fn pipeline_u8<const N: usize>(input: &[u8]) -> Result<PcuTensor<u8>, PcuExecutionError> {
    let constant = pcu::constant(const { payload_u8::<N>() })?;
    let uniform = pcu::uniform_like(input, const { 2_u8 })?;
    let sum = pcu::add(input, &constant)?;
    pcu::mul(&sum, &uniform)
}
#[pcu(crate_path=::pcu_facade)]
fn literal_u8<const N: usize>(unused: &[u8]) -> Result<PcuTensor<u8>, PcuExecutionError> {
    pcu::constant(const { payload_u8::<N>() })
}
#[pcu(crate_path=::pcu_facade)]
fn uniform_u8(input: &[u8]) -> Result<PcuTensor<u8>, PcuExecutionError> {
    pcu::uniform_like(input, const { 2_u8 })
}
#[pcu(crate_path=::pcu_facade)]
fn foreign_u8<const N: usize>(_unused: &PcuTensor<u8>) -> Result<PcuTensor<u8>, PcuExecutionError> {
    pcu::constant(const { payload_u8::<N>() })
}
impl ProducerSource for u8 {
    fn capture<const N: usize>(
        mode: pcu_facade::PcuNumericalMode,
        options: pcu_facade::PcuNumericalOptions,
    ) -> Result<pcu_facade::global::PcuCapturedTensorProgram, PcuExecutionError> {
        for builder in [
            literal_u8::__pcu_capture_entry::<N>,
            uniform_u8::__pcu_capture_entry,
        ] {
            let captured = pcu_facade::global::__pcu_capture_tensor_program::<Self, 1, _>(
                [pcu_facade::global::PcuSourceShape::Slice { length: N }],
                pcu_facade::PcuFloatUnderflowPolicy::IeeeAfterRounding,
                mode,
                options,
                builder,
            )?;
            assert!(captured.argument_indices().is_empty());
            assert!(captured.program().input_values().is_empty());
        }
        pcu_facade::global::__pcu_capture_tensor_program::<Self, 1, _>(
            [pcu_facade::global::PcuSourceShape::Slice { length: N }],
            pcu_facade::PcuFloatUnderflowPolicy::IeeeAfterRounding,
            mode,
            options,
            pipeline_u8::__pcu_capture_entry::<N>,
        )
    }
    fn pipeline<const N: usize>(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        pipeline_u8::<N>(input)
    }
    fn literal<const N: usize>(unused: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        literal_u8::<N>(unused)
    }
    fn uniform(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        uniform_u8(input)
    }
    fn unused_owner<const N: usize>(
        unused: &PcuTensor<Self>,
    ) -> Result<PcuTensor<Self>, PcuExecutionError> {
        foreign_u8::<N>(unused)
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)] // low is independently bounded to1..=3.
const fn payload_i8<const N: usize>() -> [i8; N] {
    let mut values = [0_i8; N];
    let mut index = 0;
    while index < N {
        let low = index % 3 + 1;
        values[index] = (1_i8 << 3) | (low as i8);
        index += 1;
    }
    values
}
#[pcu(crate_path=::pcu_facade)]
fn pipeline_i8<const N: usize>(input: &[i8]) -> Result<PcuTensor<i8>, PcuExecutionError> {
    let constant = pcu::constant(const { payload_i8::<N>() })?;
    let uniform = pcu::uniform_like(input, const { 2_i8 })?;
    let sum = pcu::add(input, &constant)?;
    pcu::mul(&sum, &uniform)
}
#[pcu(crate_path=::pcu_facade)]
fn literal_i8<const N: usize>(unused: &[i8]) -> Result<PcuTensor<i8>, PcuExecutionError> {
    pcu::constant(const { payload_i8::<N>() })
}
#[pcu(crate_path=::pcu_facade)]
fn uniform_i8(input: &[i8]) -> Result<PcuTensor<i8>, PcuExecutionError> {
    pcu::uniform_like(input, const { 2_i8 })
}
#[pcu(crate_path=::pcu_facade)]
fn foreign_i8<const N: usize>(_unused: &PcuTensor<i8>) -> Result<PcuTensor<i8>, PcuExecutionError> {
    pcu::constant(const { payload_i8::<N>() })
}
impl ProducerSource for i8 {
    fn capture<const N: usize>(
        mode: pcu_facade::PcuNumericalMode,
        options: pcu_facade::PcuNumericalOptions,
    ) -> Result<pcu_facade::global::PcuCapturedTensorProgram, PcuExecutionError> {
        for builder in [
            literal_i8::__pcu_capture_entry::<N>,
            uniform_i8::__pcu_capture_entry,
        ] {
            let captured = pcu_facade::global::__pcu_capture_tensor_program::<Self, 1, _>(
                [pcu_facade::global::PcuSourceShape::Slice { length: N }],
                pcu_facade::PcuFloatUnderflowPolicy::IeeeAfterRounding,
                mode,
                options,
                builder,
            )?;
            assert!(captured.argument_indices().is_empty());
            assert!(captured.program().input_values().is_empty());
        }
        pcu_facade::global::__pcu_capture_tensor_program::<Self, 1, _>(
            [pcu_facade::global::PcuSourceShape::Slice { length: N }],
            pcu_facade::PcuFloatUnderflowPolicy::IeeeAfterRounding,
            mode,
            options,
            pipeline_i8::__pcu_capture_entry::<N>,
        )
    }
    fn pipeline<const N: usize>(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        pipeline_i8::<N>(input)
    }
    fn literal<const N: usize>(unused: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        literal_i8::<N>(unused)
    }
    fn uniform(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        uniform_i8(input)
    }
    fn unused_owner<const N: usize>(
        unused: &PcuTensor<Self>,
    ) -> Result<PcuTensor<Self>, PcuExecutionError> {
        foreign_i8::<N>(unused)
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)] // low is independently bounded to1..=3.
const fn payload_u16<const N: usize>() -> [u16; N] {
    let mut values = [0_u16; N];
    let mut index = 0;
    while index < N {
        let low = index % 3 + 1;
        values[index] = (1_u16 << 11) | (low as u16);
        index += 1;
    }
    values
}
#[pcu(crate_path=::pcu_facade)]
fn pipeline_u16<const N: usize>(input: &[u16]) -> Result<PcuTensor<u16>, PcuExecutionError> {
    let constant = pcu::constant(const { payload_u16::<N>() })?;
    let uniform = pcu::uniform_like(input, const { 2_u16 })?;
    let sum = pcu::add(input, &constant)?;
    pcu::mul(&sum, &uniform)
}
#[pcu(crate_path=::pcu_facade)]
fn literal_u16<const N: usize>(unused: &[u16]) -> Result<PcuTensor<u16>, PcuExecutionError> {
    pcu::constant(const { payload_u16::<N>() })
}
#[pcu(crate_path=::pcu_facade)]
fn uniform_u16(input: &[u16]) -> Result<PcuTensor<u16>, PcuExecutionError> {
    pcu::uniform_like(input, const { 2_u16 })
}
#[pcu(crate_path=::pcu_facade)]
fn foreign_u16<const N: usize>(
    _unused: &PcuTensor<u16>,
) -> Result<PcuTensor<u16>, PcuExecutionError> {
    pcu::constant(const { payload_u16::<N>() })
}
impl ProducerSource for u16 {
    fn capture<const N: usize>(
        mode: pcu_facade::PcuNumericalMode,
        options: pcu_facade::PcuNumericalOptions,
    ) -> Result<pcu_facade::global::PcuCapturedTensorProgram, PcuExecutionError> {
        for builder in [
            literal_u16::__pcu_capture_entry::<N>,
            uniform_u16::__pcu_capture_entry,
        ] {
            let captured = pcu_facade::global::__pcu_capture_tensor_program::<Self, 1, _>(
                [pcu_facade::global::PcuSourceShape::Slice { length: N }],
                pcu_facade::PcuFloatUnderflowPolicy::IeeeAfterRounding,
                mode,
                options,
                builder,
            )?;
            assert!(captured.argument_indices().is_empty());
            assert!(captured.program().input_values().is_empty());
        }
        pcu_facade::global::__pcu_capture_tensor_program::<Self, 1, _>(
            [pcu_facade::global::PcuSourceShape::Slice { length: N }],
            pcu_facade::PcuFloatUnderflowPolicy::IeeeAfterRounding,
            mode,
            options,
            pipeline_u16::__pcu_capture_entry::<N>,
        )
    }
    fn pipeline<const N: usize>(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        pipeline_u16::<N>(input)
    }
    fn literal<const N: usize>(unused: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        literal_u16::<N>(unused)
    }
    fn uniform(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        uniform_u16(input)
    }
    fn unused_owner<const N: usize>(
        unused: &PcuTensor<Self>,
    ) -> Result<PcuTensor<Self>, PcuExecutionError> {
        foreign_u16::<N>(unused)
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)] // low is independently bounded to1..=3.
const fn payload_i16<const N: usize>() -> [i16; N] {
    let mut values = [0_i16; N];
    let mut index = 0;
    while index < N {
        let low = index % 3 + 1;
        values[index] = (1_i16 << 11) | (low as i16);
        index += 1;
    }
    values
}
#[pcu(crate_path=::pcu_facade)]
fn pipeline_i16<const N: usize>(input: &[i16]) -> Result<PcuTensor<i16>, PcuExecutionError> {
    let constant = pcu::constant(const { payload_i16::<N>() })?;
    let uniform = pcu::uniform_like(input, const { 2_i16 })?;
    let sum = pcu::add(input, &constant)?;
    pcu::mul(&sum, &uniform)
}
#[pcu(crate_path=::pcu_facade)]
fn literal_i16<const N: usize>(unused: &[i16]) -> Result<PcuTensor<i16>, PcuExecutionError> {
    pcu::constant(const { payload_i16::<N>() })
}
#[pcu(crate_path=::pcu_facade)]
fn uniform_i16(input: &[i16]) -> Result<PcuTensor<i16>, PcuExecutionError> {
    pcu::uniform_like(input, const { 2_i16 })
}
#[pcu(crate_path=::pcu_facade)]
fn foreign_i16<const N: usize>(
    _unused: &PcuTensor<i16>,
) -> Result<PcuTensor<i16>, PcuExecutionError> {
    pcu::constant(const { payload_i16::<N>() })
}
impl ProducerSource for i16 {
    fn capture<const N: usize>(
        mode: pcu_facade::PcuNumericalMode,
        options: pcu_facade::PcuNumericalOptions,
    ) -> Result<pcu_facade::global::PcuCapturedTensorProgram, PcuExecutionError> {
        for builder in [
            literal_i16::__pcu_capture_entry::<N>,
            uniform_i16::__pcu_capture_entry,
        ] {
            let captured = pcu_facade::global::__pcu_capture_tensor_program::<Self, 1, _>(
                [pcu_facade::global::PcuSourceShape::Slice { length: N }],
                pcu_facade::PcuFloatUnderflowPolicy::IeeeAfterRounding,
                mode,
                options,
                builder,
            )?;
            assert!(captured.argument_indices().is_empty());
            assert!(captured.program().input_values().is_empty());
        }
        pcu_facade::global::__pcu_capture_tensor_program::<Self, 1, _>(
            [pcu_facade::global::PcuSourceShape::Slice { length: N }],
            pcu_facade::PcuFloatUnderflowPolicy::IeeeAfterRounding,
            mode,
            options,
            pipeline_i16::__pcu_capture_entry::<N>,
        )
    }
    fn pipeline<const N: usize>(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        pipeline_i16::<N>(input)
    }
    fn literal<const N: usize>(unused: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        literal_i16::<N>(unused)
    }
    fn uniform(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        uniform_i16(input)
    }
    fn unused_owner<const N: usize>(
        unused: &PcuTensor<Self>,
    ) -> Result<PcuTensor<Self>, PcuExecutionError> {
        foreign_i16::<N>(unused)
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)] // low is independently bounded to1..=3.
const fn payload_u32<const N: usize>() -> [u32; N] {
    let mut values = [0_u32; N];
    let mut index = 0;
    while index < N {
        let low = index % 3 + 1;
        values[index] = (1_u32 << 27) | (low as u32);
        index += 1;
    }
    values
}
#[pcu(crate_path=::pcu_facade)]
fn pipeline_u32<const N: usize>(input: &[u32]) -> Result<PcuTensor<u32>, PcuExecutionError> {
    let constant = pcu::constant(const { payload_u32::<N>() })?;
    let uniform = pcu::uniform_like(input, const { 2_u32 })?;
    let sum = pcu::add(input, &constant)?;
    pcu::mul(&sum, &uniform)
}
#[pcu(crate_path=::pcu_facade)]
fn literal_u32<const N: usize>(unused: &[u32]) -> Result<PcuTensor<u32>, PcuExecutionError> {
    pcu::constant(const { payload_u32::<N>() })
}
#[pcu(crate_path=::pcu_facade)]
fn uniform_u32(input: &[u32]) -> Result<PcuTensor<u32>, PcuExecutionError> {
    pcu::uniform_like(input, const { 2_u32 })
}
#[pcu(crate_path=::pcu_facade)]
fn foreign_u32<const N: usize>(
    _unused: &PcuTensor<u32>,
) -> Result<PcuTensor<u32>, PcuExecutionError> {
    pcu::constant(const { payload_u32::<N>() })
}
impl ProducerSource for u32 {
    fn capture<const N: usize>(
        mode: pcu_facade::PcuNumericalMode,
        options: pcu_facade::PcuNumericalOptions,
    ) -> Result<pcu_facade::global::PcuCapturedTensorProgram, PcuExecutionError> {
        for builder in [
            literal_u32::__pcu_capture_entry::<N>,
            uniform_u32::__pcu_capture_entry,
        ] {
            let captured = pcu_facade::global::__pcu_capture_tensor_program::<Self, 1, _>(
                [pcu_facade::global::PcuSourceShape::Slice { length: N }],
                pcu_facade::PcuFloatUnderflowPolicy::IeeeAfterRounding,
                mode,
                options,
                builder,
            )?;
            assert!(captured.argument_indices().is_empty());
            assert!(captured.program().input_values().is_empty());
        }
        pcu_facade::global::__pcu_capture_tensor_program::<Self, 1, _>(
            [pcu_facade::global::PcuSourceShape::Slice { length: N }],
            pcu_facade::PcuFloatUnderflowPolicy::IeeeAfterRounding,
            mode,
            options,
            pipeline_u32::__pcu_capture_entry::<N>,
        )
    }
    fn pipeline<const N: usize>(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        pipeline_u32::<N>(input)
    }
    fn literal<const N: usize>(unused: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        literal_u32::<N>(unused)
    }
    fn uniform(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        uniform_u32(input)
    }
    fn unused_owner<const N: usize>(
        unused: &PcuTensor<Self>,
    ) -> Result<PcuTensor<Self>, PcuExecutionError> {
        foreign_u32::<N>(unused)
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)] // low is independently bounded to1..=3.
const fn payload_i32<const N: usize>() -> [i32; N] {
    let mut values = [0_i32; N];
    let mut index = 0;
    while index < N {
        let low = index % 3 + 1;
        values[index] = (1_i32 << 27) | (low as i32);
        index += 1;
    }
    values
}
#[pcu(crate_path=::pcu_facade)]
fn pipeline_i32<const N: usize>(input: &[i32]) -> Result<PcuTensor<i32>, PcuExecutionError> {
    let constant = pcu::constant(const { payload_i32::<N>() })?;
    let uniform = pcu::uniform_like(input, const { 2_i32 })?;
    let sum = pcu::add(input, &constant)?;
    pcu::mul(&sum, &uniform)
}
#[pcu(crate_path=::pcu_facade)]
fn literal_i32<const N: usize>(unused: &[i32]) -> Result<PcuTensor<i32>, PcuExecutionError> {
    pcu::constant(const { payload_i32::<N>() })
}
#[pcu(crate_path=::pcu_facade)]
fn uniform_i32(input: &[i32]) -> Result<PcuTensor<i32>, PcuExecutionError> {
    pcu::uniform_like(input, const { 2_i32 })
}
#[pcu(crate_path=::pcu_facade)]
fn foreign_i32<const N: usize>(
    _unused: &PcuTensor<i32>,
) -> Result<PcuTensor<i32>, PcuExecutionError> {
    pcu::constant(const { payload_i32::<N>() })
}
impl ProducerSource for i32 {
    fn capture<const N: usize>(
        mode: pcu_facade::PcuNumericalMode,
        options: pcu_facade::PcuNumericalOptions,
    ) -> Result<pcu_facade::global::PcuCapturedTensorProgram, PcuExecutionError> {
        for builder in [
            literal_i32::__pcu_capture_entry::<N>,
            uniform_i32::__pcu_capture_entry,
        ] {
            let captured = pcu_facade::global::__pcu_capture_tensor_program::<Self, 1, _>(
                [pcu_facade::global::PcuSourceShape::Slice { length: N }],
                pcu_facade::PcuFloatUnderflowPolicy::IeeeAfterRounding,
                mode,
                options,
                builder,
            )?;
            assert!(captured.argument_indices().is_empty());
            assert!(captured.program().input_values().is_empty());
        }
        pcu_facade::global::__pcu_capture_tensor_program::<Self, 1, _>(
            [pcu_facade::global::PcuSourceShape::Slice { length: N }],
            pcu_facade::PcuFloatUnderflowPolicy::IeeeAfterRounding,
            mode,
            options,
            pipeline_i32::__pcu_capture_entry::<N>,
        )
    }
    fn pipeline<const N: usize>(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        pipeline_i32::<N>(input)
    }
    fn literal<const N: usize>(unused: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        literal_i32::<N>(unused)
    }
    fn uniform(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        uniform_i32(input)
    }
    fn unused_owner<const N: usize>(
        unused: &PcuTensor<Self>,
    ) -> Result<PcuTensor<Self>, PcuExecutionError> {
        foreign_i32::<N>(unused)
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)] // low is independently bounded to1..=3.
const fn payload_u64<const N: usize>() -> [u64; N] {
    let mut values = [0_u64; N];
    let mut index = 0;
    while index < N {
        let low = index % 3 + 1;
        values[index] = (1_u64 << 59) | (low as u64);
        index += 1;
    }
    values
}
#[pcu(crate_path=::pcu_facade)]
fn pipeline_u64<const N: usize>(input: &[u64]) -> Result<PcuTensor<u64>, PcuExecutionError> {
    let constant = pcu::constant(const { payload_u64::<N>() })?;
    let uniform = pcu::uniform_like(input, const { 2_u64 })?;
    let sum = pcu::add(input, &constant)?;
    pcu::mul(&sum, &uniform)
}
#[pcu(crate_path=::pcu_facade)]
fn literal_u64<const N: usize>(unused: &[u64]) -> Result<PcuTensor<u64>, PcuExecutionError> {
    pcu::constant(const { payload_u64::<N>() })
}
#[pcu(crate_path=::pcu_facade)]
fn uniform_u64(input: &[u64]) -> Result<PcuTensor<u64>, PcuExecutionError> {
    pcu::uniform_like(input, const { 2_u64 })
}
#[pcu(crate_path=::pcu_facade)]
fn foreign_u64<const N: usize>(
    _unused: &PcuTensor<u64>,
) -> Result<PcuTensor<u64>, PcuExecutionError> {
    pcu::constant(const { payload_u64::<N>() })
}
impl ProducerSource for u64 {
    fn capture<const N: usize>(
        mode: pcu_facade::PcuNumericalMode,
        options: pcu_facade::PcuNumericalOptions,
    ) -> Result<pcu_facade::global::PcuCapturedTensorProgram, PcuExecutionError> {
        for builder in [
            literal_u64::__pcu_capture_entry::<N>,
            uniform_u64::__pcu_capture_entry,
        ] {
            let captured = pcu_facade::global::__pcu_capture_tensor_program::<Self, 1, _>(
                [pcu_facade::global::PcuSourceShape::Slice { length: N }],
                pcu_facade::PcuFloatUnderflowPolicy::IeeeAfterRounding,
                mode,
                options,
                builder,
            )?;
            assert!(captured.argument_indices().is_empty());
            assert!(captured.program().input_values().is_empty());
        }
        pcu_facade::global::__pcu_capture_tensor_program::<Self, 1, _>(
            [pcu_facade::global::PcuSourceShape::Slice { length: N }],
            pcu_facade::PcuFloatUnderflowPolicy::IeeeAfterRounding,
            mode,
            options,
            pipeline_u64::__pcu_capture_entry::<N>,
        )
    }
    fn pipeline<const N: usize>(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        pipeline_u64::<N>(input)
    }
    fn literal<const N: usize>(unused: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        literal_u64::<N>(unused)
    }
    fn uniform(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        uniform_u64(input)
    }
    fn unused_owner<const N: usize>(
        unused: &PcuTensor<Self>,
    ) -> Result<PcuTensor<Self>, PcuExecutionError> {
        foreign_u64::<N>(unused)
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)] // low is independently bounded to1..=3.
const fn payload_i64<const N: usize>() -> [i64; N] {
    let mut values = [0_i64; N];
    let mut index = 0;
    while index < N {
        let low = index % 3 + 1;
        values[index] = (1_i64 << 59) | (low as i64);
        index += 1;
    }
    values
}
#[pcu(crate_path=::pcu_facade)]
fn pipeline_i64<const N: usize>(input: &[i64]) -> Result<PcuTensor<i64>, PcuExecutionError> {
    let constant = pcu::constant(const { payload_i64::<N>() })?;
    let uniform = pcu::uniform_like(input, const { 2_i64 })?;
    let sum = pcu::add(input, &constant)?;
    pcu::mul(&sum, &uniform)
}
#[pcu(crate_path=::pcu_facade)]
fn literal_i64<const N: usize>(unused: &[i64]) -> Result<PcuTensor<i64>, PcuExecutionError> {
    pcu::constant(const { payload_i64::<N>() })
}
#[pcu(crate_path=::pcu_facade)]
fn uniform_i64(input: &[i64]) -> Result<PcuTensor<i64>, PcuExecutionError> {
    pcu::uniform_like(input, const { 2_i64 })
}
#[pcu(crate_path=::pcu_facade)]
fn foreign_i64<const N: usize>(
    _unused: &PcuTensor<i64>,
) -> Result<PcuTensor<i64>, PcuExecutionError> {
    pcu::constant(const { payload_i64::<N>() })
}
impl ProducerSource for i64 {
    fn capture<const N: usize>(
        mode: pcu_facade::PcuNumericalMode,
        options: pcu_facade::PcuNumericalOptions,
    ) -> Result<pcu_facade::global::PcuCapturedTensorProgram, PcuExecutionError> {
        for builder in [
            literal_i64::__pcu_capture_entry::<N>,
            uniform_i64::__pcu_capture_entry,
        ] {
            let captured = pcu_facade::global::__pcu_capture_tensor_program::<Self, 1, _>(
                [pcu_facade::global::PcuSourceShape::Slice { length: N }],
                pcu_facade::PcuFloatUnderflowPolicy::IeeeAfterRounding,
                mode,
                options,
                builder,
            )?;
            assert!(captured.argument_indices().is_empty());
            assert!(captured.program().input_values().is_empty());
        }
        pcu_facade::global::__pcu_capture_tensor_program::<Self, 1, _>(
            [pcu_facade::global::PcuSourceShape::Slice { length: N }],
            pcu_facade::PcuFloatUnderflowPolicy::IeeeAfterRounding,
            mode,
            options,
            pipeline_i64::__pcu_capture_entry::<N>,
        )
    }
    fn pipeline<const N: usize>(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        pipeline_i64::<N>(input)
    }
    fn literal<const N: usize>(unused: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        literal_i64::<N>(unused)
    }
    fn uniform(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        uniform_i64(input)
    }
    fn unused_owner<const N: usize>(
        unused: &PcuTensor<Self>,
    ) -> Result<PcuTensor<Self>, PcuExecutionError> {
        foreign_i64::<N>(unused)
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)] // low is independently bounded to1..=3.
const fn payload_u128<const N: usize>() -> [u128; N] {
    let mut values = [0_u128; N];
    let mut index = 0;
    while index < N {
        let low = index % 3 + 1;
        values[index] = (1_u128 << 123) | (low as u128);
        index += 1;
    }
    values
}
#[pcu(crate_path=::pcu_facade)]
fn pipeline_u128<const N: usize>(input: &[u128]) -> Result<PcuTensor<u128>, PcuExecutionError> {
    let constant = pcu::constant(const { payload_u128::<N>() })?;
    let uniform = pcu::uniform_like(input, const { 2_u128 })?;
    let sum = pcu::add(input, &constant)?;
    pcu::mul(&sum, &uniform)
}
#[pcu(crate_path=::pcu_facade)]
fn literal_u128<const N: usize>(unused: &[u128]) -> Result<PcuTensor<u128>, PcuExecutionError> {
    pcu::constant(const { payload_u128::<N>() })
}
#[pcu(crate_path=::pcu_facade)]
fn uniform_u128(input: &[u128]) -> Result<PcuTensor<u128>, PcuExecutionError> {
    pcu::uniform_like(input, const { 2_u128 })
}
#[pcu(crate_path=::pcu_facade)]
fn foreign_u128<const N: usize>(
    _unused: &PcuTensor<u128>,
) -> Result<PcuTensor<u128>, PcuExecutionError> {
    pcu::constant(const { payload_u128::<N>() })
}
impl ProducerSource for u128 {
    fn capture<const N: usize>(
        mode: pcu_facade::PcuNumericalMode,
        options: pcu_facade::PcuNumericalOptions,
    ) -> Result<pcu_facade::global::PcuCapturedTensorProgram, PcuExecutionError> {
        for builder in [
            literal_u128::__pcu_capture_entry::<N>,
            uniform_u128::__pcu_capture_entry,
        ] {
            let captured = pcu_facade::global::__pcu_capture_tensor_program::<Self, 1, _>(
                [pcu_facade::global::PcuSourceShape::Slice { length: N }],
                pcu_facade::PcuFloatUnderflowPolicy::IeeeAfterRounding,
                mode,
                options,
                builder,
            )?;
            assert!(captured.argument_indices().is_empty());
            assert!(captured.program().input_values().is_empty());
        }
        pcu_facade::global::__pcu_capture_tensor_program::<Self, 1, _>(
            [pcu_facade::global::PcuSourceShape::Slice { length: N }],
            pcu_facade::PcuFloatUnderflowPolicy::IeeeAfterRounding,
            mode,
            options,
            pipeline_u128::__pcu_capture_entry::<N>,
        )
    }
    fn pipeline<const N: usize>(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        pipeline_u128::<N>(input)
    }
    fn literal<const N: usize>(unused: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        literal_u128::<N>(unused)
    }
    fn uniform(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        uniform_u128(input)
    }
    fn unused_owner<const N: usize>(
        unused: &PcuTensor<Self>,
    ) -> Result<PcuTensor<Self>, PcuExecutionError> {
        foreign_u128::<N>(unused)
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)] // low is independently bounded to1..=3.
const fn payload_i128<const N: usize>() -> [i128; N] {
    let mut values = [0_i128; N];
    let mut index = 0;
    while index < N {
        let low = index % 3 + 1;
        values[index] = (1_i128 << 123) | (low as i128);
        index += 1;
    }
    values
}
#[pcu(crate_path=::pcu_facade)]
fn pipeline_i128<const N: usize>(input: &[i128]) -> Result<PcuTensor<i128>, PcuExecutionError> {
    let constant = pcu::constant(const { payload_i128::<N>() })?;
    let uniform = pcu::uniform_like(input, const { 2_i128 })?;
    let sum = pcu::add(input, &constant)?;
    pcu::mul(&sum, &uniform)
}
#[pcu(crate_path=::pcu_facade)]
fn literal_i128<const N: usize>(unused: &[i128]) -> Result<PcuTensor<i128>, PcuExecutionError> {
    pcu::constant(const { payload_i128::<N>() })
}
#[pcu(crate_path=::pcu_facade)]
fn uniform_i128(input: &[i128]) -> Result<PcuTensor<i128>, PcuExecutionError> {
    pcu::uniform_like(input, const { 2_i128 })
}
#[pcu(crate_path=::pcu_facade)]
fn foreign_i128<const N: usize>(
    _unused: &PcuTensor<i128>,
) -> Result<PcuTensor<i128>, PcuExecutionError> {
    pcu::constant(const { payload_i128::<N>() })
}
impl ProducerSource for i128 {
    fn capture<const N: usize>(
        mode: pcu_facade::PcuNumericalMode,
        options: pcu_facade::PcuNumericalOptions,
    ) -> Result<pcu_facade::global::PcuCapturedTensorProgram, PcuExecutionError> {
        for builder in [
            literal_i128::__pcu_capture_entry::<N>,
            uniform_i128::__pcu_capture_entry,
        ] {
            let captured = pcu_facade::global::__pcu_capture_tensor_program::<Self, 1, _>(
                [pcu_facade::global::PcuSourceShape::Slice { length: N }],
                pcu_facade::PcuFloatUnderflowPolicy::IeeeAfterRounding,
                mode,
                options,
                builder,
            )?;
            assert!(captured.argument_indices().is_empty());
            assert!(captured.program().input_values().is_empty());
        }
        pcu_facade::global::__pcu_capture_tensor_program::<Self, 1, _>(
            [pcu_facade::global::PcuSourceShape::Slice { length: N }],
            pcu_facade::PcuFloatUnderflowPolicy::IeeeAfterRounding,
            mode,
            options,
            pipeline_i128::__pcu_capture_entry::<N>,
        )
    }
    fn pipeline<const N: usize>(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        pipeline_i128::<N>(input)
    }
    fn literal<const N: usize>(unused: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        literal_i128::<N>(unused)
    }
    fn uniform(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        uniform_i128(input)
    }
    fn unused_owner<const N: usize>(
        unused: &PcuTensor<Self>,
    ) -> Result<PcuTensor<Self>, PcuExecutionError> {
        foreign_i128::<N>(unused)
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)] // low is independently bounded to1..=3.
const fn payload_u256<const N: usize>() -> [PcuU256; N] {
    let mut values = [PcuU256::from_limbs_le([0, 0, 0, 0]); N];
    let mut index = 0;
    while index < N {
        let low = index % 3 + 1;
        values[index] = PcuU256::from_limbs_le([low as u64, 0, 0, 1_u64 << 59]);
        index += 1;
    }
    values
}
#[pcu(crate_path=::pcu_facade)]
fn pipeline_u256<const N: usize>(
    input: &[PcuU256],
) -> Result<PcuTensor<PcuU256>, PcuExecutionError> {
    let constant = pcu::constant(const { payload_u256::<N>() })?;
    let uniform = pcu::uniform_like(input, const { PcuU256::from_limbs_le([2, 0, 0, 0]) })?;
    let sum = pcu::add(input, &constant)?;
    pcu::mul(&sum, &uniform)
}
#[pcu(crate_path=::pcu_facade)]
fn literal_u256<const N: usize>(
    unused: &[PcuU256],
) -> Result<PcuTensor<PcuU256>, PcuExecutionError> {
    pcu::constant(const { payload_u256::<N>() })
}
#[pcu(crate_path=::pcu_facade)]
fn uniform_u256(input: &[PcuU256]) -> Result<PcuTensor<PcuU256>, PcuExecutionError> {
    pcu::uniform_like(input, const { PcuU256::from_limbs_le([2, 0, 0, 0]) })
}
#[pcu(crate_path=::pcu_facade)]
fn foreign_u256<const N: usize>(
    _unused: &PcuTensor<PcuU256>,
) -> Result<PcuTensor<PcuU256>, PcuExecutionError> {
    pcu::constant(const { payload_u256::<N>() })
}
impl ProducerSource for PcuU256 {
    fn capture<const N: usize>(
        mode: pcu_facade::PcuNumericalMode,
        options: pcu_facade::PcuNumericalOptions,
    ) -> Result<pcu_facade::global::PcuCapturedTensorProgram, PcuExecutionError> {
        for builder in [
            literal_u256::__pcu_capture_entry::<N>,
            uniform_u256::__pcu_capture_entry,
        ] {
            let captured = pcu_facade::global::__pcu_capture_tensor_program::<Self, 1, _>(
                [pcu_facade::global::PcuSourceShape::Slice { length: N }],
                pcu_facade::PcuFloatUnderflowPolicy::IeeeAfterRounding,
                mode,
                options,
                builder,
            )?;
            assert!(captured.argument_indices().is_empty());
            assert!(captured.program().input_values().is_empty());
        }
        pcu_facade::global::__pcu_capture_tensor_program::<Self, 1, _>(
            [pcu_facade::global::PcuSourceShape::Slice { length: N }],
            pcu_facade::PcuFloatUnderflowPolicy::IeeeAfterRounding,
            mode,
            options,
            pipeline_u256::__pcu_capture_entry::<N>,
        )
    }
    fn pipeline<const N: usize>(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        pipeline_u256::<N>(input)
    }
    fn literal<const N: usize>(unused: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        literal_u256::<N>(unused)
    }
    fn uniform(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        uniform_u256(input)
    }
    fn unused_owner<const N: usize>(
        unused: &PcuTensor<Self>,
    ) -> Result<PcuTensor<Self>, PcuExecutionError> {
        foreign_u256::<N>(unused)
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)] // low is independently bounded to1..=3.
const fn payload_i256<const N: usize>() -> [PcuI256; N] {
    let mut values = [PcuI256::from_limbs_le([0, 0, 0, 0]); N];
    let mut index = 0;
    while index < N {
        let low = index % 3 + 1;
        values[index] = PcuI256::from_limbs_le([low as u64, 0, 0, 1_u64 << 59]);
        index += 1;
    }
    values
}
#[pcu(crate_path=::pcu_facade)]
fn pipeline_i256<const N: usize>(
    input: &[PcuI256],
) -> Result<PcuTensor<PcuI256>, PcuExecutionError> {
    let constant = pcu::constant(const { payload_i256::<N>() })?;
    let uniform = pcu::uniform_like(input, const { PcuI256::from_limbs_le([2, 0, 0, 0]) })?;
    let sum = pcu::add(input, &constant)?;
    pcu::mul(&sum, &uniform)
}
#[pcu(crate_path=::pcu_facade)]
fn literal_i256<const N: usize>(
    unused: &[PcuI256],
) -> Result<PcuTensor<PcuI256>, PcuExecutionError> {
    pcu::constant(const { payload_i256::<N>() })
}
#[pcu(crate_path=::pcu_facade)]
fn uniform_i256(input: &[PcuI256]) -> Result<PcuTensor<PcuI256>, PcuExecutionError> {
    pcu::uniform_like(input, const { PcuI256::from_limbs_le([2, 0, 0, 0]) })
}
#[pcu(crate_path=::pcu_facade)]
fn foreign_i256<const N: usize>(
    _unused: &PcuTensor<PcuI256>,
) -> Result<PcuTensor<PcuI256>, PcuExecutionError> {
    pcu::constant(const { payload_i256::<N>() })
}
impl ProducerSource for PcuI256 {
    fn capture<const N: usize>(
        mode: pcu_facade::PcuNumericalMode,
        options: pcu_facade::PcuNumericalOptions,
    ) -> Result<pcu_facade::global::PcuCapturedTensorProgram, PcuExecutionError> {
        for builder in [
            literal_i256::__pcu_capture_entry::<N>,
            uniform_i256::__pcu_capture_entry,
        ] {
            let captured = pcu_facade::global::__pcu_capture_tensor_program::<Self, 1, _>(
                [pcu_facade::global::PcuSourceShape::Slice { length: N }],
                pcu_facade::PcuFloatUnderflowPolicy::IeeeAfterRounding,
                mode,
                options,
                builder,
            )?;
            assert!(captured.argument_indices().is_empty());
            assert!(captured.program().input_values().is_empty());
        }
        pcu_facade::global::__pcu_capture_tensor_program::<Self, 1, _>(
            [pcu_facade::global::PcuSourceShape::Slice { length: N }],
            pcu_facade::PcuFloatUnderflowPolicy::IeeeAfterRounding,
            mode,
            options,
            pipeline_i256::__pcu_capture_entry::<N>,
        )
    }
    fn pipeline<const N: usize>(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        pipeline_i256::<N>(input)
    }
    fn literal<const N: usize>(unused: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        literal_i256::<N>(unused)
    }
    fn uniform(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        uniform_i256(input)
    }
    fn unused_owner<const N: usize>(
        unused: &PcuTensor<Self>,
    ) -> Result<PcuTensor<Self>, PcuExecutionError> {
        foreign_i256::<N>(unused)
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)] // low is independently bounded to1..=3.
const fn payload_u512<const N: usize>() -> [PcuU512; N] {
    let mut values = [PcuU512::from_limbs_le([0, 0, 0, 0, 0, 0, 0, 0]); N];
    let mut index = 0;
    while index < N {
        let low = index % 3 + 1;
        values[index] = PcuU512::from_limbs_le([low as u64, 0, 0, 0, 0, 0, 0, 1_u64 << 59]);
        index += 1;
    }
    values
}
#[pcu(crate_path=::pcu_facade)]
fn pipeline_u512<const N: usize>(
    input: &[PcuU512],
) -> Result<PcuTensor<PcuU512>, PcuExecutionError> {
    let constant = pcu::constant(const { payload_u512::<N>() })?;
    let uniform = pcu::uniform_like(
        input,
        const { PcuU512::from_limbs_le([2, 0, 0, 0, 0, 0, 0, 0]) },
    )?;
    let sum = pcu::add(input, &constant)?;
    pcu::mul(&sum, &uniform)
}
#[pcu(crate_path=::pcu_facade)]
fn literal_u512<const N: usize>(
    unused: &[PcuU512],
) -> Result<PcuTensor<PcuU512>, PcuExecutionError> {
    pcu::constant(const { payload_u512::<N>() })
}
#[pcu(crate_path=::pcu_facade)]
fn uniform_u512(input: &[PcuU512]) -> Result<PcuTensor<PcuU512>, PcuExecutionError> {
    pcu::uniform_like(
        input,
        const { PcuU512::from_limbs_le([2, 0, 0, 0, 0, 0, 0, 0]) },
    )
}
#[pcu(crate_path=::pcu_facade)]
fn foreign_u512<const N: usize>(
    _unused: &PcuTensor<PcuU512>,
) -> Result<PcuTensor<PcuU512>, PcuExecutionError> {
    pcu::constant(const { payload_u512::<N>() })
}
impl ProducerSource for PcuU512 {
    fn capture<const N: usize>(
        mode: pcu_facade::PcuNumericalMode,
        options: pcu_facade::PcuNumericalOptions,
    ) -> Result<pcu_facade::global::PcuCapturedTensorProgram, PcuExecutionError> {
        for builder in [
            literal_u512::__pcu_capture_entry::<N>,
            uniform_u512::__pcu_capture_entry,
        ] {
            let captured = pcu_facade::global::__pcu_capture_tensor_program::<Self, 1, _>(
                [pcu_facade::global::PcuSourceShape::Slice { length: N }],
                pcu_facade::PcuFloatUnderflowPolicy::IeeeAfterRounding,
                mode,
                options,
                builder,
            )?;
            assert!(captured.argument_indices().is_empty());
            assert!(captured.program().input_values().is_empty());
        }
        pcu_facade::global::__pcu_capture_tensor_program::<Self, 1, _>(
            [pcu_facade::global::PcuSourceShape::Slice { length: N }],
            pcu_facade::PcuFloatUnderflowPolicy::IeeeAfterRounding,
            mode,
            options,
            pipeline_u512::__pcu_capture_entry::<N>,
        )
    }
    fn pipeline<const N: usize>(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        pipeline_u512::<N>(input)
    }
    fn literal<const N: usize>(unused: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        literal_u512::<N>(unused)
    }
    fn uniform(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        uniform_u512(input)
    }
    fn unused_owner<const N: usize>(
        unused: &PcuTensor<Self>,
    ) -> Result<PcuTensor<Self>, PcuExecutionError> {
        foreign_u512::<N>(unused)
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)] // low is independently bounded to1..=3.
const fn payload_i512<const N: usize>() -> [PcuI512; N] {
    let mut values = [PcuI512::from_limbs_le([0, 0, 0, 0, 0, 0, 0, 0]); N];
    let mut index = 0;
    while index < N {
        let low = index % 3 + 1;
        values[index] = PcuI512::from_limbs_le([low as u64, 0, 0, 0, 0, 0, 0, 1_u64 << 59]);
        index += 1;
    }
    values
}
#[pcu(crate_path=::pcu_facade)]
fn pipeline_i512<const N: usize>(
    input: &[PcuI512],
) -> Result<PcuTensor<PcuI512>, PcuExecutionError> {
    let constant = pcu::constant(const { payload_i512::<N>() })?;
    let uniform = pcu::uniform_like(
        input,
        const { PcuI512::from_limbs_le([2, 0, 0, 0, 0, 0, 0, 0]) },
    )?;
    let sum = pcu::add(input, &constant)?;
    pcu::mul(&sum, &uniform)
}
#[pcu(crate_path=::pcu_facade)]
fn literal_i512<const N: usize>(
    unused: &[PcuI512],
) -> Result<PcuTensor<PcuI512>, PcuExecutionError> {
    pcu::constant(const { payload_i512::<N>() })
}
#[pcu(crate_path=::pcu_facade)]
fn uniform_i512(input: &[PcuI512]) -> Result<PcuTensor<PcuI512>, PcuExecutionError> {
    pcu::uniform_like(
        input,
        const { PcuI512::from_limbs_le([2, 0, 0, 0, 0, 0, 0, 0]) },
    )
}
#[pcu(crate_path=::pcu_facade)]
fn foreign_i512<const N: usize>(
    _unused: &PcuTensor<PcuI512>,
) -> Result<PcuTensor<PcuI512>, PcuExecutionError> {
    pcu::constant(const { payload_i512::<N>() })
}
impl ProducerSource for PcuI512 {
    fn capture<const N: usize>(
        mode: pcu_facade::PcuNumericalMode,
        options: pcu_facade::PcuNumericalOptions,
    ) -> Result<pcu_facade::global::PcuCapturedTensorProgram, PcuExecutionError> {
        for builder in [
            literal_i512::__pcu_capture_entry::<N>,
            uniform_i512::__pcu_capture_entry,
        ] {
            let captured = pcu_facade::global::__pcu_capture_tensor_program::<Self, 1, _>(
                [pcu_facade::global::PcuSourceShape::Slice { length: N }],
                pcu_facade::PcuFloatUnderflowPolicy::IeeeAfterRounding,
                mode,
                options,
                builder,
            )?;
            assert!(captured.argument_indices().is_empty());
            assert!(captured.program().input_values().is_empty());
        }
        pcu_facade::global::__pcu_capture_tensor_program::<Self, 1, _>(
            [pcu_facade::global::PcuSourceShape::Slice { length: N }],
            pcu_facade::PcuFloatUnderflowPolicy::IeeeAfterRounding,
            mode,
            options,
            pipeline_i512::__pcu_capture_entry::<N>,
        )
    }
    fn pipeline<const N: usize>(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        pipeline_i512::<N>(input)
    }
    fn literal<const N: usize>(unused: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        literal_i512::<N>(unused)
    }
    fn uniform(input: &[Self]) -> Result<PcuTensor<Self>, PcuExecutionError> {
        uniform_i512(input)
    }
    fn unused_owner<const N: usize>(
        unused: &PcuTensor<Self>,
    ) -> Result<PcuTensor<Self>, PcuExecutionError> {
        foreign_i512::<N>(unused)
    }
}

#[pcu(crate_path=::pcu_facade,invocations=N)]
pub fn overwrite<T: PcuScalar, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[0];
}
#[pcu(crate_path=::pcu_facade)]
pub fn identity<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu(crate_path=::pcu_facade)]
pub fn consume<T: PcuScalar>(input: PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
