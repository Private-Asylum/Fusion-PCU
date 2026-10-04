//! Literal-fed source functions; arithmetic reference controls live separately.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    global,
    PcuCheckedFloat,
    PcuScalar,
    PcuTensor,
    PcuExecutionError,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuFloatUnderflowPolicy,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    Sub,
    Div,
    Relu,
    Backward,
}
pub trait OperatorSource: PcuCheckedFloat {
    const LABEL: &'static str;
    const ONE: Self;
    const TWO: Self;
    const HALF: Self;
    const ZERO: Self;
    fn execute(
        operation: Operation,
        matrix: bool,
        input: &[Self],
    ) -> Result<PcuTensor<Self>, PcuExecutionError> {
        Self::execute_case::<0>(operation, matrix, input)
    }
    fn capture(
        operation: Operation,
        matrix: bool,
        mode: PcuNumericalMode,
        options: PcuNumericalOptions,
        uf: PcuFloatUnderflowPolicy,
    ) -> global::PcuCapturedTensorProgram {
        Self::capture_case::<0>(operation, matrix, mode, options, uf)
    }
    fn execute_case<const CASE: usize>(
        operation: Operation,
        matrix: bool,
        input: &[Self],
    ) -> Result<PcuTensor<Self>, PcuExecutionError>;
    fn capture_case<const CASE: usize>(
        operation: Operation,
        matrix: bool,
        mode: PcuNumericalMode,
        options: PcuNumericalOptions,
        uf: PcuFloatUnderflowPolicy,
    ) -> global::PcuCapturedTensorProgram;
}

mod f32 {
    #[rustfmt::skip]
    use super::{
        pcu,
        PcuTensor,
        PcuExecutionError,
    };
    const ONE: f32 = 1.0_f32;
    const TWO: f32 = 2.0_f32;
    const PATTERN: [f32; 3] = [ONE, -1.0_f32, -0.0_f32];
    const NAN: f32 = f32::from_bits(0x7fc0_0000);
    const fn payload<const N: usize, const CASE: usize>() -> [f32; N] {
        let mut values = [ONE; N];
        let mut index = 0;
        while index < N {
            values[index] = if CASE == 1 && index == 1 {
                NAN
            } else {
                PATTERN[(index + if CASE == 2 { 1 } else { 0 }) % 3]
            };
            index += 1;
        }
        values
    }
    const fn matrix_payload<const CASE: usize>() -> [[f32; 3]; 2] {
        let values = payload::<6, CASE>();
        [
            [values[0], values[1], values[2]],
            [values[3], values[4], values[5]],
        ]
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn sub<const CASE: usize>(input: &[f32; 65]) -> Result<PcuTensor<f32>, PcuExecutionError> {
        let literal = pcu::constant(const { payload::<65, CASE>() })?;
        pcu::sub(input, &literal)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn div<const CASE: usize>(input: &[f32; 65]) -> Result<PcuTensor<f32>, PcuExecutionError> {
        let uniform = pcu::uniform_like(input, const { if CASE == 1 { PATTERN[2] } else { TWO } })?;
        pcu::div(input, &uniform)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn relu<const CASE: usize>() -> Result<PcuTensor<f32>, PcuExecutionError> {
        let literal = pcu::constant(const { payload::<65, CASE>() })?;
        pcu::relu(&literal)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn backward<const CASE: usize>(
        input: &[f32; 65],
    ) -> Result<PcuTensor<f32>, PcuExecutionError> {
        let activation = pcu::constant(const { payload::<65, CASE>() })?;
        pcu::relu_backward(&activation, input)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn sub_matrix<const CASE: usize>(
        input: &[[f32; 3]; 2],
    ) -> Result<PcuTensor<f32>, PcuExecutionError> {
        let literal = pcu::constant(const { matrix_payload::<CASE>() })?;
        pcu::sub(input, &literal)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn div_matrix<const CASE: usize>(
        input: &[[f32; 3]; 2],
    ) -> Result<PcuTensor<f32>, PcuExecutionError> {
        let uniform = pcu::uniform_like(input, const { if CASE == 1 { PATTERN[2] } else { TWO } })?;
        pcu::div(input, &uniform)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn relu_matrix<const CASE: usize>() -> Result<PcuTensor<f32>, PcuExecutionError> {
        let literal = pcu::constant(const { matrix_payload::<CASE>() })?;
        pcu::relu(&literal)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn backward_matrix<const CASE: usize>(
        input: &[[f32; 3]; 2],
    ) -> Result<PcuTensor<f32>, PcuExecutionError> {
        let activation = pcu::constant(const { matrix_payload::<CASE>() })?;
        pcu::relu_backward(&activation, input)
    }
}
impl OperatorSource for f32 {
    const LABEL: &'static str = "f32";
    const ONE: Self = 1.0_f32;
    const TWO: Self = 2.0_f32;
    const HALF: Self = 0.5_f32;
    const ZERO: Self = 0.0_f32;
    fn execute_case<const CASE: usize>(
        operation: Operation,
        matrix: bool,
        input: &[Self],
    ) -> Result<PcuTensor<Self>, PcuExecutionError> {
        if matrix {
            let data = [
                [input[0], input[1], input[2]],
                [input[3], input[4], input[5]],
            ];
            match operation {
                Operation::Sub => f32::sub_matrix::<CASE>(&data),
                Operation::Div => f32::div_matrix::<CASE>(&data),
                Operation::Relu => f32::relu_matrix::<CASE>(),
                Operation::Backward => f32::backward_matrix::<CASE>(&data),
            }
        } else {
            let data: &[Self; 65] = input.try_into().unwrap();
            match operation {
                Operation::Sub => f32::sub::<CASE>(data),
                Operation::Div => f32::div::<CASE>(data),
                Operation::Relu => f32::relu::<CASE>(),
                Operation::Backward => f32::backward::<CASE>(data),
            }
        }
    }
    fn capture_case<const CASE: usize>(
        operation: Operation,
        matrix: bool,
        mode: PcuNumericalMode,
        options: PcuNumericalOptions,
        uf: PcuFloatUnderflowPolicy,
    ) -> global::PcuCapturedTensorProgram {
        if operation == Operation::Relu {
            global::__pcu_capture_tensor_program::<Self, 0, _>(
                [],
                uf,
                mode,
                options,
                if matrix {
                    f32::relu_matrix::__pcu_capture_entry::<CASE>
                } else {
                    f32::relu::__pcu_capture_entry::<CASE>
                },
            )
            .unwrap()
        } else {
            let builder = match (operation, matrix) {
                (Operation::Sub, false) => f32::sub::__pcu_capture_entry::<CASE>,
                (Operation::Sub, true) => f32::sub_matrix::__pcu_capture_entry::<CASE>,
                (Operation::Div, false) => f32::div::__pcu_capture_entry::<CASE>,
                (Operation::Div, true) => f32::div_matrix::__pcu_capture_entry::<CASE>,
                (Operation::Backward, false) => f32::backward::__pcu_capture_entry::<CASE>,
                (Operation::Backward, true) => f32::backward_matrix::__pcu_capture_entry::<CASE>,
                _ => unreachable!(),
            };
            global::__pcu_capture_tensor_program::<Self, 1, _>(
                [if matrix {
                    global::PcuSourceShape::FixedMatrix {
                        rows: 2,
                        columns: 3,
                    }
                } else {
                    global::PcuSourceShape::FixedArray { length: 65 }
                }],
                uf,
                mode,
                options,
                builder,
            )
            .unwrap()
        }
    }
}

mod f64 {
    #[rustfmt::skip]
    use super::{
        pcu,
        PcuTensor,
        PcuExecutionError,
    };
    const ONE: f64 = 1.0_f64;
    const TWO: f64 = 2.0_f64;
    const PATTERN: [f64; 3] = [ONE, -1.0_f64, -0.0_f64];
    const NAN: f64 = f64::from_bits(0x7ff8_0000_0000_0000);
    const fn payload<const N: usize, const CASE: usize>() -> [f64; N] {
        let mut values = [ONE; N];
        let mut index = 0;
        while index < N {
            values[index] = if CASE == 1 && index == 1 {
                NAN
            } else {
                PATTERN[(index + if CASE == 2 { 1 } else { 0 }) % 3]
            };
            index += 1;
        }
        values
    }
    const fn matrix_payload<const CASE: usize>() -> [[f64; 3]; 2] {
        let values = payload::<6, CASE>();
        [
            [values[0], values[1], values[2]],
            [values[3], values[4], values[5]],
        ]
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn sub<const CASE: usize>(input: &[f64; 65]) -> Result<PcuTensor<f64>, PcuExecutionError> {
        let literal = pcu::constant(const { payload::<65, CASE>() })?;
        pcu::sub(input, &literal)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn div<const CASE: usize>(input: &[f64; 65]) -> Result<PcuTensor<f64>, PcuExecutionError> {
        let uniform = pcu::uniform_like(input, const { if CASE == 1 { PATTERN[2] } else { TWO } })?;
        pcu::div(input, &uniform)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn relu<const CASE: usize>() -> Result<PcuTensor<f64>, PcuExecutionError> {
        let literal = pcu::constant(const { payload::<65, CASE>() })?;
        pcu::relu(&literal)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn backward<const CASE: usize>(
        input: &[f64; 65],
    ) -> Result<PcuTensor<f64>, PcuExecutionError> {
        let activation = pcu::constant(const { payload::<65, CASE>() })?;
        pcu::relu_backward(&activation, input)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn sub_matrix<const CASE: usize>(
        input: &[[f64; 3]; 2],
    ) -> Result<PcuTensor<f64>, PcuExecutionError> {
        let literal = pcu::constant(const { matrix_payload::<CASE>() })?;
        pcu::sub(input, &literal)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn div_matrix<const CASE: usize>(
        input: &[[f64; 3]; 2],
    ) -> Result<PcuTensor<f64>, PcuExecutionError> {
        let uniform = pcu::uniform_like(input, const { if CASE == 1 { PATTERN[2] } else { TWO } })?;
        pcu::div(input, &uniform)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn relu_matrix<const CASE: usize>() -> Result<PcuTensor<f64>, PcuExecutionError> {
        let literal = pcu::constant(const { matrix_payload::<CASE>() })?;
        pcu::relu(&literal)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn backward_matrix<const CASE: usize>(
        input: &[[f64; 3]; 2],
    ) -> Result<PcuTensor<f64>, PcuExecutionError> {
        let activation = pcu::constant(const { matrix_payload::<CASE>() })?;
        pcu::relu_backward(&activation, input)
    }
}
impl OperatorSource for f64 {
    const LABEL: &'static str = "f64";
    const ONE: Self = 1.0_f64;
    const TWO: Self = 2.0_f64;
    const HALF: Self = 0.5_f64;
    const ZERO: Self = 0.0_f64;
    fn execute_case<const CASE: usize>(
        operation: Operation,
        matrix: bool,
        input: &[Self],
    ) -> Result<PcuTensor<Self>, PcuExecutionError> {
        if matrix {
            let data = [
                [input[0], input[1], input[2]],
                [input[3], input[4], input[5]],
            ];
            match operation {
                Operation::Sub => f64::sub_matrix::<CASE>(&data),
                Operation::Div => f64::div_matrix::<CASE>(&data),
                Operation::Relu => f64::relu_matrix::<CASE>(),
                Operation::Backward => f64::backward_matrix::<CASE>(&data),
            }
        } else {
            let data: &[Self; 65] = input.try_into().unwrap();
            match operation {
                Operation::Sub => f64::sub::<CASE>(data),
                Operation::Div => f64::div::<CASE>(data),
                Operation::Relu => f64::relu::<CASE>(),
                Operation::Backward => f64::backward::<CASE>(data),
            }
        }
    }
    fn capture_case<const CASE: usize>(
        operation: Operation,
        matrix: bool,
        mode: PcuNumericalMode,
        options: PcuNumericalOptions,
        uf: PcuFloatUnderflowPolicy,
    ) -> global::PcuCapturedTensorProgram {
        if operation == Operation::Relu {
            global::__pcu_capture_tensor_program::<Self, 0, _>(
                [],
                uf,
                mode,
                options,
                if matrix {
                    f64::relu_matrix::__pcu_capture_entry::<CASE>
                } else {
                    f64::relu::__pcu_capture_entry::<CASE>
                },
            )
            .unwrap()
        } else {
            let builder = match (operation, matrix) {
                (Operation::Sub, false) => f64::sub::__pcu_capture_entry::<CASE>,
                (Operation::Sub, true) => f64::sub_matrix::__pcu_capture_entry::<CASE>,
                (Operation::Div, false) => f64::div::__pcu_capture_entry::<CASE>,
                (Operation::Div, true) => f64::div_matrix::__pcu_capture_entry::<CASE>,
                (Operation::Backward, false) => f64::backward::__pcu_capture_entry::<CASE>,
                (Operation::Backward, true) => f64::backward_matrix::__pcu_capture_entry::<CASE>,
                _ => unreachable!(),
            };
            global::__pcu_capture_tensor_program::<Self, 1, _>(
                [if matrix {
                    global::PcuSourceShape::FixedMatrix {
                        rows: 2,
                        columns: 3,
                    }
                } else {
                    global::PcuSourceShape::FixedArray { length: 65 }
                }],
                uf,
                mode,
                options,
                builder,
            )
            .unwrap()
        }
    }
}

mod f16 {
    #[rustfmt::skip]
    use super::{
        pcu,
        PcuTensor,
        PcuExecutionError,
        PcuF16Bits,
    };
    const ONE: PcuF16Bits = PcuF16Bits::from_bits(0x3c00);
    const TWO: PcuF16Bits = PcuF16Bits::from_bits(0x4000);
    const PATTERN: [PcuF16Bits; 3] = [
        ONE,
        PcuF16Bits::from_bits(0xbc00),
        PcuF16Bits::from_bits(0x8000),
    ];
    const NAN: PcuF16Bits = PcuF16Bits::from_bits(0x7e00);
    const fn payload<const N: usize, const CASE: usize>() -> [PcuF16Bits; N] {
        let mut values = [ONE; N];
        let mut index = 0;
        while index < N {
            values[index] = if CASE == 1 && index == 1 {
                NAN
            } else {
                PATTERN[(index + if CASE == 2 { 1 } else { 0 }) % 3]
            };
            index += 1;
        }
        values
    }
    const fn matrix_payload<const CASE: usize>() -> [[PcuF16Bits; 3]; 2] {
        let values = payload::<6, CASE>();
        [
            [values[0], values[1], values[2]],
            [values[3], values[4], values[5]],
        ]
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn sub<const CASE: usize>(
        input: &[PcuF16Bits; 65],
    ) -> Result<PcuTensor<PcuF16Bits>, PcuExecutionError> {
        let literal = pcu::constant(const { payload::<65, CASE>() })?;
        pcu::sub(input, &literal)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn div<const CASE: usize>(
        input: &[PcuF16Bits; 65],
    ) -> Result<PcuTensor<PcuF16Bits>, PcuExecutionError> {
        let uniform = pcu::uniform_like(input, const { if CASE == 1 { PATTERN[2] } else { TWO } })?;
        pcu::div(input, &uniform)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn relu<const CASE: usize>() -> Result<PcuTensor<PcuF16Bits>, PcuExecutionError> {
        let literal = pcu::constant(const { payload::<65, CASE>() })?;
        pcu::relu(&literal)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn backward<const CASE: usize>(
        input: &[PcuF16Bits; 65],
    ) -> Result<PcuTensor<PcuF16Bits>, PcuExecutionError> {
        let activation = pcu::constant(const { payload::<65, CASE>() })?;
        pcu::relu_backward(&activation, input)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn sub_matrix<const CASE: usize>(
        input: &[[PcuF16Bits; 3]; 2],
    ) -> Result<PcuTensor<PcuF16Bits>, PcuExecutionError> {
        let literal = pcu::constant(const { matrix_payload::<CASE>() })?;
        pcu::sub(input, &literal)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn div_matrix<const CASE: usize>(
        input: &[[PcuF16Bits; 3]; 2],
    ) -> Result<PcuTensor<PcuF16Bits>, PcuExecutionError> {
        let uniform = pcu::uniform_like(input, const { if CASE == 1 { PATTERN[2] } else { TWO } })?;
        pcu::div(input, &uniform)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn relu_matrix<const CASE: usize>() -> Result<PcuTensor<PcuF16Bits>, PcuExecutionError> {
        let literal = pcu::constant(const { matrix_payload::<CASE>() })?;
        pcu::relu(&literal)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn backward_matrix<const CASE: usize>(
        input: &[[PcuF16Bits; 3]; 2],
    ) -> Result<PcuTensor<PcuF16Bits>, PcuExecutionError> {
        let activation = pcu::constant(const { matrix_payload::<CASE>() })?;
        pcu::relu_backward(&activation, input)
    }
}
impl OperatorSource for PcuF16Bits {
    const LABEL: &'static str = "f16";
    const ONE: Self = Self::from_bits(0x3c00);
    const TWO: Self = Self::from_bits(0x4000);
    const HALF: Self = Self::from_bits(0x3800);
    const ZERO: Self = Self::from_bits(0);
    fn execute_case<const CASE: usize>(
        operation: Operation,
        matrix: bool,
        input: &[Self],
    ) -> Result<PcuTensor<Self>, PcuExecutionError> {
        if matrix {
            let data = [
                [input[0], input[1], input[2]],
                [input[3], input[4], input[5]],
            ];
            match operation {
                Operation::Sub => f16::sub_matrix::<CASE>(&data),
                Operation::Div => f16::div_matrix::<CASE>(&data),
                Operation::Relu => f16::relu_matrix::<CASE>(),
                Operation::Backward => f16::backward_matrix::<CASE>(&data),
            }
        } else {
            let data: &[Self; 65] = input.try_into().unwrap();
            match operation {
                Operation::Sub => f16::sub::<CASE>(data),
                Operation::Div => f16::div::<CASE>(data),
                Operation::Relu => f16::relu::<CASE>(),
                Operation::Backward => f16::backward::<CASE>(data),
            }
        }
    }
    fn capture_case<const CASE: usize>(
        operation: Operation,
        matrix: bool,
        mode: PcuNumericalMode,
        options: PcuNumericalOptions,
        uf: PcuFloatUnderflowPolicy,
    ) -> global::PcuCapturedTensorProgram {
        if operation == Operation::Relu {
            global::__pcu_capture_tensor_program::<Self, 0, _>(
                [],
                uf,
                mode,
                options,
                if matrix {
                    f16::relu_matrix::__pcu_capture_entry::<CASE>
                } else {
                    f16::relu::__pcu_capture_entry::<CASE>
                },
            )
            .unwrap()
        } else {
            let builder = match (operation, matrix) {
                (Operation::Sub, false) => f16::sub::__pcu_capture_entry::<CASE>,
                (Operation::Sub, true) => f16::sub_matrix::__pcu_capture_entry::<CASE>,
                (Operation::Div, false) => f16::div::__pcu_capture_entry::<CASE>,
                (Operation::Div, true) => f16::div_matrix::__pcu_capture_entry::<CASE>,
                (Operation::Backward, false) => f16::backward::__pcu_capture_entry::<CASE>,
                (Operation::Backward, true) => f16::backward_matrix::__pcu_capture_entry::<CASE>,
                _ => unreachable!(),
            };
            global::__pcu_capture_tensor_program::<Self, 1, _>(
                [if matrix {
                    global::PcuSourceShape::FixedMatrix {
                        rows: 2,
                        columns: 3,
                    }
                } else {
                    global::PcuSourceShape::FixedArray { length: 65 }
                }],
                uf,
                mode,
                options,
                builder,
            )
            .unwrap()
        }
    }
}

mod bf16 {
    #[rustfmt::skip]
    use super::{
        pcu,
        PcuTensor,
        PcuExecutionError,
        PcuBf16Bits,
    };
    const ONE: PcuBf16Bits = PcuBf16Bits::from_bits(0x3f80);
    const TWO: PcuBf16Bits = PcuBf16Bits::from_bits(0x4000);
    const PATTERN: [PcuBf16Bits; 3] = [
        ONE,
        PcuBf16Bits::from_bits(0xbf80),
        PcuBf16Bits::from_bits(0x8000),
    ];
    const NAN: PcuBf16Bits = PcuBf16Bits::from_bits(0x7fc0);
    const fn payload<const N: usize, const CASE: usize>() -> [PcuBf16Bits; N] {
        let mut values = [ONE; N];
        let mut index = 0;
        while index < N {
            values[index] = if CASE == 1 && index == 1 {
                NAN
            } else {
                PATTERN[(index + if CASE == 2 { 1 } else { 0 }) % 3]
            };
            index += 1;
        }
        values
    }
    const fn matrix_payload<const CASE: usize>() -> [[PcuBf16Bits; 3]; 2] {
        let values = payload::<6, CASE>();
        [
            [values[0], values[1], values[2]],
            [values[3], values[4], values[5]],
        ]
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn sub<const CASE: usize>(
        input: &[PcuBf16Bits; 65],
    ) -> Result<PcuTensor<PcuBf16Bits>, PcuExecutionError> {
        let literal = pcu::constant(const { payload::<65, CASE>() })?;
        pcu::sub(input, &literal)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn div<const CASE: usize>(
        input: &[PcuBf16Bits; 65],
    ) -> Result<PcuTensor<PcuBf16Bits>, PcuExecutionError> {
        let uniform = pcu::uniform_like(input, const { if CASE == 1 { PATTERN[2] } else { TWO } })?;
        pcu::div(input, &uniform)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn relu<const CASE: usize>() -> Result<PcuTensor<PcuBf16Bits>, PcuExecutionError> {
        let literal = pcu::constant(const { payload::<65, CASE>() })?;
        pcu::relu(&literal)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn backward<const CASE: usize>(
        input: &[PcuBf16Bits; 65],
    ) -> Result<PcuTensor<PcuBf16Bits>, PcuExecutionError> {
        let activation = pcu::constant(const { payload::<65, CASE>() })?;
        pcu::relu_backward(&activation, input)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn sub_matrix<const CASE: usize>(
        input: &[[PcuBf16Bits; 3]; 2],
    ) -> Result<PcuTensor<PcuBf16Bits>, PcuExecutionError> {
        let literal = pcu::constant(const { matrix_payload::<CASE>() })?;
        pcu::sub(input, &literal)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn div_matrix<const CASE: usize>(
        input: &[[PcuBf16Bits; 3]; 2],
    ) -> Result<PcuTensor<PcuBf16Bits>, PcuExecutionError> {
        let uniform = pcu::uniform_like(input, const { if CASE == 1 { PATTERN[2] } else { TWO } })?;
        pcu::div(input, &uniform)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn relu_matrix<const CASE: usize>() -> Result<PcuTensor<PcuBf16Bits>, PcuExecutionError> {
        let literal = pcu::constant(const { matrix_payload::<CASE>() })?;
        pcu::relu(&literal)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn backward_matrix<const CASE: usize>(
        input: &[[PcuBf16Bits; 3]; 2],
    ) -> Result<PcuTensor<PcuBf16Bits>, PcuExecutionError> {
        let activation = pcu::constant(const { matrix_payload::<CASE>() })?;
        pcu::relu_backward(&activation, input)
    }
}
impl OperatorSource for PcuBf16Bits {
    const LABEL: &'static str = "bf16";
    const ONE: Self = Self::from_bits(0x3f80);
    const TWO: Self = Self::from_bits(0x4000);
    const HALF: Self = Self::from_bits(0x3f00);
    const ZERO: Self = Self::from_bits(0);
    fn execute_case<const CASE: usize>(
        operation: Operation,
        matrix: bool,
        input: &[Self],
    ) -> Result<PcuTensor<Self>, PcuExecutionError> {
        if matrix {
            let data = [
                [input[0], input[1], input[2]],
                [input[3], input[4], input[5]],
            ];
            match operation {
                Operation::Sub => bf16::sub_matrix::<CASE>(&data),
                Operation::Div => bf16::div_matrix::<CASE>(&data),
                Operation::Relu => bf16::relu_matrix::<CASE>(),
                Operation::Backward => bf16::backward_matrix::<CASE>(&data),
            }
        } else {
            let data: &[Self; 65] = input.try_into().unwrap();
            match operation {
                Operation::Sub => bf16::sub::<CASE>(data),
                Operation::Div => bf16::div::<CASE>(data),
                Operation::Relu => bf16::relu::<CASE>(),
                Operation::Backward => bf16::backward::<CASE>(data),
            }
        }
    }
    fn capture_case<const CASE: usize>(
        operation: Operation,
        matrix: bool,
        mode: PcuNumericalMode,
        options: PcuNumericalOptions,
        uf: PcuFloatUnderflowPolicy,
    ) -> global::PcuCapturedTensorProgram {
        if operation == Operation::Relu {
            global::__pcu_capture_tensor_program::<Self, 0, _>(
                [],
                uf,
                mode,
                options,
                if matrix {
                    bf16::relu_matrix::__pcu_capture_entry::<CASE>
                } else {
                    bf16::relu::__pcu_capture_entry::<CASE>
                },
            )
            .unwrap()
        } else {
            let builder = match (operation, matrix) {
                (Operation::Sub, false) => bf16::sub::__pcu_capture_entry::<CASE>,
                (Operation::Sub, true) => bf16::sub_matrix::__pcu_capture_entry::<CASE>,
                (Operation::Div, false) => bf16::div::__pcu_capture_entry::<CASE>,
                (Operation::Div, true) => bf16::div_matrix::__pcu_capture_entry::<CASE>,
                (Operation::Backward, false) => bf16::backward::__pcu_capture_entry::<CASE>,
                (Operation::Backward, true) => bf16::backward_matrix::__pcu_capture_entry::<CASE>,
                _ => unreachable!(),
            };
            global::__pcu_capture_tensor_program::<Self, 1, _>(
                [if matrix {
                    global::PcuSourceShape::FixedMatrix {
                        rows: 2,
                        columns: 3,
                    }
                } else {
                    global::PcuSourceShape::FixedArray { length: 65 }
                }],
                uf,
                mode,
                options,
                builder,
            )
            .unwrap()
        }
    }
}

mod e4m3fn {
    #[rustfmt::skip]
    use super::{
        pcu,
        PcuTensor,
        PcuExecutionError,
        PcuF8E4M3FnBits,
    };
    const ONE: PcuF8E4M3FnBits = PcuF8E4M3FnBits::from_bits(0x38);
    const TWO: PcuF8E4M3FnBits = PcuF8E4M3FnBits::from_bits(0x40);
    const PATTERN: [PcuF8E4M3FnBits; 3] = [
        ONE,
        PcuF8E4M3FnBits::from_bits(0xb8),
        PcuF8E4M3FnBits::from_bits(0x80),
    ];
    const NAN: PcuF8E4M3FnBits = PcuF8E4M3FnBits::from_bits(0x7f);
    const fn payload<const N: usize, const CASE: usize>() -> [PcuF8E4M3FnBits; N] {
        let mut values = [ONE; N];
        let mut index = 0;
        while index < N {
            values[index] = if CASE == 1 && index == 1 {
                NAN
            } else {
                PATTERN[(index + if CASE == 2 { 1 } else { 0 }) % 3]
            };
            index += 1;
        }
        values
    }
    const fn matrix_payload<const CASE: usize>() -> [[PcuF8E4M3FnBits; 3]; 2] {
        let values = payload::<6, CASE>();
        [
            [values[0], values[1], values[2]],
            [values[3], values[4], values[5]],
        ]
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn sub<const CASE: usize>(
        input: &[PcuF8E4M3FnBits; 65],
    ) -> Result<PcuTensor<PcuF8E4M3FnBits>, PcuExecutionError> {
        let literal = pcu::constant(const { payload::<65, CASE>() })?;
        pcu::sub(input, &literal)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn div<const CASE: usize>(
        input: &[PcuF8E4M3FnBits; 65],
    ) -> Result<PcuTensor<PcuF8E4M3FnBits>, PcuExecutionError> {
        let uniform = pcu::uniform_like(input, const { if CASE == 1 { PATTERN[2] } else { TWO } })?;
        pcu::div(input, &uniform)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn relu<const CASE: usize>() -> Result<PcuTensor<PcuF8E4M3FnBits>, PcuExecutionError> {
        let literal = pcu::constant(const { payload::<65, CASE>() })?;
        pcu::relu(&literal)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn backward<const CASE: usize>(
        input: &[PcuF8E4M3FnBits; 65],
    ) -> Result<PcuTensor<PcuF8E4M3FnBits>, PcuExecutionError> {
        let activation = pcu::constant(const { payload::<65, CASE>() })?;
        pcu::relu_backward(&activation, input)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn sub_matrix<const CASE: usize>(
        input: &[[PcuF8E4M3FnBits; 3]; 2],
    ) -> Result<PcuTensor<PcuF8E4M3FnBits>, PcuExecutionError> {
        let literal = pcu::constant(const { matrix_payload::<CASE>() })?;
        pcu::sub(input, &literal)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn div_matrix<const CASE: usize>(
        input: &[[PcuF8E4M3FnBits; 3]; 2],
    ) -> Result<PcuTensor<PcuF8E4M3FnBits>, PcuExecutionError> {
        let uniform = pcu::uniform_like(input, const { if CASE == 1 { PATTERN[2] } else { TWO } })?;
        pcu::div(input, &uniform)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn relu_matrix<const CASE: usize>() -> Result<PcuTensor<PcuF8E4M3FnBits>, PcuExecutionError>
    {
        let literal = pcu::constant(const { matrix_payload::<CASE>() })?;
        pcu::relu(&literal)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn backward_matrix<const CASE: usize>(
        input: &[[PcuF8E4M3FnBits; 3]; 2],
    ) -> Result<PcuTensor<PcuF8E4M3FnBits>, PcuExecutionError> {
        let activation = pcu::constant(const { matrix_payload::<CASE>() })?;
        pcu::relu_backward(&activation, input)
    }
}
impl OperatorSource for PcuF8E4M3FnBits {
    const LABEL: &'static str = "e4m3fn";
    const ONE: Self = Self::from_bits(0x38);
    const TWO: Self = Self::from_bits(0x40);
    const HALF: Self = Self::from_bits(0x30);
    const ZERO: Self = Self::from_bits(0);
    fn execute_case<const CASE: usize>(
        operation: Operation,
        matrix: bool,
        input: &[Self],
    ) -> Result<PcuTensor<Self>, PcuExecutionError> {
        if matrix {
            let data = [
                [input[0], input[1], input[2]],
                [input[3], input[4], input[5]],
            ];
            match operation {
                Operation::Sub => e4m3fn::sub_matrix::<CASE>(&data),
                Operation::Div => e4m3fn::div_matrix::<CASE>(&data),
                Operation::Relu => e4m3fn::relu_matrix::<CASE>(),
                Operation::Backward => e4m3fn::backward_matrix::<CASE>(&data),
            }
        } else {
            let data: &[Self; 65] = input.try_into().unwrap();
            match operation {
                Operation::Sub => e4m3fn::sub::<CASE>(data),
                Operation::Div => e4m3fn::div::<CASE>(data),
                Operation::Relu => e4m3fn::relu::<CASE>(),
                Operation::Backward => e4m3fn::backward::<CASE>(data),
            }
        }
    }
    fn capture_case<const CASE: usize>(
        operation: Operation,
        matrix: bool,
        mode: PcuNumericalMode,
        options: PcuNumericalOptions,
        uf: PcuFloatUnderflowPolicy,
    ) -> global::PcuCapturedTensorProgram {
        if operation == Operation::Relu {
            global::__pcu_capture_tensor_program::<Self, 0, _>(
                [],
                uf,
                mode,
                options,
                if matrix {
                    e4m3fn::relu_matrix::__pcu_capture_entry::<CASE>
                } else {
                    e4m3fn::relu::__pcu_capture_entry::<CASE>
                },
            )
            .unwrap()
        } else {
            let builder = match (operation, matrix) {
                (Operation::Sub, false) => e4m3fn::sub::__pcu_capture_entry::<CASE>,
                (Operation::Sub, true) => e4m3fn::sub_matrix::__pcu_capture_entry::<CASE>,
                (Operation::Div, false) => e4m3fn::div::__pcu_capture_entry::<CASE>,
                (Operation::Div, true) => e4m3fn::div_matrix::__pcu_capture_entry::<CASE>,
                (Operation::Backward, false) => e4m3fn::backward::__pcu_capture_entry::<CASE>,
                (Operation::Backward, true) => e4m3fn::backward_matrix::__pcu_capture_entry::<CASE>,
                _ => unreachable!(),
            };
            global::__pcu_capture_tensor_program::<Self, 1, _>(
                [if matrix {
                    global::PcuSourceShape::FixedMatrix {
                        rows: 2,
                        columns: 3,
                    }
                } else {
                    global::PcuSourceShape::FixedArray { length: 65 }
                }],
                uf,
                mode,
                options,
                builder,
            )
            .unwrap()
        }
    }
}

mod e5m2 {
    #[rustfmt::skip]
    use super::{
        pcu,
        PcuTensor,
        PcuExecutionError,
        PcuF8E5M2Bits,
    };
    const ONE: PcuF8E5M2Bits = PcuF8E5M2Bits::from_bits(0x3c);
    const TWO: PcuF8E5M2Bits = PcuF8E5M2Bits::from_bits(0x40);
    const PATTERN: [PcuF8E5M2Bits; 3] = [
        ONE,
        PcuF8E5M2Bits::from_bits(0xbc),
        PcuF8E5M2Bits::from_bits(0x80),
    ];
    const NAN: PcuF8E5M2Bits = PcuF8E5M2Bits::from_bits(0x7e);
    const fn payload<const N: usize, const CASE: usize>() -> [PcuF8E5M2Bits; N] {
        let mut values = [ONE; N];
        let mut index = 0;
        while index < N {
            values[index] = if CASE == 1 && index == 1 {
                NAN
            } else {
                PATTERN[(index + if CASE == 2 { 1 } else { 0 }) % 3]
            };
            index += 1;
        }
        values
    }
    const fn matrix_payload<const CASE: usize>() -> [[PcuF8E5M2Bits; 3]; 2] {
        let values = payload::<6, CASE>();
        [
            [values[0], values[1], values[2]],
            [values[3], values[4], values[5]],
        ]
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn sub<const CASE: usize>(
        input: &[PcuF8E5M2Bits; 65],
    ) -> Result<PcuTensor<PcuF8E5M2Bits>, PcuExecutionError> {
        let literal = pcu::constant(const { payload::<65, CASE>() })?;
        pcu::sub(input, &literal)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn div<const CASE: usize>(
        input: &[PcuF8E5M2Bits; 65],
    ) -> Result<PcuTensor<PcuF8E5M2Bits>, PcuExecutionError> {
        let uniform = pcu::uniform_like(input, const { if CASE == 1 { PATTERN[2] } else { TWO } })?;
        pcu::div(input, &uniform)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn relu<const CASE: usize>() -> Result<PcuTensor<PcuF8E5M2Bits>, PcuExecutionError> {
        let literal = pcu::constant(const { payload::<65, CASE>() })?;
        pcu::relu(&literal)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn backward<const CASE: usize>(
        input: &[PcuF8E5M2Bits; 65],
    ) -> Result<PcuTensor<PcuF8E5M2Bits>, PcuExecutionError> {
        let activation = pcu::constant(const { payload::<65, CASE>() })?;
        pcu::relu_backward(&activation, input)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn sub_matrix<const CASE: usize>(
        input: &[[PcuF8E5M2Bits; 3]; 2],
    ) -> Result<PcuTensor<PcuF8E5M2Bits>, PcuExecutionError> {
        let literal = pcu::constant(const { matrix_payload::<CASE>() })?;
        pcu::sub(input, &literal)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn div_matrix<const CASE: usize>(
        input: &[[PcuF8E5M2Bits; 3]; 2],
    ) -> Result<PcuTensor<PcuF8E5M2Bits>, PcuExecutionError> {
        let uniform = pcu::uniform_like(input, const { if CASE == 1 { PATTERN[2] } else { TWO } })?;
        pcu::div(input, &uniform)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn relu_matrix<const CASE: usize>() -> Result<PcuTensor<PcuF8E5M2Bits>, PcuExecutionError> {
        let literal = pcu::constant(const { matrix_payload::<CASE>() })?;
        pcu::relu(&literal)
    }
    #[pcu(crate_path=::pcu_facade)]
    pub fn backward_matrix<const CASE: usize>(
        input: &[[PcuF8E5M2Bits; 3]; 2],
    ) -> Result<PcuTensor<PcuF8E5M2Bits>, PcuExecutionError> {
        let activation = pcu::constant(const { matrix_payload::<CASE>() })?;
        pcu::relu_backward(&activation, input)
    }
}
impl OperatorSource for PcuF8E5M2Bits {
    const LABEL: &'static str = "e5m2";
    const ONE: Self = Self::from_bits(0x3c);
    const TWO: Self = Self::from_bits(0x40);
    const HALF: Self = Self::from_bits(0x38);
    const ZERO: Self = Self::from_bits(0);
    fn execute_case<const CASE: usize>(
        operation: Operation,
        matrix: bool,
        input: &[Self],
    ) -> Result<PcuTensor<Self>, PcuExecutionError> {
        if matrix {
            let data = [
                [input[0], input[1], input[2]],
                [input[3], input[4], input[5]],
            ];
            match operation {
                Operation::Sub => e5m2::sub_matrix::<CASE>(&data),
                Operation::Div => e5m2::div_matrix::<CASE>(&data),
                Operation::Relu => e5m2::relu_matrix::<CASE>(),
                Operation::Backward => e5m2::backward_matrix::<CASE>(&data),
            }
        } else {
            let data: &[Self; 65] = input.try_into().unwrap();
            match operation {
                Operation::Sub => e5m2::sub::<CASE>(data),
                Operation::Div => e5m2::div::<CASE>(data),
                Operation::Relu => e5m2::relu::<CASE>(),
                Operation::Backward => e5m2::backward::<CASE>(data),
            }
        }
    }
    fn capture_case<const CASE: usize>(
        operation: Operation,
        matrix: bool,
        mode: PcuNumericalMode,
        options: PcuNumericalOptions,
        uf: PcuFloatUnderflowPolicy,
    ) -> global::PcuCapturedTensorProgram {
        if operation == Operation::Relu {
            global::__pcu_capture_tensor_program::<Self, 0, _>(
                [],
                uf,
                mode,
                options,
                if matrix {
                    e5m2::relu_matrix::__pcu_capture_entry::<CASE>
                } else {
                    e5m2::relu::__pcu_capture_entry::<CASE>
                },
            )
            .unwrap()
        } else {
            let builder = match (operation, matrix) {
                (Operation::Sub, false) => e5m2::sub::__pcu_capture_entry::<CASE>,
                (Operation::Sub, true) => e5m2::sub_matrix::__pcu_capture_entry::<CASE>,
                (Operation::Div, false) => e5m2::div::__pcu_capture_entry::<CASE>,
                (Operation::Div, true) => e5m2::div_matrix::__pcu_capture_entry::<CASE>,
                (Operation::Backward, false) => e5m2::backward::__pcu_capture_entry::<CASE>,
                (Operation::Backward, true) => e5m2::backward_matrix::__pcu_capture_entry::<CASE>,
                _ => unreachable!(),
            };
            global::__pcu_capture_tensor_program::<Self, 1, _>(
                [if matrix {
                    global::PcuSourceShape::FixedMatrix {
                        rows: 2,
                        columns: 3,
                    }
                } else {
                    global::PcuSourceShape::FixedArray { length: 65 }
                }],
                uf,
                mode,
                options,
                builder,
            )
            .unwrap()
        }
    }
}
#[pcu(crate_path=::pcu_facade)]
pub fn consume<T: PcuScalar>(input: PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
