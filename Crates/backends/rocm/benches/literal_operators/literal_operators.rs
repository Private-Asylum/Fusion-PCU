//! Cold source metadata and CPU reference preflight; native peers are a separate gate.
extern crate pcu_facade as fusion_pcu;
#[path = "source/source.rs"]
#[allow(dead_code)] // Fault and ownership companions are authored before native driver integration.
mod source;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuFloatUnderflowPolicy,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuExecutionFaultKind,
};
use fusion_pcu::dialect::tensor::OpDescriptor;
#[rustfmt::skip]
use source::{
    Operation,
    OperatorSource,
};
fn verify<T: OperatorSource>(
    operation: Operation,
    matrix: bool,
    mode: PcuNumericalMode,
    options: PcuNumericalOptions,
    uf: PcuFloatUnderflowPolicy,
) {
    let captured = T::capture(operation, matrix, mode, options, uf);
    assert_eq!(
        captured.input_values().len(),
        usize::from(operation != Operation::Relu)
    );
    let shape = if matrix { vec![2, 3] } else { vec![65] };
    for &value in captured.program().selected_nodes() {
        let node = captured.program().graph().node(value).unwrap();
        assert_eq!(node.scalar_type, T::TYPE);
        assert_eq!(node.shape, shape);
        if matches!(node.op, OpDescriptor::Input) {
            continue;
        }
        assert_eq!(node.numerical_options, options);
        if matches!(
            node.op,
            OpDescriptor::Constant(_) | OpDescriptor::Uniform { .. }
        ) {
            assert_eq!(node.numerical_mode, None);
            assert_eq!(node.float_underflow_policy, None);
        } else {
            assert_eq!(node.float_underflow_policy, Some(uf));
            assert_eq!(
                node.numerical_mode,
                if operation == Operation::Backward {
                    Some(mode)
                } else {
                    None
                }
            );
        }
    }
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        numerical_mode: mode,
        numerical_options: options,
        float_underflow: uf,
        ..Default::default()
    })
    .unwrap();
    let count = if matrix { 6 } else { 65 };
    let input = vec![T::ONE; count];
    let owner = T::execute(operation, matrix, &input).unwrap();
    assert_eq!(owner.shape(), shape);
    let mut observed = vec![T::TWO; count + 2];
    owner.read_into(&mut observed).unwrap();
    for (index, value) in observed[..count].iter().enumerate() {
        let expected = match operation {
            Operation::Sub => [T::ZERO, T::TWO, T::ONE][index % 3],
            Operation::Div => T::HALF,
            Operation::Relu | Operation::Backward => {
                if index % 3 == 0 {
                    T::ONE
                } else {
                    T::ZERO
                }
            }
        };
        assert_eq!(value.encode_le().as_ref(), expected.encode_le().as_ref());
    }
    for tail in &observed[count..] {
        assert_eq!(tail.encode_le().as_ref(), T::TWO.encode_le().as_ref());
    }
    let error = T::execute_case::<1>(operation, matrix, &input).unwrap_err();
    let fault = error
        .arithmetic_fault()
        .expect("exact source arithmetic fault");
    assert_eq!(
        fault.kind,
        if operation == Operation::Div {
            PcuExecutionFaultKind::DivideByZero
        } else {
            PcuExecutionFaultKind::InvalidFloatingOperand
        }
    );
    assert_eq!(fault.invocation_id, u64::from(operation != Operation::Div));
    assert!(!fault.recovered);
    let retry = T::execute(operation, matrix, &input).unwrap();
    let mut retried = vec![T::TWO; count + 2];
    retry.read_into(&mut retried).unwrap();
    for (actual, expected) in retried.iter().zip(&observed) {
        assert_eq!(actual.encode_le().as_ref(), expected.encode_le().as_ref());
    }
    let mut retained = vec![T::TWO; count + 2];
    owner.read_into(&mut retained).unwrap();
    for (actual, expected) in retained.iter().zip(&observed) {
        assert_eq!(actual.encode_le().as_ref(), expected.encode_le().as_ref());
    }
    println!(
        "literal-operator-cpu/{mode:?}/{:?}/{:?}/{uf:?}/{}/{matrix}/{operation:?}: source/cold metadata/shape/bits/tails PASS",
        options.compound_arithmetic,
        options.precision,
        T::LABEL
    );
}
fn main() {
    assert!(
        std::env::var_os("PCU_LITERAL_OPERATOR_CPU_REFERENCE").is_some(),
        "native peer qualification is not authored yet"
    );
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound_arithmetic in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for uf in [
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                ] {
                    let options = PcuNumericalOptions {
                        compound_arithmetic,
                        precision,
                        ..Default::default()
                    };
                    for matrix in [false, true] {
                        for operation in [
                            Operation::Sub,
                            Operation::Div,
                            Operation::Relu,
                            Operation::Backward,
                        ] {
                            macro_rules! width {
                                ($ty:ty) => {
                                    verify::<$ty>(operation, matrix, mode, options, uf);
                                };
                            }
                            width!(f32);
                            width!(f64);
                            width!(PcuF16Bits);
                            width!(PcuBf16Bits);
                            width!(PcuF8E4M3FnBits);
                            width!(PcuF8E5M2Bits);
                        }
                    }
                }
            }
        }
    }
}
