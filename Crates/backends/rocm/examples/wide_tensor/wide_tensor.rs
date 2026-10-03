//! Fresh checked signed512 tensor results preserve inputs and prior escaped owners on failure.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/strict_matmul/selection.rs"]
mod selection;
#[path = "../../benches/wide_tensor/source/source.rs"]
#[allow(dead_code)] // Other source functions belong to the paired benchmark and integration proof.
mod source;
use fusion_pcu::{PcuI512, PcuExecutionError, PcuExecutionFaultKind};
fn main() {
    let _device = selection::selected_device();
    let input = [PcuI512::MIN; 65];
    let one = PcuI512::from_limbs_le([1, 0, 0, 0, 0, 0, 0, 0]);
    let a = source::identity(input.as_slice()).unwrap();
    let b = source::identity(&[one; 65]).unwrap();
    let prior = source::mul(&a, &b).unwrap();
    let minus = PcuI512::from_limbs_le([u64::MAX; 8]);
    let result = source::mul(&a, &[minus; 65]);
    assert!(
        matches!(result,Err(PcuExecutionError::ArithmeticFault(f)) if f.invocation_id==0&&f.kind==PcuExecutionFaultKind::ArithmeticOverflow&&!f.recovered)
    );
    for owner in [&a, &prior] {
        let mut out = [PcuI512::ZERO; 65];
        owner.read_into(&mut out).unwrap();
        assert_eq!(out, input);
    }
    let retry = source::mul(&a, &b).unwrap();
    let mut out = [PcuI512::ZERO; 65];
    retry.read_into(&mut out).unwrap();
    assert_eq!(out, input);
    println!(
        "512-bit owned minimum multiply exact; fatal private output preserved retained siblings and fresh retry"
    );
}
