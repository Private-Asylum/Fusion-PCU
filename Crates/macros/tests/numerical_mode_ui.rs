#[test]
fn numerical_mode_flag_ui() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/numerical_mode_duplicate.rs");
    cases.compile_fail("tests/ui/numerical_mode_conflict.rs");
    cases.pass("tests/ui/numerical_mode_scalar.rs");
}
