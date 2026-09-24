#[test]
fn dispatch_macro_ui() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/unsupported_statement.rs");
    tests.compile_fail("tests/ui/unsupported_expression.rs");
    tests.compile_fail("tests/ui/invocation_overflow.rs");
    tests.compile_fail("tests/ui/invocation_zero.rs");
    #[cfg(target_pointer_width = "64")]
    tests.compile_fail("tests/ui/invocation_too_large.rs");
    tests.pass("tests/ui/renamed_crate.rs");
}
