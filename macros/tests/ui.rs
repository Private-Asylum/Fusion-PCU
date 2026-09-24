#[test]
fn dispatch_macro_ui() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/unsupported_statement.rs");
    tests.compile_fail("tests/ui/unsupported_expression.rs");
    tests.pass("tests/ui/renamed_crate.rs");
}
