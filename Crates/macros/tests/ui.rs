#[test]
fn dispatch_macro_ui() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/unsupported_statement.rs");
    tests.compile_fail("tests/ui/unsupported_expression.rs");
    tests.pass("tests/ui/grid_stride_multi_iteration.rs");
    tests.pass("tests/ui/grid_stride_symbolic_mismatch.rs");
    tests.pass("tests/ui/qualified_context.rs");
    tests.compile_fail("tests/ui/qualified_context_arguments.rs");
    tests.compile_fail("tests/ui/qualified_invocation_count_arguments.rs");
    tests.compile_fail("tests/ui/grid_stride_mutated_stride.rs");
    tests.compile_fail("tests/ui/grid_stride_control_flow.rs");
    tests.compile_fail("tests/ui/invocation_overflow.rs");
    tests.compile_fail("tests/ui/invocation_zero.rs");
    #[cfg(target_pointer_width = "64")]
    tests.compile_fail("tests/ui/invocation_too_large.rs");
    tests.pass("tests/ui/renamed_crate.rs");
}
