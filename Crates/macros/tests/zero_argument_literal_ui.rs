#[test]
fn zero_argument_producers_cannot_capture_runtime_payloads_or_unannotated_host_helpers() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/zero_argument_runtime_payload.rs");
    cases.compile_fail("tests/ui/zero_argument_unannotated_helper.rs");
    cases.compile_fail("tests/ui/zero_argument_integer_underflow_flag.rs");
}
