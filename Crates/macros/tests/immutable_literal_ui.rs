#[test]
fn immutable_payloads_require_compile_time_typed_arrays_or_scalars() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/immutable_literal_runtime.rs");
    cases.compile_fail("tests/ui/immutable_literal_runtime_in_const.rs");
    cases.compile_fail("tests/ui/immutable_literal_scalar_array.rs");
}
