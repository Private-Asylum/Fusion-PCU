#[test]
fn invocation_clamp_profile_ui() {
    trybuild::TestCases::new().compile_fail("tests/ui/owned_clamp_range_unsupported.rs");
}
