#[test]
fn immutable_shadowing_preserves_owner_and_view_provenance() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/owned_shadow_double_move.rs");
    tests.compile_fail("tests/ui/owned_shadow_old_view_after_move.rs");
    tests.compile_fail("tests/ui/owned_shadow_return_view.rs");
}
