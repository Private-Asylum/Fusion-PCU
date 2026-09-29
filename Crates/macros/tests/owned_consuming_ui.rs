#[test]
fn consuming_helper_modes_typecheck() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/owned_helper_bare_to_borrow.rs");
    tests.compile_fail("tests/ui/owned_helper_borrow_to_consume.rs");
    tests.pass("tests/ui/owned_helper_graph_temp_to_consume.rs");
    tests.compile_fail("tests/ui/owned_helper_temp_bare_fanout.rs");
    tests.compile_fail("tests/ui/owned_helper_temp_borrow_after_move.rs");
    tests.compile_fail("tests/ui/owned_helper_consume_borrowed_rvalue.rs");
    tests.compile_fail("tests/ui/owned_helper_return_borrowed_rvalue.rs");
    tests.compile_fail("tests/ui/owned_helper_return_borrowed_input_rvalue.rs");
    tests.compile_fail("tests/ui/owned_helper_return_borrowed_slice.rs");
    tests.compile_fail("tests/ui/owned_helper_return_borrowed_resident.rs");
    tests.compile_fail("tests/ui/owned_helper_return_borrowed_resident_alias.rs");
    tests.compile_fail("tests/ui/owned_helper_use_after_move.rs");
    tests.compile_fail("tests/ui/owned_helper_double_move_alias.rs");
    tests.compile_fail("tests/ui/owned_helper_borrow_view_after_move.rs");
    tests.compile_fail("tests/ui/owned_helper_return_borrow_alias.rs");
    tests.compile_fail("tests/ui/owned_multi_second_input_moved_twice.rs");
}
