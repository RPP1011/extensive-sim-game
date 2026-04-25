//! Compile-fail test: external types must NOT be able to `impl CascadeHandler<Event>`.
//! Expected error: `the trait bound `MyHandler: __sealed::Sealed` is not satisfied`.

#[test]
fn external_cascade_handler_impl_rejected() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/external_impl_rejected.rs");
}
