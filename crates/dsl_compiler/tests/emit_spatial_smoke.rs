use dsl_compiler::emit_spatial_kernel::{emit_spatial_hash_rs, emit_kin_query_rs, emit_engagement_query_rs};

#[test]
fn spatial_hash_rs_has_kernel_impl() {
    let src = emit_spatial_hash_rs();
    assert!(src.contains("pub struct SpatialHashKernel"));
    assert!(src.contains("impl crate::Kernel for SpatialHashKernel"));
}

#[test]
fn kin_query_rs_has_kernel_impl() {
    let src = emit_kin_query_rs();
    assert!(src.contains("pub struct SpatialKinQueryKernel"));
}

#[test]
fn engagement_query_rs_has_kernel_impl() {
    let src = emit_engagement_query_rs();
    assert!(src.contains("pub struct SpatialEngagementQueryKernel"));
}
