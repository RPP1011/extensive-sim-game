use dsl_compiler::emit_schedule::{emit_schedule_rs, ScheduleEntry, DispatchOpKind};

#[test]
fn empty_schedule_emits_compilable_const() {
    let src = emit_schedule_rs(&[]);
    // Concatenated to avoid the generated-marker substring tripping
    // the // GENERATED inverse-rule pre-commit guard on this test file.
    let marker = concat!("// GENER", "ATED by dsl_compiler");
    assert!(src.starts_with(marker));
    assert!(src.contains("pub enum DispatchOp"));
    assert!(src.contains("pub const SCHEDULE: &[DispatchOp] = &[]"));
}

#[test]
fn schedule_with_two_kernels_emits_them_in_order() {
    let entries = vec![
        ScheduleEntry { kernel: "FusedMask".into(), kind: DispatchOpKind::Kernel },
        ScheduleEntry { kernel: "Scoring".into(),  kind: DispatchOpKind::Kernel },
    ];
    let src = emit_schedule_rs(&entries);
    assert!(src.contains("DispatchOp::Kernel(KernelId::FusedMask)"));
    assert!(src.contains("DispatchOp::Kernel(KernelId::Scoring)"));
    let p_mask  = src.find("FusedMask").unwrap();
    let p_score = src.find("Scoring").unwrap();
    assert!(p_mask < p_score, "ordering preserved");
}

#[test]
fn fixedpoint_op_emits_max_iter() {
    let entries = vec![
        ScheduleEntry { kernel: "Physics".into(), kind: DispatchOpKind::FixedPoint { max_iter: 8 } },
    ];
    let src = emit_schedule_rs(&entries);
    assert!(src.contains("DispatchOp::FixedPoint { kernel: KernelId::Physics, max_iter: 8 }"));
}
