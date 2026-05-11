//! Replaces the previous per-fixture build pipeline with the shared
//! `dsl_compiler::build_helper` (Plan E-A1). Uses Conservative
//! schedule strategy because this fixture's per-verb mask kernels
//! aren't fusion-compatible — keeps each verb's mask kernel as its
//! own `mask_verb_<Name>` module rather than a single fused one.
fn main() {
    dsl_compiler::build_helper::emit_with_strategy(
        "village_day_cycle",
        dsl_compiler::cg::schedule::ScheduleStrategy::Conservative,
    );
}
