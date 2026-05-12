//! Pins the `@decay(rate = R, per = tick)` rate-bound contract:
//!
//! - `rate = 0.0` accepted (the "full reset every tick" idiom — the
//!   per-tick decay multiplier zeroes prior storage before the fold's
//!   event handlers add the current tick's contributions).
//! - `0.0 < rate < 1.0` accepted (legacy anchor-pattern rates).
//! - `rate = 1.0` rejected (the decay kernel becomes a no-op; callers
//!   should drop the annotation instead).
//! - `rate < 0.0` and `rate > 1.0` rejected (out of range).
//! - non-finite (NaN, ±Inf) rejected.

const TEMPLATE: &str = r#"
event Tick { }

@replayable
@gpu_amenable
event Hit {
  target: AgentId,
}

entity Particle : Agent {
  pos: vec3,
  vel: vec3,
}

@materialized(on_event = [Hit])
@decay(rate = RATE_PLACEHOLDER, per = tick)
view hits(p: Agent) -> f32 {
  initial: 0.0,
  on Hit { target: agent } where agent == p { self += 1.0 }
  clamp: [0.0, 1000000.0],
}

physics Tickle @phase(per_agent) {
  on Tick {} {
    emit Hit { target: self }
  }
}
"#;

fn try_resolve(rate: &str) -> Result<(), String> {
    let src = TEMPLATE.replace("RATE_PLACEHOLDER", rate);
    let program = dsl_compiler::parse(&src).map_err(|e| format!("parse: {e}"))?;
    dsl_ast::resolve::resolve(program)
        .map(|_| ())
        .map_err(|e| format!("resolve: {e}"))
}

#[test]
fn decay_rate_zero_is_accepted_full_reset_idiom() {
    try_resolve("0.0").expect("`@decay(rate = 0.0)` is the full-reset idiom and must resolve");
}

#[test]
fn decay_rate_in_open_interval_is_accepted() {
    for r in ["0.5", "0.95", "0.0001", "0.99999"] {
        try_resolve(r).unwrap_or_else(|e| panic!("rate={r} should resolve: {e}"));
    }
}

#[test]
fn decay_rate_one_is_rejected_no_op() {
    let err =
        try_resolve("1.0").expect_err("`@decay(rate = 1.0)` is a no-op; resolver must reject");
    assert!(
        err.contains("[0.0, 1.0)") || err.contains("rate"),
        "expected closed-lower / open-upper bound message; got: {err}"
    );
}

#[test]
fn decay_rate_out_of_range_is_rejected() {
    for r in ["-0.1", "1.5", "2.0", "-1.0"] {
        let err = try_resolve(r).expect_err(&format!("rate={r} must be rejected"));
        assert!(err.contains("rate"), "expected rate-range error; got: {err}");
    }
}

#[test]
fn decay_rate_non_finite_is_rejected() {
    // NaN and Inf round-trip via the float parser as `nan` / `inf`. The
    // resolver's `is_finite()` check rejects both.
    for r in ["nan", "inf", "-inf"] {
        // Some of these may fail to parse rather than fail to resolve;
        // either is an acceptable rejection.
        let res = try_resolve(r);
        assert!(res.is_err(), "rate={r} must be rejected (parse or resolve)");
    }
}
