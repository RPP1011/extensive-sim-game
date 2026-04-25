# Game docs

How we're using the engine + DSL + compiler specs to build the actual game. Reference specs live under `../spec/` (language, runtime, compiler, gpu, ability, economy) with live status under `../engine/status.md`; this folder explains how they compose into a playable artefact and how to extend it.

## Reading order

1. **`overview.md`** — What the game is. Layer map. Current state + compiler-first migration sequence.
2. **`compiler_progress.md`** — Live tracker: which DSL declaration kinds the compiler can emit, which legacy handlers have been retired. The ground-rule page.
3. **`feature_flow.md`** — How to add a new feature. Scenario A (compiler supports the kinds): write DSL, compile, test. Scenario B (compiler doesn't yet): extend the compiler first.
4. **`wolves_and_humans.md`** — The first end-to-end target. DSL source for the DF-like wolves+humans sim; maps to the compiler milestones that unlock each slice. Runs on legacy code until milestone 4; DSL-owned thereafter.

## Scope

This folder documents the *application layer* — the game we're building. It does not duplicate the specs; it references them. If a page here starts describing DSL grammar or engine internals in detail, that content belongs under the respective spec folder instead.
