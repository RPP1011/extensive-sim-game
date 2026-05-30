# Interactive Runtime + Engine UI + Playable Vampire Survivors (Design)

> Status: design, awaiting review. Next: `writing-plans` → implementation plan with AIS (P8).
> Builds on: `docs/superpowers/specs/2026-05-24-vampire-survivors-viewer-design.md` (DSL waves + GPU execution + voxel viewer, landed on `main`). The sim runs and is watchable; this spec makes it **playable** and builds the reusable engine capabilities that playability requires.

## 1. Goal & framing

Make `vampire_survivors` a **playable game** — a human drives the survivor (WASD), weapons auto-fire, level-ups offer an upgrade menu, with a real game UI (HUD, level-up menu, death/restart screen) over the existing 3D voxel viewer.

**This is engine expansion, not a fixture hack.** Two general, reusable engine capabilities are the marquee deliverables; vampire_survivors is their first consumer, not their boundary:

1. **A runtime input channel** — a runtime-writable buffer the DSL can read, so external (human or agent) input reaches a deterministic sim each tick. Any future `.sim` gets input for free.
2. **An engine UI layer, built in two tiers** — a reusable Rust UI framework (data-driven HUD/menu/screens rendered via egui, consumed by any fixture's viewer), and then a DSL `ui {}` block that lowers onto it, so UI is declared in `.sim` like rules, events, and views. The Rust framework lands first (reusable immediately); the DSL surface comes on top.

Everything is built to generalize. The work moves the engine forward rather than probing-and-working-around a DSL gap.

Success = a windowed run where you drive the survivor around an open arena, auto-firing weapons cull a continuously-spawning swarm, you pick upgrades as you level, and you get a death screen + restart when overwhelmed — with the HUD/menu/screens ultimately declared in the `.sim`.

## 2. Background: what exists, what's missing

**Exists (on `main`):**
- The full combat loop in `assets/sim/vampire_survivors.sim`: enemy chase, player kite, auto-firing bolt + nova, XP-on-kill (`xp(by: Agent)` materialized view, readable as a GPU storage buffer via `fold_view_xp_handles`), discrete level math (`floor(xp/xp_per_level)`), kill credit.
- GPU execution + host summon-drain (`crates/sims/src/summon_alloc.rs`) — DSL-authored waves spawn live enemies; the host already **writes per-agent SoA buffers every tick**.
- A windowed 3D voxel viewer (`crates/viewer_runtime/src/bin/vs_viewer.rs` + `vs.rs`) — top-down arena, role-colored voxel splats, pan/zoom/pause, auto-restart on death.
- `voxel_engine` (sibling crate) exposes egui-over-Vulkan: `EguiState` (`run`, `handle_window_event`, `cmd_paint`) + `SwapchainContext::present_blit_with_overlay`.

**Missing for "playable":**
1. **A runtime input channel.** The deterministic GPU sim decides everything itself; no per-tick human intent reaches a rule. The `sim_cfg` uniform (binding 5 on every fold kernel) uploads each tick via the `UploadSimCfg` op but carries only `tick` — `config` blocks otherwise bake to inline WGSL constants (the documented config-driven-sims gap). **Closing a slice of this is keystone #1.**
2. **Input-driven gameplay.** Player movement is autonomous flee-steering (`KitePlayer`); upgrades are auto-chosen by a fixed-priority `ChooseUpgrade` rule that fires *every tick* (a probe artifact) and whose choice has **no gameplay effect** — weapon damage scales off `xp`, not the chosen `UpgradeKind`. Both must become input-driven, and upgrades must actually change weapons.
3. **An engine UI layer.** `voxel_engine` has the egui *plumbing*, but there is no reusable, data-driven UI framework (HUD/menu/screen widgets bound to data) and no DSL UI surface. `viewer_runtime` has never driven egui. **Building the framework is keystone #2; the DSL `ui {}` block sits on top of it.**

## 3. The two cruxes

### 3.1 Input — a runtime-writable buffer the DSL reads (the general-infrastructure path)

All human-derived state lives **host-side** and flows into the sim each tick through one runtime-writable buffer; rules **read** it. Mutations still flow through the sim's normal event/kernel path — input is an external read, exactly like `sim_cfg.tick`. The host write is the sole determinism boundary (deterministic given the input log, the same property a replayed controller stream has).

- **Chosen — extend `sim_cfg` into a declared `input {}` block.** A new DSL `input { ... }` block (mirrors `config { ... }`) whose fields lower to `sim_cfg`-buffer **reads** instead of baked constants. Reuses the existing per-tick `UploadSimCfg` path; adds a host `set_input(...)` setter on the generated runtime. Smallest path that is also genuinely general — the first concrete slice of the config-driven-sims gap closure; any `.sim` can declare an `input` block afterward.
- **Rejected — host overwrites Agent SoA fields directly.** Zero compiler changes and determinism-safe, but a fixture-local hack: gameplay logic leaks out of the DSL into the viewer; nothing generalizes.
- **Rejected — a CPU gameplay layer in the viewer.** Abandons the DSL-as-engine premise.

**Consequence — upgrade levels live host-side too.** The human drives upgrades, so accumulated per-weapon levels are host-authoritative and pushed through the input buffer each tick (`input.bolt_level`, `input.nova_level`, …); DSL weapons read them directly. This sidesteps the **G1 fold-where-guard WALL** (per-kind upgrade tallies can't be split in a fold view today) **and** avoids any new Agent SoA column — so **no schema-hash bump** (P2 N/A).

### 3.2 UI — a data-driven model in two layers

The UI is **data-driven**, never hand-painted per fixture. A neutral UI model describes *what* to show and *what input an action writes*; a renderer turns it into egui draw calls; a per-frame data snapshot supplies the live values.

- **Layer 1 (Rust, `crates/engine_ui`):** a reusable framework with three pieces:
  - **`UiModel`** — a declarative description: HUD `Widget`s (`Bar { label, value, max, color }`, `Text { template, bindings }`, `Icon`…) and `Screen`s (`Menu { trigger, cards: [Card { label, action }] }`, `EndScreen { trigger, summary }`).
  - **`UiData`** — a per-frame snapshot of named values the viewer fills from sim readback (e.g. `hp`, `hp_max`, `tick`, `level`, `xp_frac`, `kills`, `enemies`). Bindings reference these by name; the framework never reads the sim itself.
  - **`UiAction`** — a card/button yields a named action; the framework returns picked actions to the host, which applies them to the input struct (e.g. `bolt_level += 1`). The framework owns no game state.
  - Rendering is via `voxel_engine`'s `EguiState` + `present_blit_with_overlay`. The framework is sim-agnostic and testable headlessly (model construction + binding resolution as pure logic; render is a thin egui adapter).
- **Layer 2 (DSL `ui {}` block):** a `ui {}` block in `.sim` declares HUD elements bound to views/fields/config and menu/screen actions bound to `input` fields; the compiler lowers it to a `UiModel` descriptor the runtime hands to `engine_ui`. UI becomes rules-as-data like everything else — and this *strengthens* P1 (compiler-first) by pulling UI into the DSL layer rather than hand-written Rust.

Layer 1 is fully usable on its own (a viewer builds a `UiModel` in Rust). Layer 2 replaces the hand-built model with a DSL-lowered one. The data-snapshot + action seam is identical for both, so Layer 2 is a front-end swap, not a rewrite.

## 4. Phases (one design; each phase independently verifiable and landable)

### Phase A — Input channel (keystone #1; reusable)

- **DSL surface:** a top-level `input { field: type = default, ... }` block (scalar `f32`/`u32`/`i32` fields). `input.field` is a new read expression usable anywhere `config.*` is (guards, emit amounts, `let`).
- **Lowering:** `input.field` resolves to a `sim_cfg`-buffer field read (not a `const`). Extend the `sim_cfg` uniform layout (`crates/dsl_compiler/src/cg/data_handle.rs` `SimCfgBuffer` + `UploadSimCfg` in `cg/op.rs` / `cg/schedule/synthesis.rs`) to carry the declared fields alongside `tick`, preserving std140 alignment.
- **Host API:** generated runtime gains `set_input(...)`; the per-tick `UploadSimCfg` writes current input values + tick before kernels run.
- **Verify:** a minimal `assets/sim/input_probe.sim` declares `input { drive: f32 }` + one rule moving an agent by `input.drive`; a headless test writes `drive`, steps, asserts the agent moved by the written amount. Plus a compile-gate assertion that `input.*` lowers to a buffer read, not a constant. **Proves the capability independent of VS and the viewer.**

### Phase B — Input-driven gameplay (DSL, in `vampire_survivors.sim`)

- **Movement:** rename `KitePlayer` → `PlayerControl`; drive `agents.set_pos` from `input.move_x/move_y` at speed `player_speed + input.move_level * speed_per_level`, keeping the radial arena clamp. Autonomous flee logic removed.
- **Weapon scaling:** `BoltFire` damage scales with `input.bolt_level`, period shortens with `input.bolt_rate_level`; `NovaFire` damage/radius scale with `input.nova_level`.
- **New weapons (engine-exercising content):** at least two new auto-weapons, each a new spatial query + physics rule gated by `input.<weapon>_level > 0`:
  - **Garlic/aura** — continuous damage to enemies within a tight radius every tick (reuses the nova query shape).
  - **Whip** — a directional forward strike (query filtered by direction relative to last move intent — exercises directional spatial filtering).
- **Remove** the auto-`ChooseUpgrade` rule + `UpgradeChosen`/`upgrades_total` (upgrade state is host-side now). Keep the `xp` view (HUD/level-up uses it).
- **Verify (headless):** drive a scripted input stream (move east, then bump `bolt_level`), step T ticks draining summons, assert player position tracks input; bolt damage rises with `bolt_level`; a gated weapon is inert at level 0 and fires once set; T ticks no panic (P10).

### Phase C — Engine UI framework + HUD (keystone #2, Layer 1)

- **Create `crates/engine_ui`:** `UiModel`/`UiData`/`UiAction` types + binding resolution + an egui renderer adapter over `voxel_engine`'s `EguiState`. Sim-agnostic.
- **Wire into `vs_viewer`:** construct `EguiState` on `resumed()`, forward winit events, switch the present path to `present_blit_with_overlay`, paint inside the overlay closure. Camera follows the player. Held WASD → an input vector pushed via `set_input` before each `step`.
- **First consumer:** `vs_viewer` builds a `UiModel` in Rust for the **HUD** — HP bar, run timer, level + XP-to-next bar, kill count, alive-enemy count — and fills `UiData` from readback each frame.
- **Verify:** unit tests on `UiModel` construction + binding resolution (pure, no GPU); a headless viewer smoke that steps a few ticks without panic (skips without a GPU adapter); manual run.

### Phase D — Level-up menu + death screen (Layer 1 interactive consumers)

- **Menu:** host tracks `level = floor(xp/xp_per_level)` from the `xp` readback; on increase, pause (stop `step`) and show a `Menu` screen — 3 cards (seeded RNG) from the upgrade pool (bolt damage, bolt rate, nova, move speed, + each new weapon). Picking a card emits a `UiAction` the host applies to the input struct; unpause. New weapons appear the tick their level goes 0→1.
- **Death screen:** on player death, show an `EndScreen` with a run summary (time survived, level, kills) + restart (re-seed runtime, reset host input/level state).
- **Verify:** headless unit tests of card-draw + action-application + screen-trigger logic (pure host logic); manual (level up → 3 cards → pick → weapon changes; die → summary → restart).

### Phase E — DSL `ui {}` block (Layer 2)

- **DSL surface:** a `ui {}` block declaring HUD widgets bound to views/agent-fields/config and menu/screen cards bound to `input` fields, e.g.:
  ```
  ui {
    hud {
      bar hp  { value: agents.hp(player), max: config.vs.player_hp, color: red }
      bar xp  { value: xp(player), max: config.vs.xp_per_level }
      text "Lv {level}   Kills {kills}   {time}"
    }
    menu on level_up {
      card "Bolt Damage +" { input.bolt_level += 1 }
      card "Nova +"        { input.nova_level += 1 }
      ...
    }
    screen on player_dead { summary: [time, level, kills] }
  }
  ```
- **Lowering:** the compiler emits a `UiModel` descriptor (the same shape `engine_ui` consumes); `vs_viewer` consumes the generated descriptor instead of hand-building it. The player-agent reference resolves via the existing player mana-band / designated singleton (detail for the plan).
- **Verify:** a compile-gate test that the `ui {}` block lowers to the expected descriptor; the viewer renders the DSL-declared UI identically to the Phase C/D hand-built model (swap the source, same screens).

## 5. Constitution check (for the plan's AIS, P8)

- **P1 (compiler-first)** ✅ — gameplay stays in the DSL; the input channel is sanctioned runtime I/O (like `sim_cfg.tick`). The UI framework (Phase C/D) is presentation/runtime code, not engine-rule behavior. The `ui {}` block (Phase E) *extends* the compiler-first surface — UI declared as data, lowered by the compiler.
- **P2 (schema-hash)** N/A — input fields live in the `sim_cfg` uniform, not Agent SoA; no event-variant or SoA-layout change. The `ui` descriptor and `engine_ui` touch no sim state.
- **P3 (cross-backend parity)** — `input.*` lowers to a buffer read identical on both backends; the host write is the determinism boundary (deterministic given the input log). UI is render-side, outside the sim. (VS runs GPU-only today, as all compiled `.sim` runtimes do.)
- **P5 (keyed PCG)** ✅ — spawns and the menu card-draw both use `per_agent_u32`/seeded RNG.
- **P6 (events are the mutation channel)** ✅ — input is read-only into rules; mutations still flow through events / the kernel API.
- **P10 (no panic on hot path)** — Phase B's headless driver asserts T ticks with no panic.
- **P8** — the implementation plan carries the full AIS.

## 6. Top risks

- **`sim_cfg` layout expansion** ripples to binding 5 on *every* fold kernel — preserve std140 alignment and re-verify naga accepts the widened uniform across the whole workspace. Riskiest single change; Phase A's minimal probe de-risks it before VS depends on it.
- **First egui-over-Vulkan use in this workspace.** API exists in `voxel_engine`; integration (render-pass LOAD semantics, per-frame slow path, event consumption) is the main Phase C unknown. Keep the dungeon_horde path untouched.
- **`engine_ui` ↔ `voxel_engine` coupling.** `engine_ui` needs egui types + the present-overlay seam from `voxel_engine` (a sibling repo). Decide the dependency direction in the plan (likely `engine_ui` depends on `voxel_engine`, re-exporting egui) so the framework stays in the game-engine workspace.
- **`xp` view readback accessor.** Confirm/add a `pub` accessor on the generated runtime for the `xp` primary buffer (`fold_view_xp_handles`).
- **Storage-buffer ceiling (32).** Each new weapon adds a query binding; VS already carries several. Budget weapon count against the ceiling.
- **DSL `ui {}` scope creep (Phase E).** UI declaration languages sprawl. Keep the block minimal — bars, text, menu cards, end screen — bound to existing readable values + input fields; defer layout/styling richness.

## 7. File map

- `assets/sim/vampire_survivors.sim` — Phase B gameplay rework + new weapons; Phase E `ui {}` block (modify).
- `assets/sim/input_probe.sim` — Phase A probe (create).
- `crates/dsl_ast/` parser + AST — `input {}` block + `input.field` read; later the `ui {}` block grammar (modify).
- `crates/dsl_compiler/src/` resolve + `cg/data_handle.rs` (`SimCfgBuffer`) + `cg/op.rs` / `cg/schedule/synthesis.rs` (`UploadSimCfg`) — lower `input.*`, widen the uniform, host setter; Phase E `ui {}` → `UiModel` descriptor emit (modify).
- `crates/dsl_compiler/tests/` — Phase A input-read + Phase E `ui` lowering compile-gates (create/modify).
- `crates/engine_ui/` — Phase C `UiModel`/`UiData`/`UiAction` + egui renderer adapter (create).
- `crates/sims/tests/vampire_survivors_exec.rs` — Phase B headless input-driven driver (modify).
- `crates/viewer_runtime/src/vs.rs` + `bin/vs_viewer.rs` + `vs_ui.rs` — egui wiring, WASD→`set_input`, follow-cam, HUD/menu/death via `engine_ui` (Phase C/D), then consume the DSL descriptor (Phase E) (create/modify).

## 8. Out of scope (future slices)

- The full config-driven-sims gap closure (runtime-overridable `config` blocks); this spec closes only the *input* slice.
- Per-agent input arrays (multi-agent / networked control) — input is a single shared struct here.
- Rich UI layout/styling, animation, theming in the `ui {}` block — minimal widget set only.
- Physical XP gems + magnet pickup (foundation Slice 3); cross-item weapon **evolution** (Slice 5); enemy archetypes/elites/bosses (Slice 6); meta-progression (Slice 8).
- Persisting/replaying the input stream (the determinism boundary is designed for it; not built here).
