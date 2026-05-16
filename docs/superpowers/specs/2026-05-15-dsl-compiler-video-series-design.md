# DSL Compiler — YouTube Series Design

**Date:** 2026-05-15
**Status:** Draft — pending user review
**Owner:** Ricky

A YouTube series documenting how the `dsl_compiler` crate works, from `.sim` source text to the four output artefact classes (scalar Rust, WGSL kernels, Python dataclasses, trace format). Animated explainer style, end-to-end coverage, 14 episodes across three acts.

---

## 1. Series parameters

| | |
|---|---|
| **Audience** | Compiler-curious developers — viewers comfortable reading Rust and WGSL, interested in IRs, lowering, codegen, GPU compilation. Reference points: r/ProgrammingLanguages, *Crafting Interpreters* readers, Jon Gjengset / Tsoding watchers. |
| **Episode length** | 8–15 min target, aim for 12. Short enough that single-concept episodes are achievable; long enough to develop a real demo. |
| **Cadence** | Released in batches by act (Act I, then II, then III). Not bound to a fixed weekly schedule. |
| **Scope** | End-to-end pipeline: tokens → parser → AST → resolve → frontend IR → Compute-Graph IR → schedule/fuse → emit (Rust + WGSL + Python) → schema-hash + build.rs integration. |
| **Style** | Animated explainer. Heavy use of motion graphics; real code visible but never the dominant frame; every beat has visual change. |
| **Rendering stack** | Motion Canvas (TypeScript) as the renderer + a new `--emit-viz-json` extension on `dsl_compiler` so animations consume real compiler artifacts, not hand-drawn approximations. |
| **Deliverables from this design** | Per-episode visualization specs (§5) and asset/tooling plan (§4). Narration scripts are deliberately out of scope — the user will script in their own voice. |

## 2. Series principles

These constrain every episode regardless of where it sits in the arc.

### 2.1 Hook-first per episode

The 3-act spine (§3) is a binge-watch recommendation, not a viewing requirement. Every episode is structurally hook-first so a cold-drop viewer landing from search isn't lost. The cold-drop intro template (§4.4) is itself the hook structure: title → thesis with signature-viz preview → cut into action. The body then opens directly on the signature viz, not on a "today we'll learn about" preamble.

Per-episode shape:

| | |
|---|---|
| **0:00–0:30** | Cold-drop intro (§4.4): title → thesis → pipeline locator → hard cut. |
| **0:30–10:30** | Body, in 4–6 beats. The first beat opens on the signature visualization (the hook). Each beat is its own mini-payoff. Visual change every 20–40 seconds. No 90-second uninterrupted shots. |
| **10:30–11:30** | Earned demo. Run the actual compiler. Show actual output. Connect what was built to a visible result. |
| **11:30–12:00** | End card with a tease for the next episode. One specific question, not "subscribe." |

Episode 1 deviates: no cold-drop intro at all, opens straight on the simulation. Episode 2 uses the intro but skips the pipeline locator beat (the pipeline hasn't been introduced yet). Episode 3 onward uses the full template.

### 2.2 Stimulation budget

- Visual change every 20–40 seconds, minimum. Static narration over a single frame is the kill shot.
- 3–5 "oh that's clever" moments per episode. The compiler has dozens — one per episode is plenty.
- Cold-drop survivability. Every episode states its own thesis. Cross-references between episodes are *visual* (consistent motifs, color codes) rather than narrative ("remember when we…").

### 2.3 Honest framing

The engine and compiler exist because the user wanted to design them, not because failed alternatives forced them. When introducing a design choice (lowering pass, fusion, schema-hash, parity), frame as:

> *Here is a property we wanted, here is how the design provides it.*

Not as:

> *Here is a bug that forced us to do this.*

If a real bug history is interesting on its own merits (the f32 RMW race, the ahash drift), tell it as-is. But never as the origin myth of a feature that was actually designed. This is a passion-project series; transparency about that is the value-add, not a weakness.

### 2.4 Real artifacts, not hand-drawn

Animations consume JSON dumped from the actual compiler (§4.1). When E5 shows an AST building, it builds from a real parser snapshot of a real fixture. When E12 shows fusion happening, the before/after IR is the actual fusion-pass output. This is what justifies the time investment in `--emit-viz-json` — the visuals are *true*, and the compiler's evolution is automatically reflected.

---

## 3. Episode map

Three acts. 14 episodes. Each episode card lists the title, the one-line thesis, and the signature visualization that dominates the episode (and the thumbnail).

### Act I — Why this language exists (3 eps)

Sell the project before any compiler vocabulary appears. The word "compiler" is not used in the first 5 minutes of Episode 1.

| Ep | Title | Thesis | Signature viz |
|---|---|---|---|
| 1 | A simulation worth compiling | 10,000 deterministic agents running on CPU or GPU — and it all comes from one text file. | `megaswarm_10000` running real-time; quick cuts to `duel`, `dungeon_horde`; the "wait, this is one source file" reveal at 11:00. |
| 2 | Why I wanted a compiler | A passion-project explainer: the design space, what compiler-first buys, and what it costs. Constitution shows up here as designed-in architecture. | Side-by-side: same Strike verb expressed three ways (hand-coded Rust, scripted, DSL). All three work. The DSL has the properties the user wanted; properties are labeled, not problems. |
| 3 | Reading the language | Read `duel_1v1.sim` as a story: each declaration appears as the duel needs it. By the end the viewer knows what `.sim` syntax feels like. | Source pane fills line-by-line; running duel alongside; final shot: source on left, combat on right, synchronized. |

### Act II — Frontend (5 eps)

Source text → trusted IR. Conventional compiler-frontend territory, but with this project's specific shape.

| Ep | Title | Thesis | Signature viz |
|---|---|---|---|
| 4 | Lexing in 130 lines of Rust | How `tokens.rs` turns characters into a flat token stream. | Source pane on left; `TokenStream` populates on right, colored by class; the lexer's read-cursor walks the source character-by-character. |
| 5 | Parsing without a parser generator | Recursive descent in `parser.rs`. The call stack IS the AST tree. | Tokens flow in; `ASTTree` grows branch-by-branch matching the parser's call stack. |
| 6 | The shape of an AST | `Decl`, `Spanned<T>`, why spans are everywhere. An AST is just data. | AST tree with span-arrows back to source; mouseover-style highlighting between source ranges and tree nodes. |
| 7 | Turning strings into pointers | Name resolution in `resolve.rs`. Why this is a 5K-line pass and not parser sugar. | Identifier tokens (strings) dissolving into typed IDs (`EventId`, `FieldId`); `SymbolTable` grows alongside as scopes open/close. |
| 8 | The IR is just data | The frontend `Compilation` struct as the boundary between trusted input and compile work. | AST collapses into a flat `Compilation` struct laid out in memory; typed handles inside are clickable to show their meaning. |

### Act III — Backend (6 eps)

IR → running, bit-exact code on two backends. The distinctive territory: cross-backend emission, fusion, parity, schema hash.

| Ep | Title | Thesis | Signature viz |
|---|---|---|---|
| 9 | Why we built a second IR | The Compute-Graph IR (`cg/program.rs`). Why frontend IR can't go directly to kernels. | `Compilation` struct on left; CG graph (nodes + edges) materializing on right; same information, different shape. |
| 10 | How one verb becomes three kernels | Lowering: `verb_expand.rs`, `physics.rs`, `scoring.rs`, `mask.rs`. Watch `Strike` from `duel_1v1` expand. | One `verb Strike { … }` declaration explodes into mask-kernel + physics-rule + scoring-row; each child traces parentage back to the source verb. |
| 11 | Catching GPU bugs at compile time | Well-formedness in `cg/well_formed.rs` — 5K+ lines of "is this CG-IR actually valid?" | Intentionally broken `.sim` file; the compiler's rejection animates each well-formed check firing, green/red bitmap of which checks passed. |
| 12 | Three loops, one kernel | Scheduling and fusion in `cg/schedule/fusion.rs`. The marquee "wow" episode. | `FusionDiagram`: three independent CG-nodes physically merge into one fused kernel; below them, WGSL shader text shrinks line-by-line as fusion completes. |
| 13 | Same source, two backends, bit-for-bit identical | Cross-backend emission (`cg/emit/*`): scalar Rust + WGSL from one IR. Constitution P3 + P11. | `BackendSplit`: same fixture running on CPU and GPU; per-tick state diffs always zero; reduction-determinism (sort-then-fold) animated. |
| 14 | How a schema hash stops the build | Shipping the compiler: `schema_hash.rs`, `build_helper.rs`, Python dataclasses, traces, replay. | Layout change touching `SimState`; CI failing; the schema_hash sentinel front-and-center; trace ring-buffer view; series recap (the complete four-artefact diagram). |

### Out of scope (could be Act IV later)

`ability_lower.rs`, `ability_registry.rs` (the `.ability` file DSL), `belief_decay_wgsl.rs`, `custom_agent_fields.rs`, `kernel_binding_ir.rs`, `cpu_chronicle_reference.rs`. Each is a candidate for follow-up episodes if the series finds an audience; not committed for v1.

---

## 4. Asset and tooling plan

The shared foundation under all 14 episodes. Built once, reused everywhere.

### 4.1 `--emit-viz-json` compiler extension

A new flag on `dsl_compiler` dumps structured JSON at each pipeline stage. Animations consume these dumps so the visuals are always faithful to the current compiler.

| Stage | Source module | Emits |
|---|---|---|
| `tokens` | `dsl_ast/tokens.rs` | Ordered token list, source spans, classes. |
| `ast` | `dsl_ast/parser.rs` → `ast.rs` | Full AST + incremental snapshots at N parse points (so animation can build the tree). |
| `resolve` | `dsl_ast/resolve.rs` | Symbol table, scope-stack snapshots, resolution diffs per identifier. |
| `ir` | `dsl_ast/ir.rs` | Frontend `Compilation` struct as JSON. |
| `cg` | `dsl_compiler/cg/program.rs` | Compute-graph nodes + edges, typed. |
| `well_formed` | `cg/well_formed.rs` | Per-check pass/fail, evidence per failure. |
| `schedule` | `cg/schedule/*.rs` | IR before/after each pass; fusion deltas. |
| `emit` | `cg/emit/*.rs` | Final WGSL strings + Rust strings + Python module text. |

**CLI shape.** `cargo run -p dsl_compiler -- viz-dump --fixture duel_1v1.sim --out target/viz/` — emits one JSON file per stage into the target directory. Defaults to all stages; `--stage tokens,ast` to subset.

**Schema.** Versioned (`{ "schema_version": 1, "stage": "tokens", … }`). Schemas are stable across patch versions; bumps allowed at major versions with downstream Motion Canvas readers updated in lockstep.

**Cost.** Approximately 2–4 weeks of compiler work. Cleanly additive — a new module behind a `viz-json` feature flag. Zero risk to runtime behavior. Highest-leverage item to build first because no animation reflects reality without it.

### 4.2 Motion Canvas project layout

Single monorepo at `video/` in this project. One Motion Canvas project. One folder per episode. Shared `video/src/lib/` for components, shared `video/src/data/` symlinking `target/viz/` dumps.

```
video/
  src/
    lib/              # reusable scene components (§4.3)
    data/             # symlinks to target/viz/
    episodes/
      ep01_simulation/
      ep02_why_compiler/
      …
    theme.ts          # palette, typography, motion constants
  assets/
    fonts/            # JetBrains Mono + Inter
    recordings/       # real sim screen-recordings (for E1)
    narration/        # voiceover WAVs per episode
  renders/            # mp4 + thumbnail per episode
  package.json
```

### 4.3 Reusable scene components

Each component is a Motion Canvas `Node` subclass with declarative props and a generator-based animation API. Build once; used by 6–14 episodes.

**Core (used by most episodes):**

- `CodePanel` — syntax-highlighted code (Rust / WGSL / `.sim` / `.ability`). APIs: `highlight(range)`, `insert(line)`, `fade(range)`, `callout(range, text)`.
- `SourceToOutput` — split view: source pane left, derived artifact right, animated span-arrows connecting source ranges to output ranges. Used in E5, E7, E10, E13.

**Frontend domain:**

- `TokenStream` — horizontal flow of token cards, colored by class. Built incrementally from `tokens` JSON. E4 signature.
- `ASTTree` — force-directed or layered tree, grows node-by-node from parser snapshots. E5, E6.
- `SymbolTable` — side-panel showing scopes as a stack; names → resolved IDs animate on insert. Bound to `resolve` stage JSON. E7.

**Backend domain:**

- `IRGraph` — generic node-edge graph for compute-graph IR. Force-directed layout; data-driven from `cg` JSON. E9–E13.
- `FusionDiagram` — N loop boxes merge into one; before/after WGSL panes shrink. E12 signature.
- `BackendSplit` — two-pane execution comparison: Serial Rust left, GPU WGSL right, per-tick state diff always zero. E13.

**Overlay / structural:**

- `PipelineDiagram` — canonical horizontal strip of stages (tokens → AST → IR → CG → emit). Used in every cold-drop intro as the "where we are" locator. Current stage glows.
- `FixtureRunner` — embedded mini-viewer for a real sim recording with HUD (tick counter, agent count, backend label). E1, E2, E13.
- `BeforeAfter` — two-state diff card: title above, content below, explicit `→` transition or crossfade.
- `ColdDropIntro` — the 20–30s opener template (§4.4).
- `EndCard` — last 10s: next-episode tease + subscribe overlay + series logo.
- `ThumbnailFrame` — 1280×720 still-image generator (§4.6).

### 4.4 Cold-drop intro template

Every episode opens with the same 20–30 second shape. Familiar enough that returning viewers can skip; informative enough that a search-driven cold-drop viewer doesn't bounce.

| | |
|---|---|
| 0:00–0:05 | Title card: episode number + title. Bold typography; act color in corner. |
| 0:05–0:15 | Thesis statement on screen, narrated. Signature-viz preview behind. |
| 0:15–0:25 | `PipelineDiagram` appears, current stage glowing. "We're here in the pipeline." Skipped on E1 (no intro at all) and E2 (pipeline not yet introduced). |
| 0:25–0:30 | Hard cut into the episode body — directly onto the signature visualization, no preamble. |

### 4.5 Visual language

**Concept palette** — one accent per major concept; reused episode-to-episode for visual continuity:

| | |
|---|---|
| `#4FC3F7` | tokens |
| `#81C784` | AST / frontend IR |
| `#B388FF` | CG IR |
| `#FFB74D` | emitted kernels |
| `#EF5350` | errors / diagnostics |

Background `#1B1B1B`; primary text `#FFFFFF`. Dark mode default; no light theme variant.

**Typography.** Code: JetBrains Mono (ligatures off — cleaner on small screens). Body labels and titles: Inter. Title cards: Inter Bold at large weights. No decorative fonts.

**Motion principles.** Easing: `easeOutCubic` default; no spring/bounce (cartoonish on technical content). Durations: 200–400ms for micro (highlight, callout); 600–1200ms for major (scene transition, IR transform). Content arrives by reveal (mask/fade), not by scale-pop.

### 4.6 Thumbnail system

1280×720, generated from `ThumbnailFrame`. Layout: signature-viz screenshot occupies ~75% of frame; act-colored episode badge top-right (`EP 04 · FRONTEND`); large title (4–6 words, Inter Bold) bottom or right. Series logo bottom-right corner. Same typography family as the video itself.

### 4.7 Render pipeline

- **Format:** 1080p60 H.264 mp4 (4K30 only if upload bandwidth tolerates).
- **Audio:** narration recorded separately; lined up against Motion Canvas audio markers. Music bed optional at −20 dB.
- **Captions:** generate from narration via Whisper or auto-YT; edit before upload.
- **Per-episode output:** `video/renders/ep0X/` containing `episode.mp4`, `thumbnail.png`, `captions.srt`, `description.md`.
- **Iteration speed:** Motion Canvas hot-reloads scenes during development; full render of a 12-min episode ≈ 10–30 minutes wall-clock.

### 4.8 Build sequencing

Dependency order. Each item unblocks downstream work.

1. **`--emit-viz-json` compiler extension** — nothing animates real data without this. Highest-leverage first build. ~2–4 weeks.
2. **Motion Canvas scaffold + `theme.ts`** — palette, fonts, motion constants. ~1–2 days.
3. **Core components** (`CodePanel`, `SourceToOutput`, `PipelineDiagram`, `ColdDropIntro`, `EndCard`, `ThumbnailFrame`) — the "every episode uses these" layer. ~1 week.
4. **Episode 1** — largely `FixtureRunner` + recordings; doesn't need most components yet. Forces cold-drop template and end-card to exist.
5. **Domain components per act**, on demand:
   - Act II: `TokenStream` (E4), `ASTTree` (E5/E6), `SymbolTable` (E7).
   - Act III: `IRGraph` (E9), `FusionDiagram` (E12), `BackendSplit` (E13).

---

## 5. Per-episode visualization specs

Each episode is specified beat-by-beat. Format per beat: time range, what's on screen (component names from §4.3 in `monospace`), data source (compiler JSON dump or recording), and the transition out.

Narration is *not* specified — the user authors that. The specs are a storyboard in prose: the visuals are pinned, the words are free.

### Episode 1 — A simulation worth compiling

| Time | On screen | Data source | Transition |
|---|---|---|---|
| 0:00–0:20 | `FixtureRunner` full-screen: `megaswarm_10000` running. HUD shows tick counter incrementing, agent count, frame budget. No narration yet — just the sim. | Recording: `megaswarm_10000`, 30s loop. | Cut. |
| 0:20–1:00 | Three `FixtureRunner` panes in a row: `megaswarm`, `duel_25v25`, `dungeon_horde`. Each labeled with its agent count. | Three recordings. | Crossfade. |
| 1:00–3:00 | `BackendSplit`: same fixture (`duel_1v1`) running on CPU and GPU side-by-side. Per-tick state-diff counter at bottom reads `0`. Same seed indicator. | Recordings + a real diff trace. | Pull back. |
| 3:00–6:00 | A single file icon labeled `duel_1v1.sim` on screen. Line count and byte size animate in. Content still hidden. | Static. | Fade. |
| 6:00–9:00 | Diagram: source file at center, four arrows to four artefact boxes (Rust, WGSL, Python, traces). Each box briefly shows real generated text scrolling. | `--emit-viz-json emit` for `duel_1v1.sim`. | Reorganize. |
| 9:00–10:30 | Back to `megaswarm_10000` running, but now with the four-artefact diagram as a faint overlay. | Recording + diagram. | Zoom. |
| 10:30–11:30 | The `.sim` file revealed in full. `CodePanel` pans over it once, top to bottom. No annotations. | `duel_1v1.sim` source. | Fade to end card. |
| 11:30–12:00 | `EndCard`: "Next — why I wanted a compiler for this in the first place." | — | End. |

### Episode 2 — Why I wanted a compiler

| Time | On screen | Data source | Transition |
|---|---|---|---|
| 0:00–0:30 | `ColdDropIntro` (skip pipeline locator — we haven't introduced it yet). Thesis on screen: *"You don't have to write a compiler to ship a game. So why?"* | — | Cut. |
| 0:30–2:00 | Three `CodePanel`s in a row, same gameplay rule (`Strike` damage) expressed in: hand-coded Rust, scripted DSL (Lua-shape), the `.sim` DSL. All three labeled "works." | Hand-authored stand-ins; final column is real `.sim` syntax. | Settle. |
| 2:00–4:00 | Annotated comparison table animates across the three columns: rows are properties (perf, determinism, parity, type safety, refactorability, learnability). Cells fill in. | — | Highlight DSL column. |
| 4:00–7:00 | The DSL column's wanted-properties light up one at a time; each becomes a labeled badge: *"deterministic replay," "two backends," "rules-as-data," "compile-time checking."* | — | Step back. |
| 7:00–9:00 | One constitution principle (e.g. P1) shown as a `PrincipleBadge` next to the DSL column. Narration frames it as a *designed-in* property, not as a scar. | `docs/constitution.md` P1 text. | Crossfade. |
| 9:00–10:30 | Honest counter-list: what does the DSL cost? Compiler complexity, slower iteration on rule changes, learning curve. Three items appear as cards. | — | Cut. |
| 10:30–11:30 | Earned demo: edit `duel_1v1.sim` (change a damage value); recompile; rerun. Cycle time visible. ~1–2 seconds. | Recording of the actual edit-compile-run loop. | Fade. |
| 11:30–12:00 | `EndCard`: tease E3 — "Next, what the language actually looks like." | — | End. |

### Episode 3 — Reading the language

| Time | On screen | Data source | Transition |
|---|---|---|---|
| 0:00–0:30 | `ColdDropIntro` with `PipelineDiagram` introduced for the first time (no stage glowing yet — we're outside the pipeline). | — | Cut. |
| 0:30–1:30 | Empty `CodePanel` left; empty arena right. Title: *"Two combatants. Three abilities. One duel."* | — | Settle. |
| 1:30–3:00 | `entity Combatant : Agent { … }` types into the source pane line by line; arena gets two figures with HP bars. | `duel_1v1.sim` source, by region. | Continue. |
| 3:00–4:30 | `event Damaged`, `event Healed`, `event Defeated` declarations appear. Arena gets an event-log header. | Source. | Continue. |
| 4:30–6:30 | `verb Strike { … }` typed in. `SourceToOutput`: mask, cascade, scoring sub-blocks highlighted on left; arena previews what Strike *would* do (no firing yet). | Source + parsed AST regions from `--emit-viz-json ast`. | Continue. |
| 6:30–8:00 | `verb Spell`, `verb Heal` added. Arena's "available actions" menu populates. | Source. | Continue. |
| 8:00–9:30 | `physics ApplyDamage`, `physics ApplyHeal` rules added. Arrows from events → physics rules on screen. | Source. | Continue. |
| 9:30–10:30 | `invariant` + `probe` declarations added. Frame them as the safety net. | Source. | Settle. |
| 10:30–11:30 | Earned demo: hit "compile + run." The duel plays out in the arena. Events tick into the log; HPs deplete; one combatant wins. | Real `duel_1v1` run. | Fade. |
| 11:30–12:00 | `EndCard`: "Next — how the compiler turns these characters into tokens." | — | End. |

### Episode 4 — Lexing in 130 lines of Rust

| Time | On screen | Data source | Transition |
|---|---|---|---|
| 0:00–0:30 | `ColdDropIntro` with `PipelineDiagram` — `tokens` stage glowing. | — | Cut. |
| 0:30–1:30 | `SourceToOutput`: raw source on left; on right, the goal — a flat list of typed token cards. Empty for now. | — | Settle. |
| 1:30–3:00 | Read-cursor walks through `verb Strike {`. `TokenStream` populates: KEYWORD `verb`, IDENT `Strike`, LBRACE `{`. Each card colored by class. | `--emit-viz-json tokens` for `duel_1v1.sim`, first 30 tokens. | Continue. |
| 3:00–5:00 | `CodePanel` of the actual `tokens.rs` file scrolling. Highlight the main loop. Each branch maps to a token class produced. | `crates/dsl_ast/src/tokens.rs`. | Cut. |
| 5:00–7:00 | Tricky case: `>=` vs `>`. Lookahead. Two-character scan animated. | — | Continue. |
| 7:00–8:30 | Spans: each token card grows a span-arrow back to source position. Foreshadow E11 ("errors point back here"). | `tokens` JSON has spans. | Continue. |
| 8:30–10:00 | What lexers don't do: a "next stage" hopper appears; tokens flow into it. The hopper is closed for now. | — | Cut. |
| 10:00–11:30 | Earned demo: full `duel_1v1.sim` lexed in real time. ~3000 tokens flow past in ~10 seconds (sped up). | Full `tokens` JSON, replayed. | Fade. |
| 11:30–12:00 | `EndCard`: "Next — those tokens get assembled into something with structure." | — | End. |

### Episode 5 — Parsing without a parser generator

| Time | On screen | Data source | Transition |
|---|---|---|---|
| 0:00–0:30 | `ColdDropIntro` — `parser` stage glowing on `PipelineDiagram`. | — | Cut. |
| 0:30–1:30 | `SourceToOutput`: token cards on left, empty `ASTTree` on right. Title: *"Tokens in, AST out — but how?"* | — | Settle. |
| 1:30–3:00 | Diagram: the parser as a tree of functions calling each other. `parse_program → parse_decl → parse_entity_decl → parse_field_decl`. Each function as a labeled node. | `crates/dsl_ast/src/parser.rs` function names. | Continue. |
| 3:00–5:00 | Animation: as the call stack descends, `ASTTree` nodes are constructed in real time. Call stack visible on side; tree grows on main canvas. | `--emit-viz-json ast` with snapshots at each parse-decl boundary. | Continue. |
| 5:00–7:00 | Precedence: `a + b * c` parses as `a + (b * c)`. Show the disambiguation step in the parser; tree resolves correctly. | Synthetic expression + real parser behavior. | Continue. |
| 7:00–9:00 | Error recovery: feed `verb Strike { mask: …` (truncated). Parser doesn't panic; produces a diagnostic with a span. Span-arrow back to source. | Synthetic broken input. | Continue. |
| 9:00–10:00 | The `.ability` parser split. Quick split-screen: same project, two parsers. Why? Different surface syntax. | `dsl_ast/parser.rs` vs `ability_parser.rs`. | Cut. |
| 10:00–11:30 | Earned demo: full `duel_1v1.sim` parse, `ASTTree` builds in fast-forward. Final tree visible. | Full `ast` JSON, replayed. | Fade. |
| 11:30–12:00 | `EndCard`: "Next — what's actually in an AST." | — | End. |

### Episode 6 — The shape of an AST

| Time | On screen | Data source | Transition |
|---|---|---|---|
| 0:00–0:30 | `ColdDropIntro` — `parser` stage still glowing. | — | Cut. |
| 0:30–2:00 | `CodePanel` showing the `Decl` enum definition with variants visible. Highlight one variant. | `crates/dsl_ast/src/ast.rs::Decl`. | Cut. |
| 2:00–4:00 | `Spanned<T>` wrapper. Hover demo: cursor moves over an `ASTTree` node; corresponding source range highlights on the left pane. `SourceToOutput`. | `ast` JSON spans. | Continue. |
| 4:00–6:00 | Nominal vs untyped AST. Animated comparison: typed enum (catches errors at transitions) vs untyped string-kind nodes (errors leak through). | — | Cut. |
| 6:00–8:00 | Visitor pattern: a real traversal from the codebase (find-all-verbs). Show recursion over AST in `CodePanel`; results panel populates. | `dsl_ast` source. | Continue. |
| 8:00–9:30 | What the AST isn't: it doesn't know what `Damaged` *refers to*. `EventRef("Damaged")` is just a string. Forward-pointer to E7. | — | Cut. |
| 9:30–10:30 | Earned demo: pretty-print `duel_1v1.sim`'s AST. Side-by-side with source. Each node clickable shows its variant. | Full `ast` JSON. | Fade. |
| 10:30–12:00 | `EndCard`: "Next — turning those strings into actual references." | — | End. |

### Episode 7 — Turning strings into pointers

| Time | On screen | Data source | Transition |
|---|---|---|---|
| 0:00–0:30 | `ColdDropIntro` — `resolve` stage glowing. | — | Cut. |
| 0:30–1:30 | Problem statement: AST contains `EventRef("Damaged")` (a string). Result needed: a typed `EventId(7)`. | — | Settle. |
| 1:30–3:00 | `SymbolTable` panel appears. Walk top-down through AST declarations; entries push into the table. | `--emit-viz-json resolve` scope snapshots. | Continue. |
| 3:00–5:00 | Scope rules: variables in a `verb` are local; entities and events are global. Scope-stack push/pop animated. | `resolve` JSON. | Continue. |
| 5:00–7:00 | Resolution result: each `EventRef("Damaged")` dissolves into `EventId(7)`. Animation: string fades, typed ID materializes. | `resolve` JSON. | Continue. |
| 7:00–9:00 | Three diagnostic examples in sequence: unresolved name, duplicate declaration, cycle. Each shown as a broken `.sim` file + compiler error with span-arrow. | Synthetic broken inputs. | Cut. |
| 9:00–10:00 | Why a separate pass: forward references need to see all declarations first. Diagram: two-pass collect-then-resolve. | — | Continue. |
| 10:00–11:30 | Earned demo: `duel_1v1.sim` fully resolved. `SymbolTable` full; all `EventRef` strings replaced by IDs. | Full `resolve` JSON. | Fade. |
| 11:30–12:00 | `EndCard`: "Next — the frontend's final shape." | — | End. |

### Episode 8 — The IR is just data

| Time | On screen | Data source | Transition |
|---|---|---|---|
| 0:00–0:30 | `ColdDropIntro` — `ir` stage glowing. | — | Cut. |
| 0:30–2:00 | AST on left, `Compilation` struct on right. AST collapses field-by-field into the IR. | `--emit-viz-json ir`. | Continue. |
| 2:00–4:00 | The `Compilation` struct fields walkthrough. Each field highlighted; what it normalized from the AST. | `crates/dsl_ast/src/ir.rs`. | Cut. |
| 4:00–6:00 | The boundary: above the IR, "trusted input." Below, "compile work." Diagram of frontend (parse+resolve+validate) handing off to backend. | — | Continue. |
| 6:00–8:00 | The contract: if we got here, IR is typed, named, scoped. Spans preserved but no longer load-bearing for compilation logic. | — | Cut. |
| 8:00–9:30 | What the IR enables for the backend: dataflow analyses, fusion, scheduling — tease E12. | — | Cut. |
| 9:30–11:00 | Earned demo: full frontend pipeline run from source to `Compilation`. Each stage flashes briefly (`PipelineDiagram` lights up stage-by-stage). End state: `Compilation` JSON on screen. | All frontend JSON dumps, replayed in sequence. | Fade. |
| 11:00–11:30 | Series midpoint marker: half the pipeline covered. Act III preview montage (5 seconds of clips). | Recordings. | Fade. |
| 11:30–12:00 | `EndCard`: "Next — why we have a *second* IR." | — | End. |

### Episode 9 — Why we built a second IR

| Time | On screen | Data source | Transition |
|---|---|---|---|
| 0:00–0:30 | `ColdDropIntro` — `cg` stage glowing on `PipelineDiagram`. | — | Cut. |
| 0:30–2:00 | Premise: we have a `Compilation` struct. Why isn't it enough for codegen? Annotate the struct with what it *doesn't* know: kernel boundaries, dispatch shape, dataflow. | `ir` JSON. | Cut. |
| 2:00–4:00 | The compute-graph IR. Nodes are operations; edges are data dependencies. `IRGraph` materializes alongside the `Compilation` struct — same information, different shape. | `--emit-viz-json cg`. | Continue. |
| 4:00–6:00 | The graph IS the program. Topological order = execution order. Animate a walk through the graph. | `cg` JSON. | Continue. |
| 6:00–8:00 | What this enables: dataflow analysis, fusion, scheduling. Quick tease frames pointing to E12. | — | Cut. |
| 8:00–10:00 | Live build: take `Compilation`, derive CG graph. Edges appear one at a time. | `cg` JSON, incremental construction. | Continue. |
| 10:00–11:30 | Earned demo: zoom into a single op node. Its inputs, outputs, and inline kernel implementation visible. The "atomic unit" of compute. | `cg` JSON, single node detail. | Fade. |
| 11:30–12:00 | `EndCard`: "Next — how one verb becomes three kernels." | — | End. |

### Episode 10 — How one verb becomes three kernels

| Time | On screen | Data source | Transition |
|---|---|---|---|
| 0:00–0:30 | `ColdDropIntro` — `cg` stage glowing. | — | Cut. |
| 0:30–1:30 | `CodePanel`: the `verb Strike { … }` declaration from `duel_1v1.sim`. ~8 lines. Frame: "This single declaration becomes three things." | `duel_1v1.sim` source. | Cut. |
| 1:30–3:00 | Verb expansion: three CG-IR nodes branch off the verb. Labeled: mask kernel, physics rule, scoring row. `IRGraph` with parentage edges back to the source verb. | `--emit-viz-json cg` for `duel_1v1.sim`. | Continue. |
| 3:00–5:00 | Trace 1 — mask kernel: reads `alive`, `cooldown`, target distance. `CodePanel` highlights each read; `IRGraph` highlights data dependencies. | `cg` JSON, mask sub-graph. | Continue. |
| 5:00–7:00 | Trace 2 — physics rule: consumes `StrikeIssued`, writes `Damaged`. Event flow animated. | `cg` JSON, physics sub-graph. | Continue. |
| 7:00–9:00 | Trace 3 — scoring row: computes utility from `target.hp`. Pair-field access visualized. | `cg` JSON, scoring sub-graph. | Continue. |
| 9:00–10:30 | Step back: all three sub-graphs side-by-side under the source verb. Parentage arrows. | `cg` JSON, full verb expansion. | Cut. |
| 10:30–11:30 | Earned demo: run `duel_1v1`; mark each kernel as it fires (mask, physics, scoring). Color-code the three. | Live run + scheduling JSON. | Fade. |
| 11:30–12:00 | `EndCard`: "Next — catching GPU bugs before they ever run." | — | End. |

### Episode 11 — Catching GPU bugs at compile time

| Time | On screen | Data source | Transition |
|---|---|---|---|
| 0:00–0:30 | `ColdDropIntro` — `well_formed` stage glowing. | — | Cut. |
| 0:30–1:30 | Premise: GPU stack traces don't exist. Compile-time checking saves you from runtime mystery. | — | Cut. |
| 1:30–3:00 | `cg/well_formed.rs` line count visible (5K+ lines). Category breakdown: types, dataflow, dispatch shape. | Source. | Continue. |
| 3:00–5:00 | Example 1 — read-after-write across kernels. Broken `.sim`; compiler rejects; well_formed check fires; error message with span-arrow. | Synthetic broken input + `--emit-viz-json well_formed`. | Continue. |
| 5:00–7:00 | Example 2 — incompatible reduction (float-associativity hazard). Tease P11. | Synthetic broken input. | Continue. |
| 7:00–9:00 | Example 3 — missing event handler. Verb emits `Damaged`; no physics consumes `Damaged`. Compile error. | Synthetic broken input. | Cut. |
| 9:00–10:30 | The "green bar": all well_formed checks passing for `duel_1v1`. Per-check bitmap, green wave fills across. | `well_formed` JSON for `duel_1v1`. | Continue. |
| 10:30–11:30 | Earned demo: intentionally break `duel_1v1.sim`; watch compilation fail with helpful error. Spans matter (callback to E6). | Live edit + compile failure. | Fade. |
| 11:30–12:00 | `EndCard`: "Next — the marquee episode: three loops, one kernel." | — | End. |

### Episode 12 — Three loops, one kernel (marquee)

| Time | On screen | Data source | Transition |
|---|---|---|---|
| 0:00–0:30 | `ColdDropIntro` — `schedule` stage glowing. Thesis explicitly framed "wait for it." | — | Cut. |
| 0:30–2:00 | Naive code: three separate kernels, three GPU dispatches. Timeline at bottom shows three launch overheads. | `cg` JSON pre-fusion. | Continue. |
| 2:00–4:00 | Observation: kernel A's output feeds B feeds C, all over the same agent set. Data-flow arrows highlight. | `cg` JSON. | Continue. |
| 4:00–6:00 | Fusion conditions: data-dependent shape, same iteration domain, no aliasing. Each condition shown with a checkmark or X on the example. | — | Continue. |
| 6:00–8:00 | `FusionDiagram` plays: three boxes physically merge into one fused kernel. WGSL pane below shrinks line-by-line as redundant reads/writes vanish. | `--emit-viz-json schedule` before/after. | Continue. |
| 8:00–10:00 | Before/after comparison: 3 kernels × ~50 lines → 1 kernel × ~80 lines. Run on real fixture; time the difference. | Real WGSL pre/post + timing measurement. | Continue. |
| 10:00–11:30 | Earned demo: real fusion-pass output for a non-trivial fixture (e.g. `boids.sim`'s N²-fold or `dungeon_horde`). Show real fusion deltas. | `schedule` JSON. | Fade. |
| 11:30–12:00 | `EndCard`: "Next — running the same source on two backends, bit-for-bit identical." | — | End. |

### Episode 13 — Same source, two backends, bit-for-bit identical

| Time | On screen | Data source | Transition |
|---|---|---|---|
| 0:00–0:30 | `ColdDropIntro` — `emit` stage glowing. | — | Cut. |
| 0:30–1:30 | Premise: one CG-IR emits both Rust (`SerialBackend`) and WGSL (`GpuBackend`). Different code, same semantics. | — | Cut. |
| 1:30–3:00 | The emit pass walks the CG-IR generating code per target. `SourceToOutput`: CG node on left, Rust + WGSL outputs on right (split). | `--emit-viz-json emit`. | Continue. |
| 3:00–5:00 | `BackendSplit`: same fixture running on CPU and GPU. Per-tick state diff stays zero. | Real parity test run. | Continue. |
| 5:00–7:00 | Why parity is hard: float associativity (sums in different orders), atomic-append ordering, RNG. Three drift hazards as cards. | — | Cut. |
| 7:00–9:00 | Three mechanisms: sort-then-fold (animated); per-tick `seq` for atomic order (animated); keyed PCG for RNG (`per_agent_u32(seed, agent_id, tick, purpose)` highlighted). | `crates/engine` PCG source. | Continue. |
| 9:00–10:30 | The parity test: `tests/parity_*.rs` running. Same seed, same fixture, both backends, diff is zero. | `cargo test` output. | Continue. |
| 10:30–11:30 | Earned demo: 1000-tick run, real-time diff counter stays at 0. Visible perf comparison side-by-side. | Live run. | Fade. |
| 11:30–12:00 | `EndCard`: "Next — the last piece: shipping the compiler." | — | End. |

### Episode 14 — How a schema hash stops the build

| Time | On screen | Data source | Transition |
|---|---|---|---|
| 0:00–0:30 | `ColdDropIntro` — full `PipelineDiagram` lit (we're past the pipeline, into release). | — | Cut. |
| 0:30–1:30 | Problem: someone changes `SimState`'s layout. Old snapshots and traces are now incompatible. Visualization: snapshot from yesterday tries to load — fails — corrupt state. | Recording of the failure mode. | Cut. |
| 1:30–3:00 | Schema hash: SHA-256 of the canonical layout description. Stored in `.schema_hash`. Compared on every build. Show the file. | `crates/engine/.schema_hash`. | Continue. |
| 3:00–5:00 | Demo: edit a field in `SimState`. Run the build. `schema_hash` test fails. Span-arrow to the diff. | Live edit + test run. | Continue. |
| 5:00–7:00 | The ritual: regen the hash, bump downstream consumers, document in commit. Step-by-step. | Procedure from `engine.md`. | Cut. |
| 7:00–9:00 | `build.rs` integration: every `*_runtime` crate has a 1-line build.rs calling `dsl_compiler::build_helper::emit`. Show the file. | `crates/*_runtime/build.rs`. | Continue. |
| 9:00–10:00 | Python dataclasses + traces — the 4th artefact. External training scripts consume the same compiled output. Show real pytorch dataset code. | `--emit-viz-json emit` (Python). | Cut. |
| 10:00–11:00 | Series recap: complete four-artefact diagram. Quick montage of signature visualizations from episodes 1–13. | Library of past frames. | Continue. |
| 11:00–11:30 | What's next: the `.ability` DSL, the train operator, follow-up territory. Open door — no commitment. | — | Fade. |
| 11:30–12:00 | `EndCard`: final, with subscribe overlay and acknowledgments. | — | End. |

---

## 6. Open questions

- **Voice-over workflow.** Narration recorded before or after Motion Canvas scenes are timed? Affects whether scene durations adjust to narration or narration is paced to scene cuts. Recommend recording rough scratch narration first, finalizing after scene timing is locked.
- **Music bed.** Whether to use any. Recommend none for v1 — narration + ambient sim audio is enough for compiler content. Music can be added per episode if a beat needs it.
- **Episode 1 hook.** Whether to lead with `megaswarm_10000` (most visceral) or `duel_25v25` (more legible per agent). Recommend `megaswarm` for the first 5 seconds, transition to a duel within the first minute when the human-scale matters.
- **`.ability` DSL inclusion.** Currently scoped out of v1. Revisit when the series has data on viewership / reception.
- **Series title.** Not picked. Candidates: *"How a Sim Compiles,"* *"The Compiler Behind the Game,"* *"From .sim to GPU."* Defer to the user.
- **Channel branding.** Out of scope for this design; existing channel assets (if any) determine.

---

## 7. What this design doesn't cover

- Narration scripts (deliberately the user's domain).
- Marketing / SEO / channel strategy.
- Audio recording setup.
- Implementation plan for `--emit-viz-json` (that becomes a separate plan under `docs/superpowers/plans/`; the writing-plans skill will produce it after this design is approved).
- The Motion Canvas codebase itself (likewise a separate plan).
