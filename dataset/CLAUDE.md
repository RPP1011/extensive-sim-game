# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## What this directory is

`dataset/` is a 770-file grab-bag of TOML/JSON/`.ability` data fixtures. Only a
small slice of it is actually read by code in `crates/` today. Most of the
rest is either (a) input to ML training binaries that are **excluded** from
the default workspace build (`ability-vae`, `ability_operator`,
`combat-trainer`, `world_sim_bench` — see root `Cargo.toml` `[workspace]
exclude`), or (b) legacy game-design content with no code reference found
anywhere under `crates/`. There is no README anywhere inside `dataset/` —
this file is the only orientation document. Treat every claim below as
grep-verified against `crates/**/*.rs`, not inferred from directory names.

## Path-claim correction for the root CLAUDE.md

The root `F:\Game\extensive-sim-game\CLAUDE.md` says the retired hero-template
layer lived at **`assets/hero_templates/`**. That path does not exist —
`assets/` only contains `ability_test/`, `config/`, `models/`, `sim/`. The
real hero-template directories are under `dataset/`:

- `dataset/hero_templates/` (38 files, top-level, flat) — referenced only in
  a doc-comment in `crates/dsl_ast/tests/ability_parser.rs` ("a real ability
  from `dataset/hero_templates/warrior.ability`"); the test itself uses an
  inlined string literal copied from that file, it does **not** read the file
  from disk. `dataset/hero_templates/warrior.ability` and
  `dataset/abilities/hero_templates/warrior.ability` are byte-identical
  (`diff` confirms).
- `dataset/abilities/hero_templates/` (65 files, nested inside `abilities/`)
  — this is the one actually read from disk, by
  `crates/combat-trainer/src/main.rs` (`dataset/abilities/hero_templates`)
  and by `crates/ability-vae`'s training binaries (which walk all of
  `dataset/abilities` recursively). Both `combat-trainer` and `ability-vae`
  are excluded from the default workspace, so this is not exercised by
  `cargo test`.

**Recommend fixing the root CLAUDE.md line to read `dataset/hero_templates/`
(or more precisely `dataset/abilities/hero_templates/`) instead of
`assets/hero_templates/`.**

## `dataset/abilities/lol_heroes/` — the one directory with real default-build coverage

344 files = 172 `.ability` + 172 `.toml` pairs, one pair per League of
Legends champion (`Aatrox.ability`/`Aatrox.toml`, `Ahri.ability`/`Ahri.toml`,
…). The `.ability` files are hand/LLM-authored translations of each
champion's kit into this engine's ability DSL; the `.toml` files carry
matching hero stat blocks (`[hero]`, `[stats]`, `[attack]`, `[[abilities]]`).

This is consumed directly by tests in workspace-member (default-build)
crates — confirmed by grep:

- `crates/dsl_ast/tests/ability_parser_wave_1_4.rs`, `ability_parser_wave_1_5.rs`,
  `ability_corpus_round_trip.rs` — walk
  `<workspace>/dataset/abilities/lol_heroes/*.ability` and assert every file
  parses (round-trips) under the current DSL parser. This is the parser's
  regression corpus: as DSL grammar waves land, these tests catch real-world
  syntax the hand-written unit tests don't cover.
- `crates/dsl_compiler/tests/lol_corpus_lowering.rs` — same directory, but
  exercises the AST-to-lowered-IR pass (`ast → lower`) over the whole corpus.
- `crates/engine/tests/packed_registry_lol_corpus.rs` — same directory again,
  pushed through the packed-registry emission path used at runtime.
- `crates/dsl_ast/tests/ability_corpus_smoke.rs` — different scope: this one
  walks **all of `dataset/`** (not just `lol_heroes/`) for any `.ability`
  file and reports a pass/fail tally. It's `#[ignore]`d, so it does not run
  under plain `cargo test`; run explicitly with
  `cargo test -p dsl_ast --test ability_corpus_smoke -- --ignored --nocapture`.

So: `lol_heroes/` is load-bearing parser-regression fixture data, actively
exercised by `cargo test` in the default workspace. It is also
League-of-Legends-derived content (champion names, kit shapes) — see the
game-specific flag below.

`dataset/lol_champions/*.json` (173 files, one per champion) is a separate,
much richer dataset — raw scraped LoL Wiki data (passive/Q/W/E/R
descriptions, wikitext, tooltips) per champion. No reference to
`dataset/lol_champions` was found anywhere in `crates/`; it looks like the
source material `lol_heroes/*.ability` was hand/LLM-translated from, kept
around but not read by any current code path.

## Subdirectory inventory: live vs. orphaned

Grepped `crates/**/*.rs` for `dataset/<subdir>` (and bare subdir-name path
segments) for every top-level entry. Result:

| Subdirectory | Files | Live in default workspace? | Notes |
|---|---|---|---|
| `abilities/lol_heroes/` | 344 | **Yes** — `dsl_ast`, `dsl_compiler`, `engine` tests (see above) | parser regression corpus |
| `abilities/hero_templates/` | 65 | Only via excluded ML crates (`combat-trainer`, `ability-vae`) | original fantasy-RPG class kits (warrior, mage, cleric, …) |
| `abilities/champion_templates/` | 6 | Only via excluded `ability-vae` (walks `dataset/abilities` recursively) | 3 original bespoke champions (dax_the_unbreaking, kira_long_shadow, sera_ironfang), `.ability`+`.toml` pairs |
| `abilities/classes/` | 62 | Only via excluded ML crates | `.class` files, generic class archetypes |
| `abilities/campaign/` | 10 | Only via excluded ML crates | `.ability` files tagged for non-combat campaign systems (diplomacy, economy, stealth, …) |
| `abilities/generated/` | 208 | Only via excluded ML crates | `gen_NNNN.ability` — synthetic/model-generated ability samples |
| `abilities/tier1_instant/` … `tier10_kits/` | 89 total | Only via excluded ML crates | generic hand-authored training fixtures graded by mechanical complexity, tagged in `dataset/manifest.toml` |
| `hero_templates/` (top-level) | 38 | No live reader found (see path-correction section above) | duplicate/older copy of a subset of `abilities/hero_templates/` |
| `achievements/` | 10 | **No reference found in `crates/`** | orphaned; achievement/progression design TOML (combat, diplomacy, economy, crisis, class_progression) |
| `campaign/` (top-level) | 197 | **No reference found in `crates/`** | orphaned; campaign choice-event templates (`choice_templates/*.toml` — faction negotiation, crisis response, etc.) with narrative text, options, and effects |
| `champion_scenarios/` | 13 | **No reference found in `crates/`** | orphaned; scenario TOMLs pitting hero-template parties against the bespoke champions from `abilities/champion_templates/` (e.g. `4v1_vs_dax.toml`) |
| `classes/` (top-level) | 31 | **No reference found in `crates/`** | orphaned; class-design TOMLs (alchemist, architect, artisan, bard, betrayer, …) |
| `entities/` | 22 | **No reference found in `crates/`** | orphaned; unit/entity stat TOMLs (archer, archmage, assassin, builder, climber, …) |
| `environments/` | 6 | **No reference found in `crates/`** | orphaned; `terrains/*.toml` + `scenarios/*.toml` (env-flavored, distinct from top-level `scenarios/`) |
| `fonts/` (`DejaVuSans.ttf`) | 1 | **No reference found in `crates/`** | orphaned; likely used by an external/Python plotting script, not by any Rust crate |
| `heroes/` | 35 | **No reference found in `crates/`** | orphaned; tiered hero stat TOMLs (`tier1_autoattack/archer.toml`, etc.) |
| `lol_champions/` | 173 | **No reference found in `crates/`** | orphaned but see note above — likely the scrape source for `abilities/lol_heroes/`; raw LoL Wiki JSON, third-party IP |
| `maps/` | 75 | **No reference found in `crates/`** | orphaned; `buildings/*.toml` by category (academic, defensive, economic, military, production) |
| `scenarios/` (top-level) | 6,491 | **No reference found in `crates/`** | by far the largest subdir — `curriculum/phase1-4`, `drills/phase1-5`, `mirror/`, `tier1/`, `tier2/`; all `.toml` scenario configs (`hero_templates=[...]`, seed, room_type, etc.), presumably for an external/excluded RL training pipeline, not current `crates/` code |
| `siege/` | 119 | **No reference found in `crates/`** | orphaned; siege-equipment TOMLs (ballista, battering_ram, arcane_cannon, …) |
| `stsb/` | 2 | Only via excluded `ability-vae` (`train_text_encoder.rs` downloads/reads `stsbenchmark.tsv`) | generic STS-Benchmark sentence-similarity data for text-encoder training, unrelated to game content |
| loose files: `ability_descriptions*.jsonl`, `twi_skills.json`, `manifest.toml` | 5 | `ability_descriptions*.jsonl` and `twi_skills.json` read only by excluded `ability-vae` binaries; `manifest.toml` (ability tag index) — **no reader found anywhere in `crates/`** | |

Bottom line: from a plain `cargo build` / `cargo test` in this workspace, the
**only** directory under `dataset/` that matters is `abilities/lol_heroes/`
(plus, if you explicitly run the ignored smoke test, the whole `.ability`
corpus tree gets syntax-checked). Everything else is either ML-training input
reachable only through the excluded `ability-vae` / `ability_operator` /
`combat-trainer` / `world_sim_bench` crates, or has no code reference at all
in `crates/` and should be treated as legacy/orphaned design content pending
cleanup.

## Game-specific vs. generic-engine content (for the planned engine/game-logic split)

Everything in `dataset/` is arguably "content" rather than "engine," but some
of it is generic mechanical test fixture material and some is unmistakably
themed game content:

- **Overtly game/IP-specific** (would need to leave an "engine" split
  entirely, or move behind a clearly-marked game-content boundary):
  - `abilities/lol_heroes/*` and `lol_champions/*.json` — League of Legends
    champion names, kits, and wiki text. Third-party IP, used purely as a
    parser stress-test corpus; not original content.
  - `abilities/champion_templates/*` and `champion_scenarios/*` — original
    bespoke named champions (Dax the Unbreaking, Kira Long Shadow, Sera
    Ironfang) and scenarios built around them.
  - `campaign/*` (top-level) and `achievements/*` — narrative/campaign
    systems (faction diplomacy, crisis events, guild alliances) with
    flavor text — clearly game-design content, not engine mechanics.
  - `siege/*`, `maps/*` — themed equipment/building catalogs (ballista,
    arcane cannon, academic/military/economic buildings) — setting-specific.
- **Generic-flavored but still class/hero-shaped** (fantasy RPG archetype
  names — warrior/mage/cleric/knight/etc. — reusable as *a* game's content
  but not engine-neutral test fixtures either):
  - `abilities/hero_templates/`, `hero_templates/` (top-level), `heroes/`,
    `entities/`, `classes/` (top-level), `abilities/classes/`.
- **Closer to genuinely generic engine/mechanical fixtures** (no thematic
  naming, organized purely by mechanical shape/tier):
  - `abilities/tier1_instant/` … `tier10_kits/` — files named by mechanic
    (`damage.ability`, `buff_debuff.ability`, `cc.ability`) not by
    character/setting.
  - `abilities/generated/` — synthetic model output, mechanically random.
  - `scenarios/` (top-level, 6,491 files) — parameterized by
    `hero_templates=[...]` lists (still points at the fantasy archetype
    names above) plus seed/tick/room_type — mechanical in shape but its
    content vector still names the fantasy hero templates.
  - `stsb/` — wholly generic NLP benchmark data, no game content at all.

If the engine/game-logic split is meant to strip Riot-IP-adjacent and
setting-specific content out of the tree, `abilities/lol_heroes/`,
`lol_champions/`, `champion_scenarios/`, `abilities/champion_templates/`,
`campaign/`, and `achievements/` are the clearest candidates. Note that
`abilities/lol_heroes/` is also the one directory with live default-build
test coverage (see above) — removing or relocating it requires updating the
four test files listed in that section, not just deleting the corpus.
