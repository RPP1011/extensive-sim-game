# Parser Pipeline

The ability DSL parser is built with [winnow](https://docs.rs/winnow), a parser
combinator library in the tradition of nom but with better error messages and
a simpler API.

## Pipeline Stages

```
Input text ──▶ Parser ──▶ AST ──▶ Lower ──▶ AbilityDef
                                            PassiveDef
```

### Stage 1: Parsing (`parser.rs`)

The top-level entry point:

```rust
pub fn parse_ability_file(input: &mut &str)
    -> PResult<Vec<AstItem>>
```

This parses an entire `.ability` file into a list of `AstItem` nodes — either
ability or passive declarations. The parser handles:

- Whitespace and comment stripping
- Header field parsing (target, range, cooldown, etc.)
- Effect statement parsing (delegated to `parse_effects.rs`)
- Delivery block parsing (delegated to `parse_delivery.rs`)
- Tag parsing (`[KEY: value, ...]`)
- Condition parsing (`when ...`)
- Area parsing (`in circle(...)`)

### Stage 2: AST (`ast.rs`)

The abstract syntax tree preserves the structure of the source without
interpreting it:

```rust
pub enum AstItem {
    Ability(AstAbility),
    Passive(AstPassive),
}

pub struct AstAbility {
    pub name: String,
    pub header: Vec<AstHeaderField>,
    pub body: Vec<AstStatement>,
}
```

The AST is a faithful representation of the source text structure, not the
final semantic model. For example, `cooldown: 5s` is stored as a header field
with the string `"5s"`, not yet converted to `5000u32`.

### Stage 3: Lowering (`lower.rs`)

The lowering pass converts the AST into the runtime types:

```rust
pub fn lower_file(file: &[AstItem])
    -> Result<(Vec<AbilityDef>, Vec<PassiveDef>), String>
```

Lowering performs:
- Duration string parsing (`"5s"` → `5000u32`, `"300ms"` → `300u32`)
- Target mode string mapping (`"enemy"` → `AbilityTargeting::TargetEnemy`)
- Effect type resolution (delegated to `lower_effects.rs`)
- Delivery type resolution (delegated to `lower_delivery.rs`)
- Validation (e.g., range is required for `TargetEnemy`)

### Stage 4: Emission (`emit.rs`)

The emitter goes the other direction — from `AbilityDef` back to DSL text:

```rust
pub fn emit_ability(def: &AbilityDef) -> String
```

This enables:
- **Round-trip testing** — parse → lower → emit → parse again, verify equality
- **Code generation** — programmatically create abilities and emit them as DSL
- **Pretty-printing** — normalize formatting of hand-written ability files

## Module Inventory

```
src/ai/effects/dsl/
├── parser.rs          # Top-level parser (ability/passive declarations)
├── ast.rs             # AST types
├── lower.rs           # AST → AbilityDef/PassiveDef
├── lower_effects.rs   # Effect lowering
├── lower_delivery.rs  # Delivery lowering
├── emit.rs            # AbilityDef → DSL text
├── emit_effects.rs    # Effect emission
├── emit_helpers.rs    # Emission utilities
├── parse_effects.rs   # Effect statement parser
├── parse_delivery.rs  # Delivery block parser
├── error.rs           # Error types with line/col info
├── tests.rs           # Unit tests
├── tests_roundtrip.rs # Round-trip (parse → emit → parse) tests
├── fuzz.rs            # Proptest fuzz targets
├── fuzz_generators.rs # Random ability generators
└── fuzz_roundtrip.rs  # Fuzz round-trip testing
```

## Error Reporting

Parse errors include line, column, and context:

```rust
pub struct DslError {
    pub message: String,
    pub line: usize,
    pub col: usize,
    pub context: Option<String>,
}
```

The `error.rs` module provides `offset_to_line_col()` to convert byte offsets
(from winnow) into human-readable line/column numbers, plus `extract_line()` to
show the offending source line.

## Property-Based Testing

The parser is tested with [proptest](https://docs.rs/proptest) for fuzzing:

- `fuzz_generators.rs` generates random valid `AbilityDef` structs
- `fuzz.rs` emits them to DSL text and verifies parsing succeeds
- `fuzz_roundtrip.rs` verifies that parse → emit → parse is idempotent

This catches edge cases that hand-written tests miss: unusual tag combinations,
extreme numeric values, deeply nested delivery blocks, etc.

```bash
# Run fuzz tests
cargo test fuzz -- --test-threads=1
```
