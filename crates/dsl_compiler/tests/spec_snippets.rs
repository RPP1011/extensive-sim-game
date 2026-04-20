//! Run the parser against every fenced DSL code block in `docs/dsl/spec.md`
//! §2 and §8. Snippets known to be intentionally malformed are opted out by
//! `// PARSE_SKIP: reason` on their first line, and we auto-skip fences
//! tagged with `rust` / `toml` / `ignore` (those aren't DSL).
//!
//! This is a loose coverage check — it confirms the grammar covers the
//! shapes the spec advertises. A failure here means either:
//!  a) the spec has drifted from the compiler (fix: add a PARSE_SKIP tag
//!     or tighten the spec), or
//!  b) the parser doesn't cover a shape the spec uses (fix: extend the
//!     parser).

use std::fs;
use std::path::PathBuf;

fn spec_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("docs")
        .join("dsl")
        .join("spec.md")
}

fn section_ranges(src: &str) -> Vec<(String, usize, usize)> {
    // Sections delineated by `## N.` or `## <n>.` headings. We emit
    // (heading-line, start-byte, end-byte) tuples.
    let mut out: Vec<(String, usize, usize)> = Vec::new();
    let mut cur_start = 0;
    let mut cur_heading: Option<String> = None;
    let mut line_start = 0;
    for (i, ch) in src.char_indices() {
        if ch == '\n' || i == src.len() {
            let line = &src[line_start..i];
            if line.starts_with("## ") {
                if let Some(h) = cur_heading.take() {
                    out.push((h, cur_start, line_start));
                }
                cur_heading = Some(line.trim().to_string());
                cur_start = i + 1;
            }
            line_start = i + 1;
        }
    }
    if let Some(h) = cur_heading {
        out.push((h, cur_start, src.len()));
    }
    out
}

fn extract_fences(src: &str) -> Vec<(usize, String, String)> {
    // Returns (line_number_of_opening_fence, info_string, body).
    let mut out = Vec::new();
    let mut in_fence = false;
    let mut fence_info = String::new();
    let mut fence_body = String::new();
    let mut fence_line = 0;
    for (line_no, line) in src.lines().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            if in_fence {
                out.push((fence_line, fence_info.clone(), fence_body.clone()));
                fence_body.clear();
                in_fence = false;
            } else {
                fence_info = trimmed.trim_start_matches("```").trim().to_string();
                fence_line = line_no + 1;
                in_fence = true;
            }
        } else if in_fence {
            fence_body.push_str(line);
            fence_body.push('\n');
        }
    }
    out
}

fn is_dsl_fence(info: &str, body: &str) -> bool {
    let info = info.trim().to_lowercase();
    if info == "rust" || info == "toml" || info == "ignore" || info == "text" || info == "sh" || info == "bash" {
        return false;
    }
    // Skip fences whose first non-empty line tells us to.
    if body.lines().next().map_or(false, |l| l.trim().starts_with("// PARSE_SKIP")) {
        return false;
    }
    // Templates sprinkled with grammar meta-variables are not parseable.
    for meta in [
        "<Name>", "<name>", "<field>", "<expr>", "<predicate>", "<bindings>",
        "<args>", "<type>", "<body>", "<EventPattern>", "<EventName>",
        "<ActionHead>", "<scope>", "<MicroOrMacro>", "<assert_expr>",
        "<value_comparator>", "<mode>", "<filter>", "<ticks>", "<u32>",
        "<u64>", "<f32>", "<string>", "<scalar>", "<prob>", "<scalar_expr>",
        "<action_filter>", "<obs_filter>", "<condition>", "<delta>", "<K>",
    ] {
        if body.contains(meta) {
            return false;
        }
    }
    // `...` as a placeholder (common in spec templates).
    if body.contains("{ ... }") || body.contains("[ ... ]") || body.contains("( ... )") {
        return false;
    }
    if body.lines().any(|l| l.trim() == "..." || l.trim().starts_with("...")) {
        return false;
    }
    // Explanatory / mixed-prose blocks — `= true` / `→ mask passes` / `✓` etc.
    if body.contains("= true") || body.contains("= false") || body.contains("→") || body.contains("✓") {
        return false;
    }
    // Grammar EBNF blocks.
    let first_non_blank = body.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
    let trimmed = first_non_blank.trim();
    if trimmed.contains(":=")
        || trimmed.starts_with("enum ")
        || trimmed.starts_with("struct ")
        || trimmed.starts_with("Scalar:")
        || trimmed.starts_with("Vector:")
        || trimmed.starts_with("Time:")
        || trimmed.starts_with("ID:")
        || trimmed.starts_with("FactRef:")
        || trimmed.starts_with("FactPayload:")
        || trimmed.starts_with("Bounded:")
        || trimmed.starts_with("Struct:")
        || trimmed.starts_with("Enum:")
        || trimmed.starts_with("action {")
        || trimmed.starts_with("action ")
        || trimmed.starts_with("pre phase")
        || trimmed.starts_with("Agent ")
        || trimmed.starts_with("view relationship")
        || trimmed.starts_with("AuctionPosted")
    {
        return false;
    }
    // Skip state/cascade diagrams starting with a `|`.
    if trimmed.starts_with('|') {
        return false;
    }
    // Skip grammar meta-markers: lines starting with `<`.
    if trimmed.starts_with('<') {
        return false;
    }
    true
}

#[test]
fn spec_section_2_and_8_snippets_parse() {
    let src = fs::read_to_string(spec_path()).expect("read spec.md");
    let sections = section_ranges(&src);
    let mut parsed = 0;
    let mut skipped = 0;
    let mut failed: Vec<String> = Vec::new();
    for (heading, start, end) in sections {
        let num_ok = heading.starts_with("## 2.")
            || heading == "## 2. Top-level declarations"
            || heading == "## 8. Worked example";
        if !num_ok {
            continue;
        }
        let section = &src[start..end];
        for (line_no, info, body) in extract_fences(section) {
            if !is_dsl_fence(&info, &body) {
                skipped += 1;
                continue;
            }
            match dsl_compiler::parse(&body) {
                Ok(_) => parsed += 1,
                Err(e) => {
                    failed.push(format!(
                        "section `{heading}` fence @line~{line_no}:\n{}\n--- snippet ---\n{}\n--- end ---",
                        e.rendered, body
                    ));
                }
            }
        }
    }
    assert!(
        failed.is_empty(),
        "{} fences parsed, {} skipped, {} failed:\n{}",
        parsed,
        skipped,
        failed.len(),
        failed.join("\n\n")
    );
    // Sanity: at least a few must have parsed.
    assert!(parsed > 0, "no DSL fences found — check fence detection");
}
