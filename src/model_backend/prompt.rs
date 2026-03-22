/// Utility helpers for building prompts sent to the model backend.

/// Wrap a generation request with system context and output format instructions.
pub fn format_json_generation_prompt(
    system_context: &str,
    user_instruction: &str,
    output_schema_hint: &str,
) -> String {
    let mut prompt = String::with_capacity(
        system_context.len() + user_instruction.len() + output_schema_hint.len() + 128,
    );
    prompt.push_str("### System\n");
    prompt.push_str(system_context);
    prompt.push_str("\n\n### Instruction\n");
    prompt.push_str(user_instruction);
    prompt.push_str("\n\n### Output Format\nRespond with valid JSON matching this schema:\n");
    prompt.push_str(output_schema_hint);
    prompt.push_str("\n\n### Response\n");
    prompt
}

/// Build a simple text generation prompt (no JSON schema).
pub fn format_text_prompt(system_context: &str, user_instruction: &str) -> String {
    let mut prompt =
        String::with_capacity(system_context.len() + user_instruction.len() + 64);
    prompt.push_str("### System\n");
    prompt.push_str(system_context);
    prompt.push_str("\n\n### Instruction\n");
    prompt.push_str(user_instruction);
    prompt.push_str("\n\n### Response\n");
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_prompt_contains_all_sections() {
        let prompt = format_json_generation_prompt("ctx", "do thing", r#"{"name": "string"}"#);
        assert!(prompt.contains("### System\nctx"));
        assert!(prompt.contains("### Instruction\ndo thing"));
        assert!(prompt.contains(r#"{"name": "string"}"#));
    }

    #[test]
    fn text_prompt_contains_sections() {
        let prompt = format_text_prompt("ctx", "generate text");
        assert!(prompt.contains("### System\nctx"));
        assert!(prompt.contains("### Instruction\ngenerate text"));
        assert!(prompt.contains("### Response"));
    }
}
