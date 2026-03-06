use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use base64::Engine;
use serde_json::{json, Value};

const API_ROOT: &str = "https://generativelanguage.googleapis.com/v1beta/models";

#[derive(Debug)]
pub struct GeminiOutputs {
    pub text: Option<String>,
    pub image_bytes: Option<Vec<u8>>,
}

pub fn load_dotenv_if_present(path: &Path) -> Result<(), String> {
    if !path.exists() {
        return Ok(());
    }
    let raw =
        fs::read_to_string(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    for raw_line in raw.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') || !line.contains('=') {
            continue;
        }
        let mut parts = line.splitn(2, '=');
        let key = parts.next().unwrap_or("").trim();
        let value = parts
            .next()
            .unwrap_or("")
            .trim()
            .trim_matches('"')
            .trim_matches('\'');
        if !key.is_empty() && env::var_os(key).is_none() {
            env::set_var(key, value);
        }
    }
    Ok(())
}

pub fn load_prompt(prompt: Option<&str>, prompt_file: Option<&Path>) -> Result<String, String> {
    match (prompt, prompt_file) {
        (Some(_), Some(_)) => Err("Use either --prompt or --prompt-file, not both.".to_string()),
        (Some(text), None) => Ok(text.trim().to_string()),
        (None, Some(path)) => fs::read_to_string(path)
            .map(|s| s.trim().to_string())
            .map_err(|e| format!("failed to read {}: {e}", path.display())),
        (None, None) => Err("You must provide either --prompt or --prompt-file.".to_string()),
    }
}

pub fn call_gemini(model: &str, prompt: &str, api_key: &str) -> Result<Value, String> {
    let payload = json!({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}
    });

    let url = format!("{API_ROOT}/{model}:generateContent");
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| format!("failed to build http client: {e}"))?;

    let response = client
        .post(url)
        .header("x-goog-api-key", api_key)
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .map_err(|e| format!("Gemini API request failed: {e}"))?;

    let status = response.status();
    let body = response
        .text()
        .map_err(|e| format!("failed to read Gemini response body: {e}"))?;

    if !status.is_success() {
        return Err(format!("Gemini API HTTP {}: {}", status.as_u16(), body));
    }

    serde_json::from_str(&body).map_err(|e| format!("failed to parse Gemini JSON: {e}"))
}

pub fn call_gemini_text(model: &str, prompt: &str, api_key: &str) -> Result<Value, String> {
    let payload = json!({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["TEXT"]}
    });

    let url = format!("{API_ROOT}/{model}:generateContent");
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| format!("failed to build http client: {e}"))?;

    let response = client
        .post(url)
        .header("x-goog-api-key", api_key)
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .map_err(|e| format!("Gemini API request failed: {e}"))?;

    let status = response.status();
    let body = response
        .text()
        .map_err(|e| format!("failed to read Gemini response body: {e}"))?;

    if !status.is_success() {
        return Err(format!("Gemini API HTTP {}: {}", status.as_u16(), body));
    }

    serde_json::from_str(&body).map_err(|e| format!("failed to parse Gemini JSON: {e}"))
}

pub fn call_gemini_with_reference_image(
    model: &str,
    prompt: &str,
    api_key: &str,
    reference_image_bytes: &[u8],
    reference_mime_type: &str,
) -> Result<Value, String> {
    if reference_image_bytes.is_empty() {
        return Err("reference image is empty".to_string());
    }

    let encoded_reference =
        base64::engine::general_purpose::STANDARD.encode(reference_image_bytes);
    let payload = json!({
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inlineData": {
                        "mimeType": reference_mime_type,
                        "data": encoded_reference
                    }
                }
            ]
        }],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}
    });

    let url = format!("{API_ROOT}/{model}:generateContent");
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| format!("failed to build http client: {e}"))?;

    let response = client
        .post(url)
        .header("x-goog-api-key", api_key)
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .map_err(|e| format!("Gemini API request failed: {e}"))?;

    let status = response.status();
    let body = response
        .text()
        .map_err(|e| format!("failed to read Gemini response body: {e}"))?;

    if !status.is_success() {
        return Err(format!("Gemini API HTTP {}: {}", status.as_u16(), body));
    }

    serde_json::from_str(&body).map_err(|e| format!("failed to parse Gemini JSON: {e}"))
}

pub async fn call_gemini_async(model: &str, prompt: &str, api_key: &str) -> Result<Value, String> {
    let payload = serde_json::json!({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}
    });

    let url = format!("{API_ROOT}/{model}:generateContent");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| format!("failed to build async http client: {e}"))?;

    let response = client
        .post(url)
        .header("x-goog-api-key", api_key)
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await
        .map_err(|e| format!("Gemini API request failed: {e}"))?;

    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|e| format!("failed to read Gemini response body: {e}"))?;

    if !status.is_success() {
        return Err(format!("Gemini API HTTP {}: {}", status.as_u16(), body));
    }

    serde_json::from_str(&body).map_err(|e| format!("failed to parse Gemini JSON: {e}"))
}

pub fn extract_parts(response_json: &Value) -> Result<GeminiOutputs, String> {
    let mut text_chunks: Vec<String> = Vec::new();
    let mut image_bytes: Option<Vec<u8>> = None;

    let candidates = response_json
        .get("candidates")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let Some(first) = candidates.first() else {
        return Ok(GeminiOutputs {
            text: None,
            image_bytes: None,
        });
    };

    let parts = first
        .get("content")
        .and_then(|v| v.get("parts"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    for part in parts {
        if let Some(text) = part.get("text").and_then(Value::as_str) {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                text_chunks.push(trimmed.to_string());
            }
            continue;
        }

        let inline = part
            .get("inlineData")
            .or_else(|| part.get("inline_data"))
            .and_then(Value::as_object)
            .cloned();

        if let Some(inline_obj) = inline {
            if let Some(data) = inline_obj.get("data").and_then(Value::as_str) {
                let decoded = base64::engine::general_purpose::STANDARD
                    .decode(data)
                    .map_err(|e| format!("failed to decode Gemini inline image: {e}"))?;
                image_bytes = Some(decoded);
            }
        }
    }

    let text = if text_chunks.is_empty() {
        None
    } else {
        Some(text_chunks.join("\n"))
    };

    Ok(GeminiOutputs { text, image_bytes })
}

pub fn write_outputs(
    out_path: &Path,
    image_bytes: &[u8],
    text: Option<&str>,
    save_text: bool,
) -> Result<(), String> {
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
    }
    fs::write(out_path, image_bytes)
        .map_err(|e| format!("failed to write {}: {e}", out_path.display()))?;
    println!("Saved image: {}", out_path.display());

    if save_text {
        if let Some(t) = text {
            let mut text_path = PathBuf::from(out_path);
            let suffix = out_path
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| format!("{s}.txt"))
                .unwrap_or_else(|| "txt".to_string());
            text_path.set_extension(suffix);
            fs::write(&text_path, format!("{t}\n"))
                .map_err(|e| format!("failed to write {}: {e}", text_path.display()))?;
            println!("Saved text:  {}", text_path.display());
        }
    }

    Ok(())
}
