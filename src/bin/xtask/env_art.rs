use std::fs;
use std::path::Path;
use std::process::ExitCode;

use bevy_game::mapgen_gemini;
use futures::future::join_all;
use hnsw_rs::prelude::*;
use serde::{Deserialize, Serialize};

use super::cli::{
    EnvArtStyle, MapEnvArtBuildIndexArgs, MapEnvArtCommand, MapEnvArtGenerateArgs,
    MapEnvArtQueryArgs, MapEnvArtSubcommand,
};

// ---------------------------------------------------------------------------
// Persistent index types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EnvPromptRow {
    id: String,
    title: String,
    tags: Vec<String>,
    prompt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EnvArtPersistentIndex {
    version: u32,
    embedding_model: String,
    corpus_path: String,
    corpus_fingerprint: u64,
    vector_dim: usize,
    rows: Vec<EnvPromptRow>,
    vectors: Vec<Vec<f32>>,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn run_map_env_art(args: MapEnvArtCommand) -> ExitCode {
    match args.command {
        MapEnvArtSubcommand::BuildIndex(build) => run_map_env_art_build_index(build),
        MapEnvArtSubcommand::Query(query) => run_map_env_art_query(query),
        MapEnvArtSubcommand::Generate(generate) => run_map_env_art_generate(generate),
    }
}

fn run_map_env_art_build_index(args: MapEnvArtBuildIndexArgs) -> ExitCode {
    let api_key = match load_gemini_api_key() {
        Ok(v) => v,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(2);
        }
    };
    let built = match load_or_build_env_art_index(&args.corpus, &args.index, &api_key, true) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    println!(
        "Index ready: {} vectors, dim {}, file {}",
        built.rows.len(),
        built.vector_dim,
        args.index.display()
    );
    ExitCode::SUCCESS
}

fn run_map_env_art_query(args: MapEnvArtQueryArgs) -> ExitCode {
    let api_key = match load_gemini_api_key() {
        Ok(v) => v,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(2);
        }
    };

    let index = match load_or_build_env_art_index(
        &args.corpus,
        &args.index,
        &api_key,
        args.refresh_index,
    ) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    let query_embedding = match embed_text(&args.query, &api_key) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    let hits = search_corpus_with_hnsw(&index.vectors, &query_embedding, args.top_k);
    println!("Query: {}", args.query);
    for (rank, neighbor) in hits.iter().enumerate() {
        let row = &index.rows[neighbor.d_id];
        println!(
            "#{rank_idx} distance={distance:.4} id={} :: {}",
            row.id,
            row.title,
            rank_idx = rank + 1,
            distance = neighbor.distance
        );
        println!("tags={}", row.tags.join(", "));
        println!("prompt={}", row.prompt);
        println!();
    }
    ExitCode::SUCCESS
}

fn run_map_env_art_generate(args: MapEnvArtGenerateArgs) -> ExitCode {
    let api_key = match load_gemini_api_key() {
        Ok(v) => v,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(2);
        }
    };

    let index = match load_or_build_env_art_index(
        &args.corpus,
        &args.index,
        &api_key,
        args.refresh_index,
    ) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    let query_embedding = match embed_text(&args.query, &api_key) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    let hits = search_corpus_with_hnsw(&index.vectors, &query_embedding, args.top_k);
    if hits.is_empty() {
        eprintln!("No semantic hits found for query '{}'.", args.query);
        return ExitCode::from(1);
    }

    if let Err(err) = fs::create_dir_all(&args.out_dir) {
        eprintln!("Failed to create {}: {err}", args.out_dir.display());
        return ExitCode::from(1);
    }

    // Build the per-request metadata up front so we can fan out concurrently.
    struct RequestMeta {
        idx: usize,
        id: String,
        title: String,
        tags: Vec<String>,
        distance: f32,
        composed_prompt: String,
        stem: String,
    }

    let request_metas: Vec<RequestMeta> = hits
        .into_iter()
        .take(args.count)
        .enumerate()
        .map(|(idx, neighbor)| {
            let row = &index.rows[neighbor.d_id];
            let composed_prompt = compose_environment_prompt(&row.prompt, args.style);
            let stem = format!(
                "{:02}_{}_{}",
                idx + 1,
                sanitize_stem(&row.id),
                args.style.as_slug()
            );
            RequestMeta {
                idx,
                id: row.id.clone(),
                title: row.title.clone(),
                tags: row.tags.clone(),
                distance: neighbor.distance,
                composed_prompt,
                stem,
            }
        })
        .collect();

    // Write prompt files before launching concurrent API calls.
    for meta in &request_metas {
        let out_prompt = args.out_dir.join(format!("{}.prompt.txt", meta.stem));
        if let Err(err) = fs::write(&out_prompt, format!("{}\n", meta.composed_prompt)) {
            eprintln!("Failed to write {}: {err}", out_prompt.display());
            return ExitCode::from(1);
        }
    }

    // Fan out all Gemini requests concurrently.
    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(err) => {
            eprintln!("Failed to create Tokio runtime: {err}");
            return ExitCode::from(1);
        }
    };

    let total = request_metas.len();
    let api_key_ref = &api_key;
    let model_ref = &args.model;
    let gemini_futures: Vec<_> = request_metas
        .iter()
        .map(|meta| {
            let prompt = meta.composed_prompt.clone();
            let id = meta.id.clone();
            let rank = meta.idx + 1;
            async move {
                println!("Generating [{rank}/{total}] {id}...");
                mapgen_gemini::call_gemini_async(model_ref, &prompt, api_key_ref).await
            }
        })
        .collect();
    let responses: Vec<Result<serde_json::Value, String>> = rt.block_on(join_all(gemini_futures));

    // Process results in order and write image outputs.
    let mut manifest_entries: Vec<serde_json::Value> = Vec::new();
    for (meta, response) in request_metas.iter().zip(responses.into_iter()) {
        let response_json = match response {
            Ok(v) => v,
            Err(err) => {
                eprintln!("Gemini request failed for {}: {}", meta.id, err);
                return ExitCode::from(1);
            }
        };
        let outputs = match mapgen_gemini::extract_parts(&response_json) {
            Ok(v) => v,
            Err(err) => {
                eprintln!(
                    "Failed to parse Gemini response for {}: {}",
                    meta.id, err
                );
                return ExitCode::from(1);
            }
        };
        let Some(image_bytes) = outputs.image_bytes else {
            eprintln!("No image returned for {}.", meta.id);
            if let Some(text) = outputs.text {
                eprintln!("Model text response:\n{text}");
            }
            return ExitCode::from(1);
        };

        let out_png = args.out_dir.join(format!("{}.png", meta.stem));
        let out_prompt = args.out_dir.join(format!("{}.prompt.txt", meta.stem));
        if let Err(err) =
            mapgen_gemini::write_outputs(&out_png, &image_bytes, outputs.text.as_deref(), true)
        {
            eprintln!("{err}");
            return ExitCode::from(1);
        }

        manifest_entries.push(serde_json::json!({
            "rank": meta.idx + 1,
            "distance": meta.distance,
            "id": meta.id,
            "title": meta.title,
            "tags": meta.tags,
            "image": out_png.display().to_string(),
            "prompt_file": out_prompt.display().to_string()
        }));
    }

    let manifest_path = args.out_dir.join("manifest.json");
    let manifest = serde_json::json!({
        "query": args.query,
        "style": args.style.as_slug(),
        "model": args.model,
        "results": manifest_entries
    });
    let serialized = match serde_json::to_string_pretty(&manifest) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("Failed to serialize manifest: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = fs::write(&manifest_path, format!("{serialized}\n")) {
        eprintln!("Failed to write {}: {err}", manifest_path.display());
        return ExitCode::from(1);
    }
    println!("Saved manifest: {}", manifest_path.display());

    ExitCode::SUCCESS
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn load_gemini_api_key() -> Result<String, String> {
    if let Err(err) = mapgen_gemini::load_dotenv_if_present(Path::new(".env")) {
        eprintln!("Warning: {err}");
    }
    match std::env::var("GEMINI_API_KEY") {
        Ok(v) if !v.trim().is_empty() => Ok(v),
        _ => Err("Missing GEMINI_API_KEY environment variable.".to_string()),
    }
}

fn load_or_build_env_art_index(
    corpus_path: &Path,
    index_path: &Path,
    api_key: &str,
    refresh_index: bool,
) -> Result<EnvArtPersistentIndex, String> {
    let corpus_raw = fs::read_to_string(corpus_path)
        .map_err(|err| format!("Failed to read {}: {err}", corpus_path.display()))?;
    let corpus: Vec<EnvPromptRow> = serde_json::from_str(&corpus_raw)
        .map_err(|err| format!("Failed to parse {}: {err}", corpus_path.display()))?;
    let fingerprint = fnv1a64(corpus_raw.as_bytes());
    const EMBED_MODEL: &str = "models/gemini-embedding-001";

    if !refresh_index && index_path.exists() {
        let raw = fs::read_to_string(index_path)
            .map_err(|err| format!("Failed to read {}: {err}", index_path.display()))?;
        let cached: EnvArtPersistentIndex = serde_json::from_str(&raw)
            .map_err(|err| format!("Failed to parse {}: {err}", index_path.display()))?;
        let valid = cached.version == 1
            && cached.embedding_model == EMBED_MODEL
            && cached.corpus_fingerprint == fingerprint
            && cached.rows.len() == cached.vectors.len()
            && !cached.vectors.is_empty()
            && cached.vectors.iter().all(|v| v.len() == cached.vector_dim);
        if valid {
            println!(
                "Using cached env-art index: {} ({} vectors)",
                index_path.display(),
                cached.rows.len()
            );
            return Ok(cached);
        }
        println!("Cached env-art index is stale; rebuilding...");
    }

    let vectors = embed_corpus_rows(&corpus, api_key)?;
    if vectors.is_empty() {
        return Err("Cannot build env-art index: corpus has no rows.".to_string());
    }
    let vector_dim = vectors[0].len();
    let built = EnvArtPersistentIndex {
        version: 1,
        embedding_model: EMBED_MODEL.to_string(),
        corpus_path: corpus_path.display().to_string(),
        corpus_fingerprint: fingerprint,
        vector_dim,
        rows: corpus,
        vectors,
    };
    if let Some(parent) = index_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("Failed to create {}: {err}", parent.display()))?;
    }
    let serialized = serde_json::to_string_pretty(&built)
        .map_err(|err| format!("Failed to serialize index: {err}"))?;
    fs::write(index_path, format!("{serialized}\n"))
        .map_err(|err| format!("Failed to write {}: {err}", index_path.display()))?;
    println!(
        "Saved env-art index: {} ({} vectors)",
        index_path.display(),
        built.rows.len()
    );
    Ok(built)
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn embed_corpus_rows(corpus: &[EnvPromptRow], api_key: &str) -> Result<Vec<Vec<f32>>, String> {
    let mut vectors = Vec::with_capacity(corpus.len());
    for (idx, row) in corpus.iter().enumerate() {
        let joined = format!(
            "id: {}\ntitle: {}\ntags: {}\nprompt: {}",
            row.id,
            row.title,
            row.tags.join(", "),
            row.prompt
        );
        let embedding = embed_text(&joined, api_key)?;
        println!("embedded {}/{} :: {}", idx + 1, corpus.len(), row.id);
        vectors.push(embedding);
    }
    Ok(vectors)
}

fn embed_text(text: &str, api_key: &str) -> Result<Vec<f32>, String> {
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={api_key}"
    );
    let payload = serde_json::json!({
        "model": "models/gemini-embedding-001",
        "content": { "parts": [{ "text": text }] },
        "taskType": "RETRIEVAL_DOCUMENT"
    });
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(90))
        .build()
        .map_err(|e| format!("failed to build embedding http client: {e}"))?;
    let response = client
        .post(url)
        .json(&payload)
        .send()
        .map_err(|e| format!("Gemini embedding request failed: {e}"))?;
    let status = response.status();
    let body = response
        .text()
        .map_err(|e| format!("failed to read embedding response: {e}"))?;
    if !status.is_success() {
        return Err(format!("Gemini embedding HTTP {}: {}", status.as_u16(), body));
    }
    let json: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| format!("failed to parse embedding JSON: {e}"))?;
    let values = json
        .get("embedding")
        .and_then(|v| v.get("values"))
        .and_then(|v| v.as_array())
        .ok_or_else(|| "embedding response missing values array".to_string())?;
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        let as_f32 = value
            .as_f64()
            .ok_or_else(|| "embedding value is not numeric".to_string())?
            as f32;
        out.push(as_f32);
    }
    Ok(out)
}

fn search_corpus_with_hnsw(
    vectorized: &[Vec<f32>],
    query_embedding: &[f32],
    top_k: usize,
) -> Vec<Neighbour> {
    if vectorized.is_empty() || top_k == 0 {
        return Vec::new();
    }
    if query_embedding.len() != vectorized[0].len() {
        return Vec::new();
    }
    let max_connections = 24;
    let max_elements = vectorized.len();
    let max_layers = 16;
    let ef_construction = 200;
    let hnsw: Hnsw<f32, DistCosine> = Hnsw::new(
        max_connections,
        max_elements,
        max_layers,
        ef_construction,
        DistCosine {},
    );
    for (idx, vec) in vectorized.iter().enumerate() {
        hnsw.insert((vec, idx));
    }
    let ef_search = (top_k.max(32)).min(256);
    hnsw.search(query_embedding, top_k, ef_search)
}

fn compose_environment_prompt(base_prompt: &str, style: EnvArtStyle) -> String {
    format!(
        "{base_prompt}\nStyle direction: {}.\nHard constraints: environment only, no people, no humanoids, no foreground creatures, no text, no watermark, no UI.",
        style.as_prompt_suffix()
    )
}

fn sanitize_stem(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}
