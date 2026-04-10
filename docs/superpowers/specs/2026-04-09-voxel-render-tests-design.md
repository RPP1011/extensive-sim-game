# Voxel Render Test Suite

## Goal

A test suite that validates voxel terrain generation through GPU rendering, covering three levels: pixel-level feature assertions, multi-viewpoint gallery output, and LLM-based scene understanding via Ollama.

## Architecture

```
tests/
  render_test_helpers/
    mod.rs              -- shared render, image, and Ollama utilities
  voxel_render_features.rs   -- per-feature pixel assertion tests
  voxel_render_gallery.rs    -- multi-viewpoint contact sheet generator
  voxel_render_understand.rs -- LLM identification + subjective quality tests
```

All tests require `--features app` (Vulkan GPU). LLM tests skip gracefully when Ollama is unreachable. All rendered images saved to `generated/render_tests/`.

## Shared Helpers (`tests/render_test_helpers/mod.rs`)

### Terrain rendering

- `try_vulkan() -> Option<VulkanContext>` -- headless Vulkan init, returns None if no GPU
- `generate_biome_world(terrain: Terrain, seed: u64) -> (VoxelWorld, Vec<ChunkPos>)` -- generates a 6x6x6 chunk block in the specified biome by scanning the region plan for a matching cell
- `generate_feature_world(feature: Feature, seed: u64) -> (VoxelWorld, Vec<ChunkPos>, CameraSetup)` -- generates chunks targeting a specific feature (cave, river, settlement, etc.) with an appropriate camera position
- `render_world(ctx, alloc, renderer, world, positions, camera) -> Vec<[u8;4]>` -- packs chunks into a grid, uploads to GPU, renders, returns RGBA pixels
- `CameraSetup` -- struct with position, look_at, and descriptive label

### Image utilities

- `save_png(pixels, w, h, path)` -- write RGBA pixels to PNG file
- `make_contact_sheet(images: Vec<(RgbaImage, String)>, cols: usize) -> RgbaImage` -- composite labeled images into a grid with text labels rendered as simple pixel font
- `pixel_stats(pixels) -> PixelStats` -- compute color distribution stats: % black, unique color buckets (quantized), dominant hue, presence of color families (green, brown, gray, blue, white, etc.)

### Ollama client

- `ollama_available() -> bool` -- ping `OLLAMA_URL` (default `http://localhost:11434`), returns false if unreachable
- `ollama_judge(png_bytes: &[u8], prompt: &str) -> Option<String>` -- POST to `/api/generate` with base64 image, returns None if Ollama unavailable. Model from `OLLAMA_MODEL` env var, default `qwen2.5vl:7b`.
- `parse_json_response(response: &str) -> Option<serde_json::Value>` -- extract JSON from markdown-fenced or bare response

### Constants

- `RENDER_W: u32 = 320`, `RENDER_H: u32 = 240` -- test render resolution
- `GALLERY_RENDER_W: u32 = 320`, `GALLERY_RENDER_H: u32 = 240` -- gallery tile resolution

## 1. Per-Feature Coverage Tests (`voxel_render_features.rs`)

Each test generates a specific terrain feature at a known seed, renders it, saves a PNG, and asserts pixel-level properties. These are deterministic and do not require Ollama.

| Test | Generation | Camera | Assertions |
|------|-----------|--------|------------|
| `caves_have_interior` | Chunk block at cave depth (z well below surface) | Inside cave looking horizontally | Dark pixels present (>10% below brightness 40); non-uniform (not solid black) |
| `rivers_have_water` | Chunk block at a river crossing (scan region plan for river cell) | Top-down looking at surface | Blue-family pixels present (>5% of non-sky pixels) |
| `ore_veins_at_depth` | Deep chunk block (z << surface) | Close-up inside stone | Colored pixels distinct from gray stone (gold/copper/iron hues) |
| `forest_has_vegetation` | Forest biome surface | 45-degree overhead | Green-family pixels dominate surface area (>15%) |
| `desert_is_sandy` | Desert biome surface | 45-degree overhead | Tan/sandy pixels dominate (>20%); minimal green |
| `mountains_have_elevation` | Mountain biome | Side view showing elevation | Vertical pixel variation; terrain spans >50% of frame height |
| `snow_at_peaks` | Mountain biome, high elevation | Looking at peak | White/near-white pixels present (>5%) |
| `buildings_have_structure` | Settlement chunk | Overhead | Rectangular color regions; distinct from surrounding terrain colors |
| `flying_islands_in_sky` | Sky-level chunk near island feature | Side view | Solid terrain pixels above a gap of empty/sky pixels |

Each test:
1. Calls `generate_feature_world(feature, seed)`
2. Calls `render_world(...)` to get pixels
3. Calls `save_png(...)` to `generated/render_tests/<test_name>.png`
4. Runs `pixel_stats()` and asserts thresholds

## 2. Multi-Viewpoint Gallery (`voxel_render_gallery.rs`)

Generates a contact sheet for manual inspection. One test function, always passes (unless rendering crashes).

**Biomes:** Forest, Desert, Mountains, Plains, Tundra (5 rows)

**Viewpoints per biome** (5 columns):
1. Top-down (directly above, looking straight down)
2. North 45-degree (looking south)
3. East 45-degree (looking west)
4. Ground-level (eye height, looking at horizon)
5. Close-up (zoomed into surface detail)

Output: `generated/render_tests/gallery.png` -- 5x5 grid = 25 tiles, each 320x240, with row/column labels.

The gallery uses a single seed (42) for reproducibility. Each biome is generated once as a shared world, then rendered from 5 camera positions.

## 3. LLM Understanding Tests (`voxel_render_understand.rs`)

All tests in this file skip if Ollama is unreachable. None fail the build (LLM assertions use a soft-assert pattern that logs warnings but doesn't panic).

### 3a. Calibration controls (run first)

Three control images rendered in Rust (no terrain gen needed):

| Control | How generated | Expected rating | Purpose |
|---------|--------------|-----------------|---------|
| Flat plane | Single-material 96^3 grid, all dirt | visual_interest <= 3, layout_variety <= 3 | Detect sycophancy |
| Random noise | Random material per voxel | biome_coherence <= 3 | Detect indiscriminate praise |
| Known-good settlement | Rich settlement scene with buildings + terrain | All scores >= 4 | Detect excessive harshness |

If both flat plane and random noise score >6 on any metric, print:
```
WARNING: LLM calibration failed -- model may not discriminate. Quality scores unreliable.
```
If known-good scores below 3 on all metrics, print:
```
WARNING: LLM calibration failed -- model too harsh. Quality scores unreliable.
```

Calibration results saved to `generated/render_tests/calibration.json`.

### 3b. Biome identification tests

Render each biome (Forest, Desert, Mountains, Ocean/coast, Plains, Tundra) and ask the model to identify it from a constrained choice list.

Prompt:
```
This is a rendered 3D voxel terrain scene from a game. Which of these biomes does it most resemble?
Choose exactly one: forest, desert, mountains, ocean, plains, tundra, cave, settlement

Respond with just the biome name, nothing else.
```

Acceptance: the model's answer matches the generated biome. Log pass/fail per biome. Overall identification accuracy logged but does not fail the build (we observed `qwen2.5vl:7b` can hallucinate on synthetic voxel images).

Results saved to `generated/render_tests/identification.json`:
```json
{
  "model": "qwen2.5vl:7b",
  "results": [
    {"biome": "forest", "predicted": "forest", "pass": true},
    {"biome": "desert", "predicted": "plains", "pass": false}
  ],
  "accuracy": 0.83
}
```

### 3c. Subjective quality assessment

For each biome gallery render (the 5 biomes from section 2, using the north-45 viewpoint), ask for quality ratings.

Prompt:
```
You are evaluating a rendered 3D voxel terrain image for a game.

Rate the following on a scale of 1-10 (1=worst, 10=best). Be critical and honest.
A flat featureless plane should score 1-2. Random noise should score 1-2.
Interesting terrain with varied elevation, features, and coherent materials should score 6-10.

1. Visual Interest: How visually appealing and interesting is the scene?
2. Layout Variety: How varied is the spatial layout? Flat and monotonous (low) vs interesting terrain features (high)?
3. Biome Coherence: Does this look like a believable natural environment?

Respond in exactly this JSON format, no other text:
{"visual_interest": N, "layout_variety": N, "biome_coherence": N, "one_line_description": "..."}
```

Results saved to `generated/render_tests/quality_report.json`:
```json
{
  "seed": 42,
  "model": "qwen2.5vl:7b",
  "calibration_passed": true,
  "scores": [
    {
      "scene": "forest_north45",
      "visual_interest": 6,
      "layout_variety": 5,
      "biome_coherence": 7,
      "description": "Dense forest canopy with varied elevation"
    }
  ]
}
```

These scores never fail the build. They exist as data for tracking terrain gen quality over time.

## Dependencies (dev-only)

Added to `[dev-dependencies]` in Cargo.toml:

- `reqwest = { version = "0.12", features = ["blocking", "json"] }` -- Ollama HTTP client
- `base64 = "0.22"` -- encode PNG for Ollama API
- `serde_json` -- already present, parse LLM responses

`image` is already a dependency.

## Running

```bash
# All render tests (LLM tests skip if no Ollama)
cargo test --test voxel_render_features --test voxel_render_gallery --test voxel_render_understand --features app -- --nocapture

# Just feature tests (no Ollama needed)
cargo test --test voxel_render_features --features app -- --nocapture

# Just gallery generation
cargo test --test voxel_render_gallery --features app -- --nocapture

# LLM tests with custom model
OLLAMA_MODEL=qwen2.5vl:7b cargo test --test voxel_render_understand --features app -- --nocapture

# Single feature test
cargo test --test voxel_render_features caves_have_interior --features app -- --nocapture
```

## Output files

All in `generated/render_tests/`:
- `<feature_name>.png` -- per-feature test renders
- `gallery.png` -- contact sheet
- `calibration_flat.png`, `calibration_noise.png`, `calibration_good.png` -- control images
- `calibration.json` -- calibration scores
- `identification.json` -- biome ID results
- `quality_report.json` -- subjective scores
