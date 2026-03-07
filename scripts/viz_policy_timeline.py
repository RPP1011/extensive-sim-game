#!/usr/bin/env python3
"""Generate an interactive policy activation timeline visualization.

Usage:
    # With oracle dataset:
    uv run --with numpy --with torch scripts/viz_policy_timeline.py \\
        generated/student_model.json --dataset generated/oracle_dataset.jsonl

    # Synthetic combat scenario (no dataset needed):
    uv run --with numpy --with torch scripts/viz_policy_timeline.py \\
        generated/student_model.json -o generated/reports/policy_timeline.html

Loads the trained student policy and runs inference on sequential samples,
showing the softmax distribution over action classes as a stacked area chart.
If no dataset is provided, generates a synthetic combat scenario by smoothly
varying key game-state features to simulate a fight progression.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


ACTION_NAMES_10 = [
    "AttackNearest", "AttackWeakest", "DamageAbility", "HealAbility",
    "CcAbility", "DefenseAbility", "UtilityAbility", "MoveToward", "MoveAway", "Hold",
]

ACTION_NAMES_5 = [
    "AttackNearest", "AttackWeakest", "MoveToward", "MoveAway", "Hold",
]

ACTION_COLORS_10 = [
    "#ef4444",  # AttackNearest - red
    "#f97316",  # AttackWeakest - orange
    "#a855f7",  # DamageAbility - purple
    "#22c55e",  # HealAbility - green
    "#06b6d4",  # CcAbility - cyan
    "#3b82f6",  # DefenseAbility - blue
    "#eab308",  # UtilityAbility - yellow
    "#14b8a6",  # MoveToward - teal
    "#f43f5e",  # MoveAway - rose
    "#78716c",  # Hold - stone
]


def load_mlp(weights_path: str):
    """Load the student MLP model from exported JSON weights."""
    with open(weights_path) as f:
        data = json.load(f)

    # Support two export formats:
    # 1. {"layers": [{"w":..., "b":...}, ...]} (train_student.py export)
    # 2. {"w1":..., "b1":..., "w2":..., "b2":..., ...} (Rust-side export)

    if "layers" in data:
        layers_data = data["layers"]
        layers = []
        for i, ld in enumerate(layers_data):
            w = torch.tensor(ld["w"], dtype=torch.float32)
            b = torch.tensor(ld["b"], dtype=torch.float32)
            linear = nn.Linear(w.shape[0], w.shape[1])
            linear.weight = nn.Parameter(w.T)
            linear.bias = nn.Parameter(b)
            layers.append(linear)
            if i < len(layers_data) - 1:
                layers.append(nn.ReLU())
    else:
        # w1/b1, w2/b2, ... format
        layers = []
        i = 1
        while f"w{i}" in data:
            w = torch.tensor(data[f"w{i}"], dtype=torch.float32)
            b = torch.tensor(data[f"b{i}"], dtype=torch.float32)
            # w is [in, out] in this format
            linear = nn.Linear(w.shape[0], w.shape[1])
            linear.weight = nn.Parameter(w.T)
            linear.bias = nn.Parameter(b)
            layers.append(linear)
            # Check if there's a next layer (add ReLU between layers)
            if f"w{i+1}" in data:
                layers.append(nn.ReLU())
            i += 1

    model = nn.Sequential(*layers)
    model.eval()
    return model


def load_samples(dataset_path: str, max_samples: int = 500):
    """Load sequential samples from the oracle JSONL dataset."""
    features = []
    labels = []
    metadata = []

    with open(dataset_path) as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            sample = json.loads(line)
            features.append(sample["features"])
            labels.append(sample["label"])
            metadata.append({
                "tick": sample.get("tick", i),
                "unit_id": sample.get("unit_id", 0),
                "scenario": sample.get("scenario", ""),
            })

    return (np.array(features, dtype=np.float32),
            np.array(labels, dtype=np.int64),
            metadata)


def generate_synthetic_scenario(n_steps: int = 300, input_dim: int = 115, seed: int = 42):
    """Generate a synthetic combat scenario by smoothly varying game-state features.

    Simulates a fight with 4 phases (approach, engagement, crisis, cleanup)
    using smooth curves. Works for any input_dim by filling features with
    meaningful combat-like patterns.
    """
    rng = np.random.RandomState(seed)
    features = np.zeros((n_steps, input_dim), dtype=np.float32)
    t = np.linspace(0, 1, n_steps)

    for i in range(n_steps):
        f = features[i]
        phase = t[i]
        noise = rng.randn(input_dim) * 0.02

        # Fill all features with smooth, varied curves
        for j in range(input_dim):
            # Mix of sinusoidal patterns at different frequencies per feature
            freq1 = 1.0 + (j % 7) * 0.5
            freq2 = 2.0 + (j % 5) * 0.3
            base_val = 0.5 + 0.3 * np.sin(phase * np.pi * freq1 + j * 0.7)
            base_val += 0.1 * np.cos(phase * np.pi * freq2 + j * 1.3)
            f[j] = np.clip(base_val + noise[j], 0.0, 1.0)

        # Override first few features with meaningful combat patterns
        # Feature 0: self HP — drops mid-fight
        hp_curve = 1.0 - 0.5 * np.sin(phase * np.pi) ** 2 - 0.15 * phase
        f[0] = np.clip(hp_curve + noise[0], 0.05, 1.0)

        # Feature 1: resource/shield — depletes over time
        if input_dim > 1:
            f[1] = np.clip(0.8 - 0.3 * phase + noise[1], 0.0, 1.0)

        # Make some features represent approaching enemies
        if input_dim > 3:
            f[3] = np.clip(0.9 - 0.7 * phase, 0.1, 1.0)  # distance-like

        # Make a feature represent game phase directly
        if input_dim > min(input_dim - 1, 10):
            f[min(input_dim - 1, 10)] = phase

    # Generate oracle-like labels based on phase heuristics
    n_classes = 10 if input_dim >= 48 else 5
    labels = np.zeros(n_steps, dtype=np.int64)
    for i in range(n_steps):
        phase = t[i]
        r = rng.rand()
        if n_classes >= 10:
            if phase < 0.2:
                labels[i] = 7 if r < 0.6 else (0 if r < 0.8 else 9)
            elif phase < 0.5:
                if r < 0.25: labels[i] = 0
                elif r < 0.45: labels[i] = 2
                elif r < 0.6: labels[i] = 4
                elif r < 0.75: labels[i] = 1
                else: labels[i] = 5
            elif phase < 0.75:
                if r < 0.3: labels[i] = 3
                elif r < 0.5: labels[i] = 5
                elif r < 0.7: labels[i] = 8
                else: labels[i] = 6
            else:
                if r < 0.4: labels[i] = 0
                elif r < 0.65: labels[i] = 2
                elif r < 0.8: labels[i] = 1
                else: labels[i] = 7
        else:
            if phase < 0.3:
                labels[i] = 2 if r < 0.6 else 0
            elif phase < 0.7:
                labels[i] = 0 if r < 0.4 else (1 if r < 0.7 else 3)
            else:
                labels[i] = 0 if r < 0.5 else (4 if r < 0.8 else 2)

    metadata = [{"tick": i, "unit_id": 0, "scenario": "synthetic"} for i in range(n_steps)]
    return features, labels, metadata


def build_html(softmax_probs: np.ndarray, labels: np.ndarray, metadata: list,
               action_names: list, action_colors: list) -> str:
    """Build interactive stacked area chart + confidence timeline."""
    n_steps, n_actions = softmax_probs.shape

    # Prepare data as JSON
    probs_list = softmax_probs.tolist()
    labels_list = labels.tolist()

    # Extract feature highlights: HP%, nearest enemy distance from raw features
    # (indices 0 and 59 in the 115-feature student model)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Policy Activation Timeline</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: ui-sans-serif, system-ui, sans-serif; background: #111318; color: #e8eaf0; }}
.header {{ padding: 16px 24px; border-bottom: 1px solid #2a2e3a; }}
.header h1 {{ font-size: 20px; font-weight: 600; }}
.header p {{ font-size: 13px; color: #8b8fa3; margin-top: 4px; }}
.main {{ display: flex; flex-direction: column; height: calc(100vh - 80px); }}
.chart-area {{ flex: 1; position: relative; padding: 12px; }}
canvas {{ width: 100%; height: 100%; }}
.controls {{ display: flex; gap: 16px; padding: 8px 24px; border-top: 1px solid #2a2e3a; align-items: center; flex-wrap: wrap; }}
.controls label {{ font-size: 13px; color: #8b8fa3; }}
.controls select, .controls input {{ background: #1a1d28; border: 1px solid #2a2e3a; color: #e8eaf0; border-radius: 4px; padding: 4px 8px; font-size: 13px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 12px; padding: 8px 24px; border-bottom: 1px solid #2a2e3a; }}
.legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 12px; cursor: pointer; }}
.legend-item.dimmed {{ opacity: 0.25; }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 2px; }}
.tooltip {{ position: fixed; pointer-events: none; background: #1a1d28; border: 1px solid #3a3e4a; border-radius: 6px; padding: 10px 14px; font-size: 12px; display: none; z-index: 10; box-shadow: 0 4px 12px rgba(0,0,0,0.4); min-width: 180px; }}
.tooltip .row {{ display: flex; justify-content: space-between; gap: 12px; padding: 1px 0; }}
.tooltip .bar {{ height: 4px; border-radius: 2px; margin-top: 2px; }}
.info-bar {{ display: flex; gap: 24px; padding: 6px 24px; font-size: 12px; color: #8b8fa3; border-bottom: 1px solid #2a2e3a; }}
.info-bar .metric {{ }}
.info-bar .metric span {{ color: #e8eaf0; font-weight: 500; }}
</style></head>
<body>
<div class="header">
  <h1>Policy Activation Timeline</h1>
  <p>Softmax distribution over {n_actions} action classes across {n_steps} sequential samples &mdash; stacked area chart</p>
</div>
<div class="legend" id="legend"></div>
<div class="info-bar" id="info-bar">
  <div class="metric">Accuracy: <span id="accuracy">-</span></div>
  <div class="metric">Avg confidence: <span id="avg-conf">-</span></div>
  <div class="metric">Most common: <span id="most-common">-</span></div>
</div>
<div class="main">
  <div class="chart-area">
    <canvas id="canvas"></canvas>
  </div>
  <div class="controls">
    <label>View:
      <select id="view-mode">
        <option value="stacked">Stacked Area</option>
        <option value="lines">Line Plot</option>
        <option value="heatmap">Heatmap</option>
      </select>
    </label>
    <label>Range:
      <input type="range" id="range-start" min="0" max="{n_steps - 1}" value="0" style="width:120px" />
      &mdash;
      <input type="range" id="range-end" min="0" max="{n_steps - 1}" value="{n_steps - 1}" style="width:120px" />
    </label>
    <label>Smoothing:
      <input type="range" id="smoothing" min="1" max="20" value="1" style="width:80px" />
    </label>
  </div>
</div>
<div class="tooltip" id="tooltip"></div>
<script>
const PROBS = {json.dumps(probs_list)};
const LABELS = {json.dumps(labels_list)};
const ACTIONS = {json.dumps(action_names)};
const COLORS = {json.dumps(action_colors[:n_actions])};
const N = PROBS.length;
const K = ACTIONS.length;

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');
let dpr = window.devicePixelRatio || 1;
let W, H;
let viewMode = 'stacked';
let rangeStart = 0, rangeEnd = N - 1;
let smoothing = 1;
let activeActions = new Set(Array.from({{length: K}}, (_, i) => i));
let hoveredStep = -1;

// Build legend
const legend = document.getElementById('legend');
for (let i = 0; i < K; i++) {{
  const item = document.createElement('div');
  item.className = 'legend-item';
  item.innerHTML = `<span class="legend-dot" style="background:${{COLORS[i]}}"></span>${{ACTIONS[i]}}`;
  item.addEventListener('click', () => {{
    if (activeActions.has(i)) {{ activeActions.delete(i); item.classList.add('dimmed'); }}
    else {{ activeActions.add(i); item.classList.remove('dimmed'); }}
    draw();
  }});
  legend.appendChild(item);
}}

// Stats
let correct = 0;
let totalConf = 0;
const classCounts = new Array(K).fill(0);
for (let t = 0; t < N; t++) {{
  const pred = PROBS[t].indexOf(Math.max(...PROBS[t]));
  if (pred === LABELS[t]) correct++;
  totalConf += Math.max(...PROBS[t]);
  classCounts[LABELS[t]]++;
}}
document.getElementById('accuracy').textContent = (correct / N * 100).toFixed(1) + '%';
document.getElementById('avg-conf').textContent = (totalConf / N * 100).toFixed(1) + '%';
const mostCommonIdx = classCounts.indexOf(Math.max(...classCounts));
document.getElementById('most-common').textContent = ACTIONS[mostCommonIdx] + ' (' + (classCounts[mostCommonIdx] / N * 100).toFixed(0) + '%)';

function getSmoothed() {{
  if (smoothing <= 1) return PROBS;
  const out = [];
  for (let t = 0; t < N; t++) {{
    const row = new Array(K).fill(0);
    let count = 0;
    for (let s = Math.max(0, t - smoothing + 1); s <= t; s++) {{
      for (let k = 0; k < K; k++) row[k] += PROBS[s][k];
      count++;
    }}
    for (let k = 0; k < K; k++) row[k] /= count;
    out.push(row);
  }}
  return out;
}}

function resize() {{
  const rect = canvas.parentElement.getBoundingClientRect();
  W = rect.width;
  H = rect.height;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}}

const MARGIN = {{ left: 40, right: 16, top: 12, bottom: 28 }};

function draw() {{
  ctx.clearRect(0, 0, W, H);
  const probs = getSmoothed();
  const s = rangeStart, e = rangeEnd;
  const steps = e - s + 1;
  if (steps < 2) return;

  const cw = W - MARGIN.left - MARGIN.right;
  const ch = H - MARGIN.top - MARGIN.bottom;

  // Grid
  ctx.strokeStyle = '#2a2e3a';
  ctx.lineWidth = 0.5;
  for (let y = 0; y <= 4; y++) {{
    const py = MARGIN.top + ch * (1 - y / 4);
    ctx.beginPath(); ctx.moveTo(MARGIN.left, py); ctx.lineTo(W - MARGIN.right, py); ctx.stroke();
    ctx.fillStyle = '#555';
    ctx.font = '10px ui-sans-serif, system-ui, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText((y * 25) + '%', MARGIN.left - 4, py + 3);
  }}

  // X axis ticks
  ctx.textAlign = 'center';
  const tickStep = Math.max(1, Math.floor(steps / 8));
  for (let i = 0; i < steps; i += tickStep) {{
    const x = MARGIN.left + (i / (steps - 1)) * cw;
    ctx.fillStyle = '#555';
    ctx.fillText(String(s + i), x, H - 4);
  }}

  if (viewMode === 'stacked') {{
    drawStacked(probs, s, e, cw, ch);
  }} else if (viewMode === 'lines') {{
    drawLines(probs, s, e, cw, ch);
  }} else {{
    drawHeatmap(probs, s, e, cw, ch);
  }}

  // Oracle label markers (bottom strip)
  const stripH = 4;
  for (let i = 0; i < steps; i++) {{
    const t = s + i;
    const x = MARGIN.left + (i / (steps - 1)) * cw;
    ctx.fillStyle = COLORS[LABELS[t]] || '#444';
    ctx.fillRect(x - 1, MARGIN.top + ch + 2, Math.max(cw / steps, 2), stripH);
  }}

  // Hover line
  if (hoveredStep >= s && hoveredStep <= e) {{
    const x = MARGIN.left + ((hoveredStep - s) / (steps - 1)) * cw;
    ctx.strokeStyle = 'rgba(255,255,255,0.4)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(x, MARGIN.top); ctx.lineTo(x, MARGIN.top + ch); ctx.stroke();
    ctx.setLineDash([]);
  }}
}}

function drawStacked(probs, s, e, cw, ch) {{
  const steps = e - s + 1;
  // Draw in reverse order so first actions are on top
  for (let k = K - 1; k >= 0; k--) {{
    if (!activeActions.has(k)) continue;
    ctx.beginPath();
    for (let i = 0; i < steps; i++) {{
      const t = s + i;
      const x = MARGIN.left + (i / (steps - 1)) * cw;
      let cumTop = 0;
      for (let j = 0; j <= k; j++) {{
        if (activeActions.has(j)) cumTop += probs[t][j];
      }}
      const y = MARGIN.top + ch * (1 - cumTop);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }}
    // Close bottom
    for (let i = steps - 1; i >= 0; i--) {{
      const t = s + i;
      const x = MARGIN.left + (i / (steps - 1)) * cw;
      let cumBot = 0;
      for (let j = 0; j < k; j++) {{
        if (activeActions.has(j)) cumBot += probs[t][j];
      }}
      const y = MARGIN.top + ch * (1 - cumBot);
      ctx.lineTo(x, y);
    }}
    ctx.closePath();
    ctx.fillStyle = COLORS[k] + 'cc';
    ctx.fill();
  }}
}}

function drawLines(probs, s, e, cw, ch) {{
  const steps = e - s + 1;
  for (let k = 0; k < K; k++) {{
    if (!activeActions.has(k)) continue;
    ctx.beginPath();
    ctx.strokeStyle = COLORS[k];
    ctx.lineWidth = 1.5;
    for (let i = 0; i < steps; i++) {{
      const t = s + i;
      const x = MARGIN.left + (i / (steps - 1)) * cw;
      const y = MARGIN.top + ch * (1 - probs[t][k]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }}
    ctx.stroke();
  }}
}}

function drawHeatmap(probs, s, e, cw, ch) {{
  const steps = e - s + 1;
  const cellW = cw / steps;
  const cellH = ch / K;
  for (let k = 0; k < K; k++) {{
    if (!activeActions.has(k)) continue;
    for (let i = 0; i < steps; i++) {{
      const t = s + i;
      const x = MARGIN.left + i * cellW;
      const y = MARGIN.top + k * cellH;
      const v = probs[t][k];
      const alpha = Math.min(1, v * 2);
      ctx.fillStyle = COLORS[k];
      ctx.globalAlpha = 0.1 + alpha * 0.9;
      ctx.fillRect(x, y, cellW + 0.5, cellH);
    }}
  }}
  ctx.globalAlpha = 1.0;
  // Row labels
  ctx.font = '10px ui-sans-serif, system-ui, sans-serif';
  ctx.fillStyle = '#e8eaf0';
  ctx.textAlign = 'right';
  for (let k = 0; k < K; k++) {{
    ctx.fillText(ACTIONS[k], MARGIN.left - 4, MARGIN.top + k * cellH + cellH / 2 + 3);
  }}
}}

// Interaction
canvas.addEventListener('mousemove', (ev) => {{
  const rect = canvas.getBoundingClientRect();
  const mx = ev.clientX - rect.left;
  const cw = W - MARGIN.left - MARGIN.right;
  const steps = rangeEnd - rangeStart + 1;
  const frac = (mx - MARGIN.left) / cw;
  const step = Math.round(frac * (steps - 1)) + rangeStart;

  if (step >= rangeStart && step <= rangeEnd && step < N) {{
    hoveredStep = step;
    draw();

    let html = `<div style="font-weight:600;margin-bottom:4px">Step ${{step}}</div>`;
    html += `<div style="color:#8b8fa3;margin-bottom:4px">Oracle: <span style="color:${{COLORS[LABELS[step]]}}">${{ACTIONS[LABELS[step]]}}</span></div>`;
    const sorted = Array.from({{length: K}}, (_, i) => i).sort((a, b) => PROBS[step][b] - PROBS[step][a]);
    for (const k of sorted) {{
      const pct = (PROBS[step][k] * 100).toFixed(1);
      html += `<div class="row"><span style="color:${{COLORS[k]}}">${{ACTIONS[k]}}</span><span>${{pct}}%</span></div>`;
      html += `<div class="bar" style="width:${{pct}}%;background:${{COLORS[k]}}"></div>`;
    }}
    tooltip.innerHTML = html;
    tooltip.style.display = 'block';
    tooltip.style.left = (ev.clientX + 14) + 'px';
    tooltip.style.top = (ev.clientY - 10) + 'px';
  }} else {{
    hoveredStep = -1;
    tooltip.style.display = 'none';
    draw();
  }}
}});

canvas.addEventListener('mouseleave', () => {{
  hoveredStep = -1;
  tooltip.style.display = 'none';
  draw();
}});

// Controls
document.getElementById('view-mode').addEventListener('change', (e) => {{
  viewMode = e.target.value;
  draw();
}});
document.getElementById('range-start').addEventListener('input', (e) => {{
  rangeStart = parseInt(e.target.value);
  if (rangeStart >= rangeEnd) rangeStart = rangeEnd - 1;
  draw();
}});
document.getElementById('range-end').addEventListener('input', (e) => {{
  rangeEnd = parseInt(e.target.value);
  if (rangeEnd <= rangeStart) rangeEnd = rangeStart + 1;
  draw();
}});
document.getElementById('smoothing').addEventListener('input', (e) => {{
  smoothing = parseInt(e.target.value);
  draw();
}});

window.addEventListener('resize', resize);
resize();
</script>
</body></html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate policy activation timeline")
    parser.add_argument("model", help="Path to student_model.json")
    parser.add_argument("--dataset", default=None, help="Path to oracle_dataset.jsonl (optional)")
    parser.add_argument("-o", "--output", default="generated/reports/policy_timeline.html")
    parser.add_argument("--samples", type=int, default=300, help="Number of sequential samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading model from {args.model}...")
    model = load_mlp(args.model)

    # Detect input dim from model
    first_layer = list(model.children())[0]
    input_dim = first_layer.in_features
    print(f"  Input dim: {input_dim}")

    if args.dataset:
        print(f"Loading dataset from {args.dataset}...")
        features, labels, metadata = load_samples(args.dataset, args.samples)
    else:
        print(f"Generating synthetic combat scenario ({args.samples} steps)...")
        features, labels, metadata = generate_synthetic_scenario(args.samples, input_dim, args.seed)
    print(f"  {len(features)} samples, {features.shape[1]} features")

    n_classes = int(labels.max()) + 1
    action_names = ACTION_NAMES_5 if n_classes <= 5 else ACTION_NAMES_10
    action_colors = ACTION_COLORS_10[:n_classes]
    print(f"  {n_classes} classes: {action_names}")

    print(f"\nRunning inference...")
    with torch.no_grad():
        X = torch.from_numpy(features)
        logits = model(X)
        probs = torch.softmax(logits, dim=1).numpy()

    preds = probs.argmax(axis=1)
    acc = (preds == labels).mean()
    print(f"  Accuracy: {acc:.1%}")

    print(f"\nGenerating HTML...")
    html = build_html(probs, labels, metadata, action_names, action_colors)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"  Written to {args.output} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
