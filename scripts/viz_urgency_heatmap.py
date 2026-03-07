#!/usr/bin/env python3
"""Generate interactive urgency heatmaps for each ability evaluator category.

Usage:
    uv run --with numpy --with torch scripts/viz_urgency_heatmap.py \\
        generated/ability_eval_weights.json
    uv run --with numpy --with torch scripts/viz_urgency_heatmap.py \\
        generated/ability_eval_weights_v4_healctx.json \\
        -o generated/reports/urgency_heatmaps.html --resolution 50

Loads trained per-category MLP weights and generates urgency surfaces by
sweeping two feature dimensions while holding others at baseline values.
Produces an interactive HTML report with heatmaps for all categories.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Category feature definitions (matching features.rs / features_aoe.rs)
# ---------------------------------------------------------------------------

# Per-category: (feature_count, feature_names)
# Only listing the most important features for axis sweeps
CATEGORY_FEATURES = {
    "damage_unit": {
        "count": 34,  # 10 base + 24 per-target (3x8)
        "names": [
            "self_hp%", "self_resource%", "ability_range", "cast_time",
            "in_range_count", "numeric_advantage", "cover_bonus", "elevation",
            "hostile_zones", "friendly_zones",
            # target 0
            "t0_distance", "t0_hp%", "t0_dps", "t0_is_healer", "t0_casting", "t0_has_reflect", "t0_cover", "t0_elev_adv",
            # target 1
            "t1_distance", "t1_hp%", "t1_dps", "t1_is_healer", "t1_casting", "t1_has_reflect", "t1_cover", "t1_elev_adv",
            # target 2
            "t2_distance", "t2_hp%", "t2_dps", "t2_is_healer", "t2_casting", "t2_has_reflect", "t2_cover", "t2_elev_adv",
        ],
        "defaults": [0.8, 1.0, 0.6, 0.1, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.3, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.5, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.7, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        "interesting_pairs": [(0, 11), (10, 11), (0, 10), (4, 5)],
    },
    "cc_unit": {
        "count": 28,  # 10 base + 18 per-target (3x6)
        "names": [
            "self_hp%", "ability_range", "cast_time", "team_hp_avg",
            "ally_critical", "numeric_advantage", "cover_bonus", "elevation",
            "hostile_zones", "friendly_zones",
            "t0_distance", "t0_dps", "t0_is_healer", "t0_casting", "t0_already_ccd", "t0_hp%",
            "t1_distance", "t1_dps", "t1_is_healer", "t1_casting", "t1_already_ccd", "t1_hp%",
            "t2_distance", "t2_dps", "t2_is_healer", "t2_casting", "t2_already_ccd", "t2_hp%",
        ],
        "defaults": [0.8, 0.6, 0.1, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.3, 0.3, 0.0, 0.0, 0.0, 0.7,
                     0.5, 0.2, 0.0, 0.0, 0.0, 0.8,
                     0.7, 0.1, 0.0, 0.0, 0.0, 0.9],
        "interesting_pairs": [(0, 3), (10, 11), (10, 14), (4, 5)],
    },
    "heal_unit": {
        "count": 25,  # 10 base + 15 per-ally (3x5)
        "names": [
            "self_hp%", "ability_range", "cast_time", "team_hp_avg",
            "enemy_count", "self_in_danger", "cover_bonus", "elevation",
            "hostile_zones", "friendly_zones",
            "a0_hp%", "a0_distance", "a0_has_hot", "a0_threats", "a0_hostile_zones",
            "a1_hp%", "a1_distance", "a1_has_hot", "a1_threats", "a1_hostile_zones",
            "a2_hp%", "a2_distance", "a2_has_hot", "a2_threats", "a2_hostile_zones",
        ],
        "defaults": [0.8, 0.6, 0.1, 0.7, 0.375, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.4, 0.3, 0.0, 0.25, 0.0,
                     0.6, 0.4, 0.0, 0.0, 0.0,
                     0.8, 0.5, 0.0, 0.0, 0.0],
        "interesting_pairs": [(0, 10), (10, 13), (3, 10), (5, 10)],
    },
    "damage_aoe": {
        "count": 21,
        "names": [
            "self_hp%", "self_resource%", "ability_range", "cast_time",
            "nearest_enemy_dist", "enemy_count", "enemies_at_best", "allies_at_best",
            "numeric_advantage", "dist_to_best_pos",
            "top1_hp%", "top1_dps", "top2_hp%", "top2_dps",
            "enemy_spread", "cover_bonus", "elevation", "hostile_zones", "friendly_zones",
            "pos_hostile", "pos_friendly",
        ],
        "defaults": [0.8, 1.0, 0.6, 0.1, 0.3, 0.5, 0.5, 0.0, 0.0, 0.3,
                     0.7, 0.3, 0.8, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "interesting_pairs": [(0, 6), (4, 6), (5, 14), (6, 7)],
    },
    "heal_aoe": {
        "count": 14,
        "names": [
            "self_hp%", "self_resource%", "ability_range", "cast_time",
            "team_hp_avg", "team_hp_min", "nearest_enemy_dist", "threats",
            "ally_count", "enemy_count", "cover_bonus", "elevation",
            "hostile_zones", "friendly_zones",
        ],
        "defaults": [0.8, 1.0, 0.6, 0.1, 0.7, 0.4, 0.5, 0.25, 0.5, 0.375, 0.0, 0.0, 0.0, 0.0],
        "interesting_pairs": [(0, 4), (4, 5), (0, 7), (6, 7)],
    },
    "defense": {
        "count": 14,
        "names": [
            "self_hp%", "self_resource%", "ability_range", "cast_time",
            "team_hp_avg", "team_hp_min", "nearest_enemy_dist", "threats",
            "ally_count", "enemy_count", "cover_bonus", "elevation",
            "hostile_zones", "friendly_zones",
        ],
        "defaults": [0.8, 1.0, 0.6, 0.1, 0.7, 0.4, 0.5, 0.25, 0.5, 0.375, 0.0, 0.0, 0.0, 0.0],
        "interesting_pairs": [(0, 7), (0, 6), (4, 7), (5, 7)],
    },
    "utility": {
        "count": 14,
        "names": [
            "self_hp%", "self_resource%", "ability_range", "cast_time",
            "team_hp_avg", "team_hp_min", "nearest_enemy_dist", "threats",
            "ally_count", "enemy_count", "cover_bonus", "elevation",
            "hostile_zones", "friendly_zones",
        ],
        "defaults": [0.8, 1.0, 0.6, 0.1, 0.7, 0.4, 0.5, 0.25, 0.5, 0.375, 0.0, 0.0, 0.0, 0.0],
        "interesting_pairs": [(0, 6), (0, 7), (6, 7), (4, 5)],
    },
    "summon": {
        "count": 16,
        "names": [
            "self_hp%", "self_resource%", "ability_range", "cast_time",
            "team_hp_avg", "nearest_enemy_dist", "ally_count", "enemy_count",
            "existing_summons", "numeric_advantage", "outnumbered", "is_ccd",
            "cover_bonus", "elevation", "hostile_zones", "friendly_zones",
        ],
        "defaults": [0.8, 1.0, 0.6, 0.1, 0.7, 0.5, 0.5, 0.375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "interesting_pairs": [(0, 8), (8, 9), (5, 10), (6, 7)],
    },
    "obstacle": {
        "count": 20,
        "names": [
            "self_hp%", "self_resource%", "ability_range", "cast_time",
            "team_hp_avg", "ally_critical", "nearest_enemy_dist", "enemy_count",
            "ally_count", "numeric_advantage", "existing_obstacles",
            "cover_bonus", "elevation", "hostile_zones", "friendly_zones",
            "pos_hostile", "pos_friendly", "pos_blocked",
            "enemies_nearby_wall", "enemy_spread",
        ],
        "defaults": [0.8, 1.0, 0.6, 0.1, 0.7, 0.0, 0.5, 0.375, 0.5, 0.0, 0.1,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.3],
        "interesting_pairs": [(0, 6), (5, 6), (10, 18), (6, 19)],
    },
}


def load_category_model(weights: dict, cat_name: str):
    """Build a PyTorch model from exported category weights."""
    cat_weights = weights[cat_name]
    layers_data = cat_weights["layers"]

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

    model = nn.Sequential(*layers)
    model.eval()
    return model


def generate_heatmap(model, cat_info: dict, feat_x: int, feat_y: int,
                     resolution: int = 40) -> dict:
    """Generate urgency surface by sweeping two features."""
    defaults = np.array(cat_info["defaults"], dtype=np.float32)
    n_feat = len(defaults)

    xs = np.linspace(0.0, 1.0, resolution)
    ys = np.linspace(0.0, 1.0, resolution)

    grid = np.tile(defaults, (resolution * resolution, 1))
    idx = 0
    for yi in range(resolution):
        for xi in range(resolution):
            grid[idx, feat_x] = xs[xi]
            grid[idx, feat_y] = ys[yi]
            idx += 1

    with torch.no_grad():
        out = model(torch.from_numpy(grid))
        urgency = torch.sigmoid(out[:, 0]).numpy()

    return {
        "urgency": urgency.reshape(resolution, resolution).tolist(),
        "x_name": cat_info["names"][feat_x],
        "y_name": cat_info["names"][feat_y],
        "feat_x": feat_x,
        "feat_y": feat_y,
    }


def build_html(all_heatmaps: dict, resolution: int) -> str:
    """Build interactive HTML with heatmap grids for all categories."""
    heatmaps_json = json.dumps(all_heatmaps)

    cat_list = sorted(all_heatmaps.keys())

    cat_colors = {
        "damage_unit": "#ef4444", "damage_aoe": "#f97316", "cc_unit": "#a855f7",
        "heal_unit": "#22c55e", "heal_aoe": "#10b981", "defense": "#3b82f6",
        "utility": "#eab308", "summon": "#06b6d4", "obstacle": "#78716c",
    }

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Ability Evaluator Urgency Heatmaps</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: ui-sans-serif, system-ui, sans-serif; background: #111318; color: #e8eaf0; }}
.header {{ padding: 16px 24px; border-bottom: 1px solid #2a2e3a; }}
.header h1 {{ font-size: 20px; font-weight: 600; }}
.header p {{ font-size: 13px; color: #8b8fa3; margin-top: 4px; }}
.tabs {{ display: flex; gap: 4px; padding: 8px 24px; border-bottom: 1px solid #2a2e3a; flex-wrap: wrap; }}
.tab {{ padding: 6px 14px; border-radius: 4px; cursor: pointer; font-size: 13px; border: 1px solid transparent; }}
.tab:hover {{ background: #1e2130; }}
.tab.active {{ background: #1e2130; border-color: #3a3e4a; }}
.content {{ padding: 16px 24px; }}
.grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; max-width: 1200px; }}
.heatmap-card {{ background: #1a1d28; border: 1px solid #2a2e3a; border-radius: 8px; padding: 12px; }}
.heatmap-card h3 {{ font-size: 13px; color: #8b8fa3; margin-bottom: 8px; }}
.heatmap-card canvas {{ width: 100%; aspect-ratio: 1; cursor: crosshair; border-radius: 4px; }}
.tooltip {{ position: fixed; pointer-events: none; background: #1a1d28; border: 1px solid #3a3e4a; border-radius: 6px; padding: 8px 12px; font-size: 12px; display: none; z-index: 10; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }}
.color-bar {{ display: flex; align-items: center; gap: 8px; margin: 12px 0 0 0; font-size: 11px; color: #8b8fa3; }}
.color-bar .gradient {{ flex: 1; height: 12px; border-radius: 3px; }}
.pair-selector {{ display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }}
.pair-btn {{ padding: 4px 10px; border-radius: 4px; border: 1px solid #2a2e3a; background: #1a1d28; color: #8b8fa3; cursor: pointer; font-size: 12px; }}
.pair-btn:hover {{ border-color: #4a6fa5; }}
.pair-btn.active {{ background: #2a3548; border-color: #4a6fa5; color: #e8eaf0; }}
@media (max-width: 800px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style></head>
<body>
<div class="header">
  <h1>Ability Evaluator Urgency Heatmaps</h1>
  <p>Per-category MLP urgency output sweeping pairs of input features (others held at baseline)</p>
</div>
<div class="tabs" id="tabs"></div>
<div class="content" id="content"></div>
<div class="tooltip" id="tooltip"></div>
<script>
const ALL_DATA = {heatmaps_json};
const RES = {resolution};
const CAT_COLORS = {json.dumps(cat_colors)};
const CATS = {json.dumps(cat_list)};
const tooltip = document.getElementById('tooltip');

let activeCat = CATS[0];

// Build tabs
const tabsEl = document.getElementById('tabs');
for (const cat of CATS) {{
  const tab = document.createElement('div');
  tab.className = 'tab' + (cat === activeCat ? ' active' : '');
  tab.textContent = cat;
  tab.style.borderLeftColor = CAT_COLORS[cat];
  tab.addEventListener('click', () => {{
    activeCat = cat;
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    renderCategory(cat);
  }});
  tabsEl.appendChild(tab);
}}

function urgencyColor(v) {{
  // Dark blue (low) -> yellow -> red (high)
  if (v < 0.5) {{
    const t = v * 2;
    const r = Math.round(20 + t * 220);
    const g = Math.round(20 + t * 200);
    const b = Math.round(80 + (1 - t) * 120);
    return `rgb(${{r}},${{g}},${{b}})`;
  }} else {{
    const t = (v - 0.5) * 2;
    const r = Math.round(240);
    const g = Math.round(220 - t * 180);
    const b = Math.round(20);
    return `rgb(${{r}},${{g}},${{b}})`;
  }}
}}

function renderCategory(cat) {{
  const contentEl = document.getElementById('content');
  const catData = ALL_DATA[cat];
  if (!catData) {{
    contentEl.innerHTML = `<p style="color:#8b8fa3">No data for ${{cat}}</p>`;
    return;
  }}

  let html = '<div class="grid">';
  for (let pi = 0; pi < catData.length; pi++) {{
    const hm = catData[pi];
    html += `<div class="heatmap-card">
      <h3>${{hm.y_name}} vs ${{hm.x_name}}</h3>
      <canvas id="hm-${{pi}}" data-idx="${{pi}}"></canvas>
      <div class="color-bar">
        <span>0.0</span>
        <div class="gradient" style="background:linear-gradient(to right, rgb(20,20,200), rgb(240,220,20), rgb(240,40,20))"></div>
        <span>1.0</span>
      </div>
    </div>`;
  }}
  html += '</div>';
  contentEl.innerHTML = html;

  // Draw each heatmap
  for (let pi = 0; pi < catData.length; pi++) {{
    drawHeatmap(cat, pi);
  }}
}}

function drawHeatmap(cat, idx) {{
  const canvas = document.getElementById(`hm-${{idx}}`);
  if (!canvas) return;
  const hm = ALL_DATA[cat][idx];
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const size = rect.width;
  canvas.width = size * dpr;
  canvas.height = size * dpr;
  canvas.style.height = size + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const margin = {{ left: 44, bottom: 28, top: 8, right: 8 }};
  const cw = size - margin.left - margin.right;
  const ch = size - margin.top - margin.bottom;
  const cellW = cw / RES;
  const cellH = ch / RES;

  // Draw cells
  for (let yi = 0; yi < RES; yi++) {{
    for (let xi = 0; xi < RES; xi++) {{
      const v = hm.urgency[yi][xi];
      ctx.fillStyle = urgencyColor(v);
      const x = margin.left + xi * cellW;
      const y = margin.top + (RES - 1 - yi) * cellH;  // flip Y so 0 is bottom
      ctx.fillRect(x, y, cellW + 0.5, cellH + 0.5);
    }}
  }}

  // Axes
  ctx.font = '10px ui-sans-serif, system-ui, sans-serif';
  ctx.fillStyle = '#8b8fa3';
  ctx.textAlign = 'center';
  for (let i = 0; i <= 4; i++) {{
    const v = (i / 4).toFixed(1);
    ctx.fillText(v, margin.left + (i / 4) * cw, size - 4);
  }}
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {{
    const v = (i / 4).toFixed(1);
    ctx.fillText(v, margin.left - 4, margin.top + (1 - i / 4) * ch + 3);
  }}

  // Axis labels
  ctx.fillStyle = '#6b7080';
  ctx.textAlign = 'center';
  ctx.fillText(hm.x_name, margin.left + cw / 2, size - 14 + 24);
  ctx.save();
  ctx.translate(10, margin.top + ch / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(hm.y_name, 0, 0);
  ctx.restore();

  // Hover
  canvas.addEventListener('mousemove', (ev) => {{
    const r = canvas.getBoundingClientRect();
    const mx = ev.clientX - r.left;
    const my = ev.clientY - r.top;
    const xi = Math.floor((mx - margin.left) / cellW);
    const yi = RES - 1 - Math.floor((my - margin.top) / cellH);
    if (xi >= 0 && xi < RES && yi >= 0 && yi < RES) {{
      const v = hm.urgency[yi][xi];
      const xv = (xi / (RES - 1)).toFixed(2);
      const yv = (yi / (RES - 1)).toFixed(2);
      tooltip.innerHTML = `<div>${{hm.x_name}}: ${{xv}}</div><div>${{hm.y_name}}: ${{yv}}</div><div style="color:${{urgencyColor(v)}}; font-weight:600">urgency: ${{v.toFixed(3)}}</div>`;
      tooltip.style.display = 'block';
      tooltip.style.left = (ev.clientX + 14) + 'px';
      tooltip.style.top = (ev.clientY - 10) + 'px';
    }} else {{
      tooltip.style.display = 'none';
    }}
  }});
  canvas.addEventListener('mouseleave', () => {{
    tooltip.style.display = 'none';
  }});
}}

renderCategory(activeCat);
// Re-render on resize
window.addEventListener('resize', () => renderCategory(activeCat));
</script>
</body></html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate urgency heatmap visualization")
    parser.add_argument("weights", help="Path to ability_eval_weights.json")
    parser.add_argument("-o", "--output", default="generated/reports/urgency_heatmaps.html")
    parser.add_argument("--resolution", type=int, default=40, help="Grid resolution per axis")
    args = parser.parse_args()

    print(f"Loading weights from {args.weights}...")
    with open(args.weights) as f:
        weights = json.load(f)
    print(f"  Categories: {list(weights.keys())}")

    all_heatmaps = {}

    for cat_name in sorted(weights.keys()):
        cat_info = CATEGORY_FEATURES.get(cat_name)
        if cat_info is None:
            print(f"  Skipping {cat_name}: no feature definition")
            continue

        # Check if weight input dim matches our expected feature count
        first_layer = weights[cat_name]["layers"][0]
        actual_dim = len(first_layer["w"])
        expected_dim = cat_info["count"]

        # Adjust defaults if needed
        if actual_dim != expected_dim:
            print(f"  {cat_name}: expected {expected_dim} features, got {actual_dim} — adjusting")
            if actual_dim > expected_dim:
                cat_info["defaults"] = cat_info["defaults"] + [0.0] * (actual_dim - expected_dim)
                cat_info["names"] = cat_info["names"] + [f"feat_{i}" for i in range(expected_dim, actual_dim)]
            else:
                cat_info["defaults"] = cat_info["defaults"][:actual_dim]
                cat_info["names"] = cat_info["names"][:actual_dim]
            cat_info["count"] = actual_dim
            # Re-validate interesting pairs
            cat_info["interesting_pairs"] = [
                (a, b) for a, b in cat_info["interesting_pairs"]
                if a < actual_dim and b < actual_dim
            ]

        print(f"\n  {cat_name} ({actual_dim} features):")
        model = load_category_model(weights, cat_name)

        heatmaps = []
        for feat_x, feat_y in cat_info["interesting_pairs"]:
            hm = generate_heatmap(model, cat_info, feat_x, feat_y, args.resolution)
            print(f"    {hm['x_name']} vs {hm['y_name']}: "
                  f"urgency range [{np.min(hm['urgency']):.3f}, {np.max(hm['urgency']):.3f}]")
            heatmaps.append(hm)

        all_heatmaps[cat_name] = heatmaps

    print(f"\nGenerating HTML...")
    html = build_html(all_heatmaps, args.resolution)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"  Written to {args.output} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
