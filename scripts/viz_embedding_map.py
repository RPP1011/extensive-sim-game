#!/usr/bin/env python3
"""Generate an interactive 2D embedding map of ability embeddings.

Usage:
    uv run --with numpy --with scikit-learn scripts/viz_embedding_map.py generated/ability_encoder_embeddings.json
    uv run --with numpy --with scikit-learn scripts/viz_embedding_map.py generated/ability_encoder_embeddings.json -o generated/reports/embedding_map.html

Reads the 32-dim ability embeddings and projects them to 2D via t-SNE,
then generates a self-contained interactive HTML report.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# t-SNE implementation (lightweight, no sklearn dependency required at runtime
# but we use sklearn if available for speed)
# ---------------------------------------------------------------------------

def tsne_project(embeddings: np.ndarray, perplexity: float = 15.0, n_iter: int = 1000,
                 lr: float = 200.0, seed: int = 42) -> np.ndarray:
    """Project high-dim embeddings to 2D via t-SNE."""
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=min(perplexity, len(embeddings) - 1),
                     max_iter=n_iter, learning_rate=lr, random_state=seed, init="pca")
        return tsne.fit_transform(embeddings)
    except ImportError:
        # Fallback: PCA projection (much simpler but still useful)
        print("  sklearn not available, falling back to PCA projection")
        return pca_project(embeddings, seed)


def pca_project(embeddings: np.ndarray, seed: int = 42) -> np.ndarray:
    """Simple PCA to 2D."""
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Take top 2 eigenvectors
    idx = np.argsort(eigenvalues)[::-1][:2]
    components = eigenvectors[:, idx]
    return centered @ components


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    "damage_unit": "#ef4444",    # red
    "damage_aoe": "#f97316",     # orange
    "cc_unit": "#a855f7",        # purple
    "heal_unit": "#22c55e",      # green
    "heal_aoe": "#10b981",       # emerald
    "defense": "#3b82f6",        # blue
    "utility": "#eab308",        # yellow
    "summon": "#06b6d4",         # cyan
    "obstacle": "#78716c",       # stone
}


def build_html(abilities: list, coords_2d: np.ndarray) -> str:
    """Build a self-contained interactive HTML visualization."""
    # Prepare data as JSON for embedding in HTML
    points = []
    for i, ab in enumerate(abilities):
        points.append({
            "x": float(coords_2d[i, 0]),
            "y": float(coords_2d[i, 1]),
            "hero": ab["hero"],
            "ability": ab["ability"],
            "category": ab["category"],
        })

    categories = sorted(set(p["category"] for p in points))
    heroes = sorted(set(p["hero"] for p in points))
    cat_colors_json = json.dumps(CATEGORY_COLORS)
    points_json = json.dumps(points)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Ability Embedding Map</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: ui-sans-serif, system-ui, sans-serif; background: #111318; color: #e8eaf0; }}
.header {{ padding: 16px 24px; border-bottom: 1px solid #2a2e3a; }}
.header h1 {{ font-size: 20px; font-weight: 600; }}
.header p {{ font-size: 13px; color: #8b8fa3; margin-top: 4px; }}
.container {{ display: flex; height: calc(100vh - 80px); }}
.sidebar {{ width: 240px; padding: 12px; border-right: 1px solid #2a2e3a; overflow-y: auto; flex-shrink: 0; }}
.sidebar h3 {{ font-size: 13px; color: #8b8fa3; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }}
.legend-item {{ display: flex; align-items: center; gap: 8px; padding: 4px 6px; border-radius: 4px; cursor: pointer; font-size: 13px; margin-bottom: 2px; }}
.legend-item:hover {{ background: #1e2130; }}
.legend-item.dimmed {{ opacity: 0.25; }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
.legend-count {{ color: #8b8fa3; margin-left: auto; font-size: 11px; }}
.canvas-wrap {{ flex: 1; position: relative; }}
canvas {{ width: 100%; height: 100%; cursor: crosshair; }}
.tooltip {{ position: fixed; pointer-events: none; background: #1a1d28; border: 1px solid #3a3e4a; border-radius: 6px; padding: 8px 12px; font-size: 13px; display: none; z-index: 10; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }}
.tooltip .hero {{ color: #8b8fa3; font-size: 11px; }}
.tooltip .cat {{ font-size: 11px; margin-top: 2px; }}
.search-box {{ width: 100%; padding: 6px 8px; background: #1a1d28; border: 1px solid #2a2e3a; border-radius: 4px; color: #e8eaf0; font-size: 13px; margin-bottom: 12px; outline: none; }}
.search-box:focus {{ border-color: #4a6fa5; }}
.stats {{ font-size: 11px; color: #8b8fa3; padding: 8px 0; border-top: 1px solid #2a2e3a; margin-top: 8px; }}
</style></head>
<body>
<div class="header">
  <h1>Ability Embedding Map</h1>
  <p>t-SNE projection of {len(points)} ability embeddings (32-dim &rarr; 2D) &mdash; colored by category</p>
</div>
<div class="container">
  <div class="sidebar">
    <input class="search-box" id="search" placeholder="Search abilities..." />
    <h3>Categories</h3>
    <div id="legend"></div>
    <div class="stats" id="stats"></div>
  </div>
  <div class="canvas-wrap">
    <canvas id="canvas"></canvas>
  </div>
</div>
<div class="tooltip" id="tooltip">
  <div class="name" id="tt-name"></div>
  <div class="hero" id="tt-hero"></div>
  <div class="cat" id="tt-cat"></div>
</div>
<script>
const POINTS = {points_json};
const CAT_COLORS = {cat_colors_json};
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');
const searchBox = document.getElementById('search');

let dpr = window.devicePixelRatio || 1;
let W, H;
let activeCategories = new Set(Object.keys(CAT_COLORS));
let searchFilter = '';
let hoveredIdx = -1;

// Compute bounds
let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
for (const p of POINTS) {{
  if (p.x < minX) minX = p.x;
  if (p.x > maxX) maxX = p.x;
  if (p.y < minY) minY = p.y;
  if (p.y > maxY) maxY = p.y;
}}
const pad = 0.08;
const rangeX = (maxX - minX) || 1;
const rangeY = (maxY - minY) || 1;
minX -= rangeX * pad; maxX += rangeX * pad;
minY -= rangeY * pad; maxY += rangeY * pad;

function toScreen(px, py) {{
  return [
    ((px - minX) / (maxX - minX)) * W,
    ((py - minY) / (maxY - minY)) * H,
  ];
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

function isVisible(p, idx) {{
  if (!activeCategories.has(p.category)) return false;
  if (searchFilter && !p.ability.toLowerCase().includes(searchFilter) &&
      !p.hero.toLowerCase().includes(searchFilter)) return false;
  return true;
}}

function draw() {{
  ctx.clearRect(0, 0, W, H);

  // Draw dimmed points first
  for (let i = 0; i < POINTS.length; i++) {{
    const p = POINTS[i];
    if (isVisible(p, i)) continue;
    const [sx, sy] = toScreen(p.x, p.y);
    ctx.beginPath();
    ctx.arc(sx, sy, 3, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(60,60,70,0.3)';
    ctx.fill();
  }}

  // Draw active points
  for (let i = 0; i < POINTS.length; i++) {{
    const p = POINTS[i];
    if (!isVisible(p, i)) continue;
    const [sx, sy] = toScreen(p.x, p.y);
    const r = (i === hoveredIdx) ? 8 : 5;
    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.fillStyle = CAT_COLORS[p.category] || '#888';
    ctx.globalAlpha = (i === hoveredIdx) ? 1.0 : 0.8;
    ctx.fill();
    ctx.globalAlpha = 1.0;
    if (i === hoveredIdx) {{
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
    }}
  }}

  // Draw labels for search matches when few results
  if (searchFilter) {{
    const matches = POINTS.filter((p, i) => isVisible(p, i));
    if (matches.length <= 20) {{
      ctx.font = '11px ui-sans-serif, system-ui, sans-serif';
      ctx.fillStyle = '#e8eaf0';
      ctx.textAlign = 'left';
      for (const p of matches) {{
        const [sx, sy] = toScreen(p.x, p.y);
        ctx.fillText(p.ability, sx + 8, sy + 3);
      }}
    }}
  }}
}}

// Legend
const legend = document.getElementById('legend');
const catCounts = {{}};
for (const p of POINTS) catCounts[p.category] = (catCounts[p.category] || 0) + 1;
for (const cat of Object.keys(CAT_COLORS).sort()) {{
  const item = document.createElement('div');
  item.className = 'legend-item';
  item.innerHTML = `<span class="legend-dot" style="background:${{CAT_COLORS[cat]}}"></span>${{cat}}<span class="legend-count">${{catCounts[cat] || 0}}</span>`;
  item.addEventListener('click', () => {{
    if (activeCategories.has(cat)) {{
      activeCategories.delete(cat);
      item.classList.add('dimmed');
    }} else {{
      activeCategories.add(cat);
      item.classList.remove('dimmed');
    }}
    draw();
  }});
  legend.appendChild(item);
}}

// Stats
document.getElementById('stats').textContent =
  `${{POINTS.length}} abilities from ${{new Set(POINTS.map(p => p.hero)).size}} heroes`;

// Search
searchBox.addEventListener('input', (e) => {{
  searchFilter = e.target.value.toLowerCase();
  draw();
}});

// Hover
canvas.addEventListener('mousemove', (e) => {{
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  let closest = -1, closestDist = 15;
  for (let i = 0; i < POINTS.length; i++) {{
    if (!isVisible(POINTS[i], i)) continue;
    const [sx, sy] = toScreen(POINTS[i].x, POINTS[i].y);
    const d = Math.hypot(sx - mx, sy - my);
    if (d < closestDist) {{ closestDist = d; closest = i; }}
  }}
  if (closest !== hoveredIdx) {{
    hoveredIdx = closest;
    draw();
    if (closest >= 0) {{
      const p = POINTS[closest];
      document.getElementById('tt-name').textContent = p.ability;
      document.getElementById('tt-hero').textContent = p.hero;
      document.getElementById('tt-cat').textContent = p.category;
      document.getElementById('tt-cat').style.color = CAT_COLORS[p.category];
      tooltip.style.display = 'block';
      tooltip.style.left = (e.clientX + 14) + 'px';
      tooltip.style.top = (e.clientY - 10) + 'px';
    }} else {{
      tooltip.style.display = 'none';
    }}
  }} else if (closest >= 0) {{
    tooltip.style.left = (e.clientX + 14) + 'px';
    tooltip.style.top = (e.clientY - 10) + 'px';
  }}
}});

canvas.addEventListener('mouseleave', () => {{
  hoveredIdx = -1;
  tooltip.style.display = 'none';
  draw();
}});

window.addEventListener('resize', resize);
resize();
</script>
</body></html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate ability embedding map visualization")
    parser.add_argument("embeddings", help="Path to ability_encoder_embeddings.json")
    parser.add_argument("-o", "--output", default="generated/reports/embedding_map.html")
    parser.add_argument("--perplexity", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading embeddings from {args.embeddings}...")
    with open(args.embeddings) as f:
        abilities = json.load(f)
    print(f"  {len(abilities)} abilities loaded")

    embeddings = np.array([a["embedding"] for a in abilities], dtype=np.float32)
    print(f"  Embedding shape: {embeddings.shape}")

    # Count per category
    from collections import Counter
    cats = Counter(a["category"] for a in abilities)
    for cat, count in sorted(cats.items()):
        print(f"    {cat}: {count}")

    print(f"\nRunning t-SNE (perplexity={args.perplexity})...")
    coords = tsne_project(embeddings, perplexity=args.perplexity, seed=args.seed)
    print(f"  Projection shape: {coords.shape}")

    print(f"\nGenerating HTML...")
    html = build_html(abilities, coords)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"  Written to {args.output} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
