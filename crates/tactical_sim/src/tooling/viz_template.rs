use crate::sim::{ReplayResult, Team};

pub fn build_visualization_html(
    title: &str,
    subtitle: &str,
    replay: &ReplayResult,
    event_rows: &str,
    frame_rows: &str,
    obstacle_rows: &str,
    seed: u64,
    ticks: u32,
) -> String {
    let max_tick = replay.final_state.tick;

    format!(
        r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>{title}</title>
<style>
  body {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin: 16px; background: #111318; color: #e8eaf0; }}
  .header {{ display:flex; align-items:flex-end; justify-content:space-between; gap: 10px; margin-bottom: 8px; }}
  .tabs {{ display:flex; gap:6px; margin: 8px 0 10px; }}
  .tab-btn {{ background:#1b2233; border:1px solid #3a4259; color:#dbe4ff; padding:6px 10px; cursor:pointer; }}
  .tab-btn.active {{ background:#2a3552; }}
  .pane {{ display:none; }}
  .pane.active {{ display:block; }}
  .controls {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-bottom: 12px; }}
  .map-controls {{ display: grid; grid-template-columns: 1fr auto auto auto; gap: 8px; align-items: center; margin: 8px 0 10px; }}
  input, select {{ background: #1e2230; color: #eef; border: 1px solid #3a4259; padding: 6px 8px; }}
  button {{ background: #1e2230; color: #eef; border: 1px solid #3a4259; padding: 6px 10px; cursor: pointer; }}
  #timeline {{ display: grid; grid-template-columns: repeat(100, 1fr); gap: 1px; margin: 10px 0 16px; }}
  #abilityTimeline {{ display: grid; grid-template-columns: repeat(100, 1fr); gap: 1px; margin: 8px 0 10px; }}
  .bar {{ height: 10px; background: #3f4a67; }}
  .ability-stack {{ display:flex; flex-direction:column; justify-content:flex-end; height:52px; background:#151b2a; }}
  .ability-seg {{ width:100%; }}
  #map-wrap {{ border: 1px solid #2a3042; background: #151927; padding: 8px; margin-bottom: 10px; }}
  #map {{ width: 100%; max-width: 900px; height: 360px; background: #0f1320; border: 1px solid #2f3850; }}
  .legend {{ color: #9ca7c5; font-size: 12px; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th, td {{ border-bottom: 1px solid #2a3042; text-align: left; padding: 4px 6px; }}
  tr:hover {{ background: #1d2232; }}
  .meta {{ color: #9ca7c5; margin-bottom: 8px; }}
  .kpis {{ display:grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 8px; margin: 10px 0; }}
  .kpi {{ background:#151b2a; border:1px solid #2a3042; padding:8px; }}
  .kpi .label {{ color:#9ca7c5; font-size:11px; }}
  .kpi .value {{ font-size:18px; }}
</style>
</head>
<body>
  <div class="header">
    <div>
      <h2 style="margin:0;">{title}</h2>
      <div class="meta">{subtitle}</div>
    </div>
    <div class="meta">seed={seed}, ticks={ticks}, event_hash={event_hash:016x}, state_hash={state_hash:016x}, max_tick={max_tick}</div>
  </div>
  <div class="tabs">
    <button class="tab-btn active" data-tab="map">Map</button>
    <button class="tab-btn" data-tab="events">Events</button>
    <button class="tab-btn" data-tab="metrics">Metrics</button>
  </div>
  <div id="pane-map" class="pane active">
    <div class="controls">
      <input id="search" placeholder="search detail/kind/source/target" />
      <select id="kind"></select>
      <select id="unit"></select>
    </div>
    <div id="map-wrap">
      <div class="map-controls">
        <input id="tick" type="range" min="0" max="{max_tick}" value="0" />
        <span id="tickLabel">tick 0</span>
        <span id="tickStats"></span>
        <select id="speedSel">
          <option value="220">0.5x</option>
          <option value="120" selected>1x</option>
          <option value="70">1.7x</option>
          <option value="40">3x</option>
        </select>
        <button id="playBtn">play</button>
      </div>
      <div class="map-controls">
        <label>link window</label>
        <input id="depth" type="range" min="1" max="30" value="12" />
        <span id="depthLabel">12</span>
      </div>
      <canvas id="map" width="900" height="360"></canvas>
      <div class="legend">units: blue=Hero, red=Enemy | links: red=damage, green=heal, amber=casts, cyan=reposition | walls: hatched blocks | gate: highlighted opening</div>
    </div>
    <div id="timeline"></div>
    <div class="legend">ability timeline: amber=abilities, green=heals, blue=attacks</div>
    <div id="abilityTimeline"></div>
  </div>
  <div id="pane-events" class="pane">
    <table>
      <thead>
        <tr><th>tick</th><th>kind</th><th>src</th><th>dst</th><th>value</th><th>detail</th></tr>
      </thead>
      <tbody id="rows"></tbody>
    </table>
  </div>
  <div id="pane-metrics" class="pane">
    <div class="kpis">
      <div class="kpi"><div class="label">Winner</div><div class="value">{winner}</div></div>
      <div class="kpi"><div class="label">First Death</div><div class="value">{first_death}</div></div>
      <div class="kpi"><div class="label">Casts Completed</div><div class="value">{casts_completed}</div></div>
      <div class="kpi"><div class="label">Heals Completed</div><div class="value">{heals_completed}</div></div>
      <div class="kpi"><div class="label">Repositions</div><div class="value">{repositions}</div></div>
      <div class="kpi"><div class="label">Invariant Violations</div><div class="value">{invariants}</div></div>
      <div class="kpi"><div class="label">Hero Alive</div><div class="value">{hero_alive}</div></div>
      <div class="kpi"><div class="label">Enemy Alive</div><div class="value">{enemy_alive}</div></div>
    </div>
    <h4>Ability Usage By Unit</h4>
    <table>
      <thead>
        <tr><th>unit</th><th>ability starts</th><th>heal starts</th><th>attack starts</th></tr>
      </thead>
      <tbody id="abilityUnitRows"></tbody>
    </table>
  </div>
<script>
const raw = `{rows}`;
const lines = raw.trim().split('\n').filter(Boolean);
const data = lines.map(line => {{
  const [tick, kind, src, dst, value, ...detail] = line.split('\t');
  return {{ tick: Number(tick), kind, src, dst, value, detail: detail.join('\t') }};
}});"#,
        title = title,
        subtitle = subtitle,
        seed = seed,
        ticks = ticks,
        event_hash = replay.event_log_hash,
        state_hash = replay.final_state_hash,
        max_tick = max_tick,
        winner = format!("{:?}", replay.metrics.winner),
        first_death = replay
            .metrics
            .tick_to_first_death
            .map_or_else(|| "-".to_string(), |t| t.to_string()),
        casts_completed = replay.metrics.casts_completed,
        heals_completed = replay.metrics.heals_completed,
        repositions = replay.metrics.reposition_for_range_events,
        invariants = replay.metrics.invariant_violations,
        hero_alive = replay
            .final_state
            .units
            .iter()
            .filter(|u| u.team == Team::Hero && u.hp > 0)
            .count(),
        enemy_alive = replay
            .final_state
            .units
            .iter()
            .filter(|u| u.team == Team::Enemy && u.hp > 0)
            .count(),
        rows = event_rows
            .replace('\\', r"\\")
            .replace('`', r"\`")
            .replace('$', r"\$"),
    ) + &build_visualization_js(frame_rows, obstacle_rows)
}

fn build_visualization_js(frame_rows: &str, obstacle_rows: &str) -> String {
    format!(
        r#"
const rawFrames = `{frame_rows}`;
const frameLines = rawFrames.trim().split('\n').filter(Boolean);
const frameData = frameLines.map(line => {{
  const [tick, id, team, hp, x, y] = line.split('\t');
  return {{ tick: Number(tick), id: Number(id), team, hp: Number(hp), x: Number(x), y: Number(y) }};
}});
const rawObstacles = `{obstacle_rows}`;
const obstacleData = rawObstacles.trim() ? rawObstacles.trim().split('\n').map(line => {{
  const [min_x,max_x,min_y,max_y] = line.split('\t').map(Number);
  return {{min_x,max_x,min_y,max_y}};
}}) : [];
const framesByTick = new Map();
for (const f of frameData) {{
  if (!framesByTick.has(f.tick)) framesByTick.set(f.tick, []);
  framesByTick.get(f.tick).push(f);
}}
for (const list of framesByTick.values()) list.sort((a,b) => a.id - b.id);

const kinds = ['ALL', ...new Set(data.map(d => d.kind)).values()];
const units = ['ALL', ...new Set(data.flatMap(d => [d.src, d.dst]).filter(v => v !== '-')).values()].sort((a,b)=>Number(a)-Number(b));
const kindSel = document.getElementById('kind'), unitSel = document.getElementById('unit');
for (const k of kinds) {{ const o=document.createElement('option'); o.value=k; o.textContent=k; kindSel.appendChild(o); }}
for (const u of units) {{ const o=document.createElement('option'); o.value=u; o.textContent=u; unitSel.appendChild(o); }}

const rowsEl = document.getElementById('rows'), timelineEl = document.getElementById('timeline');
const abilityTimelineEl = document.getElementById('abilityTimeline'), abilityUnitRowsEl = document.getElementById('abilityUnitRows');
const searchEl = document.getElementById('search'), tickEl = document.getElementById('tick');
const tickLabelEl = document.getElementById('tickLabel'), tickStatsEl = document.getElementById('tickStats');
const playBtn = document.getElementById('playBtn'), speedSel = document.getElementById('speedSel');
const depthEl = document.getElementById('depth'), depthLabelEl = document.getElementById('depthLabel');
const mapEl = document.getElementById('map'), ctx = mapEl.getContext('2d');

const world = frameData.reduce((acc, f) => {{ acc.minX = Math.min(acc.minX, f.x); acc.maxX = Math.max(acc.maxX, f.x); acc.minY = Math.min(acc.minY, f.y); acc.maxY = Math.max(acc.maxY, f.y); return acc; }}, {{ minX: 0, maxX: 1, minY: 0, maxY: 1 }});
const pad = 0.5;
world.minX -= pad; world.maxX += pad;
world.minY -= pad; world.maxY += pad;
if (Math.abs(world.maxX - world.minX) < 0.001) {{ world.maxX += 1; world.minX -= 1; }}
if (Math.abs(world.maxY - world.minY) < 0.001) {{ world.maxY += 1; world.minY -= 1; }}

function mapToCanvas(x, y) {{
  const w = mapEl.width, h = mapEl.height;
  const px = ((x - world.minX) / (world.maxX - world.minX)) * (w - 40) + 20;
  const py = h - ((((y - world.minY) / (world.maxY - world.minY)) * (h - 40)) + 20);
  return [px, py];
}}

function approxEq(a, b, eps) {{
  return Math.abs(a - b) <= eps;
}}

function detectChokepoints(obstacles) {{
  const eps = 0.12;
  const groups = [];
  for (const o of obstacles) {{
    let group = null;
    for (const g of groups) {{
      if (approxEq(g.min_x, o.min_x, eps) && approxEq(g.max_x, o.max_x, eps)) {{
        group = g;
        break;
      }}
    }}
    if (!group) {{
      group = {{ min_x: o.min_x, max_x: o.max_x, segs: [] }};
      groups.push(group);
    }}
    group.segs.push(o);
  }}

  const gates = [];
  for (const g of groups) {{
    if (g.segs.length < 2) continue;
    g.segs.sort((a, b) => a.min_y - b.min_y);
    for (let i = 0; i < g.segs.length - 1; i++) {{
      const a = g.segs[i];
      const b = g.segs[i + 1];
      const gap = b.min_y - a.max_y;
      if (gap > 0.25) {{
        gates.push({{
          min_x: g.min_x,
          max_x: g.max_x,
          min_y: a.max_y,
          max_y: b.min_y
        }});
      }}
    }}
  }}
  return gates;
}}

const chokepoints = detectChokepoints(obstacleData);

function eventColor(kind) {{
  if (kind === 'DamageApplied') return '#ff6b6b';
  if (kind === 'HealApplied') return '#4ade80';
  if (kind === 'ControlApplied' || kind === 'ControlCastStarted' || kind === 'UnitControlled') return '#c084fc';
  if (kind.includes('CastStarted')) return '#fbbf24';
  if (kind === 'AttackRepositioned' || kind === 'Moved') return '#22d3ee';
  return '#94a3b8';
}}

function drawMap(currentTick) {{
  ctx.clearRect(0, 0, mapEl.width, mapEl.height);
  ctx.fillStyle = '#0f1320';
  ctx.fillRect(0, 0, mapEl.width, mapEl.height);

  ctx.strokeStyle = '#1f273a'; ctx.lineWidth = 1;
  for (let i=0; i<=10; i++) {{
    const x = 20 + (i/10) * (mapEl.width - 40);
    const y = 20 + (i/10) * (mapEl.height - 40);
    ctx.beginPath(); ctx.moveTo(x, 20); ctx.lineTo(x, mapEl.height - 20); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(20, y); ctx.lineTo(mapEl.width - 20, y); ctx.stroke();
  }}

  for (const o of obstacleData) {{
    const [ax, ay] = mapToCanvas(o.min_x, o.min_y);
    const [bx, by] = mapToCanvas(o.max_x, o.max_y);
    const minX = Math.min(ax, bx), maxX = Math.max(ax, bx);
    const minY = Math.min(ay, by), maxY = Math.max(ay, by);
    ctx.fillStyle = 'rgba(170, 178, 204, 0.48)';
    ctx.fillRect(minX, minY, maxX-minX, maxY-minY);
    ctx.strokeStyle = 'rgba(224, 230, 255, 0.82)';
    ctx.lineWidth = 1.6;
    ctx.strokeRect(minX, minY, maxX-minX, maxY-minY);

    // Diagonal hatching helps blocked regions stand out from path trails.
    ctx.save();
    ctx.beginPath();
    ctx.rect(minX, minY, maxX-minX, maxY-minY);
    ctx.clip();
    ctx.strokeStyle = 'rgba(230, 236, 255, 0.28)';
    ctx.lineWidth = 1.0;
    const step = 8;
    for (let x = minX - (maxY - minY); x < maxX + (maxY - minY); x += step) {{
      ctx.beginPath();
      ctx.moveTo(x, minY);
      ctx.lineTo(x + (maxY - minY), maxY);
      ctx.stroke();
    }}
    ctx.restore();
  }}

  for (const gate of chokepoints) {{
    const [ax, ay] = mapToCanvas(gate.min_x, gate.min_y);
    const [bx, by] = mapToCanvas(gate.max_x, gate.max_y);
    const minX = Math.min(ax, bx), maxX = Math.max(ax, bx);
    const minY = Math.min(ay, by), maxY = Math.max(ay, by);
    const cx = (minX + maxX) * 0.5;
    const cy = (minY + maxY) * 0.5;
    const width = Math.max(16, maxX - minX + 10);
    const height = Math.max(24, maxY - minY);
    ctx.strokeStyle = 'rgba(251, 191, 36, 0.95)';
    ctx.lineWidth = 2.4;
    ctx.strokeRect(cx - width * 0.5, cy - height * 0.5, width, height);
    ctx.beginPath();
    ctx.moveTo(cx - width * 0.5, cy);
    ctx.lineTo(cx + width * 0.5, cy);
    ctx.stroke();
    ctx.fillStyle = 'rgba(251, 191, 36, 0.95)';
    ctx.font = 'bold 11px ui-monospace, monospace';
    ctx.fillText('CHOKEPOINT GATE', cx + 10, cy - 8);
  }}

  const unitsNow = framesByTick.get(currentTick) || [];
  const byId = new Map(unitsNow.map(u => [String(u.id), u]));
  const recentDepth = Number(depthEl.value || 12);
  const eventsNow = data.filter(d => d.tick === currentTick);
  const recentEvents = data.filter(d => d.tick <= currentTick && d.tick > currentTick - recentDepth);

  // Unit trails for recent motion context.
  for (const u of unitsNow) {{
    const trail = [];
    for (let t = Math.max(0, currentTick - 24); t <= currentTick; t++) {{
      const at = (framesByTick.get(t) || []).find(x => x.id === u.id);
      if (at) trail.push(at);
    }}
    if (trail.length >= 2) {{
      ctx.strokeStyle = u.team === 'Hero' ? '#3b82f6' : '#ef4444';
      ctx.globalAlpha = 0.35;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i=0; i<trail.length; i++) {{
        const [tx, ty] = mapToCanvas(trail[i].x, trail[i].y);
        if (i === 0) ctx.moveTo(tx, ty); else ctx.lineTo(tx, ty);
      }}
      ctx.stroke();
      ctx.globalAlpha = 1.0;
    }}
  }}

  for (const e of recentEvents) {{
    if (e.src === '-' || e.dst === '-') continue;
    const a = byId.get(e.src), b = byId.get(e.dst);
    if (!a || !b) continue;
    const [ax, ay] = mapToCanvas(a.x, a.y);
    const [bx, by] = mapToCanvas(b.x, b.y);
    ctx.strokeStyle = eventColor(e.kind);
    const age = Math.max(0, currentTick - e.tick);
    ctx.globalAlpha = Math.max(0.18, 1.0 - age / recentDepth);
    ctx.lineWidth = e.tick === currentTick ? 2.6 : 1.4;
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    ctx.globalAlpha = 1.0;
  }}

  for (const u of unitsNow) {{
    const [x, y] = mapToCanvas(u.x, u.y);
    ctx.fillStyle = u.team === 'Hero' ? '#60a5fa' : '#f87171';
    ctx.beginPath(); ctx.arc(x, y, u.hp > 0 ? 8 : 5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#e5e7eb'; ctx.font = '12px ui-monospace, monospace';
    ctx.fillText(`${{u.id}} (hp:${{u.hp}})`, x + 10, y - 10);
  }}
}}

function render() {{
  const q = searchEl.value.trim().toLowerCase();
  const kind = kindSel.value;
  const unit = unitSel.value;
  const filtered = data.filter(d => {{
    if (kind !== 'ALL' && d.kind !== kind) return false;
    if (unit !== 'ALL' && d.src !== unit && d.dst !== unit) return false;
    if (q && !(`${{d.kind}} ${{d.src}} ${{d.dst}} ${{d.detail}}`.toLowerCase().includes(q))) return false;
    return true;
  }});

  rowsEl.innerHTML = filtered.map(d => `<tr><td>${{d.tick}}</td><td>${{d.kind}}</td><td>${{d.src}}</td><td>${{d.dst}}</td><td>${{d.value}}</td><td>${{d.detail}}</td></tr>`).join('');

  const maxTick = Math.max(...data.map(d => d.tick), 1);
  const bins = new Array(100).fill(0);
  for (const d of filtered) {{
    const idx = Math.min(99, Math.floor((d.tick / maxTick) * 100));
    bins[idx] += 1;
  }}
  const maxBin = Math.max(...bins, 1);
  timelineEl.innerHTML = bins.map(v => `<div class='bar' title='${{v}} events' style='height:${{Math.max(3, (v/maxBin)*42)}}px'></div>`).join('');

  const abilityBins = new Array(100).fill(0), healBins = new Array(100).fill(0), attackBins = new Array(100).fill(0);
  for (const d of filtered) {{
    const idx = Math.min(99, Math.floor((d.tick / maxTick) * 100));
    if (d.kind === 'AbilityCastStarted' || d.kind === 'ControlCastStarted') abilityBins[idx] += 1;
    if (d.kind === 'HealCastStarted') healBins[idx] += 1;
    if (d.kind === 'CastStarted') attackBins[idx] += 1;
  }}
  const maxAbilityBin = Math.max(
    ...abilityBins.map((v, i) => v + healBins[i] + attackBins[i]),
    1
  );
  abilityTimelineEl.innerHTML = abilityBins.map((_, i) => {{
    const a = abilityBins[i], h = healBins[i], atk = attackBins[i];
    const aH = Math.round((a / maxAbilityBin) * 50);
    const hH = Math.round((h / maxAbilityBin) * 50);
    const atkH = Math.round((atk / maxAbilityBin) * 50);
    return `<div class='ability-stack' title='abilities=${{a}} heals=${{h}} attacks=${{atk}}'>
      <div class='ability-seg' style='height:${{Math.max(0, atkH)}}px;background:#60a5fa;'></div>
      <div class='ability-seg' style='height:${{Math.max(0, hH)}}px;background:#4ade80;'></div>
      <div class='ability-seg' style='height:${{Math.max(0, aH)}}px;background:#fbbf24;'></div>
    </div>`;
  }}).join('');

  const perUnit = new Map();
  for (const d of data) {{
    if (!perUnit.has(d.src) && d.src !== '-') perUnit.set(d.src, {{ ability: 0, heal: 0, attack: 0 }});
    if (d.src === '-') continue;
    if (d.kind === 'AbilityCastStarted' || d.kind === 'ControlCastStarted') perUnit.get(d.src).ability += 1;
    if (d.kind === 'HealCastStarted') perUnit.get(d.src).heal += 1;
    if (d.kind === 'CastStarted') perUnit.get(d.src).attack += 1;
  }}
  const unitRows = Array.from(perUnit.entries())
    .sort((a,b) => Number(a[0]) - Number(b[0]))
    .map(([unit, v]) => `<tr><td>${{unit}}</td><td>${{v.ability}}</td><td>${{v.heal}}</td><td>${{v.attack}}</td></tr>`)
    .join('');
  abilityUnitRowsEl.innerHTML = unitRows;

  const tick = Number(tickEl.value);
  tickLabelEl.textContent = `tick ${{tick}}`;
  depthLabelEl.textContent = String(depthEl.value);
  const depth = Number(depthEl.value || 12);
  tickStatsEl.textContent = `${{data.filter(d=>d.tick===tick).length}} events @ tick | ${{data.filter(d=>d.tick<=tick && d.tick>tick-depth).length}} in last ${{depth}}`;
  drawMap(tick);
}}

searchEl.addEventListener('input', render);
kindSel.addEventListener('change', render);
unitSel.addEventListener('change', render);
tickEl.addEventListener('input', render);
depthEl.addEventListener('input', render);
let playTimer = null;
playBtn.addEventListener('click', () => {{
  if (playTimer) {{
    clearInterval(playTimer);
    playTimer = null;
    playBtn.textContent = 'play';
    return;
  }}
  playBtn.textContent = 'pause';
  playTimer = setInterval(() => {{
    let t = Number(tickEl.value) + 1;
    if (t > Number(tickEl.max)) t = 0;
    tickEl.value = String(t);
    render();
  }}, Number(speedSel.value || 120));
}});
speedSel.addEventListener('change', () => {{
  if (playTimer) {{
    clearInterval(playTimer);
    playTimer = null;
    playBtn.textContent = 'play';
  }}
}});
for (const btn of document.querySelectorAll('.tab-btn')) {{
  btn.addEventListener('click', () => {{
    for (const b of document.querySelectorAll('.tab-btn')) b.classList.remove('active');
    btn.classList.add('active');
    const tab = btn.dataset.tab;
    for (const pane of document.querySelectorAll('.pane')) pane.classList.remove('active');
    document.getElementById(`pane-${{tab}}`).classList.add('active');
  }});
}}
render();
</script>
</body>
</html>"#,
        frame_rows = frame_rows
            .replace('\\', r"\\")
            .replace('`', r"\`")
            .replace('$', r"\$"),
        obstacle_rows = obstacle_rows
            .replace('\\', r"\\")
            .replace('`', r"\`")
            .replace('$', r"\$")
    )
}
