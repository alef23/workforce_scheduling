from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import zarr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye un dashboard HTML estatico desde logs y buffers."
    )
    parser.add_argument(
        "--reports-dir",
        default="datasets/reports",
        help="Directorio con JSONL de mcts_generation. Default: datasets/reports.",
    )
    parser.add_argument(
        "--raw-path",
        default="datasets/raw/trajectories.zarr",
        help="TrajectoryBuffer raw. Default: datasets/raw/trajectories.zarr.",
    )
    parser.add_argument(
        "--stock-path",
        default="datasets/derived/stock_adjusted/trajectories.zarr",
        help="TrajectoryBuffer stock_adjusted.",
    )
    parser.add_argument(
        "--sample-path",
        default="datasets/samples/samples.zarr",
        help="SampleBuffer actual. Default: datasets/samples/samples.zarr.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="modules/evaluators/resnet/checkpoints",
        help="Directorio de checkpoints ResNet.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="HTML destino. Default: <reports-dir>/training_dashboard.html.",
    )
    parser.add_argument(
        "--max-sample-scan",
        type=int,
        default=200_000,
        help="Maximo de samples a leer para resumen. Default: 200000.",
    )
    parser.add_argument(
        "--max-trajectory-preview",
        type=int,
        default=20,
        help="Cantidad de trayectorias a listar como preview. Default: 20.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    output_path = Path(args.output) if args.output else reports_dir / "training_dashboard.html"

    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "paths": {
            "reports_dir": str(reports_dir),
            "raw_path": str(args.raw_path),
            "stock_path": str(args.stock_path),
            "sample_path": str(args.sample_path),
            "checkpoint_dir": str(args.checkpoint_dir),
        },
        "logs": read_training_logs(reports_dir),
        "buffers": {
            "raw": summarize_trajectory_buffer(
                args.raw_path,
                max_preview=int(args.max_trajectory_preview),
            ),
            "stock": summarize_trajectory_buffer(
                args.stock_path,
                max_preview=int(args.max_trajectory_preview),
            ),
            "samples": summarize_sample_buffer(
                args.sample_path,
                max_scan=int(args.max_sample_scan),
            ),
        },
        "checkpoints": summarize_checkpoints(args.checkpoint_dir),
    }
    data["derived"] = build_derived_summary(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_dashboard(data), encoding="utf-8")
    print(f"[dashboard] output={output_path}", flush=True)


def read_training_logs(reports_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        "runs": read_jsonl(reports_dir / "mcts_generation_runs.jsonl"),
        "cycles": read_jsonl(reports_dir / "mcts_generation_cycles.jsonl"),
        "learner_steps": read_jsonl(reports_dir / "mcts_generation_learner_steps.jsonl"),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                rows.append(
                    {
                        "event": "decode_error",
                        "path": str(path),
                        "line_number": line_number,
                        "error": str(exc),
                    }
                )
    return rows


def summarize_trajectory_buffer(
    store_path: str | Path,
    max_preview: int,
) -> dict[str, Any]:
    path = Path(store_path)
    if not path.exists():
        return {"exists": False, "path": str(path)}

    try:
        root = zarr.open_group(store=str(path), mode="r")
        group = root["trajectories"]
        ids = sorted(str(key) for key in group.group_keys())
        lengths = []
        rewards = []
        expansion_count = 0
        stock_reduced_count = 0
        preview = []

        for index, trajectory_id in enumerate(ids):
            trajectory_group = group[trajectory_id]
            attrs = dict(trajectory_group.attrs)
            length = _safe_int(attrs.get("length"))
            reward = _safe_float(attrs.get("final_reward"))
            has_expansion = _attr_bool(attrs.get("metadata.has_expansion_mode"))
            stock_reduced = _attr_bool(attrs.get("metadata.stock_was_reduced"))

            if length is not None:
                lengths.append(length)
            if reward is not None:
                rewards.append(reward)
            if has_expansion:
                expansion_count += 1
            if stock_reduced:
                stock_reduced_count += 1

            if index < max_preview:
                preview.append(
                    {
                        "trajectory_id": trajectory_id,
                        "length": length,
                        "final_reward": reward,
                        "stage": attrs.get("metadata.stage"),
                        "source_trajectory_id": attrs.get("metadata.source_trajectory_id"),
                        "stock_was_reduced": stock_reduced,
                        "output_stock": attrs.get("metadata.output_stock"),
                        "has_expansion_mode": has_expansion,
                        "first_expansion_step": attrs.get("metadata.first_expansion_step"),
                    }
                )

        return {
            "exists": True,
            "path": str(path),
            "count": len(ids),
            "length": summary_stats(lengths),
            "final_reward": summary_stats(rewards),
            "has_expansion_count": expansion_count,
            "stock_reduced_count": stock_reduced_count,
            "preview": preview,
        }
    except Exception as exc:
        return {"exists": True, "path": str(path), "error": str(exc)}


def summarize_sample_buffer(
    store_path: str | Path,
    max_scan: int,
) -> dict[str, Any]:
    path = Path(store_path)
    if not path.exists():
        return {"exists": False, "path": str(path)}

    try:
        root = zarr.open_group(store=str(path), mode="r")
        group = root["samples"]
        length = int(group.attrs.get("length", 0))
        if length <= 0:
            return {
                "exists": True,
                "path": str(path),
                "length": 0,
            }

        scan_count = min(length, int(max_scan))
        indices = np.arange(scan_count, dtype=int)
        scan_limited = scan_count < length

        sample_source_counts = read_counter(group, "sample_source", indices, str)
        trajectory_count = len(set(read_array(group, "trajectory_id", indices, str)))
        source_trajectory_count = len(
            set(read_array(group, "source_trajectory_id", indices, str))
        )
        policy_weights = read_array(group, "policy_weight", indices, float)
        values = read_array(group, "value", indices, float)
        expansion_mode = read_array(group, "expansion_mode", indices, bool)

        preview = []
        preview_indices = np.arange(min(25, scan_count), dtype=int)
        if len(preview_indices) > 0:
            trajectory_ids = read_array(group, "trajectory_id", preview_indices, str)
            step_indices = read_array(group, "step_index", preview_indices, int)
            sources = read_array(group, "sample_source", preview_indices, str)
            actions = read_array(group, "action_id", preview_indices, int)
            pweights = read_array(group, "policy_weight", preview_indices, float)
            vals = read_array(group, "value", preview_indices, float)
            for i in range(len(preview_indices)):
                preview.append(
                    {
                        "sample_index": int(preview_indices[i]),
                        "trajectory_id": trajectory_ids[i],
                        "step_index": int(step_indices[i]),
                        "sample_source": sources[i],
                        "action_id": int(actions[i]),
                        "policy_weight": float(pweights[i]),
                        "value": float(vals[i]),
                    }
                )

        return {
            "exists": True,
            "path": str(path),
            "length": length,
            "scan_count": scan_count,
            "scan_limited": scan_limited,
            "sample_source_counts": dict(sample_source_counts),
            "trajectory_count_scanned": trajectory_count,
            "source_trajectory_count_scanned": source_trajectory_count,
            "policy_weight": summary_stats(policy_weights),
            "value": summary_stats(values),
            "expansion_mode_count_scanned": int(np.asarray(expansion_mode, dtype=bool).sum()),
            "preview": preview,
        }
    except Exception as exc:
        return {"exists": True, "path": str(path), "error": str(exc)}


def summarize_checkpoints(checkpoint_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(checkpoint_dir)
    if not path.exists():
        return []

    rows = []
    for checkpoint in sorted(path.glob("*.pt")):
        rows.append(
            {
                "path": str(checkpoint),
                "name": checkpoint.name,
                "size_mb": round(checkpoint.stat().st_size / (1024 * 1024), 2),
                "modified_at": datetime.fromtimestamp(
                    checkpoint.stat().st_mtime
                ).isoformat(timespec="seconds"),
                "global_step_from_name": parse_checkpoint_step(checkpoint.name),
            }
        )
    return rows


def build_derived_summary(data: dict[str, Any]) -> dict[str, Any]:
    logs = data["logs"]
    runs = logs["runs"]
    cycles = logs["cycles"]
    learner_steps = logs["learner_steps"]

    run_ids = sorted({row.get("run_id") for row in runs + cycles + learner_steps if row.get("run_id")})
    latest_run = runs[-1] if runs else None
    latest_cycle = cycles[-1] if cycles else None
    latest_step = learner_steps[-1] if learner_steps else None
    latest_checkpoint = data["checkpoints"][-1] if data["checkpoints"] else None

    total_samples_logged = sum(
        int((row.get("report") or {}).get("saved_samples", 0))
        for row in runs
    )
    total_mcts_jobs = sum(
        int((row.get("report") or {}).get("used_mcts_jobs", 0))
        for row in runs
    )
    total_reweighted_jobs = sum(
        int((row.get("report") or {}).get("reweighted_jobs", 0))
        for row in runs
    )

    cycles_by_run = defaultdict(list)
    for cycle in cycles:
        cycles_by_run[cycle.get("run_id")].append(cycle)

    return {
        "run_ids": run_ids,
        "latest_run": latest_run,
        "latest_cycle": latest_cycle,
        "latest_learner_step": latest_step,
        "latest_checkpoint": latest_checkpoint,
        "total_runs": len(runs),
        "total_cycles": len(cycles),
        "total_learner_steps": len(learner_steps),
        "total_samples_logged": total_samples_logged,
        "total_mcts_jobs": total_mcts_jobs,
        "total_reweighted_jobs": total_reweighted_jobs,
        "cycles_by_run": {key: value for key, value in cycles_by_run.items()},
    }


def read_array(group, name: str, indices: np.ndarray, dtype):
    if name not in group:
        return np.asarray([], dtype=dtype)
    return group[name][indices].astype(dtype)


def read_counter(group, name: str, indices: np.ndarray, dtype) -> Counter:
    if name not in group:
        return Counter()
    return Counter(group[name][indices].astype(dtype))


def summary_stats(values) -> dict[str, Any]:
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"count": 0}
    output = {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
    }
    if len(values) > 1:
        output["median"] = statistics.median(values)
    return output


def parse_checkpoint_step(name: str) -> int | None:
    stem = Path(name).stem
    suffix = stem.rsplit("_", 1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return None


def _safe_int(value) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _attr_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def render_dashboard(data: dict[str, Any]) -> str:
    encoded = json.dumps(data, ensure_ascii=False)
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e")
    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Workforce Training Dashboard</title>
  <style>
    :root {{
      --bg: #f6f7f8;
      --panel: #ffffff;
      --text: #20242a;
      --muted: #667085;
      --line: #d9dee7;
      --accent: #116a5b;
      --accent-2: #b54708;
      --accent-3: #344054;
      --danger: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 22px 28px 14px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    h1 {{ margin: 0 0 4px; font-size: 24px; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    h3 {{ margin: 0 0 8px; font-size: 14px; }}
    main {{ padding: 20px 28px 40px; }}
    .subtle {{ color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 16px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-height: 92px;
    }}
    .metric {{ font-size: 26px; font-weight: 700; letter-spacing: 0; }}
    .label {{ color: var(--muted); margin-top: 4px; }}
    .toolbar {{
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      margin-top: 12px;
    }}
    select, button {{
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      padding: 8px 10px;
      color: var(--text);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 7px 8px;
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    .two-col {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    .chart {{
      width: 100%;
      min-height: 220px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
    }}
    svg {{ display: block; width: 100%; height: 240px; }}
    .pill {{
      display: inline-block;
      padding: 2px 7px;
      border-radius: 999px;
      background: #eef4f2;
      color: var(--accent);
      font-size: 12px;
      font-weight: 600;
    }}
    .warn {{ color: var(--danger); }}
    pre {{
      white-space: pre-wrap;
      overflow: auto;
      background: #f2f4f7;
      padding: 10px;
      border-radius: 6px;
      border: 1px solid var(--line);
      max-height: 280px;
    }}
    @media (max-width: 1100px) {{
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .two-col {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 700px) {{
      header, main {{ padding-left: 14px; padding-right: 14px; }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Workforce Training Dashboard</h1>
    <div class="subtle">Snapshot generado: <span id="generatedAt"></span></div>
    <div class="toolbar">
      <label>Run <select id="runSelect"></select></label>
      <button id="showAll">Mostrar todo</button>
      <span class="subtle">Para actualizar, volver a ejecutar <code>scripts/build_training_dashboard.py</code>.</span>
    </div>
  </header>
  <main>
    <section class="grid" id="cards"></section>

    <section class="two-col">
      <div class="panel">
        <h2>Loss del learner</h2>
        <div class="chart" id="lossChart"></div>
      </div>
      <div class="panel">
        <h2>Samples por ciclo</h2>
        <div class="chart" id="cycleChart"></div>
      </div>
    </section>

    <section class="two-col">
      <div class="panel">
        <h2>Runs</h2>
        <div id="runsTable"></div>
      </div>
      <div class="panel">
        <h2>Ciclos</h2>
        <div id="cyclesTable"></div>
      </div>
    </section>

    <section class="panel">
      <h2>Buffers Zarr</h2>
      <div id="buffers"></div>
    </section>

    <section class="two-col">
      <div class="panel">
        <h2>SampleBuffer actual</h2>
        <div id="sampleBuffer"></div>
      </div>
      <div class="panel">
        <h2>Checkpoints</h2>
        <div id="checkpoints"></div>
      </div>
    </section>

    <section class="panel">
      <h2>Detalles crudos</h2>
      <pre id="rawDetails"></pre>
    </section>
  </main>
  <script id="dashboard-data" type="application/json">{encoded}</script>
  <script>
    const DATA = JSON.parse(document.getElementById('dashboard-data').textContent);
    let selectedRun = 'ALL';

    function fmt(value, digits = 3) {{
      if (value === null || value === undefined) return '-';
      if (typeof value === 'number') {{
        if (Math.abs(value) >= 1000) return value.toLocaleString();
        return Number.isInteger(value) ? String(value) : value.toFixed(digits);
      }}
      return String(value);
    }}

    function rowsForRun(rows) {{
      if (selectedRun === 'ALL') return rows;
      return rows.filter(row => row.run_id === selectedRun);
    }}

    function initRunSelect() {{
      const select = document.getElementById('runSelect');
      const runIds = DATA.derived.run_ids || [];
      select.innerHTML = '<option value="ALL">Todos</option>' + runIds.map(id => `<option value="${{id}}">${{id}}</option>`).join('');
      if (runIds.length) {{
        selectedRun = runIds[runIds.length - 1];
        select.value = selectedRun;
      }}
      select.addEventListener('change', () => {{
        selectedRun = select.value;
        render();
      }});
      document.getElementById('showAll').addEventListener('click', () => {{
        selectedRun = 'ALL';
        select.value = 'ALL';
        render();
      }});
    }}

    function renderCards() {{
      const runs = rowsForRun(DATA.logs.runs);
      const cycles = rowsForRun(DATA.logs.cycles);
      const steps = rowsForRun(DATA.logs.learner_steps);
      const latestStep = steps[steps.length - 1]?.metric;
      const savedSamples = runs.reduce((acc, row) => acc + (row.report?.saved_samples || 0), 0);
      const mctsJobs = runs.reduce((acc, row) => acc + (row.report?.used_mcts_jobs || 0), 0);
      const reweightedJobs = runs.reduce((acc, row) => acc + (row.report?.reweighted_jobs || 0), 0);
      const failedJobs = runs.reduce((acc, row) => acc + (row.report?.failed_jobs || 0), 0);

      const cards = [
        ['Runs', runs.length],
        ['Ciclos', cycles.length],
        ['Learner steps', steps.length],
        ['Samples logueados', savedSamples],
        ['Jobs MCTS', mctsJobs],
        ['Jobs reweighted', reweightedJobs],
        ['Failed jobs', failedJobs],
        ['Ultima loss', latestStep?.loss],
      ];

      document.getElementById('cards').innerHTML = cards.map(([label, value]) => `
        <div class="card">
          <div class="metric">${{fmt(value)}}</div>
          <div class="label">${{label}}</div>
        </div>
      `).join('');
    }}

    function lineChart(containerId, rows, series) {{
      const el = document.getElementById(containerId);
      if (!rows.length) {{
        el.innerHTML = '<div class="panel subtle">Sin datos.</div>';
        return;
      }}
      const width = 900, height = 240, pad = 36;
      const xs = rows.map((_, i) => i);
      const values = rows.flatMap(row => series.map(s => row.metric?.[s.key] ?? row[s.key]).filter(v => typeof v === 'number'));
      const minY = Math.min(...values);
      const maxY = Math.max(...values);
      const spanY = maxY - minY || 1;
      const x = i => pad + (i / Math.max(1, rows.length - 1)) * (width - 2 * pad);
      const y = v => height - pad - ((v - minY) / spanY) * (height - 2 * pad);

      const paths = series.map(s => {{
        const points = rows.map((row, i) => {{
          const value = row.metric?.[s.key] ?? row[s.key];
          return `${{x(i)}},${{y(value)}}`;
        }}).join(' ');
        return `<polyline points="${{points}}" fill="none" stroke="${{s.color}}" stroke-width="2"/>`;
      }}).join('');
      const legend = series.map((s, i) => `<text x="${{pad + i * 130}}" y="18" fill="${{s.color}}" font-size="12">${{s.label}}</text>`).join('');
      el.innerHTML = `<svg viewBox="0 0 ${{width}} ${{height}}" preserveAspectRatio="none">
        ${{legend}}
        <line x1="${{pad}}" y1="${{height-pad}}" x2="${{width-pad}}" y2="${{height-pad}}" stroke="#d9dee7"/>
        <line x1="${{pad}}" y1="${{pad}}" x2="${{pad}}" y2="${{height-pad}}" stroke="#d9dee7"/>
        <text x="4" y="${{pad}}" font-size="11" fill="#667085">${{fmt(maxY)}}</text>
        <text x="4" y="${{height-pad}}" font-size="11" fill="#667085">${{fmt(minY)}}</text>
        ${{paths}}
      </svg>`;
    }}

    function barChart(containerId, cycles) {{
      const el = document.getElementById(containerId);
      if (!cycles.length) {{
        el.innerHTML = '<div class="panel subtle">Sin datos.</div>';
        return;
      }}
      const width = 900, height = 240, pad = 36;
      const maxV = Math.max(...cycles.map(row => row.cycle?.saved_samples || 0), 1);
      const barW = (width - 2 * pad) / cycles.length;
      const bars = cycles.map((row, i) => {{
        const v = row.cycle?.saved_samples || 0;
        const h = (v / maxV) * (height - 2 * pad);
        const x = pad + i * barW + 2;
        const y = height - pad - h;
        return `<rect x="${{x}}" y="${{y}}" width="${{Math.max(2, barW - 4)}}" height="${{h}}" fill="#116a5b">
          <title>${{row.run_id}} ciclo ${{row.cycle?.cycle_index}}: ${{v}} samples</title>
        </rect>`;
      }}).join('');
      el.innerHTML = `<svg viewBox="0 0 ${{width}} ${{height}}" preserveAspectRatio="none">
        <line x1="${{pad}}" y1="${{height-pad}}" x2="${{width-pad}}" y2="${{height-pad}}" stroke="#d9dee7"/>
        <line x1="${{pad}}" y1="${{pad}}" x2="${{pad}}" y2="${{height-pad}}" stroke="#d9dee7"/>
        <text x="4" y="${{pad}}" font-size="11" fill="#667085">${{fmt(maxV, 0)}}</text>
        ${{bars}}
      </svg>`;
    }}

    function table(rows, columns) {{
      if (!rows.length) return '<p class="subtle">Sin datos.</p>';
      return `<table><thead><tr>${{columns.map(c => `<th>${{c.label}}</th>`).join('')}}</tr></thead><tbody>` +
        rows.map(row => `<tr>${{columns.map(c => `<td>${{c.render ? c.render(row) : fmt(row[c.key])}}</td>`).join('')}}</tr>`).join('') +
        '</tbody></table>';
    }}

    function renderTables() {{
      const runs = rowsForRun(DATA.logs.runs).slice(-20).reverse();
      const cycles = rowsForRun(DATA.logs.cycles).slice(-30).reverse();
      document.getElementById('runsTable').innerHTML = table(runs, [
        {{label: 'Run', render: r => `<span class="pill">${{r.run_id}}</span>`}},
        {{label: 'Workers', render: r => fmt(r.args?.workers)}},
        {{label: 'Device', render: r => fmt(r.args?.device)}},
        {{label: 'Jobs', render: r => fmt(r.report?.completed_jobs)}},
        {{label: 'Failed', render: r => fmt(r.report?.failed_jobs)}},
        {{label: 'Samples', render: r => fmt(r.report?.saved_samples)}},
        {{label: 'MCTS', render: r => fmt(r.report?.used_mcts_jobs)}},
        {{label: 'Reweighted', render: r => fmt(r.report?.reweighted_jobs)}},
        {{label: 'Estado', key: 'status'}},
      ]);
      document.getElementById('cyclesTable').innerHTML = table(cycles, [
        {{label: 'Run', render: r => `<span class="pill">${{r.run_id}}</span>`}},
        {{label: 'Ciclo', render: r => fmt(r.cycle?.cycle_index)}},
        {{label: 'Jobs', render: r => fmt(r.cycle?.completed_jobs)}},
        {{label: 'Samples', render: r => fmt(r.cycle?.saved_samples)}},
        {{label: 'MCTS', render: r => fmt(r.cycle?.used_mcts_jobs)}},
        {{label: 'Reweighted', render: r => fmt(r.cycle?.reweighted_jobs)}},
        {{label: 'Step', render: r => fmt(r.learner?.global_step)}},
        {{label: 'Loss', render: r => fmt(r.learner?.last_metric?.loss)}},
      ]);
    }}

    function renderBuffers() {{
      const buffers = DATA.buffers;
      document.getElementById('buffers').innerHTML = table([
        ['Raw', buffers.raw],
        ['Stock', buffers.stock],
      ], [
        {{label: 'Buffer', render: r => r[0]}},
        {{label: 'Existe', render: r => r[1].exists ? 'si' : 'no'}},
        {{label: 'Trayectorias', render: r => fmt(r[1].count)}},
        {{label: 'Len media', render: r => fmt(r[1].length?.mean)}},
        {{label: 'Reward medio', render: r => fmt(r[1].final_reward?.mean)}},
        {{label: 'Expansion', render: r => fmt(r[1].has_expansion_count)}},
        {{label: 'Stock reduced', render: r => fmt(r[1].stock_reduced_count)}},
      ]);
    }}

    function renderSampleBuffer() {{
      const s = DATA.buffers.samples;
      const counts = s.sample_source_counts || {{}};
      document.getElementById('sampleBuffer').innerHTML = `
        <div class="grid">
          <div class="card"><div class="metric">${{fmt(s.length)}}</div><div class="label">Samples</div></div>
          <div class="card"><div class="metric">${{fmt(s.trajectory_count_scanned)}}</div><div class="label">Trajectories escaneadas</div></div>
          <div class="card"><div class="metric">${{fmt(s.expansion_mode_count_scanned)}}</div><div class="label">Expansion samples</div></div>
          <div class="card"><div class="metric">${{fmt(s.policy_weight?.mean)}}</div><div class="label">Policy weight medio</div></div>
        </div>
        <h3>Fuentes</h3>
        ${{table(Object.entries(counts), [
          {{label: 'Fuente', render: r => r[0]}},
          {{label: 'Count', render: r => fmt(r[1])}},
        ])}}
        <h3>Preview</h3>
        ${{table(s.preview || [], [
          {{label: '#', key: 'sample_index'}},
          {{label: 'Trajectory', key: 'trajectory_id'}},
          {{label: 'Step', key: 'step_index'}},
          {{label: 'Source', key: 'sample_source'}},
          {{label: 'Action', key: 'action_id'}},
          {{label: 'Weight', key: 'policy_weight'}},
          {{label: 'Value', key: 'value'}},
        ])}}
      `;
    }}

    function renderCheckpoints() {{
      document.getElementById('checkpoints').innerHTML = table((DATA.checkpoints || []).slice().reverse(), [
        {{label: 'Nombre', key: 'name'}},
        {{label: 'Step', key: 'global_step_from_name'}},
        {{label: 'MB', key: 'size_mb'}},
        {{label: 'Modificado', key: 'modified_at'}},
      ]);
    }}

    function render() {{
      document.getElementById('generatedAt').textContent = DATA.generated_at;
      renderCards();
      const steps = rowsForRun(DATA.logs.learner_steps);
      lineChart('lossChart', steps, [
        {{key: 'loss', label: 'loss', color: '#116a5b'}},
        {{key: 'policy_loss', label: 'policy', color: '#b54708'}},
        {{key: 'value_loss', label: 'value', color: '#344054'}},
      ]);
      barChart('cycleChart', rowsForRun(DATA.logs.cycles));
      renderTables();
      renderBuffers();
      renderSampleBuffer();
      renderCheckpoints();
      document.getElementById('rawDetails').textContent = JSON.stringify({{
        paths: DATA.paths,
        derived: DATA.derived,
        latest_run: DATA.derived.latest_run,
        latest_cycle: DATA.derived.latest_cycle,
      }}, null, 2);
    }}

    initRunSelect();
    render();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
