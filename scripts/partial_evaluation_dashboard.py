from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numcodecs import Zstd


def read_partial_test_dataset(store_path: str | Path) -> dict[str, Any]:
    path = Path(store_path)
    if not path.exists():
        return {"exists": False, "path": str(path), "trajectories": []}

    try:
        trajectories_dir = path / "trajectories"
        rows = []
        for trajectory_dir in sorted(
            item for item in trajectories_dir.iterdir() if item.is_dir()
        ):
            trajectory_id = trajectory_dir.name
            node = json.loads(
                (trajectory_dir / "zarr.json").read_text(encoding="utf-8")
            )
            attrs = {
                str(key): _jsonable(value)
                for key, value in node.get("attributes", {}).items()
            }
            action_ids = _read_local_zarr_v3_1d(
                trajectory_dir / "action_id"
            ).astype(int)
            resources_total = int(
                np.count_nonzero(np.isin(action_ids, (0, 1, 2)))
            )
            resource_counts = [
                int(np.count_nonzero(action_ids == modality_id))
                for modality_id in (0, 1, 2)
            ]
            expansion_mode = _attr_bool(
                attrs.get("metadata.has_expansion_mode")
            )
            rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "final_reward": float(attrs["final_reward"]),
                    "resources_total": resources_total,
                    "resources_mod_4": int(resource_counts[0]),
                    "resources_mod_6": int(resource_counts[1]),
                    "resources_mod_8": int(resource_counts[2]),
                    "initial_demand_total": int(
                        attrs["metadata.initial_demand_total"]
                    ),
                    "states_count": int(attrs["length"]),
                    "has_expansion_mode": expansion_mode,
                    "stock_was_reduced": _attr_bool(
                        attrs.get("metadata.stock_was_reduced")
                    ),
                    "metadata": attrs,
                }
            )
        return {
            "exists": True,
            "path": str(path),
            "trajectories": rows,
        }
    except Exception as exc:
        return {
            "exists": True,
            "path": str(path),
            "trajectories": [],
            "error": str(exc),
        }


def _read_local_zarr_v3_1d(array_path: Path) -> np.ndarray:
    metadata = json.loads(
        (array_path / "zarr.json").read_text(encoding="utf-8")
    )
    shape = int(metadata["shape"][0])
    chunk_size = int(
        metadata["chunk_grid"]["configuration"]["chunk_shape"][0]
    )
    dtype = np.dtype(metadata["data_type"]).newbyteorder("<")
    chunks = []
    chunk_count = (shape + chunk_size - 1) // chunk_size
    decoder = Zstd()

    for chunk_index in range(chunk_count):
        chunk_path = array_path / "c" / str(chunk_index)
        if chunk_path.exists():
            raw = decoder.decode(chunk_path.read_bytes())
            chunk = np.frombuffer(raw, dtype=dtype)
        else:
            chunk = np.full(
                (chunk_size,),
                metadata.get("fill_value", 0),
                dtype=dtype,
            )
        chunks.append(chunk)

    return np.concatenate(chunks)[:shape]


def render_partial_evaluation_dashboard(data: dict[str, Any]) -> str:
    encoded = json.dumps(data, ensure_ascii=False)
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e")
    return _PARTIAL_DASHBOARD_TEMPLATE.replace("__DASHBOARD_DATA__", encoded)


def merge_partial_evaluation_metadata(
    dataset: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    source_by_id = {
        str(row["trajectory_id"]): row
        for row in dataset.get("trajectories", [])
    }
    merged_rows = []
    for row in evaluation.get("trajectories", []):
        source = source_by_id.get(str(row.get("source_trajectory_id")), {})
        merged_rows.append(
            {
                **row,
                "source_resources_total": source.get("resources_total"),
                "source_initial_demand_total": source.get("initial_demand_total"),
                "source_has_expansion_mode": source.get("has_expansion_mode"),
                "source_stock_was_reduced": source.get("stock_was_reduced"),
                "source_metadata": source.get("metadata", {}),
            }
        )
    return {**evaluation, "trajectories": merged_rows}


def _attr_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value


_PARTIAL_DASHBOARD_TEMPLATE = r"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Evaluacion parcial · Workforce</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f6f8; --panel: #fff; --ink: #182026; --muted: #66737f;
      --line: #dce2e7; --green: #087f5b; --orange: #d9480f;
      --blue: #1864ab; --red: #c92a2a; --gold: #e67700;
      --soft: #eef3f2; --shadow: 0 14px 35px rgba(28, 39, 49, .08);
    }
    :root[data-theme="dark"] {
      color-scheme: dark;
      --bg: #101418; --panel: #191f24; --ink: #eef2f4; --muted: #a7b1ba;
      --line: #354049; --green: #63d6b4; --orange: #ff9b67;
      --blue: #74b9ff; --red: #ff8787; --gold: #ffd166;
      --soft: #222a30; --shadow: 0 16px 38px rgba(0, 0, 0, .28);
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--ink);
      font: 14px/1.45 Inter, ui-sans-serif, system-ui, sans-serif; }
    header { position: sticky; top: 0; z-index: 10; padding: 16px 24px;
      background: color-mix(in srgb, var(--panel) 94%, transparent);
      backdrop-filter: blur(12px); border-bottom: 1px solid var(--line); }
    .header-row, .section-head, .filter-row, .pager {
      display: flex; align-items: center; justify-content: space-between; gap: 12px;
      flex-wrap: wrap;
    }
    h1 { margin: 0; font-size: 25px; letter-spacing: 0; }
    h2 { margin: 0; font-size: 19px; letter-spacing: 0; }
    h3 { margin: 0 0 10px; font-size: 14px; letter-spacing: 0; }
    .subtitle, .muted { color: var(--muted); }
    main { max-width: 1540px; margin: 0 auto; padding: 22px 24px 48px; }
    section { margin-bottom: 28px; }
    .section-head { margin-bottom: 12px; }
    .section-index { color: var(--green); font-weight: 800; margin-right: 7px; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
    .charts { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
    .panel, .metric { background: var(--panel); border: 1px solid var(--line);
      border-radius: 8px; box-shadow: var(--shadow); }
    .panel { padding: 15px; min-width: 0; }
    .metric { padding: 15px; min-height: 100px; }
    .metric-value { font-size: 27px; font-weight: 800; }
    .metric-label { color: var(--muted); margin-top: 4px; }
    .metric-note { color: var(--muted); font-size: 12px; margin-top: 7px; }
    .chart { position: relative; min-height: 285px; overflow: hidden; }
    svg { width: 100%; height: 255px; display: block; overflow: visible; }
    svg text { fill: var(--muted); font-size: 11px; }
    .axis { stroke: var(--line); }
    .tooltip { position: fixed; display: none; z-index: 30; pointer-events: none;
      max-width: 280px; padding: 9px 11px; border-radius: 7px;
      background: var(--panel); border: 1px solid var(--line); box-shadow: var(--shadow); }
    .filters { padding: 13px 15px; margin-bottom: 14px; position: sticky; top: 83px; z-index: 7; }
    label { color: var(--muted); font-size: 12px; font-weight: 700; }
    select, input, button, a.button { border: 1px solid var(--line); border-radius: 6px;
      background: var(--panel); color: var(--ink); padding: 8px 10px; font: inherit; }
    select { min-width: 185px; margin-left: 5px; }
    button, a.button { cursor: pointer; text-decoration: none; font-weight: 700; }
    button:hover, a.button:hover { border-color: var(--green); }
    .theme-button { width: 38px; height: 38px; padding: 0; }
    .status { display: inline-flex; align-items: center; gap: 6px; padding: 4px 8px;
      border-radius: 999px; background: var(--soft); color: var(--green); font-weight: 800; }
    .legend { display: flex; flex-wrap: wrap; gap: 10px; margin: 4px 0 8px; }
    .legend-item { display: inline-flex; align-items: center; gap: 5px; color: var(--muted); font-size: 12px; }
    .legend-dot { width: 9px; height: 9px; border-radius: 50%; }
    .table-panel { overflow: hidden; }
    .table-toolbar { display: flex; gap: 9px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; }
    .table-toolbar input { min-width: min(380px, 100%); }
    .table-wrap { overflow: auto; max-height: 680px; border: 1px solid var(--line); border-radius: 7px; }
    table { width: 100%; border-collapse: collapse; white-space: nowrap; font-size: 12px; }
    th { position: sticky; top: 0; z-index: 2; background: var(--soft); cursor: pointer; }
    th, td { padding: 8px 9px; border-bottom: 1px solid var(--line); text-align: left; }
    tbody tr:hover { background: var(--soft); }
    .positive { color: var(--green); font-weight: 800; }
    .negative { color: var(--red); font-weight: 800; }
    dialog { width: min(880px, calc(100vw - 30px)); max-height: 82vh; border: 1px solid var(--line);
      border-radius: 8px; background: var(--panel); color: var(--ink); box-shadow: var(--shadow); }
    dialog::backdrop { background: rgba(8, 15, 20, .62); }
    pre { overflow: auto; padding: 12px; border-radius: 7px; background: var(--soft); font-size: 12px; }
    @media (max-width: 1050px) { .grid { grid-template-columns: repeat(2, 1fr); } .charts { grid-template-columns: 1fr; } }
    @media (max-width: 650px) { header, main { padding-left: 12px; padding-right: 12px; }
      .grid { grid-template-columns: 1fr; } .filters { top: 104px; } select { min-width: 150px; } }
  </style>
</head>
<body>
  <header>
    <div class="header-row">
      <div>
        <h1>Evaluacion parcial</h1>
        <div class="subtitle">Mismo test set, distintos checkpoints y profundidades desde el final.</div>
      </div>
      <div class="header-row">
        <a class="button" href="model_dashboard.html">Entrenamiento</a>
        <button class="theme-button" id="themeButton" title="Cambiar tema">◐</button>
      </div>
    </div>
  </header>
  <main>
    <section>
      <div class="section-head">
        <h2><span class="section-index">01</span>Descripcion del test dataset</h2>
        <span class="status" id="datasetStatus"></span>
      </div>
      <div class="grid" id="datasetCards"></div>
      <div class="charts">
        <div class="panel"><h3>Scores originales</h3><div class="chart" id="originalScoreChart"></div></div>
        <div class="panel"><h3>Scores positivos vs negativos</h3><div class="chart" id="scoreSignChart"></div></div>
        <div class="panel"><h3>Cantidad de recursos utilizados</h3><div class="chart" id="resourceChart"></div></div>
        <div class="panel"><h3>Expansion mode vs normal</h3><div class="chart" id="expansionChart"></div></div>
        <div class="panel"><h3>Demanda inicial total</h3><div class="chart" id="demandChart"></div></div>
        <div class="panel"><h3>Estados por trayectoria</h3><div class="chart" id="stateChart"></div></div>
      </div>
    </section>

    <section>
      <div class="section-head">
        <h2><span class="section-index">02</span>Resultados MCTS + ResNet</h2>
        <span class="muted" id="filterCount"></span>
      </div>
      <div class="panel filters">
        <div class="filter-row">
          <div>
            <label>Checkpoint <select id="checkpointFilter"></select></label>
            <label>Tail <select id="tailFilter"></select></label>
          </div>
          <button id="resetFilters">Restablecer filtros</button>
        </div>
      </div>
      <div class="grid" id="evaluationCards"></div>
      <div class="charts">
        <div class="panel"><h3>Scores generados por MCTS</h3><div class="chart" id="mctsScoreChart"></div></div>
        <div class="panel"><h3>Better or equal rate por tail y modelo</h3><div class="chart" id="betterRateChart"></div></div>
      </div>
    </section>

    <section>
      <div class="section-head">
        <h2><span class="section-index">03</span>Detalles</h2>
        <span class="muted">Click en los encabezados para ordenar.</span>
      </div>
      <div class="panel table-panel">
        <div class="table-toolbar">
          <input id="tableSearch" type="search" placeholder="Buscar trayectoria, run, checkpoint...">
          <label>Filas <select id="pageSize"><option>25</option><option selected>50</option><option>100</option></select></label>
          <button id="exportCsv">Exportar CSV filtrado</button>
        </div>
        <div class="table-wrap"><table><thead id="detailHead"></thead><tbody id="detailBody"></tbody></table></div>
        <div class="pager">
          <span id="pageInfo" class="muted"></span>
          <div><button id="prevPage">Anterior</button> <button id="nextPage">Siguiente</button></div>
        </div>
      </div>
    </section>
  </main>
  <div class="tooltip" id="tooltip"></div>
  <dialog id="metadataDialog"><div class="section-head"><h2>Metadata completa</h2><button id="closeDialog">Cerrar</button></div><pre id="metadataContent"></pre></dialog>
  <script id="dashboard-data" type="application/json">__DASHBOARD_DATA__</script>
  <script>
    const DATA = JSON.parse(document.getElementById('dashboard-data').textContent);
    const datasetRows = DATA.dataset?.trajectories || [];
    const evaluationRows = DATA.evaluation?.trajectories || [];
    const evaluationRuns = DATA.evaluation?.runs || [];
    const palette = ['#087f5b','#1864ab','#d9480f','#7048e8','#c2255c','#e67700','#0b7285','#5c940d'];
    let filteredRows = [], tableRows = [], page = 1, sortKey = 'checkpoint_step', sortDirection = 'desc';
    const tooltip = document.getElementById('tooltip');
    const checkpointFilter = document.getElementById('checkpointFilter');
    const tailFilter = document.getElementById('tailFilter');
    const resetFilters = document.getElementById('resetFilters');
    const tableSearch = document.getElementById('tableSearch');
    const pageSize = document.getElementById('pageSize');
    const prevPage = document.getElementById('prevPage');
    const nextPage = document.getElementById('nextPage');
    const exportCsv = document.getElementById('exportCsv');
    const detailHead = document.getElementById('detailHead');
    const detailBody = document.getElementById('detailBody');
    const pageInfo = document.getElementById('pageInfo');
    const metadataDialog = document.getElementById('metadataDialog');
    const metadataContent = document.getElementById('metadataContent');
    const closeDialog = document.getElementById('closeDialog');

    const fmt = (value, digits=3) => {
      if (value === null || value === undefined || Number.isNaN(value)) return '-';
      if (typeof value !== 'number') return String(value);
      return new Intl.NumberFormat('es-AR', {
        minimumFractionDigits: Number.isInteger(value) ? 0 : digits,
        maximumFractionDigits: Number.isInteger(value) ? 0 : digits,
        useGrouping: true,
      }).format(value);
    };
    const mean = values => values.length ? values.reduce((a,b)=>a+b,0)/values.length : null;
    const median = values => {
      if (!values.length) return null;
      const sorted = [...values].sort((a,b)=>a-b), mid = Math.floor(sorted.length/2);
      return sorted.length % 2 ? sorted[mid] : (sorted[mid-1]+sorted[mid])/2;
    };
    const ratio = (n,d) => d ? n/d : null;
    const checkpointLabel = row => `ckpt ${row.checkpoint_step ?? '?'} · ${String(row.checkpoint_path || '').split('/').pop()}`;
    const showTooltip = (event, html) => {
      tooltip.innerHTML = html; tooltip.style.display = 'block';
      tooltip.style.left = `${Math.min(event.clientX + 14, innerWidth - 300)}px`;
      tooltip.style.top = `${Math.min(event.clientY + 14, innerHeight - 120)}px`;
    };
    const hideTooltip = () => tooltip.style.display = 'none';

    function setupTheme() {
      const stored = localStorage.getItem('workforce-dashboard-theme') || 'light';
      document.documentElement.dataset.theme = stored;
      document.getElementById('themeButton').onclick = () => {
        const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
        document.documentElement.dataset.theme = next;
        localStorage.setItem('workforce-dashboard-theme', next);
      };
    }

    function stats(values) {
      const clean = values.map(Number).filter(Number.isFinite);
      return {count: clean.length, min: clean.length ? Math.min(...clean) : null,
        max: clean.length ? Math.max(...clean) : null, mean: mean(clean), median: median(clean)};
    }

    function cards(target, items) {
      document.getElementById(target).innerHTML = items.map(([label,value,note]) =>
        `<div class="metric"><div class="metric-value">${fmt(value)}</div><div class="metric-label">${label}</div>${note ? `<div class="metric-note">${note}</div>` : ''}</div>`
      ).join('');
    }

    function histogram(targetId, values, color, bins=14) {
      const target = document.getElementById(targetId);
      const clean = values.map(Number).filter(Number.isFinite);
      if (!clean.length) { target.innerHTML = '<p class="muted">Sin datos para los filtros actuales.</p>'; return; }
      const width=760,height=245,pad={l:48,r:18,t:15,b:34};
      let min=Math.min(...clean), max=Math.max(...clean); if (min===max) { min-=.5; max+=.5; }
      const counts=Array(bins).fill(0), step=(max-min)/bins;
      clean.forEach(v => counts[Math.min(bins-1,Math.floor((v-min)/step))]++);
      const maxCount=Math.max(...counts,1), barW=(width-pad.l-pad.r)/bins;
      const sx=v=>pad.l+(v-min)/(max-min)*(width-pad.l-pad.r);
      const sy=v=>height-pad.b-v/maxCount*(height-pad.t-pad.b);
      const bandwidth=Math.max(step*1.35,(max-min)/Math.sqrt(clean.length)/2);
      const kde=Array.from({length:90},(_,i)=>{
        const x=min+(max-min)*i/89;
        const density=clean.reduce((sum,v)=>sum+Math.exp(-.5*((x-v)/bandwidth)**2),0);
        return {x,density};
      });
      const kdeMax=Math.max(...kde.map(d=>d.density),1);
      const line=kde.map((d,i)=>`${i?'L':'M'}${sx(d.x)},${height-pad.b-d.density/kdeMax*(height-pad.t-pad.b)}`).join(' ');
      target.innerHTML=`<svg viewBox="0 0 ${width} ${height}">
        <line class="axis" x1="${pad.l}" y1="${height-pad.b}" x2="${width-pad.r}" y2="${height-pad.b}"/>
        ${counts.map((count,i)=>{const x=pad.l+i*barW+1,y=sy(count);return `<rect data-bin="${i}" x="${x}" y="${y}" width="${Math.max(1,barW-2)}" height="${height-pad.b-y}" fill="${color}" opacity=".58"/>`;}).join('')}
        <path d="${line}" fill="none" stroke="${color}" stroke-width="3"/>
        <text x="${pad.l}" y="${height-10}">${fmt(min,2)}</text><text x="${width-pad.r}" y="${height-10}" text-anchor="end">${fmt(max,2)}</text>
        <text x="7" y="${pad.t+5}">${maxCount}</text>
      </svg><div class="legend"><span class="legend-item"><i class="legend-dot" style="background:${color}"></i>Histograma</span><span class="legend-item">— Densidad estimada</span></div>`;
      target.querySelectorAll('rect[data-bin]').forEach((bar,i)=>{
        bar.onmousemove=e=>showTooltip(e,`<strong>${fmt(min+i*step,3)} – ${fmt(min+(i+1)*step,3)}</strong>${counts[i]} casos · ${fmt(counts[i]/clean.length*100,1)}%`);
        bar.onmouseleave=hideTooltip;
      });
    }

    function splitBars(targetId, entries) {
      const target=document.getElementById(targetId), total=entries.reduce((s,e)=>s+e.value,0);
      target.innerHTML=`<svg viewBox="0 0 760 245">${entries.map((entry,i)=>{
        const w=total ? entry.value/total*610 : 0, y=45+i*72;
        return `<text x="15" y="${y+18}">${entry.label}</text><rect x="135" y="${y}" width="610" height="28" rx="4" fill="var(--soft)"/><rect x="135" y="${y}" width="${w}" height="28" rx="4" fill="${entry.color}"/><text x="145" y="${y+19}" fill="white">${fmt(ratio(entry.value,total)*100,1)}% · ${entry.value}</text>`;
      }).join('')}</svg>`;
    }

    function lineChart(targetId, rows) {
      const target=document.getElementById(targetId);
      const grouped=new Map();
      rows.forEach(row=>{
        const key=checkpointLabel(row), tail=Number(row.requested_tail_states);
        if (!Number.isFinite(tail)) return;
        if (!grouped.has(key)) grouped.set(key,new Map());
        const bucket=grouped.get(key).get(tail)||{better:0,total:0};
        if (row.better_than_original_count !== undefined) {
          bucket.better += Number(row.better_than_original_count||0)+Number(row.same_as_original_count||0);
          bucket.total += Number(row.completed_jobs||row.total_jobs||0);
        } else {
          bucket.better += Number(row.value_error)>=0 ? 1 : 0;
          bucket.total += 1;
        }
        grouped.get(key).set(tail,bucket);
      });
      if (!grouped.size) { target.innerHTML='<p class="muted">Sin datos.</p>'; return; }
      const allTails=[...new Set(rows.map(r=>Number(r.requested_tail_states)).filter(Number.isFinite))].sort((a,b)=>a-b);
      const series=[...grouped.entries()].map(([label,buckets],index)=>({label,color:palette[index%palette.length],
        points:[...buckets.entries()].sort((a,b)=>a[0]-b[0]).map(([tail,bucket])=>({tail,rate:bucket.total?bucket.better/bucket.total:0,count:bucket.total}))}));
      const visibleRates=series.flatMap(seriesItem=>seriesItem.points.map(point=>point.rate));
      const rawMinY=Math.min(...visibleRates),rawMaxY=Math.max(...visibleRates);
      const rateSpan=rawMaxY-rawMinY;
      const yPadding=rateSpan > 0 ? Math.max(.015,rateSpan*.22) : .05;
      const minY=Math.max(0,rawMinY-yPadding),maxY=Math.min(1,rawMaxY+yPadding);
      const effectiveMaxY=maxY > minY ? maxY : Math.min(1,minY+.1);
      const width=760,height=245,pad={l:58,r:20,t:18,b:38},minX=Math.min(...allTails),maxX=Math.max(...allTails);
      const sx=x=>pad.l+(x-minX)/(Math.max(1,maxX-minX))*(width-pad.l-pad.r);
      const sy=y=>height-pad.b-(y-minY)/(effectiveMaxY-minY)*(height-pad.t-pad.b);
      const yTicks=Array.from({length:5},(_,index)=>minY+(effectiveMaxY-minY)*index/4);
      target.innerHTML=`<svg viewBox="0 0 ${width} ${height}">
        ${yTicks.map(v=>`<line class="axis" x1="${pad.l}" y1="${sy(v)}" x2="${width-pad.r}" y2="${sy(v)}"/><text x="5" y="${sy(v)+4}">${fmt(v*100,1)}%</text>`).join('')}
        ${series.map(s=>`<path d="${s.points.map((p,i)=>`${i?'L':'M'}${sx(p.tail)},${sy(p.rate)}`).join(' ')}" fill="none" stroke="${s.color}" stroke-width="2.5"/>${s.points.map(p=>`<circle data-label="${s.label}" data-tail="${p.tail}" data-rate="${p.rate}" data-count="${p.count}" cx="${sx(p.tail)}" cy="${sy(p.rate)}" r="5" fill="${s.color}"/>`).join('')}`).join('')}
        <text x="${pad.l}" y="${height-10}">${minX}</text><text x="${width-pad.r}" y="${height-10}" text-anchor="end">${maxX}</text>
      </svg><div class="legend">${series.map(s=>`<span class="legend-item"><i class="legend-dot" style="background:${s.color}"></i>${s.label}</span>`).join('')}<span class="legend-item">Escala visible: ${fmt(minY*100,1)}% – ${fmt(effectiveMaxY*100,1)}%</span></div>`;
      target.querySelectorAll('circle').forEach(dot=>{
        dot.onmousemove=e=>showTooltip(e,`<strong>${dot.dataset.label}</strong>tail=${dot.dataset.tail}<br>better/equal=${fmt(Number(dot.dataset.rate)*100,1)}%<br>n=${dot.dataset.count}`);
        dot.onmouseleave=hideTooltip;
      });
    }

    function renderDataset() {
      const rewards=datasetRows.map(r=>r.final_reward), resources=datasetRows.map(r=>r.resources_total);
      const demand=datasetRows.map(r=>r.initial_demand_total), states=datasetRows.map(r=>r.states_count);
      const stateStats=stats(states), positive=rewards.filter(v=>v>=0).length, expansion=datasetRows.filter(r=>r.has_expansion_mode).length;
      document.getElementById('datasetStatus').textContent=`${datasetRows.length} trayectorias fijas`;
      cards('datasetCards',[
        ['Trayectorias',datasetRows.length,DATA.dataset?.path],
        ['Estados mediana',stateStats.median,`Min ${fmt(stateStats.min)} · Max ${fmt(stateStats.max)}`],
        ['Score medio',mean(rewards),`Mediana ${fmt(median(rewards))}`],
        ['Cantidad media de recursos',mean(resources),`Cada seleccion de modalidad cuenta como 1 recurso`],
      ]);
      histogram('originalScoreChart',rewards,'#087f5b');
      histogram('resourceChart',resources,'#1864ab');
      histogram('demandChart',demand,'#e67700');
      splitBars('scoreSignChart',[{label:'Positivo o cero',value:positive,color:'#087f5b'},{label:'Negativo',value:rewards.length-positive,color:'#c92a2a'}]);
      splitBars('expansionChart',[{label:'Expansion',value:expansion,color:'#d9480f'},{label:'Normal',value:datasetRows.length-expansion,color:'#1864ab'}]);
      histogram('stateChart',states,'#7048e8',12);
    }

    function setupFilters() {
      const allEvaluationRows=[...evaluationRows,...evaluationRuns];
      const checkpoints=[...new Map(allEvaluationRows.map(r=>[checkpointLabel(r),r])).keys()];
      const tails=[...new Set(allEvaluationRows.map(r=>r.requested_tail_states).filter(v=>v!==null&&v!==undefined))].sort((a,b)=>a-b);
      checkpointFilter.innerHTML='<option value="ALL">Todos los modelos</option>'+checkpoints.map(v=>`<option>${v}</option>`).join('');
      tailFilter.innerHTML='<option value="ALL">Todos los tails</option>'+tails.map(v=>`<option value="${v}">${v}</option>`).join('');
      checkpointFilter.onchange=applyFilters; tailFilter.onchange=applyFilters;
      resetFilters.onclick=()=>{checkpointFilter.value='ALL';tailFilter.value='ALL';applyFilters();};
      tableSearch.oninput=()=>{page=1;renderTable();}; pageSize.onchange=()=>{page=1;renderTable();};
      prevPage.onclick=()=>{page=Math.max(1,page-1);renderTable();}; nextPage.onclick=()=>{page++;renderTable();};
      exportCsv.onclick=exportFilteredCsv;
    }

    function applyFilters() {
      filteredRows=evaluationRows.filter(row=>(checkpointFilter.value==='ALL'||checkpointLabel(row)===checkpointFilter.value)
        &&(tailFilter.value==='ALL'||String(row.requested_tail_states)===tailFilter.value));
      document.getElementById('filterCount').textContent=`${filteredRows.length} evaluaciones visibles`;
      const rewards=filteredRows.map(r=>Number(r.final_reward)).filter(Number.isFinite);
      const better=filteredRows.filter(r=>Number(r.value_error)>=0).length;
      const positive=filteredRows.filter(r=>Number(r.final_reward)>=0).length;
      cards('evaluationCards',[
        ['Evaluaciones',filteredRows.length,'Según filtros activos'],
        ['Score medio',mean(rewards),`Mediana ${fmt(median(rewards))}`],
        ['Better or equal rate',ratio(better,filteredRows.length)*100,'Comparado con el score original'],
        ['Positive rate',ratio(positive,filteredRows.length)*100,'Score MCTS mayor o igual a cero'],
      ]);
      histogram('mctsScoreChart',rewards,'#d9480f');
      lineChart('betterRateChart',(evaluationRuns.length ? evaluationRuns : evaluationRows).filter(row=>
        (checkpointFilter.value==='ALL'||checkpointLabel(row)===checkpointFilter.value)
        &&(tailFilter.value==='ALL'||String(row.requested_tail_states)===tailFilter.value)
      ));
      page=1; renderTable();
    }

    const columns=[
      ['trajectory_id','Trayectoria'],['run_id','Run'],['checkpoint_step','Checkpoint'],
      ['requested_tail_states','Tail'],['effective_tail_states','Tail efectivo'],
      ['source_start_index','Inicio'],['source_trajectory_length','Longitud fuente'],
      ['states_count','Estados MCTS'],['final_reward','Score MCTS'],['original_value','Original'],
      ['value_error','Error'],['source_resources_total','Recursos fuente'],
      ['source_initial_demand_total','Demanda fuente'],['source_has_expansion_mode','Expansion'],
      ['elapsed_seconds','Segundos'],['metadata','Metadata']
    ];
    function renderTable() {
      const query=tableSearch.value.trim().toLowerCase();
      tableRows=filteredRows.filter(row=>!query||JSON.stringify(row).toLowerCase().includes(query));
      tableRows.sort((a,b)=>{
        const av=a[sortKey],bv=b[sortKey]; if(av===bv)return 0;
        return (av??'')<(bv??'')?-sortDirection:sortDirection;
      });
      const size=Number(pageSize.value),pages=Math.max(1,Math.ceil(tableRows.length/size)); page=Math.min(page,pages);
      const visible=tableRows.slice((page-1)*size,page*size);
      detailHead.innerHTML=`<tr>${columns.map(([key,label])=>`<th data-key="${key}">${label}${sortKey===key?(sortDirection===1?' ↑':' ↓'):''}</th>`).join('')}</tr>`;
      detailBody.innerHTML=visible.map((row,index)=>`<tr>${columns.map(([key])=>{
        if(key==='metadata')return `<td><button data-meta="${(page-1)*size+index}">Ver</button></td>`;
        const value=row[key],cls=key==='value_error'?(Number(value)>=0?'positive':'negative'):'';
        return `<td class="${cls}">${fmt(value,key.includes('reward')||key.includes('value')||key==='elapsed_seconds'?4:2)}</td>`;
      }).join('')}</tr>`).join('');
      detailHead.querySelectorAll('th').forEach(th=>th.onclick=()=>{const key=th.dataset.key;if(key==='metadata')return;
        if(sortKey===key)sortDirection*=-1;else{sortKey=key;sortDirection=1;}renderTable();});
      detailBody.querySelectorAll('[data-meta]').forEach(button=>button.onclick=()=>{
        metadataContent.textContent=JSON.stringify(tableRows[Number(button.dataset.meta)],null,2);metadataDialog.showModal();});
      pageInfo.textContent=`Pagina ${page} de ${pages} · ${tableRows.length} filas`;
      prevPage.disabled=page<=1;nextPage.disabled=page>=pages;
    }

    function exportFilteredCsv() {
      const keys=columns.filter(([key])=>key!=='metadata').map(([key])=>key);
      const lines=[keys.join(','),...tableRows.map(row=>keys.map(key=>`"${String(row[key]??'').replaceAll('"','""')}"`).join(','))];
      const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([lines.join('\n')],{type:'text/csv'}));
      link.download='partial_evaluation_filtered.csv';link.click();URL.revokeObjectURL(link.href);
    }

    closeDialog.onclick=()=>metadataDialog.close();
    setupTheme(); renderDataset(); setupFilters(); applyFilters();
  </script>
</body>
</html>"""
