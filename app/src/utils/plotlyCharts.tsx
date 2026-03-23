/**
 * Plotly Chart Engine — 32 chart types
 * Pipeline: type detection → aggregation → validation → auto-fix → render
 */
import Plot from 'react-plotly.js';

type Row = Record<string, any>;
type Agg = 'auto' | 'count' | 'sum' | 'mean' | 'min' | 'max';

// ── DATA TYPE DETECTION ───────────────────────────────────────────────────────
export function isNumericCol(data: Row[], col: string): boolean {
  const vals = data.map(r => r[col]).filter(v => v !== null && v !== undefined && v !== '');
  if (!vals.length) return false;
  return vals.every(v => !isNaN(Number(v)));
}
export function isCategoricalCol(data: Row[], col: string): boolean {
  return !isNumericCol(data, col);
}

// ── CORE TRANSFORM ────────────────────────────────────────────────────────────
export function transformData(data: Row[], x: string, y?: string, agg: Agg = 'auto'): Row[] {
  if (!data.length || !x) return [];
  let effectiveAgg = agg;
  if (agg === 'auto') effectiveAgg = (!y || isCategoricalCol(data, y)) ? 'count' : 'sum';
  const groups: Record<string, number[]> = {};
  for (const row of data) {
    const raw = row[x];
    const key = (raw === null || raw === undefined || raw === '' || String(raw).toLowerCase() === 'null') ? 'Unknown' : String(raw);
    if (!groups[key]) groups[key] = [];
    const val = y ? Number(row[y]) : 1;
    groups[key].push(isNaN(val) ? 0 : val);
  }
  return Object.entries(groups).map(([key, vals]) => {
    let value: number;
    if (effectiveAgg === 'count') value = vals.length;
    else if (effectiveAgg === 'sum') value = vals.reduce((a, b) => a + b, 0);
    else if (effectiveAgg === 'mean') value = vals.reduce((a, b) => a + b, 0) / vals.length;
    else if (effectiveAgg === 'min') value = Math.min(...vals);
    else value = Math.max(...vals);
    return { [x]: key, value };
  });
}

export function autoFixData(data: Row[], _x?: string, y?: string): Row[] {
  if (!y) return data;
  const sample = data.slice(0, 20).map(r => r[y]).filter(v => v !== null && v !== undefined && v !== '');
  if (sample.every(v => !isNaN(Number(v))) && isCategoricalCol(data, y))
    return data.map(r => ({ ...r, [y]: Number(r[y]) }));
  return data;
}

export function validateChart(data: Row[], chart: string, x?: string, y?: string): { ok: boolean; msg: string } {
  if (!data.length) return { ok: false, msg: 'No data loaded' };
  if (chart === 'scatter' && (!x || !y)) return { ok: false, msg: 'Scatter requires X and Y' };
  if (chart === 'kpi' && !y) return { ok: false, msg: 'KPI requires a numeric Y' };
  return { ok: true, msg: 'OK' };
}

// ── COMMON STYLE ──────────────────────────────────────────────────────────────
const BASE: any = {
  template: 'plotly_dark', paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
  font: { family: 'Arial, sans-serif', size: 12, color: '#e2e8f0' },
  hovermode: 'closest', dragmode: 'pan',
  transition: { duration: 600, easing: 'cubic-in-out' },
  margin: { l: 52, r: 16, t: 38, b: 52 }, showlegend: false,
};
const AXIS: any = {
  gridcolor: 'rgba(255,255,255,0.07)', linecolor: 'rgba(255,255,255,0.12)',
  tickfont: { size: 10 }, zerolinecolor: 'rgba(255,255,255,0.08)',
};
export const CFG: any = { responsive: true, displayModeBar: false, doubleClick: false, scrollZoom: false };

const BLUES = ['#1d4ed8','#2563eb','#3b82f6','#60a5fa','#93c5fd','#bfdbfe','#1e40af','#1e3a8a'];
const TEALS = ['#0f766e','#0d9488','#14b8a6','#2dd4bf','#5eead4','#99f6e4','#134e4a','#042f2e'];
const MULTI = ['#3b82f6','#f59e0b','#10b981','#ef4444','#8b5cf6','#ec4899','#06b6d4','#84cc16'];

interface ChartProps { data: Row[]; x: string; y: string; z?: string; title?: string; height?: number; agg?: Agg; }

// ── 1. BAR CHART ──────────────────────────────────────────────────────────────
export function PlotlyBarChart({ data, x, y, title, height = 280, agg = 'auto' }: ChartProps) {
  const fixed = autoFixData(data, x, y);
  const aggData = transformData(fixed, x, isNumericCol(fixed, y) ? y : undefined, agg);
  return (
    <Plot data={[{ type: 'bar', x: aggData.map(r => r[x]), y: aggData.map(r => r.value),
      marker: { color: aggData.map((_, i) => BLUES[i % BLUES.length]) },
      texttemplate: '%{y}', textposition: 'outside',
      hovertemplate: `<b>%{x}</b><br>${y || 'count'}: %{y}<extra></extra>` }]}
      layout={{ ...BASE, title: { text: title || `${y} vs ${x}`, font: { size: 13 } }, height,
        xaxis: { ...AXIS, title: { text: x } }, yaxis: { ...AXIS, title: { text: isNumericCol(fixed, y) ? y : 'count' } } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 2. GROUPED BAR ────────────────────────────────────────────────────────────
export function PlotlyGroupedBarChart({ data, x, y, z, title, height = 280 }: ChartProps) {
  const fixed = autoFixData(data, x, y);
  const groups = z ? [...new Set(fixed.map(r => r[z]))] : ['All'];
  const traces: any[] = groups.map((g, i) => {
    const subset = z ? fixed.filter(r => r[z] === g) : fixed;
    const agg = transformData(subset, x, isNumericCol(fixed, y) ? y : undefined, 'sum');
    return { type: 'bar', name: String(g), x: agg.map(r => r[x]), y: agg.map(r => r.value),
      marker: { color: MULTI[i % MULTI.length] } };
  });
  return (
    <Plot data={traces}
      layout={{ ...BASE, barmode: 'group', showlegend: !!z, title: { text: title || `${y} by ${x}`, font: { size: 13 } }, height,
        xaxis: { ...AXIS, title: { text: x } }, yaxis: { ...AXIS, title: { text: y } } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 3. STACKED BAR ────────────────────────────────────────────────────────────
export function PlotlyStackedBarChart({ data, x, y, z, title, height = 280 }: ChartProps) {
  const fixed = autoFixData(data, x, y);
  const groups = z ? [...new Set(fixed.map(r => r[z]))] : ['All'];
  const traces: any[] = groups.map((g, i) => {
    const subset = z ? fixed.filter(r => r[z] === g) : fixed;
    const agg = transformData(subset, x, isNumericCol(fixed, y) ? y : undefined, 'sum');
    return { type: 'bar', name: String(g), x: agg.map(r => r[x]), y: agg.map(r => r.value),
      marker: { color: MULTI[i % MULTI.length] } };
  });
  return (
    <Plot data={traces}
      layout={{ ...BASE, barmode: 'stack', showlegend: !!z, title: { text: title || `${y} stacked by ${z || x}`, font: { size: 13 } }, height,
        xaxis: { ...AXIS, title: { text: x } }, yaxis: { ...AXIS, title: { text: y } } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 4. 100% STACKED BAR ───────────────────────────────────────────────────────
export function PlotlyStackedBar100Chart({ data, x, y, z, title, height = 280 }: ChartProps) {
  const fixed = autoFixData(data, x, y);
  const groups = z ? [...new Set(fixed.map(r => r[z]))] : ['All'];
  const traces: any[] = groups.map((g, i) => {
    const subset = z ? fixed.filter(r => r[z] === g) : fixed;
    const agg = transformData(subset, x, isNumericCol(fixed, y) ? y : undefined, 'sum');
    return { type: 'bar', name: String(g), x: agg.map(r => r[x]), y: agg.map(r => r.value),
      marker: { color: MULTI[i % MULTI.length] } };
  });
  return (
    <Plot data={traces}
      layout={{ ...BASE, barmode: 'relative', barnorm: 'percent', showlegend: !!z,
        title: { text: title || `100% Stacked`, font: { size: 13 } }, height,
        xaxis: { ...AXIS, title: { text: x } }, yaxis: { ...AXIS, title: { text: '%' } } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 5. HORIZONTAL BAR ─────────────────────────────────────────────────────────
export function PlotlyHorizontalBarChart({ data, x, y, title, height = 280, agg = 'auto' }: ChartProps) {
  const fixed = autoFixData(data, x, y);
  const aggData = transformData(fixed, x, isNumericCol(fixed, y) ? y : undefined, agg);
  return (
    <Plot data={[{ type: 'bar', orientation: 'h',
      y: aggData.map(r => r[x]), x: aggData.map(r => r.value),
      marker: { color: aggData.map((_, i) => BLUES[i % BLUES.length]) },
      hovertemplate: `<b>%{y}</b><br>${y || 'count'}: %{x}<extra></extra>` }]}
      layout={{ ...BASE, title: { text: title || `${y} by ${x}`, font: { size: 13 } }, height,
        xaxis: { ...AXIS, title: { text: isNumericCol(fixed, y) ? y : 'count' } },
        yaxis: { ...AXIS, title: { text: x }, automargin: true } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 6. LINE CHART ─────────────────────────────────────────────────────────────
export function PlotlyLineChart({ data, x, y, title, height = 280, agg = 'auto' }: ChartProps) {
  const fixed = autoFixData(data, x, y);
  const useAgg = !isNumericCol(fixed, y);
  const aggData = useAgg ? transformData(fixed, x, undefined, 'count') : transformData(fixed, x, y, agg);
  const yKey = useAgg ? 'value' : y; const yLabel = useAgg ? 'count' : y;
  return (
    <Plot data={[{ type: 'scatter', mode: 'lines+markers', x: aggData.map(r => r[x]), y: aggData.map(r => r[yKey]),
      line: { width: 3, color: '#3b82f6' }, marker: { size: 8, color: '#60a5fa' },
      hovertemplate: `<b>%{x}</b><br>${yLabel}: %{y}<extra></extra>` }]}
      layout={{ ...BASE, title: { text: title || `${yLabel} over ${x}`, font: { size: 13 } }, height,
        xaxis: { ...AXIS, title: { text: x } }, yaxis: { ...AXIS, title: { text: yLabel } } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 7. MULTI-LINE CHART ───────────────────────────────────────────────────────
export function PlotlyMultiLineChart({ data, x, y, z, title, height = 280 }: ChartProps) {
  const fixed = autoFixData(data, x, y);
  const groups = z ? [...new Set(fixed.map(r => r[z]))] : ['All'];
  const traces: any[] = groups.map((g, i) => {
    const subset = z ? fixed.filter(r => r[z] === g) : fixed;
    const agg = transformData(subset, x, isNumericCol(fixed, y) ? y : undefined, 'sum');
    return { type: 'scatter', mode: 'lines+markers', name: String(g),
      x: agg.map(r => r[x]), y: agg.map(r => r.value),
      line: { width: 2, color: MULTI[i % MULTI.length] }, marker: { size: 6 } };
  });
  return (
    <Plot data={traces}
      layout={{ ...BASE, showlegend: true, title: { text: title || `${y} by ${z || x}`, font: { size: 13 } }, height,
        xaxis: { ...AXIS, title: { text: x } }, yaxis: { ...AXIS, title: { text: y } } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 8. SPLINE CHART ───────────────────────────────────────────────────────────
export function PlotlySplineChart({ data, x, y, title, height = 280, agg = 'auto' }: ChartProps) {
  const fixed = autoFixData(data, x, y);
  const useAgg = !isNumericCol(fixed, y);
  const aggData = useAgg ? transformData(fixed, x, undefined, 'count') : transformData(fixed, x, y, agg);
  const yKey = useAgg ? 'value' : y; const yLabel = useAgg ? 'count' : y;
  return (
    <Plot data={[{ type: 'scatter', mode: 'lines+markers', x: aggData.map(r => r[x]), y: aggData.map(r => r[yKey]),
      line: { width: 3, color: '#8b5cf6', shape: 'spline' }, marker: { size: 7, color: '#a78bfa' },
      hovertemplate: `<b>%{x}</b><br>${yLabel}: %{y}<extra></extra>` }]}
      layout={{ ...BASE, title: { text: title || `${yLabel} Spline`, font: { size: 13 } }, height,
        xaxis: { ...AXIS, title: { text: x } }, yaxis: { ...AXIS, title: { text: yLabel } } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 9. STEP LINE CHART ────────────────────────────────────────────────────────
export function PlotlyStepLineChart({ data, x, y, title, height = 280, agg = 'auto' }: ChartProps) {
  const fixed = autoFixData(data, x, y);
  const useAgg = !isNumericCol(fixed, y);
  const aggData = useAgg ? transformData(fixed, x, undefined, 'count') : transformData(fixed, x, y, agg);
  const yKey = useAgg ? 'value' : y; const yLabel = useAgg ? 'count' : y;
  return (
    <Plot data={[{ type: 'scatter', mode: 'lines', x: aggData.map(r => r[x]), y: aggData.map(r => r[yKey]),
      line: { width: 3, color: '#10b981', shape: 'hv' },
      hovertemplate: `<b>%{x}</b><br>${yLabel}: %{y}<extra></extra>` }]}
      layout={{ ...BASE, title: { text: title || `${yLabel} Step`, font: { size: 13 } }, height,
        xaxis: { ...AXIS, title: { text: x } }, yaxis: { ...AXIS, title: { text: yLabel } } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 10. AREA CHART ────────────────────────────────────────────────────────────
export function PlotlyAreaChart({ data, x, y, title, height = 280, agg = 'auto' }: ChartProps) {
  const fixed = autoFixData(data, x, y);
  const useAgg = !isNumericCol(fixed, y);
  const aggData = useAgg ? transformData(fixed, x, undefined, 'count') : transformData(fixed, x, y, agg);
  const yKey = useAgg ? 'value' : y; const yLabel = useAgg ? 'count' : y;
  return (
    <Plot data={[{ type: 'scatter', mode: 'lines', x: aggData.map(r => r[x]), y: aggData.map(r => r[yKey]),
      fill: 'tozeroy', fillcolor: 'rgba(59,130,246,0.22)', line: { width: 2, color: '#3b82f6' },
      hovertemplate: `<b>%{x}</b><br>${yLabel}: %{y}<extra></extra>` }]}
      layout={{ ...BASE, title: { text: title || `${yLabel} Area`, font: { size: 13 } }, height,
        xaxis: { ...AXIS, title: { text: x } }, yaxis: { ...AXIS, title: { text: yLabel } } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 11. COMBO CHART (Bar + Line) ──────────────────────────────────────────────
export function PlotlyComboChart({ data, x, y, z, title, height = 280 }: ChartProps) {
  const fixed = autoFixData(data, x, y);
  const barAgg = transformData(fixed, x, isNumericCol(fixed, y) ? y : undefined, 'sum');
  const lineAgg = z && isNumericCol(fixed, z)
    ? transformData(fixed, x, z, 'mean')
    : barAgg.map(r => ({ ...r, value: r.value * 0.6 }));
  return (
    <Plot data={[
      { type: 'bar', name: y || 'Value', x: barAgg.map(r => r[x]), y: barAgg.map(r => r.value),
        marker: { color: '#3b82f6' }, yaxis: 'y' },
      { type: 'scatter', mode: 'lines+markers', name: z || 'Trend',
        x: lineAgg.map(r => (r as any)[x]), y: lineAgg.map(r => r.value),
        line: { color: '#f59e0b', width: 3 }, marker: { size: 7 }, yaxis: 'y2' },
    ]}
      layout={{ ...BASE, showlegend: true, title: { text: title || `Combo: ${y}`, font: { size: 13 } }, height,
        xaxis: { ...AXIS, title: { text: x } },
        yaxis: { ...AXIS, title: { text: y } },
        yaxis2: { ...AXIS, title: { text: z || 'Trend' }, overlaying: 'y', side: 'right' } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 12. PIE CHART ─────────────────────────────────────────────────────────────
export function PlotlyPieChart({ data, x, y, title, height = 280, agg = 'auto' }: ChartProps) {
  const fixed = autoFixData(data, x, y);
  const useY = y && isNumericCol(fixed, y);
  const aggData = transformData(fixed, x, useY ? y : undefined, useY ? agg : 'count');
  return (
    <Plot data={[{ type: 'pie', labels: aggData.map(r => r[x]), values: aggData.map(r => r.value),
      textinfo: 'percent+label', textposition: 'outside', automargin: true,
      pull: aggData.map(() => 0.04), marker: { colors: aggData.map((_, i) => BLUES[i % BLUES.length]) },
      hovertemplate: '<b>%{label}</b><br>%{value} (%{percent})<extra></extra>' }]}
      layout={{ ...BASE, title: { text: title || `${x} Distribution`, font: { size: 13 } }, height,
        showlegend: true, legend: { orientation: 'h', y: -0.15 }, margin: { l: 16, r: 16, t: 38, b: 60 } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 13. DONUT CHART ───────────────────────────────────────────────────────────
export function PlotlyDonutChart({ data, x, y, title, height = 280, agg = 'auto' }: ChartProps) {
  const fixed = autoFixData(data, x, y);
  const useY = y && isNumericCol(fixed, y);
  const aggData = transformData(fixed, x, useY ? y : undefined, useY ? agg : 'count');
  const total = aggData.reduce((s, r) => s + r.value, 0);
  return (
    <Plot data={[{ type: 'pie', labels: aggData.map(r => r[x]), values: aggData.map(r => r.value),
      hole: 0.5, textinfo: 'percent', textposition: 'inside', automargin: true,
      marker: { colors: aggData.map((_, i) => TEALS[i % TEALS.length]) },
      hovertemplate: '<b>%{label}</b><br>%{value} (%{percent})<extra></extra>' }]}
      layout={{ ...BASE, title: { text: title || `${x} Distribution`, font: { size: 13 } }, height,
        showlegend: true, legend: { orientation: 'h', y: -0.15 }, margin: { l: 16, r: 16, t: 38, b: 60 },
        annotations: [{ text: `<b>${total.toLocaleString(undefined, { maximumFractionDigits: 0 })}</b>`,
          showarrow: false, font: { size: 14, color: '#e2e8f0' }, x: 0.5, y: 0.5, xref: 'paper', yref: 'paper' }] }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 14. SCATTER CHART ─────────────────────────────────────────────────────────
export function PlotlyScatterChart({ data, x, y, z, title, height = 280 }: ChartProps) {
  const fixed = autoFixData(data, x, y);
  if (!isNumericCol(fixed, x) || !isNumericCol(fixed, y))
    return <PlotlyBarChart data={fixed} x={x} y={y} z={z} title={title} height={height} />;
  const traces: any[] = z && isCategoricalCol(fixed, z)
    ? Array.from(new Set(fixed.map(r => r[z]))).map((val, i) => ({
        type: 'scatter', mode: 'markers', name: String(val),
        x: fixed.filter(r => r[z] === val).map(r => Number(r[x])),
        y: fixed.filter(r => r[z] === val).map(r => Number(r[y])),
        marker: { size: 9, color: BLUES[i % BLUES.length], opacity: 0.8 } }))
    : [{ type: 'scatter', mode: 'markers',
        x: fixed.map(r => Number(r[x])), y: fixed.map(r => Number(r[y])),
        marker: { size: 9, color: fixed.map((_, i) => BLUES[i % BLUES.length]), opacity: 0.8 } }];
  return (
    <Plot data={traces}
      layout={{ ...BASE, title: { text: title || `${y} vs ${x}`, font: { size: 13 } }, height, showlegend: !!z,
        xaxis: { ...AXIS, title: { text: x } }, yaxis: { ...AXIS, title: { text: y } } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 15. BUBBLE CHART ──────────────────────────────────────────────────────────
export function PlotlyBubbleChart({ data, x, y, z, title, height = 280 }: ChartProps) {
  const fixed = autoFixData(data, x, y);
  const sizeCol = z && isNumericCol(fixed, z) ? z : null;
  const sizes = sizeCol
    ? (() => { const vals = fixed.map(r => Number(r[sizeCol])); const mx = Math.max(...vals) || 1; return vals.map(v => 8 + (v / mx) * 40); })()
    : fixed.map(() => 18);
  return (
    <Plot data={[{ type: 'scatter', mode: 'markers',
      x: fixed.map(r => isNumericCol(fixed, x) ? Number(r[x]) : r[x]),
      y: fixed.map(r => isNumericCol(fixed, y) ? Number(r[y]) : r[y]),
      marker: { size: sizes, color: sizes, colorscale: 'Blues', opacity: 0.75, showscale: false },
      hovertemplate: `${x}: %{x}<br>${y}: %{y}${sizeCol ? `<br>${sizeCol}: %{marker.size}` : ''}<extra></extra>` }]}
      layout={{ ...BASE, title: { text: title || `Bubble: ${y} vs ${x}`, font: { size: 13 } }, height,
        xaxis: { ...AXIS, title: { text: x } }, yaxis: { ...AXIS, title: { text: y } } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 16. HISTOGRAM ─────────────────────────────────────────────────────────────
export function PlotlyHistogram({ data, x, y, title, height = 280 }: ChartProps) {
  const col = x || y;
  const vals = data.map(r => isNumericCol(data, col) ? Number(r[col]) : r[col]);
  return (
    <Plot data={[{ type: 'histogram', x: vals, marker: { color: '#3b82f6', opacity: 0.8 },
      hovertemplate: 'Range: %{x}<br>Count: %{y}<extra></extra>' }]}
      layout={{ ...BASE, title: { text: title || `${col} Distribution`, font: { size: 13 } }, height,
        xaxis: { ...AXIS, title: { text: col } }, yaxis: { ...AXIS, title: { text: 'Count' } },
        bargap: 0.05 }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 17. BOX PLOT ──────────────────────────────────────────────────────────────
export function PlotlyBoxPlot({ data, x, y, z, title, height = 280 }: ChartProps) {
  const fixed = autoFixData(data, x, y);
  const yCol = isNumericCol(fixed, y) ? y : x;
  const groups = z ? [...new Set(fixed.map(r => r[z]))] : [null];
  const traces: any[] = groups.map((g, i) => {
    const subset = g !== null ? fixed.filter(r => r[z!] === g) : fixed;
    return { type: 'box', name: g !== null ? String(g) : yCol,
      y: subset.map(r => Number(r[yCol])), marker: { color: MULTI[i % MULTI.length] },
      boxmean: true };
  });
  return (
    <Plot data={traces}
      layout={{ ...BASE, showlegend: !!z, title: { text: title || `${yCol} Distribution`, font: { size: 13 } }, height,
        yaxis: { ...AXIS, title: { text: yCol } } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 18. VIOLIN PLOT ───────────────────────────────────────────────────────────
export function PlotlyViolinPlot({ data, x, y, z, title, height = 280 }: ChartProps) {
  const fixed = autoFixData(data, x, y);
  const yCol = isNumericCol(fixed, y) ? y : x;
  const groups = z ? [...new Set(fixed.map(r => r[z]))] : [null];
  const traces: any[] = groups.map((g, i) => {
    const subset = g !== null ? fixed.filter(r => r[z!] === g) : fixed;
    return { type: 'violin', name: g !== null ? String(g) : yCol,
      y: subset.map(r => Number(r[yCol])), marker: { color: MULTI[i % MULTI.length] },
      box: { visible: true }, meanline: { visible: true } };
  });
  return (
    <Plot data={traces}
      layout={{ ...BASE, showlegend: !!z, title: { text: title || `${yCol} Violin`, font: { size: 13 } }, height,
        yaxis: { ...AXIS, title: { text: yCol } } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 19. TREEMAP ───────────────────────────────────────────────────────────────
export function PlotlyTreemapChart({ data, x, y, z, title, height = 280 }: ChartProps) {
  const fixed = autoFixData(data, x, y);
  const aggData = transformData(fixed, x, isNumericCol(fixed, y) ? y : undefined, 'sum');
  const parent = z ? [...new Set(fixed.map(r => String(r[z])))] : [];
  const labels = aggData.map(r => String(r[x]));
  const parents = labels.map((_, i) => parent[i % parent.length] || '');
  const values = aggData.map(r => r.value);
  return (
    <Plot data={[{ type: 'treemap', labels, parents, values,
      textinfo: 'label+value+percent parent',
      marker: { colorscale: 'Blues', showscale: false } }]}
      layout={{ ...BASE, title: { text: title || `${x} Treemap`, font: { size: 13 } }, height,
        margin: { l: 8, r: 8, t: 38, b: 8 } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 20. SUNBURST ──────────────────────────────────────────────────────────────
export function PlotlySunburstChart({ data, x, y, z, title, height = 280 }: ChartProps) {
  const fixed = autoFixData(data, x, y);
  const aggData = transformData(fixed, x, isNumericCol(fixed, y) ? y : undefined, 'sum');
  const parent = z ? [...new Set(fixed.map(r => String(r[z])))] : [];
  const labels = aggData.map(r => String(r[x]));
  const parents = labels.map((_, i) => parent[i % parent.length] || '');
  return (
    <Plot data={[{ type: 'sunburst', labels, parents, values: aggData.map(r => r.value),
      branchvalues: 'total', textinfo: 'label+percent parent' }]}
      layout={{ ...BASE, title: { text: title || `${x} Sunburst`, font: { size: 13 } }, height,
        margin: { l: 8, r: 8, t: 38, b: 8 } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 21. WATERFALL ─────────────────────────────────────────────────────────────
export function PlotlyWaterfallChart({ data, x, y, title, height = 280, agg = 'auto' }: ChartProps) {
  const fixed = autoFixData(data, x, y);
  const aggData = transformData(fixed, x, isNumericCol(fixed, y) ? y : undefined, agg);
  return (
    <Plot data={[{ type: 'waterfall', x: aggData.map(r => r[x]), y: aggData.map(r => r.value),
      connector: { line: { color: 'rgba(255,255,255,0.2)' } },
      increasing: { marker: { color: '#10b981' } }, decreasing: { marker: { color: '#ef4444' } },
      totals: { marker: { color: '#3b82f6' } },
      hovertemplate: `<b>%{x}</b><br>${y || 'value'}: %{y}<extra></extra>` }]}
      layout={{ ...BASE, title: { text: title || `${y} Waterfall`, font: { size: 13 } }, height,
        xaxis: { ...AXIS, title: { text: x } }, yaxis: { ...AXIS, title: { text: y || 'value' } } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 22. FUNNEL ────────────────────────────────────────────────────────────────
export function PlotlyFunnelChart({ data, x, y, title, height = 280, agg = 'auto' }: ChartProps) {
  const fixed = autoFixData(data, x, y);
  const aggData = transformData(fixed, x, isNumericCol(fixed, y) ? y : undefined, agg)
    .sort((a, b) => b.value - a.value);
  return (
    <Plot data={[{ type: 'funnel', y: aggData.map(r => r[x]), x: aggData.map(r => r.value),
      textinfo: 'value+percent initial',
      marker: { color: aggData.map((_, i) => BLUES[i % BLUES.length]) } }]}
      layout={{ ...BASE, title: { text: title || `${x} Funnel`, font: { size: 13 } }, height,
        margin: { l: 120, r: 16, t: 38, b: 16 } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 23. HEATMAP ───────────────────────────────────────────────────────────────
export function PlotlyHeatmap({ data, x, y, z, title, height = 280 }: ChartProps) {
  const fixed = autoFixData(data, x, y);
  const xs = [...new Set(fixed.map(r => String(r[x])))];
  const ys = [...new Set(fixed.map(r => String(r[y])))];
  const zCol = z && isNumericCol(fixed, z) ? z : null;
  const matrix = ys.map(yv =>
    xs.map(xv => {
      const matches = fixed.filter(r => String(r[x]) === xv && String(r[y]) === yv);
      if (!matches.length) return 0;
      return zCol ? matches.reduce((s, r) => s + Number(r[zCol]), 0) / matches.length : matches.length;
    })
  );
  return (
    <Plot data={[{ type: 'heatmap', x: xs, y: ys, z: matrix,
      colorscale: 'Blues', showscale: true,
      hovertemplate: `${x}: %{x}<br>${y}: %{y}<br>value: %{z}<extra></extra>` }]}
      layout={{ ...BASE, title: { text: title || `${x} × ${y} Heatmap`, font: { size: 13 } }, height,
        xaxis: { ...AXIS, title: { text: x } }, yaxis: { ...AXIS, title: { text: y } },
        margin: { l: 80, r: 16, t: 38, b: 60 } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 24. CALENDAR HEATMAP ──────────────────────────────────────────────────────
export function PlotlyCalendarHeatmap({ data, x, y, title, height = 280 }: ChartProps) {
  // x = date column, y = value column
  const fixed = autoFixData(data, x, y);
  const dateMap: Record<string, number> = {};
  for (const row of fixed) {
    const d = String(row[x]).split('T')[0];
    const v = isNumericCol(fixed, y) ? Number(row[y]) : 1;
    dateMap[d] = (dateMap[d] || 0) + (isNaN(v) ? 1 : v);
  }
  const dates = Object.keys(dateMap).sort();
  const values = dates.map(d => dateMap[d]);
  return (
    <Plot data={[{ type: 'scatter', mode: 'markers', x: dates, y: values,
      marker: { color: values, colorscale: 'Blues', size: 10, showscale: true },
      hovertemplate: 'Date: %{x}<br>Value: %{y}<extra></extra>' }]}
      layout={{ ...BASE, title: { text: title || `${x} Calendar`, font: { size: 13 } }, height,
        xaxis: { ...AXIS, title: { text: 'Date' } }, yaxis: { ...AXIS, title: { text: y || 'value' } } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 25. RADAR CHART ───────────────────────────────────────────────────────────
export function PlotlyRadarChart({ data, x, y, z, title, height = 280 }: ChartProps) {
  const fixed = autoFixData(data, x, y);
  const groups = z ? [...new Set(fixed.map(r => r[z]))] : [null];
  const traces: any[] = groups.map((g, i) => {
    const subset = g !== null ? fixed.filter(r => r[z!] === g) : fixed;
    const agg = transformData(subset, x, isNumericCol(fixed, y) ? y : undefined, 'sum');
    const theta = agg.map(r => String(r[x]));
    const r = agg.map(row => row.value);
    return { type: 'scatterpolar', fill: 'toself', name: g !== null ? String(g) : y,
      theta: [...theta, theta[0]], r: [...r, r[0]],
      line: { color: MULTI[i % MULTI.length] } };
  });
  return (
    <Plot data={traces}
      layout={{ ...BASE, showlegend: !!z, title: { text: title || `${y} Radar`, font: { size: 13 } }, height,
        polar: { radialaxis: { visible: true, gridcolor: 'rgba(255,255,255,0.1)' },
          angularaxis: { gridcolor: 'rgba(255,255,255,0.1)' },
          bgcolor: 'transparent' },
        margin: { l: 40, r: 40, t: 38, b: 40 } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 26. GAUGE CHART ───────────────────────────────────────────────────────────
export function PlotlyGaugeChart({ data, y, title, height = 280, agg = 'sum' }: { data: Row[]; y: string; title?: string; height?: number; agg?: Agg }) {
  const fixed = autoFixData(data, undefined, y);
  const nums = fixed.map(r => Number(r[y])).filter(v => !isNaN(v));
  let value = 0;
  if (agg === 'sum') value = nums.reduce((a, b) => a + b, 0);
  else if (agg === 'mean') value = nums.reduce((a, b) => a + b, 0) / (nums.length || 1);
  else if (agg === 'min') value = Math.min(...nums);
  else if (agg === 'max') value = Math.max(...nums);
  const max = Math.max(...nums) * 1.2 || 100;
  return (
    <Plot data={[{ type: 'indicator', mode: 'gauge+number+delta',
      value, title: { text: title || y, font: { size: 14 } },
      delta: { reference: value * 0.8 },
      gauge: { axis: { range: [0, max], tickcolor: '#e2e8f0' },
        bar: { color: '#3b82f6' },
        steps: [{ range: [0, max * 0.5], color: 'rgba(59,130,246,0.15)' },
                { range: [max * 0.5, max * 0.8], color: 'rgba(59,130,246,0.3)' }],
        threshold: { line: { color: '#f59e0b', width: 3 }, thickness: 0.75, value: value * 0.9 } } }]}
      layout={{ template: 'plotly_dark', paper_bgcolor: 'transparent', height,
        margin: { l: 20, r: 20, t: 50, b: 20 } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 27. PROGRESS BAR (Gauge variant) ─────────────────────────────────────────
export function PlotlyProgressChart({ data, y, title, height = 280, agg = 'mean' }: { data: Row[]; y: string; title?: string; height?: number; agg?: Agg }) {
  const fixed = autoFixData(data, undefined, y);
  const nums = fixed.map(r => Number(r[y])).filter(v => !isNaN(v));
  let value = 0;
  if (agg === 'mean') value = nums.reduce((a, b) => a + b, 0) / (nums.length || 1);
  else if (agg === 'sum') value = nums.reduce((a, b) => a + b, 0);
  else if (agg === 'max') value = Math.max(...nums);
  const pct = Math.min(100, Math.max(0, value));
  return (
    <Plot data={[{ type: 'indicator', mode: 'gauge+number',
      value: pct, title: { text: title || y, font: { size: 14 } },
      gauge: { axis: { range: [0, 100], ticksuffix: '%', tickcolor: '#e2e8f0' },
        bar: { color: pct >= 75 ? '#10b981' : pct >= 40 ? '#f59e0b' : '#ef4444' },
        bgcolor: 'rgba(255,255,255,0.05)', bordercolor: 'rgba(255,255,255,0.1)' } }]}
      layout={{ template: 'plotly_dark', paper_bgcolor: 'transparent', height,
        margin: { l: 20, r: 20, t: 50, b: 20 } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 28. KPI INDICATOR ─────────────────────────────────────────────────────────
export function PlotlyKPI({ data, y, title, height = 280, agg = 'sum' }: { data: Row[]; y: string; title?: string; height?: number; agg?: Agg }) {
  const fixed = autoFixData(data, undefined, y);
  let value = 0; let label = y;
  if (!y || !data.length) { value = 0; }
  else if (!isNumericCol(fixed, y)) {
    value = fixed.filter(r => r[y] !== null && r[y] !== undefined && r[y] !== '').length;
    label = `count(${y})`;
  } else {
    const nums = fixed.map(r => Number(r[y])).filter(v => !isNaN(v));
    if (agg === 'sum') value = nums.reduce((a, b) => a + b, 0);
    else if (agg === 'mean') value = nums.reduce((a, b) => a + b, 0) / (nums.length || 1);
    else if (agg === 'min') value = Math.min(...nums);
    else if (agg === 'max') value = Math.max(...nums);
    else value = nums.reduce((a, b) => a + b, 0);
  }
  return (
    <Plot data={[{ type: 'indicator', mode: 'number+delta', value,
      title: { text: title || label, font: { size: 15 } },
      delta: { reference: value * 0.8, relative: true, valueformat: '.1%' },
      number: { font: { size: 44 }, valueformat: ',.2f' },
      domain: { x: [0, 1], y: [0, 1] } }]}
      layout={{ template: 'plotly_dark', paper_bgcolor: 'transparent', height,
        margin: { l: 20, r: 20, t: 40, b: 20 } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 29. PARALLEL COORDINATES ──────────────────────────────────────────────────
export function PlotlyParallelCoords({ data, x, y, z, title, height = 280 }: ChartProps) {
  const fixed = autoFixData(data, x, y);
  // Build dimensions from all numeric columns
  const numCols = Object.keys(fixed[0] || {}).filter(c => isNumericCol(fixed, c)).slice(0, 8);
  if (numCols.length < 2) return (
    <div className="flex items-center justify-center h-full text-sm" style={{ color: 'var(--text-muted)' }}>
      Parallel Coords needs 2+ numeric columns
    </div>
  );
  const dimensions = numCols.map(col => ({
    label: col, values: fixed.map(r => Number(r[col])),
    range: [Math.min(...fixed.map(r => Number(r[col]))), Math.max(...fixed.map(r => Number(r[col])))]
  }));
  const colorCol = z && isNumericCol(fixed, z) ? z : numCols[0];
  return (
    <Plot data={[{ type: 'parcoords', dimensions,
      line: { color: fixed.map(r => Number(r[colorCol])), colorscale: 'Blues', showscale: false } }]}
      layout={{ ...BASE, title: { text: title || 'Parallel Coordinates', font: { size: 13 } }, height,
        margin: { l: 60, r: 60, t: 50, b: 20 } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 30. CHOROPLETH MAP ────────────────────────────────────────────────────────
export function PlotlyChoroplethMap({ data, x, y, title, height = 280, agg = 'sum' }: ChartProps) {
  const fixed = autoFixData(data, x, y);
  const aggData = transformData(fixed, x, isNumericCol(fixed, y) ? y : undefined, agg);
  return (
    <Plot data={[{ type: 'choropleth', locations: aggData.map(r => String(r[x])),
      z: aggData.map(r => r.value), locationmode: 'country names',
      colorscale: 'Blues', showscale: true,
      hovertemplate: '<b>%{location}</b><br>Value: %{z}<extra></extra>' }]}
      layout={{ ...BASE, title: { text: title || `${y} by ${x}`, font: { size: 13 } }, height,
        geo: { showframe: false, showcoastlines: true, coastlinecolor: 'rgba(255,255,255,0.2)',
          bgcolor: 'transparent', showland: true, landcolor: 'rgba(255,255,255,0.05)',
          showocean: true, oceancolor: 'rgba(59,130,246,0.08)' },
        margin: { l: 0, r: 0, t: 38, b: 0 } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 31. BUBBLE MAP ────────────────────────────────────────────────────────────
export function PlotlyBubbleMap({ data, x, y, title, height = 280 }: ChartProps) {
  // x = location/country name, y = value (bubble size), z = optional color
  const fixed = autoFixData(data, x, y);
  const aggData = transformData(fixed, x, isNumericCol(fixed, y) ? y : undefined, 'sum');
  const maxVal = Math.max(...aggData.map(r => r.value)) || 1;
  return (
    <Plot data={[{ type: 'scattergeo', locations: aggData.map(r => String(r[x])),
      locationmode: 'country names',
      marker: { size: aggData.map(r => 8 + (r.value / maxVal) * 40),
        color: aggData.map(r => r.value), colorscale: 'Blues', showscale: true, opacity: 0.7 },
      text: aggData.map(r => `${r[x]}: ${r.value}`),
      hovertemplate: '<b>%{location}</b><br>Value: %{marker.color}<extra></extra>' }]}
      layout={{ ...BASE, title: { text: title || `${y} Bubble Map`, font: { size: 13 } }, height,
        geo: { showframe: false, showcoastlines: true, coastlinecolor: 'rgba(255,255,255,0.2)',
          bgcolor: 'transparent', showland: true, landcolor: 'rgba(255,255,255,0.05)' },
        margin: { l: 0, r: 0, t: 38, b: 0 } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── 32. DATA TABLE ────────────────────────────────────────────────────────────
export function PlotlyDataTable({ data, height = 280 }: { data: Row[]; height?: number }) {
  if (!data?.length) return (
    <div className="flex items-center justify-center h-full text-sm" style={{ color: 'var(--text-muted)' }}>No data</div>
  );
  const cols = Object.keys(data[0]);
  return (
    <Plot data={[{ type: 'table',
      header: { values: cols.map(c => `<b>${c}</b>`), align: 'left',
        fill: { color: 'rgba(0,0,0,0.55)' }, font: { color: 'white', size: 12 },
        line: { color: 'rgba(255,255,255,0.1)', width: 1 } },
      cells: { values: cols.map(c => data.map(r => r[c] ?? '')), align: 'left',
        fill: { color: ['rgba(25,25,40,0.8)', 'rgba(18,18,30,0.8)'] },
        font: { color: '#e2e8f0', size: 11 }, line: { color: 'rgba(255,255,255,0.06)', width: 1 } } }]}
      layout={{ paper_bgcolor: 'transparent', height, margin: { l: 6, r: 6, t: 6, b: 6 } }}
      config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
  );
}

// ── AUTO CHART TYPE DETECTOR ──────────────────────────────────────────────────
export function detectChartType(data: Row[], x: string, y: string): string {
  if (!data?.length) return 'bar';
  const xNum = isNumericCol(data, x); const yNum = isNumericCol(data, y);
  const xIsDate = data.slice(0, 5).some(r => !isNaN(Date.parse(String(r[x]))));
  if (xNum && yNum) return 'scatter';
  if (xIsDate && yNum) return 'line';
  if (!xNum && yNum && new Set(data.map(r => r[x])).size <= 12) return 'pie';
  return 'bar';
}
