# Dashboard SveltePlot Refactoring Plan

> **Scope:** Migrate all dashboard visualizations from **LayerChart + raw D3/SVG** to **SveltePlot** (`svelteplot` v0.14.x).
> **Goal:** Declarative, maintainable, theme-aware charts with superior UX.
> **Status:** ✅ **COMPLETE** — All 5 phases implemented and visually verified (2026-05-12).
> **Goal:** Declarative, maintainable, theme-aware charts with superior UX.

---

## Legacy Audit

### Current Dependencies (to remove)

| Package | Role | Used By |
|---|---|---|
| `layerchart` | High-level chart wrappers | `RocCurveChart`, `SensitivityChart`, `ThroughputChart`, `HistogramChart`, `ScatterChart` |
| `d3-scale` / `@types/d3-scale` | Manual scale construction | Sparkline, potential future use |
| `d3-array` | `min`/`max`/`bin` utilities | HistogramChart manual binning |
| `d3-shape` | Line/area generators | Sparkline polyline |
| `d3-format` | Number formatting | Tooltip values |
| `d3-time` / `d3-time-format` | Time axis formatting | ThroughputChart labels |

### Current Chart Components — Shortcomings

#### 1. `Sparkline.svelte` (raw SVG polyline)
- **Location:** `src/lib/components/Sparkline.svelte`
- **Issues:** No axes, no tooltips, no responsive sizing, manual coordinate math, anomaly dots lack hover context.
- **Data:** `data: {frame_count, anomaly_count}[]`

#### 2. `ThroughputChart.svelte` (LayerChart `BarChart`)
- **Location:** `src/lib/components/charts/ThroughputChart.svelte`
- **Issues:** Stacked bar via `seriesLayout="stack"` — hardcoded colors, manual date label formatting in JS, no axis labels, no grid.
- **Data:** `buckets: {timestamp, frame_count, anomaly_count}[]`

#### 3. `HistogramChart.svelte` (LayerChart `BarChart`)
- **Location:** `src/lib/components/charts/HistogramChart.svelte`
- **Issues:** Manual binning logic in `$derived` (reimplements `d3.bin`), hardcoded brand color, no axis titles.
- **Data:** `values: number[]`

#### 4. `ScatterChart.svelte` (LayerChart `ScatterChart`)
- **Location:** `src/lib/components/charts/ScatterChart.svelte`
- **Issues:** Two-series split done manually (filter nominal/anomalous), hardcoded colors, tooltip limited to x/y.
- **Data:** `points: {x, y, a: boolean}[]`

#### 5. `RocCurveChart.svelte` (LayerChart `LineChart`)
- **Location:** `src/lib/components/charts/RocCurveChart.svelte`
- **Issues:** Diagonal reference line requires manual `randomLine` array + second series, current operating point not rendered as a dot, hardcoded colors.
- **Data:** `rocData: {fpr, tpr}[]`, `currentFpr`, `currentTpr`

#### 6. `SensitivityChart.svelte` (LayerChart `BarChart`)
- **Location:** `src/lib/components/charts/SensitivityChart.svelte`
- **Issues:** Manual downsampling (`i % 2 === 0`), active threshold bar not visually highlighted, no axis title.
- **Data:** `sweepData: {threshold, f1_score}[]`, `currentThreshold`

### Pages Using Charts

| Page | Charts Used |
|---|---|
| `/dashboard` (Home) | Sparkline (planned, not wired) |
| `/dashboard/insights` | ThroughputChart, ScatterChart, HistogramChart |
| `/dashboard/ml` | RocCurveChart, SensitivityChart |
| `/dashboard/operations` | Raw inline Gantt (CSS `position: absolute`) |
| `/dashboard/live` | None (feed cards only) |

---

## Phase 1 — Setup & Dependency Management

### 1.1 Install SveltePlot

```bash
cd frontend && bun add svelteplot
```

### 1.2 Remove Legacy Dependencies

```bash
bun remove layerchart d3-array d3-format d3-scale d3-shape d3-time d3-time-format
bun remove -d @types/d3-scale
```

> **Note:** `date-fns` can stay — it's used elsewhere and SveltePlot doesn't conflict.

### 1.3 Remove LayerChart Theme Bridge

In `src/app.css`, delete the `.lc-root-container` block (lines 68–78):

```diff
-/* ── LayerChart theme bridge ──
-   Maps project tokens to the CSS vars LayerChart expects. */
-.lc-root-container {
-  --color-primary: var(--color-brand);
-  --color-secondary: var(--color-muted);
-  --color-success: var(--color-ok);
-  --color-surface-100: var(--color-paper);
-  --color-surface-200: var(--color-surface);
-  --color-surface-300: var(--color-border);
-  --color-surface-content: var(--color-ink);
-}
```

### 1.4 Create Shared SveltePlot Theme & Defaults

Create `src/lib/chart-theme.ts`:

```ts
// Shared SveltePlot defaults for the Watchdog dashboard.
export const BRAND = '#b12142';
export const MUTED = '#6c7a96';
export const OK    = 'var(--color-ok)';
export const INK   = 'var(--color-ink)';
export const INK3  = 'var(--color-ink-3)';
export const BORDER = 'var(--color-border)';

export const WATCHDOG_COLORS = {
  nominal: MUTED,
  anomaly: BRAND,
  reference: INK3,
};

// Standard margins for dashboard cards
export const CARD_MARGIN = { top: 20, right: 20, bottom: 40, left: 50 };
```

### 1.5 Set Plot Defaults in Dashboard Layout

In `src/routes/(dashboard)/+layout.svelte`, add `setPlotDefaults` at the top of the script:

```svelte
<script lang="ts">
  import { setPlotDefaults } from 'svelteplot';
  setPlotDefaults({
    style: {
      fontSize: 11,
      fontFamily: '"IBM Plex Sans", sans-serif',
    },
    color: { scheme: 'observable10' },
  });
  // ... rest of existing code
</script>
```

---

## Phase 2 — Data Layer & Transformations

### 2.1 Principle: Push Transformation to SveltePlot

Legacy code performs heavy client-side data wrangling in `$derived` blocks (manual binning, filtering, label formatting). SveltePlot's built-in transforms (`binX`, `groupX`, `sort`, `filter`) handle this declaratively. The goal is to **pass raw API data directly to SveltePlot marks**.

### 2.2 Data Shape Contracts (No Change to API)

The existing API responses are clean. No backend changes required:

```
API Response → $derived (minimal reshape) → <Plot> data prop → Mark transform
```

### 2.3 Transformations Moving into SveltePlot

| Chart | Legacy Client Transform | SveltePlot Replacement |
|---|---|---|
| HistogramChart | Manual bin loop (20 lines) | `<RectY>` with `binX({ y: 'count' })` |
| ThroughputChart | Manual date label formatting | `<BarY>` with temporal x-scale, auto-ticks |
| SensitivityChart | Manual downsampling `i % 2` | Remove; `<BarY>` handles density natively |
| ScatterChart | Manual `.filter(p => !p.a)` split | Single `<Dot>` with `fill` channel → boolean |
| RocCurveChart | Manual sort + diagonal array | `<Line>` for curve + `<Line>` for diagonal |

### 2.4 Shared Data Utilities

Create `src/lib/data/transforms.ts`:

```ts
/** Map throughput bucket to a Date for SveltePlot temporal scale. */
export function parseBucketTimestamp(b: { timestamp: string }) {
  return { ...b, date: new Date(b.timestamp) };
}

/** Derive nominal count from total - anomalies. */
export function addNominalCount(b: { frame_count: number; anomaly_count: number }) {
  return { ...b, nominal: b.frame_count - b.anomaly_count };
}
```

---

## Phase 3 — Component-by-Component Migration

### 3.1 Sparkline → `SparklinePlot.svelte`

**File:** `src/lib/components/charts/SparklinePlot.svelte`

```svelte
<script lang="ts">
  import { Plot, AreaY, Dot } from 'svelteplot';
  let { data = [], width = 200, height = 40 } = $props();
</script>

<Plot {width} {height} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}
      x={{ axis: null }} y={{ axis: null, domain: [0, undefined] }}>
  <AreaY {data} x={(_, i) => i} y="frame_count"
         fill="var(--color-brand)" fillOpacity={0.15}
         stroke="var(--color-brand)" strokeWidth={1.5} curve="natural" />
  <Dot data={data.filter(d => d.anomaly_count > 0)}
       x={(_, i) => i} y="frame_count"
       fill="var(--color-brand)" r={3} />
</Plot>
```

**Improvements:** Area fill, anomaly dots as grammar marks, responsive sizing.

---

### 3.2 ThroughputChart → `ThroughputPlot.svelte`

**File:** `src/lib/components/charts/ThroughputPlot.svelte`

```svelte
<script lang="ts">
  import { Plot, BarY, Tip } from 'svelteplot';
  import { parseBucketTimestamp, addNominalCount } from '$lib/data/transforms';

  let { buckets, bucketType } = $props<{
    buckets: { timestamp: string; frame_count: number; anomaly_count: number }[];
    bucketType: string;
  }>();

  let longData = $derived(
    (buckets ?? []).map(b => parseBucketTimestamp(addNominalCount(b)))
      .flatMap(d => [
        { date: d.date, count: d.nominal, type: 'Nominal' },
        { date: d.date, count: d.anomaly_count, type: 'Anomaly' },
      ])
  );
</script>

<Plot height={400} x={{ type: 'band', label: null }}
      y={{ grid: true, label: 'Frames' }}
      color={{ domain: ['Nominal', 'Anomaly'], range: ['#6c7a96', '#b12142'] }}>
  <BarY data={longData} x="date" y="count" fill="type"
        offset="stack" fillOpacity={0.7} rx={2} />
  <Tip data={longData} x="date" y="count" fill="type" />
</Plot>
```

**Improvements:** Temporal axis with auto-ticks, `<Tip>` tooltips, stacking via `offset="stack"`, grid lines.

---

### 3.3 HistogramChart → `HistogramPlot.svelte`

**File:** `src/lib/components/charts/HistogramPlot.svelte`

```svelte
<script lang="ts">
  import { Plot, RectY } from 'svelteplot';
  import { binX } from 'svelteplot/transforms';

  let { values, xAxisLabel, yAxisLabel, binCount = 20 } = $props<{
    values: number[]; xAxisLabel?: string; yAxisLabel?: string; binCount?: number;
  }>();

  let data = $derived((values ?? []).map(v => ({ value: v })));
</script>

<Plot height={280}
      x={{ label: xAxisLabel ?? null }}
      y={{ grid: true, label: yAxisLabel ?? 'Frequency' }}>
  <RectY data={data}
         {...binX({ y: 'count' }, { x: 'value', thresholds: binCount })}
         fill="#b12142" fillOpacity={0.55} rx={2} />
</Plot>
```

**Improvements:** Eliminates 20 lines of manual binning — `binX` handles it declaratively.

---

### 3.4 ScatterChart → `ScatterPlot.svelte`

**File:** `src/lib/components/charts/ScatterPlot.svelte`

```svelte
<script lang="ts">
  import { Plot, Dot, Tip } from 'svelteplot';

  let { points, xAxisLabel, yAxisLabel } = $props<{
    points: { x: number; y: number; a: boolean }[];
    xAxisLabel?: string; yAxisLabel?: string;
  }>();
</script>

<Plot height={280}
      x={{ grid: true, label: xAxisLabel, nice: true }}
      y={{ grid: true, label: yAxisLabel, nice: true }}
      color={{ domain: [false, true], range: ['#6c7a96', '#b12142'], legend: true }}>
  <Dot data={points} x="x" y="y" fill="a"
       r={d => d.a ? 4.5 : 3.5}
       fillOpacity={d => d.a ? 0.85 : 0.35} stroke="none" />
  <Tip data={points} x="x" y="y" fill="a" />
</Plot>
```

**Improvements:** Single `<Dot>` with `fill` mapped to anomaly flag — no manual series split. Auto color legend.

---

### 3.5 RocCurveChart → `RocCurvePlot.svelte`

**File:** `src/lib/components/charts/RocCurvePlot.svelte`

```svelte
<script lang="ts">
  import { Plot, Line, Dot, Text } from 'svelteplot';

  let { rocData, currentFpr, currentTpr } = $props<{
    rocData: { fpr: number; tpr: number }[];
    currentFpr: number; currentTpr: number;
  }>();

  let sorted = $derived([...(rocData ?? [])].sort((a, b) => a.fpr - b.fpr));
  let diag = [{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }];
  let opPoint = $derived([{ fpr: currentFpr, tpr: currentTpr }]);
</script>

<Plot height={320} aspectRatio={1}
      x={{ domain: [0, 1], label: 'False Positive Rate', grid: true }}
      y={{ domain: [0, 1], label: 'True Positive Rate', grid: true }}>
  <Line data={diag} x="fpr" y="tpr"
        stroke="#94a3b8" strokeWidth={1} strokeDasharray="4 4" strokeOpacity={0.5} />
  <Line data={sorted} x="fpr" y="tpr" stroke="#b12142" strokeWidth={2} />
  {#if currentFpr != null && currentTpr != null}
    <Dot data={opPoint} x="fpr" y="tpr"
         fill="#b12142" r={6} stroke="white" strokeWidth={2} />
    <Text data={opPoint} x="fpr" y="tpr"
          text={d => `(${d.fpr.toFixed(2)}, ${d.tpr.toFixed(2)})`}
          dy={-12} fontSize={10} fill="var(--color-ink-2)" />
  {/if}
</Plot>
```

**Improvements:** Operating point as prominent dot + label, 1:1 aspect ratio, proper axis labels.

---

### 3.6 SensitivityChart → `SensitivityPlot.svelte`

**File:** `src/lib/components/charts/SensitivityPlot.svelte`

```svelte
<script lang="ts">
  import { Plot, BarY, Tip } from 'svelteplot';

  let { sweepData, currentThreshold } = $props<{
    sweepData: { threshold: number; f1_score: number }[];
    currentThreshold: number;
  }>();
</script>

<Plot height={300}
      x={{ type: 'band', label: 'Threshold' }}
      y={{ grid: true, label: 'F1 Score', domain: [0, 1] }}>
  <BarY data={sweepData} x={d => d.threshold.toFixed(2)} y="f1_score"
        fill={d => Math.abs(d.threshold - currentThreshold) < 0.02 ? '#b12142' : '#6c7a96'}
        fillOpacity={d => Math.abs(d.threshold - currentThreshold) < 0.02 ? 0.9 : 0.45}
        rx={2} />
  <Tip data={sweepData} x={d => d.threshold.toFixed(2)} y="f1_score" />
</Plot>
```

**Improvements:** No downsampling, active threshold bar highlighted, Y clamped [0,1].

---

### 3.7 Operations Gantt → `PassTimelinePlot.svelte`

**File:** `src/lib/components/charts/PassTimelinePlot.svelte`

Replaces the fragile CSS-positioned inline Gantt with a proper temporal bar chart:

```svelte
<script lang="ts">
  import { Plot, BarX, RuleX, Tip } from 'svelteplot';

  let { passes, now, endTime } = $props<{
    passes: { satellite: string; aos: Date; los: Date; max_elevation: number }[];
    now: Date; endTime: Date;
  }>();
</script>

<Plot height={Math.max(200, passes.length * 8 + 80)}
      x={{ type: 'utc', domain: [now, endTime], label: null }}
      y={{ type: 'band', label: null }}>
  <RuleX data={[now]} x={d => d}
         stroke="var(--color-ok)" strokeWidth={2} strokeDasharray="4 2" />
  <BarX data={passes} x1="aos" x2="los" y="satellite"
        fill="#b12142" fillOpacity={0.8} rx={3} />
  <Tip data={passes} x1="aos" x2="los" y="satellite" />
</Plot>
```

**Improvements:** Temporal x-axis, band y-axis, "Now" rule line, native tooltips.

---

### 3.8 NEW: Live Anomaly Score Timeline

**File:** `src/lib/components/charts/AnomalyTimelinePlot.svelte`

Missing from current Live Watcher — adds real-time anomaly score visualization:

```svelte
<script lang="ts">
  import { Plot, Dot, Line, RuleY } from 'svelteplot';

  let { frames, threshold } = $props<{
    frames: { timestamp: string; anomaly_score: number; is_anomaly: boolean }[];
    threshold: number;
  }>();

  let plotData = $derived(frames.map(f => ({ ...f, date: new Date(f.timestamp) })));
</script>

<Plot height={200}
      x={{ type: 'utc', label: null }}
      y={{ label: 'Score', grid: true }}
      color={{ domain: [false, true], range: ['#6c7a96', '#b12142'] }}>
  <RuleY data={[threshold]} y={d => d}
         stroke="#b12142" strokeDasharray="6 3" strokeOpacity={0.6} />
  <Line data={plotData} x="date" y="anomaly_score"
        stroke="var(--color-ink-3)" strokeWidth={1} strokeOpacity={0.4} />
  <Dot data={plotData} x="date" y="anomaly_score" fill="is_anomaly"
       r={d => d.is_anomaly ? 5 : 3} />
</Plot>
```

---

## Phase 4 — Interactivity & State Integration

### 4.1 Tooltips
All charts use SveltePlot `<Tip>` mark (replaces verbose `Tooltip.Root > Header > List > Item`).

### 4.2 Cross-Filtering
Insights page reactive state (`noradId`, `bucket`, `limit`) drives `$effect` → fetch → `$derived` data → `<Plot>`. No extra wiring needed.

### 4.3 Responsive Sizing
`<Plot>` fills container width by default. Wrap in `<div class="h-[400px] w-full">`.

### 4.4 Theme Integration
SveltePlot renders SVG. Theme tokens work via `var(--color-*)` in stroke/fill props. Add to `app.css`:

```css
.svelteplot-plot line, .svelteplot-plot .tick text {
  stroke: var(--color-ink-3);
  fill: var(--color-ink-2);
}
```

---

## Phase 5 — Final Polish & Verification

### 5.1 File Cleanup Checklist

| Action | File |
|---|---|
| **Delete** | `src/lib/components/Sparkline.svelte` |
| **Delete** | `src/lib/components/charts/RocCurveChart.svelte` |
| **Delete** | `src/lib/components/charts/SensitivityChart.svelte` |
| **Delete** | `src/lib/components/charts/ThroughputChart.svelte` |
| **Delete** | `src/lib/components/charts/HistogramChart.svelte` |
| **Delete** | `src/lib/components/charts/ScatterChart.svelte` |
| **Create** | `src/lib/chart-theme.ts` |
| **Create** | `src/lib/data/transforms.ts` |
| **Create** | `src/lib/components/charts/SparklinePlot.svelte` |
| **Create** | `src/lib/components/charts/ThroughputPlot.svelte` |
| **Create** | `src/lib/components/charts/HistogramPlot.svelte` |
| **Create** | `src/lib/components/charts/ScatterPlot.svelte` |
| **Create** | `src/lib/components/charts/RocCurvePlot.svelte` |
| **Create** | `src/lib/components/charts/SensitivityPlot.svelte` |
| **Create** | `src/lib/components/charts/PassTimelinePlot.svelte` |
| **Create** | `src/lib/components/charts/AnomalyTimelinePlot.svelte` |
| **Edit** | `src/app.css` — remove `.lc-root-container`, add SveltePlot overrides |
| **Edit** | `src/routes/(dashboard)/+layout.svelte` — add `setPlotDefaults` |
| **Edit** | `dashboard/insights/+page.svelte` — swap chart imports |
| **Edit** | `dashboard/ml/+page.svelte` — swap chart imports |
| **Edit** | `dashboard/operations/+page.svelte` — replace inline Gantt |
| **Edit** | `dashboard/live/+page.svelte` — add AnomalyTimelinePlot |

### 5.2 Visual QA Checklist

- [ ] Dark mode: axis text, grid lines, tooltips use theme tokens
- [ ] Light mode: same verification
- [ ] Responsive: charts resize 320px → 1920px
- [ ] Empty states: placeholder when API returns empty
- [ ] Tooltips: hover on all mark types
- [ ] No D3 imports in final bundle

### 5.3 Execution Order & Estimates

```
Phase 1 (Setup)          →  ~30 min
Phase 2 (Data Layer)     →  ~20 min
Phase 3.1–3.3 (Core)     →  ~1.5 hr
Phase 3.4–3.6 (ML/EDA)   →  ~1.5 hr
Phase 3.7–3.8 (New)      →  ~1 hr
Phase 4 (Interactivity)  →  ~45 min
Phase 5 (Polish)         →  ~1 hr
Total                    →  ~6.5 hours
```
