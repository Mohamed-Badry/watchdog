# Dashboard SveltePlot Guidelines & Best Practices

> **Scope:** Guidelines for creating and maintaining dashboard visualizations using **SveltePlot** (`svelteplot` v0.14.x).
> **Goal:** Declarative, maintainable, theme-aware, and responsive charts with superior UX.

---

## 1. Core Principles

### Push Transformation to SveltePlot
Avoid heavy client-side data wrangling in `$derived` blocks (manual binning, filtering, label formatting). SveltePlot's built-in transforms (`binX`, `groupX`, `sort`, `filter`) handle this declaratively.
**Goal:** Pass raw API data directly to SveltePlot marks whenever possible.

### Shared SveltePlot Theme & Defaults
Global defaults are set in `src/routes/(dashboard)/+layout.svelte` via `setPlotDefaults`.
Colors and margin constants should be consistently used (e.g., `var(--color-brand)`, `var(--color-muted)`).

---

## 2. Component Design Guidelines

### Interactivity & State Integration
- **Tooltips:** All charts should use the SveltePlot `<Tip>` mark for declarative hover contexts.
- **Cross-Filtering:** Let reactive state (`noradId`, `bucket`, `limit`) drive `$effect` → fetch → `$derived` data → `<Plot>`. No extra wiring needed.

### Responsive Sizing (Mobile-First)
- `<Plot>` fills container width by default. Wrap the plot in a responsive container (e.g., `<div class="h-[400px] w-full">`).
- **Margins:** Keep them minimal (e.g., `<Plot marginLeft={60} ...>`). Large static margins (e.g., `marginLeft={150}`) break on mobile devices and must be avoided. Use reasonable baseline values that fit on a 320px viewport, or dynamically adjust.
- **Heights:** Use fixed pixel heights (e.g., `height={300}`) for predictability, letting the width remain `100%`.

### Theme Integration
- SveltePlot renders SVG. Theme tokens work natively via CSS variables (e.g., `stroke="var(--color-ink-3)"`) in stroke/fill props.
- Global overrides are present in `src/app.css` to ensure axis ticks, lines, and grids match dark/light modes.

---

## 3. Common Transformations 

| Use Case | Legacy Client Transform | SveltePlot Replacement |
|---|---|---|
| **Histograms** | Manual bin loop | `<RectY>` with `binX({ y: 'count' })` |
| **Time Series** | Manual date label formatting | `<BarY>` or `<Line>` with temporal x-scale, auto-ticks |
| **Density/Downsampling** | Manual downsampling | Let SveltePlot handle density natively; avoid manual array slicing |
| **Categorical Splits** | Manual `.filter` splits | Single mark with `fill` channel mapped to a boolean or category |
| **Reference Lines** | Manual array data | `<RuleY>` or `<RuleX>` |

---

## 4. Visual QA Checklist for New Charts

- [ ] **Dark mode:** axis text, grid lines, tooltips use theme tokens
- [ ] **Light mode:** same verification
- [ ] **Responsive:** charts resize cleanly from 320px to 1920px without horizontal overflow
- [ ] **Empty states:** placeholder shown when API returns empty data
- [ ] **Tooltips:** hover context is available on all relevant marks
- [ ] **Data processing:** minimal JavaScript mapping; relying on SveltePlot's Grammar of Graphics transforms
