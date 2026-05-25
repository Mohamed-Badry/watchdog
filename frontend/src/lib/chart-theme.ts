/**
 * Shared SveltePlot defaults for the Watchdog dashboard.
 * Color constants, margin presets, and scale configs used across all chart components.
 *
 * ALL chart colors MUST be defined here — never hardcode hex in .svelte files.
 */

// ── Core palette ────────────────────────────────────────────────────────────
export const BRAND = '#b12142';
export const BRAND_GLOW = 'rgba(177, 33, 66, 0.6)';
export const MUTED = '#6c7a96';
export const OK = 'var(--color-ok)';
export const INK = 'var(--color-ink)';
export const INK2 = 'var(--color-ink-2)';
export const INK3 = 'var(--color-ink-3)';
export const BORDER = 'var(--color-border)';

// ── Semantic chart colors ───────────────────────────────────────────────────
export const WATCHDOG_COLORS = {
  nominal: MUTED,
  anomaly: BRAND,
  reference: '#94a3b8',
  threshold: BRAND,
  success: '#34d399',
} as const;

// ── Data series palette ─────────────────────────────────────────────────────
/** Voltage, primary EDA line color */
export const SERIES_VOLTAGE = '#4361ee';
/** Temperature — battery */
export const SERIES_TEMP_BATT = '#e64848';
/** Temperature — panel Z */
export const SERIES_TEMP_PANEL = '#2ec4b6';
/** Battery current, secondary blue */
export const SERIES_CURRENT = '#3a86ff';
/** Histogram / distribution accent */
export const SERIES_HISTOGRAM = '#9b59b6';
/** Neutral / z-score baseline */
export const SERIES_BASELINE = '#94a3b8';
/** Amber accent (One-Class SVM in ROC, alerts) */
export const SERIES_AMBER = '#f59e0b';
/** Green secondary */
export const SERIES_GREEN = '#8ac926';
/** Teal secondary */
export const SERIES_TEAL = '#20b2aa';
/** Z-Score dark line (dark navy for sensitivity sweep) */
export const SERIES_ZSCORE = '#092e4b';

// ── ML chart colors ─────────────────────────────────────────────────────────
/** Feature contribution: normal reconstruction */
export const ML_NORMAL = '#4f7fb5';
/** Feature contribution: faulty reconstruction */
export const ML_FAULT = '#b13a4b';

// ── Correlation heatmap ─────────────────────────────────────────────────────
export const CORR_NEGATIVE = '#2166ac';
export const CORR_NEUTRAL = '#f7f7f7';
export const CORR_POSITIVE = '#b2182b';

// ── Per-feature distribution palette ────────────────────────────────────────
export const FEATURE_COLORS: Record<string, string> = {
  batt_voltage: SERIES_VOLTAGE,
  batt_current: SERIES_CURRENT,
  temp_batt_a: SERIES_TEMP_BATT,
  temp_batt_b: SERIES_TEAL,
  temp_panel_z: SERIES_GREEN,
};

// ── Standard margins ────────────────────────────────────────────────────────
/** Margins for full-width dashboard chart cards */
export const CARD_MARGIN = { top: 24, right: 24, bottom: 44, left: 64 };

/** Compact margins for smaller inline charts */
export const COMPACT_MARGIN = { top: 12, right: 12, bottom: 32, left: 56 };

/** Margins for sparklines (no axes) */
export const SPARK_MARGIN = { top: 4, right: 4, bottom: 4, left: 4 };

// ── Chart heights ───────────────────────────────────────────────────────────
export const CHART_HEIGHT = {
  full: 360,
  half: 260,
  compact: 200,
  sparkline: 48,
} as const;
