/**
 * Shared SveltePlot defaults for the Watchdog dashboard.
 * Color constants, margin presets, and scale configs used across all chart components.
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
