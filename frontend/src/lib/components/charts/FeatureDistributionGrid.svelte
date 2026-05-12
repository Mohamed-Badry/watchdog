<script lang="ts">
  /**
   * Feature Distribution Grid — Multi-panel histograms with stats
   * Reproduces docs/figures/feature_distributions.png
   *
   * Shows distribution of each golden feature with:
   * - Histogram bars
   * - Mean (μ) and ±2σ reference lines
   * - Per-feature color coding
   */
  import { Plot, RectY, RuleX, Text } from 'svelteplot';
  import { binX } from 'svelteplot/transforms';

  type FrameFeatures = Record<string, number | null>;

  let { frames = [] } = $props<{ frames: FrameFeatures[] }>();

  const FEATURES = [
    { key: 'batt_voltage', label: 'Battery Voltage (V)', color: '#4361ee' },
    { key: 'batt_current', label: 'Battery Current (A)', color: '#3a86ff' },
    { key: 'temp_batt_a', label: 'Temp Batt A (°C)', color: '#2ec4b6' },
    { key: 'temp_batt_b', label: 'Temp Batt B (°C)', color: '#20b2aa' },
    { key: 'temp_panel_z', label: 'Panel Z Temp (°C)', color: '#8ac926' },
  ] as const;

  function computeStats(values: number[]) {
    if (values.length === 0) return { mean: 0, std: 0 };
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
    return { mean, std: Math.sqrt(variance) };
  }

  let panels = $derived(
    FEATURES.map(f => {
      const values = frames
        .map((frame: FrameFeatures) => frame[f.key])
        .filter((v: number | null): v is number => v != null && !isNaN(v));
      const stats = computeStats(values);
      const data = values.map((v: number) => ({ value: v }));
      const binned = data.length > 0
        ? binX({ data, x: 'value' }, { y: 'count', thresholds: 30 })
        : { data: [] };
      return { ...f, values, stats, binned };
    })
  );
</script>

<div class="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
  {#each panels as panel}
    <div class="rounded-2xl border border-border bg-panel/50 p-4 backdrop-blur">
      <div class="mb-2 flex items-baseline justify-between">
        <span class="text-xs font-semibold text-ink-2">{panel.label}</span>
        {#if panel.values.length > 0}
          <span class="font-mono text-[0.6rem] text-ink-3">
            μ={panel.stats.mean.toFixed(2)} σ={panel.stats.std.toFixed(2)}
          </span>
        {/if}
      </div>
      {#if panel.values.length > 0}
        <Plot
          height={160}
          x={{ label: false, nice: true }}
          y={{ label: false, grid: true }}
          marginTop={8} marginRight={8} marginBottom={24} marginLeft={36}
        >
          <RectY {...panel.binned}
                 fill={panel.color} fillOpacity={0.55}
                 stroke={panel.color} strokeWidth={0.5} strokeOpacity={0.3} />

          <!-- Mean line -->
          <RuleX data={[panel.stats.mean]} x={d => d}
                 stroke={panel.color} strokeWidth={2} strokeOpacity={0.8} />

          <!-- ±2σ reference lines -->
          <RuleX data={[panel.stats.mean - 2 * panel.stats.std]} x={d => d}
                 stroke="var(--color-brand)" strokeWidth={1}
                 strokeDasharray="4 3" strokeOpacity={0.5} />
          <RuleX data={[panel.stats.mean + 2 * panel.stats.std]} x={d => d}
                 stroke="var(--color-brand)" strokeWidth={1}
                 strokeDasharray="4 3" strokeOpacity={0.5} />
        </Plot>
        <div class="mt-1 flex justify-center gap-4 text-[0.55rem] text-ink-3">
          <span class="flex items-center gap-1">
            <span class="inline-block h-px w-3" style="background: {panel.color}"></span> μ
          </span>
          <span class="flex items-center gap-1">
            <span class="inline-block h-px w-3 border-t border-dashed border-brand opacity-50"></span> ±2σ
          </span>
        </div>
      {:else}
        <div class="flex h-40 items-center justify-center text-xs text-ink-3">No data</div>
      {/if}
    </div>
  {/each}
</div>
