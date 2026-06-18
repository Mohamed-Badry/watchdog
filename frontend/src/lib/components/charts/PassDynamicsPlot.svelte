<script lang="ts">
  import ResponsivePlot from './ResponsivePlot.svelte';
  /**
   * Pass Dynamics — Dual-axis time series within a single pass
   * Reproduces docs/figures/pass_dynamics_micro.png
   *
   * Top: Battery Voltage with filled area
   * Bottom: Battery Current (red) + Panel Z Temp (green dashed)
   */
  import { Line, AreaY } from 'svelteplot';
  import { SERIES_VOLTAGE, SERIES_TEMP_BATT, SERIES_TEMP_PANEL } from '$lib/chart-theme';

  type Frame = {
    timestamp: string;
    batt_voltage?: number | null;
    batt_current?: number | null;
    temp_panel_z?: number | null;
  };

  let { frames = [] } = $props<{ frames: Frame[] }>();

  let plotData = $derived(
    frames
      .filter((f: Frame) => f.timestamp)
      .map((f: Frame) => ({ ...f, date: new Date(f.timestamp) }))
      .sort((a: { date: Date }, b: { date: Date }) => a.date.getTime() - b.date.getTime())
  );
</script>

<div class="flex flex-col gap-4">
  <!-- Voltage panel -->
  <div>
    <p class="mb-1 text-[0.6rem] font-semibold uppercase tracking-wider text-ink-3">
      Battery Voltage (V)
    </p>
    {#if plotData.length > 0}
      <ResponsivePlot height={140}
        x={{ type: 'utc', label: false }}
        y={{ label: false, grid: true, nice: true }}
        marginTop={12} marginRight={12} marginBottom={20} marginLeft={46}>
        <AreaY data={plotData} x="date" y="batt_voltage"
               fill={SERIES_VOLTAGE} fillOpacity={0.12} />
        <Line data={plotData} x="date" y="batt_voltage"
              stroke={SERIES_VOLTAGE} strokeWidth={2} />
      </ResponsivePlot>
    {/if}
  </div>

  <!-- Current + Panel Temp panel -->
  <div>
    <p class="mb-1 text-[0.6rem] font-semibold uppercase tracking-wider text-ink-3">
      Current (A) &amp; Panel Temp (°C)
    </p>
    {#if plotData.length > 0}
      <ResponsivePlot height={160}
        x={{ type: 'utc', label: false }}
        y={{ label: false, grid: true, nice: true }}
        marginTop={12} marginRight={12} marginBottom={32} marginLeft={46}>
        <AreaY data={plotData} x="date" y="batt_current"
               fill={SERIES_TEMP_BATT} fillOpacity={0.08} />
        <Line data={plotData} x="date" y="batt_current"
              stroke={SERIES_TEMP_BATT} strokeWidth={2} />
        <Line data={plotData} x="date" y="temp_panel_z"
              stroke={SERIES_TEMP_PANEL} strokeWidth={1.5}
              strokeDasharray="6 3" />
      </ResponsivePlot>
      <div class="mt-1 flex items-center justify-center gap-5 text-[0.55rem] text-ink-3">
        <span class="flex items-center gap-1">
          <span class="inline-block h-0.5 w-4" style="background: {SERIES_TEMP_BATT}"></span> Battery Current
        </span>
        <span class="flex items-center gap-1">
          <span class="inline-block h-0.5 w-4 border-t border-dashed" style="border-color: {SERIES_TEMP_PANEL}"></span> Panel Z Temp
        </span>
      </div>
    {/if}
  </div>
</div>
