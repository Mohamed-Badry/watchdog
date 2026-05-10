<script lang="ts">
  /**
   * Pass Dynamics — Dual-axis time series within a single pass
   * Reproduces docs/figures/pass_dynamics_micro.png
   *
   * Top: Battery Voltage with filled area
   * Bottom: Battery Current (red) + Panel Z Temp (green dashed)
   */
  import { Plot, Line, AreaY } from 'svelteplot';

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
      <Plot height={140}
        x={{ type: 'utc', label: false }}
        y={{ label: 'V', grid: true, nice: true }}
        marginTop={8} marginRight={12} marginBottom={20} marginLeft={44}>
        <AreaY data={plotData} x="date" y="batt_voltage"
               fill="#4361ee" fillOpacity={0.12} />
        <Line data={plotData} x="date" y="batt_voltage"
              stroke="#4361ee" strokeWidth={2} />
      </Plot>
    {/if}
  </div>

  <!-- Current + Panel Temp panel -->
  <div>
    <p class="mb-1 text-[0.6rem] font-semibold uppercase tracking-wider text-ink-3">
      Current (A) &amp; Panel Temp (°C)
    </p>
    {#if plotData.length > 0}
      <Plot height={160}
        x={{ type: 'utc', label: 'Time (UTC)' }}
        y={{ label: 'A / °C', grid: true, nice: true }}
        marginTop={8} marginRight={12} marginBottom={32} marginLeft={44}>
        <AreaY data={plotData} x="date" y="batt_current"
               fill="#e64848" fillOpacity={0.08} />
        <Line data={plotData} x="date" y="batt_current"
              stroke="#e64848" strokeWidth={2} />
        <Line data={plotData} x="date" y="temp_panel_z"
              stroke="#2ec4b6" strokeWidth={1.5}
              strokeDasharray="6 3" />
      </Plot>
      <div class="mt-1 flex items-center justify-center gap-5 text-[0.55rem] text-ink-3">
        <span class="flex items-center gap-1">
          <span class="inline-block h-0.5 w-4" style="background: #e64848"></span> Battery Current
        </span>
        <span class="flex items-center gap-1">
          <span class="inline-block h-0.5 w-4 border-t border-dashed" style="border-color: #2ec4b6"></span> Panel Z Temp
        </span>
      </div>
    {/if}
  </div>
</div>
