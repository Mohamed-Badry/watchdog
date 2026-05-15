<script lang="ts">
  /**
   * Macro-Scale Health — Dual-panel: Voltage with ±2σ band + Temperature overlay
   * Reproduces docs/figures/timeseries_macro_7month.png
   *
   * Top: Daily average battery voltage with ±2σ band
   * Bottom: Daily average temp_batt_a (red) + temp_panel_z (green)
   */
  import { Plot, Line, AreaY } from 'svelteplot';
  import { COMPACT_MARGIN } from '$lib/chart-theme';

  type Frame = {
    timestamp: string;
    batt_voltage?: number | null;
    temp_batt_a?: number | null;
    temp_panel_z?: number | null;
  };

  let { frames = [] } = $props<{ frames: Frame[] }>();

  // Group frames by day and compute daily stats
  let dailyStats = $derived(() => {
    const byDay = new Map<string, Frame[]>();
    for (const f of frames) {
      if (!f.timestamp) continue;
      const day = f.timestamp.slice(0, 10); // YYYY-MM-DD
      if (!byDay.has(day)) byDay.set(day, []);
      byDay.get(day)!.push(f);
    }

    const result: {
      date: Date;
      v_mean: number; v_sigma_low: number; v_sigma_high: number;
      t_batt: number; t_panel: number;
    }[] = [];

    for (const [day, dayFrames] of [...byDay.entries()].sort()) {
      const volts = dayFrames.map(f => f.batt_voltage).filter((v): v is number => v != null);
      const tBatt = dayFrames.map(f => f.temp_batt_a).filter((v): v is number => v != null);
      const tPanel = dayFrames.map(f => f.temp_panel_z).filter((v): v is number => v != null);

      if (volts.length === 0) continue;

      const mean = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;
      const std = (arr: number[], avg: number) =>
        Math.sqrt(arr.reduce((a, b) => a + (b - avg) ** 2, 0) / arr.length);
      const vMean = mean(volts);
      const vStd = std(volts, vMean);

      result.push({
        date: new Date(day),
        v_mean: vMean,
        v_sigma_low: vMean - 2 * vStd,
        v_sigma_high: vMean + 2 * vStd,
        t_batt: tBatt.length > 0 ? mean(tBatt) : NaN,
        t_panel: tPanel.length > 0 ? mean(tPanel) : NaN,
      });
    }
    return result;
  });
</script>

<div class="flex flex-col gap-6">
  <!-- Top: Voltage panel -->
  <div>
    <p class="mb-2 text-xs font-semibold uppercase tracking-wider text-ink-3">
      Battery Voltage (V) — Daily Average with ±2σ Band
    </p>
    {#if dailyStats().length > 0}
        <Plot
          height={200}
          x={{ type: 'utc', label: false }}
          y={{ label: false, grid: true, nice: true }}
          marginTop={COMPACT_MARGIN.top + 8}
          marginRight={COMPACT_MARGIN.right + 8}
          marginBottom={COMPACT_MARGIN.bottom}
          marginLeft={COMPACT_MARGIN.left + 10}
        >
          <!-- ±2σ band -->
          <AreaY data={dailyStats()} x="date" y1="v_sigma_low" y2="v_sigma_high"
                 fill="#4361ee" fillOpacity={0.16} />

          <!-- Mean and ±2σ lines -->
          <Line data={dailyStats()} x="date" y="v_mean"
                stroke="#4361ee" strokeWidth={2} />
          <Line data={dailyStats()} x="date" y="v_sigma_low"
                stroke="#4361ee" strokeWidth={1.2}
                strokeDasharray="5 4" strokeOpacity={0.55} />
          <Line data={dailyStats()} x="date" y="v_sigma_high"
                stroke="#4361ee" strokeWidth={1.2}
                strokeDasharray="5 4" strokeOpacity={0.55} />
        </Plot>
        <div class="mt-2 flex items-center justify-center gap-5 text-xs text-ink-3">
          <span class="flex items-center gap-1.5">
            <span class="inline-block h-0.5 w-4 rounded" style="background: #4361ee"></span>
            Daily Mean
          </span>
          <span class="flex items-center gap-1.5">
            <span class="inline-block h-px w-4 border-t border-dashed" style="border-color: #4361ee; opacity: 0.65"></span>
            ±2σ
          </span>
        </div>
    {:else}
      <div class="flex h-48 items-center justify-center text-xs text-ink-3">Insufficient data</div>
    {/if}
  </div>

  <!-- Bottom: Temperature overlay -->
  <div>
    <p class="mb-2 text-xs font-semibold uppercase tracking-wider text-ink-3">
      Temperature (°C) — Daily Averages
    </p>
    {#if dailyStats().length > 0}
      {@const tempData = dailyStats().filter(d => !isNaN(d.t_batt) || !isNaN(d.t_panel))}
        <Plot
          height={200}
          x={{ type: 'utc', label: false }}
          y={{ label: false, grid: true, nice: true }}
          marginTop={COMPACT_MARGIN.top + 8}
          marginRight={COMPACT_MARGIN.right + 8}
          marginBottom={COMPACT_MARGIN.bottom + 4}
          marginLeft={COMPACT_MARGIN.left + 10}
        >
        <!-- Battery temp -->
        <Line data={tempData.filter(d => !isNaN(d.t_batt))} x="date" y="t_batt"
              stroke="#e64848" strokeWidth={1.8} />

        <!-- Panel Z temp -->
        <Line data={tempData.filter(d => !isNaN(d.t_panel))} x="date" y="t_panel"
              stroke="#2ec4b6" strokeWidth={1.8} />
      </Plot>
      <div class="mt-2 flex items-center justify-center gap-6 text-xs text-ink-3">
        <span class="flex items-center gap-1.5">
          <span class="inline-block h-0.5 w-4 rounded" style="background: #e64848"></span>
          Daily Avg Batt Temp A
        </span>
        <span class="flex items-center gap-1.5">
          <span class="inline-block h-0.5 w-4 rounded" style="background: #2ec4b6"></span>
          Daily Avg Panel Z Temp
        </span>
      </div>
    {:else}
      <div class="flex h-48 items-center justify-center text-xs text-ink-3">Insufficient data</div>
    {/if}
  </div>
</div>
