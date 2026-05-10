<script lang="ts">
  /**
   * Macro-Scale Health — Dual-panel: Voltage with confidence band + Temperature overlay
   * Reproduces docs/figures/timeseries_macro_7month.png
   *
   * Top: Daily average battery voltage with min/max band
   * Bottom: Daily average temp_batt_a (red) + temp_panel_z (green)
   */
  import { Plot, Line, AreaY, RuleY } from 'svelteplot';
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
      v_mean: number; v_min: number; v_max: number;
      t_batt: number; t_panel: number;
    }[] = [];

    for (const [day, dayFrames] of [...byDay.entries()].sort()) {
      const volts = dayFrames.map(f => f.batt_voltage).filter((v): v is number => v != null);
      const tBatt = dayFrames.map(f => f.temp_batt_a).filter((v): v is number => v != null);
      const tPanel = dayFrames.map(f => f.temp_panel_z).filter((v): v is number => v != null);

      if (volts.length === 0) continue;

      const mean = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;

      result.push({
        date: new Date(day),
        v_mean: mean(volts),
        v_min: Math.min(...volts),
        v_max: Math.max(...volts),
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
      Battery Voltage (V) — Daily Average with Min/Max Band
    </p>
    {#if dailyStats().length > 0}
      <Plot
        height={200}
        x={{ type: 'utc', label: false }}
        y={{ label: 'Voltage (V)', grid: true, nice: true }}
        marginTop={COMPACT_MARGIN.top}
        marginRight={COMPACT_MARGIN.right}
        marginBottom={COMPACT_MARGIN.bottom}
        marginLeft={COMPACT_MARGIN.left + 4}
      >
        <!-- Confidence band (min to max) -->
        <AreaY data={dailyStats()} x="date" y1="v_min" y2="v_max"
               fill="#4361ee" fillOpacity={0.15} />

        <!-- Mean line -->
        <Line data={dailyStats()} x="date" y="v_mean"
              stroke="#4361ee" strokeWidth={2} />
      </Plot>
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
        x={{ type: 'utc', label: 'Date' }}
        y={{ label: 'Temperature (°C)', grid: true, nice: true }}
        marginTop={COMPACT_MARGIN.top}
        marginRight={COMPACT_MARGIN.right}
        marginBottom={COMPACT_MARGIN.bottom}
        marginLeft={COMPACT_MARGIN.left + 4}
      >
        <!-- Battery temp -->
        <Line data={tempData.filter(d => !isNaN(d.t_batt))} x="date" y="t_batt"
              stroke="#e64848" strokeWidth={1.8} />

        <!-- Panel Z temp -->
        <Line data={tempData.filter(d => !isNaN(d.t_panel))} x="date" y="t_panel"
              stroke="#2ec4b6" strokeWidth={1.8} />
      </Plot>
      <div class="mt-2 flex items-center justify-center gap-6 text-[0.6rem] text-ink-3">
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
