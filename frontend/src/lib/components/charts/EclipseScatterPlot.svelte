<script lang="ts">
  /**
   * Eclipse Scatter — Physics Verification: Day/Night Operational States
   * Reproduces docs/figures/eclipse_scatter.png
   *
   * X: Solar Panel Z Temperature (°C) — orbit phase indicator
   * Y: Battery Current (A) — charging vs discharging
   * Color: Battery Voltage (V) — continuous colorbar
   * Reference: Dashed Eclipse Boundary at ~15°C
   */
  import { Plot, Dot, RuleX, RuleY } from 'svelteplot';
  import { COMPACT_MARGIN } from '$lib/chart-theme';

  type Frame = {
    temp_panel_z: number;
    batt_current: number;
    batt_voltage: number;
  };

  let { frames = [] } = $props<{ frames: Frame[] }>();

  // Eclipse boundary temperature threshold (~15°C)
  const ECLIPSE_BOUNDARY = 15;

  let plotData = $derived(
    frames.filter(
      (frame: Frame) =>
        Number.isFinite(frame.temp_panel_z) &&
        Number.isFinite(frame.batt_current) &&
        Number.isFinite(frame.batt_voltage) &&
        frame.batt_voltage <= 5 &&
        Math.abs(frame.batt_current) <= 1
    )
  );
</script>

<div class="flex flex-col gap-4">
  <div class="h-full w-full">
    {#if plotData.length > 0}
      <Plot
        height={380}
        x={{ label: 'Solar Panel Z Temperature (°C)', grid: true, nice: true }}
        y={{ label: 'Battery Current (A)', grid: true, domain: [-0.65, 0.55] }}
        color={{ type: 'linear', scheme: 'magma', label: 'Battery Voltage (V)', legend: true }}
        marginTop={COMPACT_MARGIN.top + 6}
        marginRight={64}
        marginBottom={COMPACT_MARGIN.bottom + 12}
        marginLeft={COMPACT_MARGIN.left + 12}
      >
        <!-- Eclipse boundary -->
        <RuleX data={[ECLIPSE_BOUNDARY]}
               stroke="cyan" strokeWidth={2}
               strokeDasharray="8 4" strokeOpacity={0.7} />

        <RuleY data={[0]}
               stroke="var(--color-border)" strokeWidth={1.2}
               strokeOpacity={0.9} />

        <!-- Data points colored by voltage -->
        <Dot data={plotData} x="temp_panel_z" y="batt_current" fill="batt_voltage"
             r={3.5} fillOpacity={0.82}
             stroke="var(--color-ink)" strokeWidth={0.4} strokeOpacity={0.25} />
      </Plot>
    {:else}
      <div class="flex h-[380px] items-center justify-center text-xs text-ink-3">
        Insufficient clean day/night telemetry
      </div>
    {/if}
  </div>

  <div class="flex items-center justify-center gap-6 text-[0.65rem] uppercase tracking-wider text-ink-3">
    <span>← Eclipse (Discharging)</span>
    <span class="mx-2 h-px w-8 border-t-2 border-dashed" style="border-color: cyan; opacity: 0.5"></span>
    <span>Sunlight (Charging) →</span>
  </div>
</div>
