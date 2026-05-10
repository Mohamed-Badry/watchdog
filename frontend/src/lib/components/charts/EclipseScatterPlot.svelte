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
  import { Plot, Dot, RuleX, Text, ColorLegend } from 'svelteplot';
  import { COMPACT_MARGIN } from '$lib/chart-theme';

  type Frame = {
    temp_panel_z: number;
    batt_current: number;
    batt_voltage: number;
  };

  let { frames = [] } = $props<{ frames: Frame[] }>();

  // Eclipse boundary temperature threshold (~15°C)
  const ECLIPSE_BOUNDARY = 15;
</script>

<div class="flex flex-col gap-4">
  <div class="h-full w-full">
    <Plot
      height={380}
      x={{ label: 'Solar Panel Z Temperature (°C)', grid: true, nice: true }}
      y={{ label: 'Battery Current (A)', grid: true, nice: true }}
      color={{ type: 'linear', scheme: 'purd', label: 'Battery Voltage (V)', legend: true }}
      marginTop={COMPACT_MARGIN.top}
      marginRight={64}
      marginBottom={COMPACT_MARGIN.bottom + 10}
      marginLeft={COMPACT_MARGIN.left + 8}
    >
      <!-- Eclipse boundary -->
      <RuleX data={[ECLIPSE_BOUNDARY]} x={d => d}
             stroke="cyan" strokeWidth={2}
             strokeDasharray="8 4" strokeOpacity={0.6} />

      <!-- Data points colored by voltage -->
      <Dot data={frames} x="temp_panel_z" y="batt_current" fill="batt_voltage"
           r={3} fillOpacity={0.65}
           stroke="none" />

      <!-- Eclipse label -->
      <Text data={[{ x: ECLIPSE_BOUNDARY - 3, y: 0.4, label: 'Eclipse Boundary (~15°C)' }]}
            x="x" y="y" text="label"
            fill="cyan" fillOpacity={0.7}
            fontSize={9} fontWeight="600"
            textAnchor="end" />
    </Plot>
  </div>

  <div class="flex items-center justify-center gap-6 text-[0.65rem] uppercase tracking-wider text-ink-3">
    <span>← Eclipse (Discharging)</span>
    <span class="mx-2 h-px w-8 border-t-2 border-dashed" style="border-color: cyan; opacity: 0.5"></span>
    <span>Sunlight (Charging) →</span>
  </div>
</div>
