<script lang="ts">
  import ResponsivePlot from './ResponsivePlot.svelte';
  /**
   * Sensitivity Sweep — VAE vs Z-Score Baseline crossover
   * Reproduces docs/figures/sensitivity_sweep.png
   * Static benchmark data from Python analysis.
   */
  import { Line, Dot, RuleY } from 'svelteplot';
  import { SERIES_TEMP_BATT, SERIES_BASELINE, SERIES_ZSCORE } from '$lib/chart-theme';

  const thermal = {
    vae: [{x:0.5,y:.45},{x:1,y:.49},{x:2,y:.58},{x:3,y:.71},{x:5,y:.81},{x:7,y:.93},{x:10,y:1},{x:15,y:1},{x:20,y:1},{x:30,y:1},{x:45,y:1}],
    zs:  [{x:0.5,y:.43},{x:1,y:.41},{x:2,y:.42},{x:3,y:.49},{x:5,y:.60},{x:7,y:.88},{x:10,y:.93},{x:15,y:1},{x:20,y:1},{x:30,y:1},{x:45,y:1}],
  };
  const panel = {
    vae: [{x:0,y:.70},{x:-.05,y:.75},{x:-.1,y:.76},{x:-.15,y:.79},{x:-.2,y:.84},{x:-.3,y:.89},{x:-.4,y:.97},{x:-.5,y:.99},{x:-.6,y:1},{x:-.8,y:1}],
    zs:  [{x:0,y:.55},{x:-.05,y:.55},{x:-.1,y:.55},{x:-.15,y:.55},{x:-.2,y:.55},{x:-.3,y:.60},{x:-.4,y:.75},{x:-.5,y:.88},{x:-.6,y:.95},{x:-.8,y:1}],
  };
</script>

<div class="grid gap-6 md:grid-cols-2">
  <!-- Thermal Runaway -->
  <div>
    <p class="mb-2 text-center text-xs font-semibold text-ink-2">Thermal Runaway Detection</p>
    <ResponsivePlot height={280}
      x={{ label: 'Thermal Runaway Δ (°C)', labelAnchor: 'center', nice: true }}
      y={{ label: false, domain: [0.4, 1.05], grid: true }}
      marginTop={16} marginRight={12} marginBottom={44} marginLeft={50}>
      <RuleY data={[0.5]} stroke={SERIES_BASELINE} strokeDasharray="4 3" strokeOpacity={0.3} />
      <Line data={thermal.vae} x="x" y="y" stroke={SERIES_TEMP_BATT} strokeWidth={2} />
      <Dot data={thermal.vae} x="x" y="y" fill={SERIES_TEMP_BATT} r={4} />
      <Line data={thermal.zs} x="x" y="y" stroke={SERIES_ZSCORE} strokeWidth={2} strokeDasharray="8 4" />
      <Dot data={thermal.zs} x="x" y="y" fill={SERIES_ZSCORE} r={4} symbol="square" />
    </ResponsivePlot>
  </div>

  <!-- Panel Failure -->
  <div>
    <p class="mb-2 text-center text-xs font-semibold text-ink-2">Panel Failure Detection</p>
    <ResponsivePlot height={280}
      x={{ label: 'Forced Current During Sunlight (A)', labelAnchor: 'center', nice: true }}
      y={{ label: false, domain: [0.4, 1.05], grid: true }}
      marginTop={16} marginRight={12} marginBottom={44} marginLeft={50}>
      <RuleY data={[0.5]} stroke={SERIES_BASELINE} strokeDasharray="4 3" strokeOpacity={0.3} />
      <Line data={panel.vae} x="x" y="y" stroke={SERIES_TEMP_BATT} strokeWidth={2} />
      <Dot data={panel.vae} x="x" y="y" fill={SERIES_TEMP_BATT} r={4} />
      <Line data={panel.zs} x="x" y="y" stroke={SERIES_ZSCORE} strokeWidth={2} strokeDasharray="8 4" />
      <Dot data={panel.zs} x="x" y="y" fill={SERIES_ZSCORE} r={4} symbol="square" />
    </ResponsivePlot>
  </div>
</div>

<div class="mt-3 flex items-center justify-center gap-6 text-[0.65rem] text-ink-2">
  <span class="flex items-center gap-1.5">
    <span class="inline-block h-0.5 w-5 rounded" style="background: {SERIES_TEMP_BATT}"></span>
    VAE
  </span>
  <span class="flex items-center gap-1.5">
    <span class="inline-block h-0.5 w-5 rounded border-t-2 border-dashed" style="border-color: {SERIES_ZSCORE}"></span>
    Z-Score Baseline
  </span>
  <span class="flex items-center gap-1.5">
    <span class="inline-block h-px w-5 border-t border-dashed" style="border-color: {SERIES_BASELINE}; opacity: 0.4"></span>
    Random Chance
  </span>
</div>
