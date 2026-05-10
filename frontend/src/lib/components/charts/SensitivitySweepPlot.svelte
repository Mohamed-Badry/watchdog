<script lang="ts">
  /**
   * Sensitivity Sweep — VAE vs Z-Score Baseline crossover
   * Reproduces docs/figures/sensitivity_sweep.png
   * Static benchmark data from Python analysis.
   */
  import { Plot, Line, Dot, RuleY } from 'svelteplot';

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
    <Plot height={280}
      x={{ label: 'Thermal Runaway Δ (°C)', nice: true }}
      y={{ label: 'AUROC', domain: [0.4, 1.05], grid: true }}
      marginTop={12} marginRight={12} marginBottom={40} marginLeft={48}>
      <RuleY data={[0.5]} y={d=>d} stroke="#94a3b8" strokeDasharray="4 3" strokeOpacity={0.3} />
      <Line data={thermal.vae} x="x" y="y" stroke="#e64848" strokeWidth={2} />
      <Dot data={thermal.vae} x="x" y="y" fill="#e64848" r={4} />
      <Line data={thermal.zs} x="x" y="y" stroke="#092e4b" strokeWidth={2} strokeDasharray="8 4" />
      <Dot data={thermal.zs} x="x" y="y" fill="#092e4b" r={4} symbol="square" />
    </Plot>
  </div>

  <!-- Panel Failure -->
  <div>
    <p class="mb-2 text-center text-xs font-semibold text-ink-2">Panel Failure Detection</p>
    <Plot height={280}
      x={{ label: 'Forced Current During Sunlight (A)', nice: true }}
      y={{ label: 'AUROC', domain: [0.4, 1.05], grid: true }}
      marginTop={12} marginRight={12} marginBottom={40} marginLeft={48}>
      <RuleY data={[0.5]} y={d=>d} stroke="#94a3b8" strokeDasharray="4 3" strokeOpacity={0.3} />
      <Line data={panel.vae} x="x" y="y" stroke="#e64848" strokeWidth={2} />
      <Dot data={panel.vae} x="x" y="y" fill="#e64848" r={4} />
      <Line data={panel.zs} x="x" y="y" stroke="#092e4b" strokeWidth={2} strokeDasharray="8 4" />
      <Dot data={panel.zs} x="x" y="y" fill="#092e4b" r={4} symbol="square" />
    </Plot>
  </div>
</div>

<div class="mt-3 flex items-center justify-center gap-6 text-[0.65rem] text-ink-2">
  <span class="flex items-center gap-1.5">
    <span class="inline-block h-0.5 w-5 rounded" style="background: #e64848"></span>
    VAE
  </span>
  <span class="flex items-center gap-1.5">
    <span class="inline-block h-0.5 w-5 rounded border-t-2 border-dashed" style="border-color: #092e4b"></span>
    Z-Score Baseline
  </span>
  <span class="flex items-center gap-1.5">
    <span class="inline-block h-px w-5 border-t border-dashed" style="border-color: #94a3b8; opacity: 0.4"></span>
    Random Chance
  </span>
</div>
