<script lang="ts">
  import ResponsivePlot from './ResponsivePlot.svelte';
  /**
   * Feature Contribution — Per-feature reconstruction error by fault type
   * Reproduces docs/figures/ae_feature_contribution.png
   * Static benchmark data from Python analysis.
   */
  import { BarY } from 'svelteplot';
  import { 
    CONTRIBUTION_NORMAL, 
    CONTRIBUTION_FAULT, 
    CONTRIBUTION_NORMAL_COLOR, 
    CONTRIBUTION_FAULT_COLOR, 
    CONTRIBUTION_LABELS, 
    CONTRIBUTION_FAULTS 
  } from '$lib/data/benchmarks';

  const normalBarInset = { left: 3, right: 18 };
  const faultBarInset = { left: 18, right: 3 };
</script>

<div class="grid gap-6 md:grid-cols-3">
  {#each CONTRIBUTION_FAULTS as fault}
    <div>
      <p class="mb-2 text-center text-xs font-semibold text-ink-2">{fault.name}</p>
      <ResponsivePlot height={238}
        x={{ type: 'band', label: false, domain: CONTRIBUTION_LABELS, tickRotate: -45 }}
        y={{ label: false, grid: true, nice: true }}
        color={{ domain: [CONTRIBUTION_NORMAL, CONTRIBUTION_FAULT], scheme: [CONTRIBUTION_NORMAL_COLOR, CONTRIBUTION_FAULT_COLOR] }}
        marginTop={12} marginRight={10} marginBottom={54} marginLeft={44}>
        <BarY data={fault.data} x="feature" y1={0} y2="error" fill="type"
              fillOpacity={0.82}
              insetLeft={(d) => d.type === CONTRIBUTION_NORMAL ? normalBarInset.left : faultBarInset.left}
              insetRight={(d) => d.type === CONTRIBUTION_NORMAL ? normalBarInset.right : faultBarInset.right}
              borderRadius={{ topLeft: 2, topRight: 2 }} />
      </ResponsivePlot>
    </div>
  {/each}
</div>

<div class="mt-3 flex items-center justify-center gap-6 text-[0.65rem] text-ink-2">
  <span class="flex items-center gap-1.5">
    <span class="inline-block h-2.5 w-2.5 rounded-sm" style="background: {CONTRIBUTION_NORMAL_COLOR}; opacity: 0.82"></span>
    Normal Baseline
  </span>
  <span class="flex items-center gap-1.5">
    <span class="inline-block h-2.5 w-2.5 rounded-sm" style="background: {CONTRIBUTION_FAULT_COLOR}; opacity: 0.82"></span>
    Injected Fault
  </span>
</div>
