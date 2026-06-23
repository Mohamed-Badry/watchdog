<script lang="ts">
  import ResponsivePlot from './ResponsivePlot.svelte';
  /**
   * Multi-model ROC comparison with AUROC labels.
   * Uses static benchmark data from the Python analysis.
   */
  import { Line } from 'svelteplot';
  import { SERIES_BASELINE } from '$lib/chart-theme';
  import { ROC_MODELS, ROC_DIAG } from '$lib/data/benchmarks';
</script>

<div class="h-full w-full mx-auto">
  <ResponsivePlot height={450}
    x={{ domain: [0, 1], label: 'False Positive Rate', labelAnchor: 'center', grid: true }}
    y={{ domain: [0, 1], label: 'True Positive Rate (Recall)', grid: true }}
    marginTop={28} marginRight={28} marginBottom={44} marginLeft={52}>
    <Line data={ROC_DIAG} x="fpr" y="tpr" stroke={SERIES_BASELINE} strokeWidth={1} strokeDasharray="6 4" strokeOpacity={0.3} />
    {#each ROC_MODELS as m}
      <Line data={m.pts} x="fpr" y="tpr" stroke={m.color} strokeWidth={2.5} />
    {/each}
  </ResponsivePlot>
</div>
<div class="mt-4 flex flex-wrap items-center justify-center gap-x-5 gap-y-2 text-xs text-ink-2">
  <span class="flex items-center gap-1.5">
    <span class="inline-block h-px w-5 border-t border-dashed" style="border-color: {SERIES_BASELINE}; opacity: 0.5"></span>
    Random baseline
  </span>
  {#each ROC_MODELS as m}
    <span class="flex items-center gap-1.5">
      <span class="inline-block h-0.5 w-5 rounded" style="background: {m.color}"></span>
      {m.name} <span class="font-mono text-ink-3">(AUROC={m.auroc.toFixed(3)})</span>
    </span>
  {/each}
</div>
