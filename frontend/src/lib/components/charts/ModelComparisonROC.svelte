<script lang="ts">
  /**
   * Multi-model ROC comparison with AUROC labels.
   * Uses static benchmark data from the Python analysis.
   */
  import { Plot, Line } from 'svelteplot';

  const MODELS = [
    { name: 'Autoencoder', auroc: 0.844, color: '#2ec4b6',
      pts: [{fpr:0,tpr:0},{fpr:.005,tpr:.38},{fpr:.02,tpr:.46},{fpr:.05,tpr:.53},{fpr:.1,tpr:.59},{fpr:.15,tpr:.65},{fpr:.2,tpr:.71}] },
    { name: 'Elliptic Env', auroc: 0.869, color: '#94a3b8',
      pts: [{fpr:0,tpr:0},{fpr:.005,tpr:.57},{fpr:.02,tpr:.60},{fpr:.05,tpr:.63},{fpr:.1,tpr:.67},{fpr:.15,tpr:.73},{fpr:.2,tpr:.79}] },
    { name: 'One-Class SVM', auroc: 0.813, color: '#f59e0b',
      pts: [{fpr:0,tpr:0},{fpr:.005,tpr:.59},{fpr:.02,tpr:.61},{fpr:.05,tpr:.63},{fpr:.1,tpr:.65},{fpr:.15,tpr:.67},{fpr:.2,tpr:.69}] },
    { name: 'Isolation Forest', auroc: 0.809, color: '#3a86ff',
      pts: [{fpr:0,tpr:0},{fpr:.005,tpr:.05},{fpr:.02,tpr:.25},{fpr:.05,tpr:.37},{fpr:.1,tpr:.48},{fpr:.15,tpr:.58},{fpr:.2,tpr:.65}] },
  ];
  const diag = [{fpr:0,tpr:0},{fpr:.2,tpr:.2}];
</script>

<div class="h-full w-full">
  <Plot height={380}
    x={{ domain: [0, 0.2], label: 'False Positive Rate', grid: true }}
    y={{ domain: [0, 1], label: 'True Positive Rate (Recall)', grid: true }}
    marginTop={12} marginRight={16} marginBottom={44} marginLeft={52}>
    <Line data={diag} x="fpr" y="tpr" stroke="#94a3b8" strokeWidth={1} strokeDasharray="6 4" strokeOpacity={0.3} />
    {#each MODELS as m}
      <Line data={m.pts} x="fpr" y="tpr" stroke={m.color} strokeWidth={2.5} />
    {/each}
  </Plot>
</div>
<div class="mt-3 flex flex-wrap items-center justify-center gap-x-5 gap-y-2 text-[0.65rem] text-ink-2">
  {#each MODELS as m}
    <span class="flex items-center gap-1.5">
      <span class="inline-block h-0.5 w-5 rounded" style="background: {m.color}"></span>
      {m.name} <span class="font-mono text-ink-3">(AUROC={m.auroc.toFixed(3)})</span>
    </span>
  {/each}
</div>
