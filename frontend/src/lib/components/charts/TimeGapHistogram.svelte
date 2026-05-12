<script lang="ts">
  /**
   * Time Gap Distribution — Intra-pass frame cadence histogram
   * Reproduces docs/figures/time_gap_distribution.png
   */
  import { Plot, RectY, RuleX, Text } from 'svelteplot';
  import { binX } from 'svelteplot/transforms';

  let { timestamps = [] } = $props<{ timestamps: string[] }>();

  let gaps = $derived(() => {
    const sorted = timestamps
      .map((t: string) => new Date(t).getTime())
      .filter((t: number) => !isNaN(t))
      .sort((a: number, b: number) => a - b);

    const result: { gap: number }[] = [];
    for (let i = 1; i < sorted.length; i++) {
      const diffSec = (sorted[i] - sorted[i - 1]) / 1000;
      // Only intra-pass gaps (< 5 min)
      if (diffSec > 0 && diffSec < 300) {
        result.push({ gap: diffSec });
      }
    }
    return result;
  });

  let median = $derived(() => {
    const vals = gaps().map(g => g.gap).sort((a, b) => a - b);
    if (vals.length === 0) return 0;
    const mid = Math.floor(vals.length / 2);
    return vals.length % 2 ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2;
  });

  let binned = $derived(
    gaps().length > 0
      ? binX({ data: gaps(), x: 'gap' }, { y: 'count', thresholds: 30 })
      : { data: [] }
  );
</script>

<div class="h-full w-full">
  {#if gaps().length > 0}
    <Plot height={240}
      x={{ label: 'Seconds Between Frames', nice: true }}
      y={{ label: 'Count', grid: true }}
      marginTop={12} marginRight={12} marginBottom={40} marginLeft={44}>

      <RectY {...binned}
             fill="#9b59b6" fillOpacity={0.65}
             stroke="#9b59b6" strokeWidth={0.5} strokeOpacity={0.3} />

      <!-- Median reference line -->
      <RuleX data={[median()]} x={d => d}
             stroke="#e64848" strokeWidth={2}
             strokeDasharray="6 3" strokeOpacity={0.7} />
    </Plot>
    <div class="mt-2 flex items-center justify-center gap-4 text-[0.6rem] text-ink-3">
      <span class="flex items-center gap-1.5">
        <span class="inline-block h-px w-4 border-t-2 border-dashed" style="border-color: #e64848; opacity: 0.7"></span>
        Median: {median().toFixed(0)}s
      </span>
      <span>{gaps().length.toLocaleString()} intra-pass gaps</span>
    </div>
  {:else}
    <div class="flex h-60 items-center justify-center text-xs text-ink-3">No timing data available</div>
  {/if}
</div>
