<script lang="ts">
  /**
   * Inter-Pass Time Gap Distribution
   * Shows the massive blackout periods between line-of-sight passes.
   */
  import { Plot, RectY, RuleX } from 'svelteplot';
  import { binX } from 'svelteplot/transforms';
  import { BRAND } from '$lib/chart-theme';

  let { timestamps = [] } = $props<{ timestamps: string[] }>();

  // Calculate gaps between consecutive frames in hours, filtering only gaps > 1 hour
  let gaps = $derived(() => {
    if (!timestamps || timestamps.length < 2) return [];
    
    // Ensure chronological order
    const sorted = [...timestamps].sort((a, b) => new Date(a).getTime() - new Date(b).getTime());
    
    const gapList = [];
    for (let i = 1; i < sorted.length; i++) {
      const prev = new Date(sorted[i - 1]).getTime();
      const curr = new Date(sorted[i]).getTime();
      const diffHours = (curr - prev) / 1000 / 3600;
      
      // If the gap is greater than 1 hour, we consider it a new pass
      if (diffHours > 1) {
        gapList.push({ gap: diffHours, count: 1 });
      }
    }
    return gapList;
  });

  let median = $derived(() => {
    const data = gaps();
    if (data.length === 0) return 0;
    const sorted = [...data].map(d => d.gap).sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  });

  // Binning transform for the histogram
  let binned = $derived(
    gaps().length > 0
      ? binX({ data: gaps(), x: 'gap' }, { y: 'count', thresholds: 20 })
      : { data: [] }
  );
</script>

<div class="w-full">
  {#if gaps().length > 0}
    <Plot height={320}
      x={{ label: 'Hours Between Passes', nice: true }}
      y={{ label: 'Count', grid: true }}
      marginTop={12} marginRight={12} marginBottom={40} marginLeft={44}>

      <RectY {...binned}
             fill="#94a3b8" fillOpacity={0.6}
             stroke="var(--color-panel)" strokeWidth={1} />

      <!-- Median reference line -->
      <RuleX data={[median()]}
             stroke="#94a3b8" strokeWidth={2}
             strokeDasharray="6 3" strokeOpacity={0.7} />
    </Plot>
    <div class="mt-2 flex items-center justify-center gap-4 text-xs text-ink-3">
      <span class="flex items-center gap-1.5">
        <span class="inline-block h-px w-4 border-t-2 border-dashed" style="border-color: #94a3b8; opacity: 0.7"></span>
        Median: {median().toFixed(1)} hrs
      </span>
      <span>{gaps().length.toLocaleString()} inter-pass gaps</span>
    </div>
  {:else}
    <div class="flex h-[320px] items-center justify-center text-xs text-ink-3">No inter-pass timing data available</div>
  {/if}
</div>
