<script lang="ts">
  import ResponsivePlot from './ResponsivePlot.svelte';
  /**
   * Inter-Pass Time Gap Distribution
   * Shows the massive blackout periods between line-of-sight passes.
   */
  import { RectY, RuleX } from 'svelteplot';
  import { binX } from 'svelteplot/transforms';
  import { SERIES_BASELINE } from '$lib/chart-theme';
  import { median } from '$lib/data/statistics';

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

  let medianValue = $derived(() => {
    return median(gaps().map(d => d.gap));
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
    <ResponsivePlot height={320}
      x={{ label: 'Hours Between Passes', labelAnchor: 'center', nice: true }}
      y={{ label: 'Count', grid: true }}
      marginTop={12} marginRight={12} marginBottom={40} marginLeft={44}>

      <RectY {...binned}
             fill={SERIES_BASELINE} fillOpacity={0.6}
             stroke="var(--color-panel)" strokeWidth={1} />

      <!-- Median reference line -->
      <RuleX data={[medianValue()]}
             stroke={SERIES_BASELINE} strokeWidth={2}
             strokeDasharray="6 3" strokeOpacity={0.7} />
    </ResponsivePlot>
    <div class="mt-2 flex items-center justify-center gap-4 text-xs text-ink-3">
      <span class="flex items-center gap-1.5">
        <span class="inline-block h-px w-4 border-t-2 border-dashed" style="border-color: {SERIES_BASELINE}; opacity: 0.7"></span>
        Median: {medianValue().toFixed(1)} hrs
      </span>
      <span>{gaps().length.toLocaleString()} inter-pass gaps</span>
    </div>
  {:else}
    <div class="flex h-[320px] items-center justify-center text-xs text-ink-3">No inter-pass timing data available</div>
  {/if}
</div>
