<script lang="ts">
  import { Plot, Dot, Line, RuleY } from 'svelteplot';
  import { BRAND, MUTED, COMPACT_MARGIN } from '$lib/chart-theme';

  let { frames, threshold } = $props<{
    frames: { timestamp: string; anomaly_score: number; is_anomaly: boolean }[];
    threshold: number;
  }>();

  let plotData = $derived(
    (frames ?? []).map((f: { timestamp: string; anomaly_score: number; is_anomaly: boolean }) => ({
      ...f,
      date: new Date(f.timestamp),
    }))
  );
</script>

<div class="h-full w-full">
  <Plot
    height={220}
    x={{ type: 'utc', label: false }}
    y={{ label: 'Anomaly Score', grid: true }}
    color={{ domain: [false, true], range: [MUTED, BRAND] }}
    marginTop={COMPACT_MARGIN.top}
    marginRight={COMPACT_MARGIN.right}
    marginBottom={COMPACT_MARGIN.bottom}
    marginLeft={COMPACT_MARGIN.left}
  >
    <!-- Threshold reference line with label context -->
    <RuleY data={[threshold]} y={d => d}
           stroke={BRAND} strokeDasharray="6 3" strokeOpacity={0.5} />

    <!-- Connecting line -->
    <Line data={plotData} x="date" y="anomaly_score"
          stroke="var(--color-ink-3)" strokeWidth={1} strokeOpacity={0.3}
          curve="monotone-x" />

    <!-- Score dots colored by anomaly status -->
    <Dot data={plotData} x="date" y="anomaly_score" fill="is_anomaly"
         r={d => d.is_anomaly ? 5.5 : 3}
         stroke={d => d.is_anomaly ? BRAND : 'none'}
         strokeWidth={d => d.is_anomaly ? 1.5 : 0}
         strokeOpacity={0.4} />
  </Plot>
</div>

<div class="mt-2 flex items-center justify-center gap-5 text-xs text-ink-3">
  <span class="flex items-center gap-1.5">
    <span class="inline-block h-2 w-2 rounded-full" style="background: {MUTED}"></span>
    Normal
  </span>
  <span class="flex items-center gap-1.5">
    <span class="inline-block h-2 w-2 rounded-full bg-brand"></span>
    Anomaly
  </span>
  <span class="flex items-center gap-1.5">
    <span class="inline-block h-px w-4 border-t border-dashed border-brand opacity-50"></span>
    Threshold ({threshold?.toFixed(2)})
  </span>
</div>
