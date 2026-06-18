<script lang="ts">
  import ResponsivePlot from './ResponsivePlot.svelte';
  import { BarX, RuleX } from 'svelteplot';
  import { BRAND, COMPACT_MARGIN } from '$lib/chart-theme';

  type Pass = { satellite: string; aos: Date; los: Date; max_elevation: number };

  let { passes, now, endTime } = $props<{
    passes: Pass[];
    now: Date;
    endTime: Date;
  }>();

  let satelliteRows = $derived.by((): string[] => {
    const rows = new Set<string>();
    for (const pass of passes ?? []) {
      rows.add(pass.satellite);
    }
    return [...rows];
  });
  let chartHeight = $derived(Math.max(170, satelliteRows.length * 38 + 76));
</script>

<div class="h-full w-full">
  <ResponsivePlot
    height={chartHeight}
    x={{ type: 'utc', domain: [now, endTime], label: false }}
    y={{ type: 'band', domain: satelliteRows, label: false }}
    marginTop={COMPACT_MARGIN.top + 4}
    marginRight={COMPACT_MARGIN.right}
    marginBottom={COMPACT_MARGIN.bottom}
    marginLeft={60}
  >
    <!-- "Now" reference line -->
    <RuleX data={[now]}
           stroke="var(--color-ok)" strokeWidth={2}
           strokeDasharray="4 2" strokeOpacity={0.7} />

    <!-- Pass duration bars -->
    <BarX data={passes ?? []} x1="aos" x2="los" y="satellite"
          fill={BRAND} fillOpacity={0.8}
          stroke={BRAND} strokeWidth={0.5} strokeOpacity={0.3}
          title={(d: Pass) => `${d.satellite}: ${d.max_elevation}° max elev`} />
  </ResponsivePlot>
</div>

<div class="mb-5 flex items-center justify-center gap-5 text-xs text-ink-3">
  <span class="flex items-center gap-1.5">
    <span class="inline-block h-px w-4 border-t-2 border-dashed" style="border-color: var(--color-ok)"></span>
    Current Time
  </span>
  <span class="flex items-center gap-1.5">
    <span class="inline-block h-2.5 w-4 rounded-sm bg-brand opacity-80"></span>
    Pass Window
  </span>
</div>
