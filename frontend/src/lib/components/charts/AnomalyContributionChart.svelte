<script lang="ts">
  import { Plot, BarX } from 'svelteplot';

  let { actual = {}, expected = {} } = $props<{ actual: Record<string, number>, expected: Record<string, number> }>();

  let data = $derived(() => {
     const pts = [];
     for (const f of Object.keys(expected)) {
        const delta = Math.abs(Number(actual[f]) - Number(expected[f]));
        pts.push({ feature: f, delta });
     }
     return pts.sort((a, b) => b.delta - a.delta);
  });

  let domainY = $derived(data().map(d => d.feature));
</script>

<div class="w-full">
  {#if Object.keys(expected).length > 0}
    <Plot height={200}
      x={{ label: 'Delta (Absolute Error)', grid: true }}
      y={{ domain: domainY, label: false }}
      marginLeft={100} marginRight={20} marginTop={10} marginBottom={40}>
      <BarX data={data()} x="delta" y="feature" fill="#B12142" fillOpacity={0.85} />
    </Plot>
  {:else}
    <div class="flex h-[200px] items-center justify-center text-sm text-ink-3">No data available.</div>
  {/if}
</div>
