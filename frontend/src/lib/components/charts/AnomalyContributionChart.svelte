<script lang="ts">
  import { Plot, BarX } from 'svelteplot';

  let { contributions = {} } = $props<{ contributions: Record<string, number> }>();

  let data = $derived(
    Object.entries(contributions)
      .map(([feature, error]) => ({ feature, error: Number(error) }))
      .sort((a, b) => b.error - a.error)
  );

  let domainY = $derived(data.map(d => d.feature));
</script>

<div class="w-full">
  {#if data.length > 0}
    <Plot height={240}
      x={{ label: 'Reconstruction Error (Delta)', grid: true }}
      y={{ domain: domainY, label: false }}
      marginLeft={120} marginRight={20} marginTop={10} marginBottom={40}>
      <BarX {data} x="error" y="feature" fill="#B12142" fillOpacity={0.85} />
    </Plot>
  {:else}
    <div class="flex h-[240px] items-center justify-center text-sm text-ink-3">No contribution data available.</div>
  {/if}
</div>
