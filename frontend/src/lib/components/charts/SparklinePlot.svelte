<script lang="ts">
  import { Plot, AreaY, Dot } from 'svelteplot';

  type FrameData = { frame_count: number; anomaly_count: number };

  let { data = [], height = 48 } = $props<{
    data: FrameData[];
    height?: number;
  }>();
  
  let width = $state(0);

  let indexedData = $derived(
    data.map((d: FrameData, i: number) => ({ ...d, idx: i }))
  );
  let anomalyData = $derived(
    indexedData.filter((d: FrameData & { idx: number }) => d.anomaly_count > 0)
  );
</script>

<div class="w-full" bind:clientWidth={width}>
  {#if width > 0}
    <Plot {width} {height}
          marginTop={4} marginRight={4} marginBottom={4} marginLeft={4}
          x={{ axis: false }} y={{ axis: false }}>
    <!-- Gradient area fill -->
    <AreaY data={indexedData} x="idx" y="frame_count"
           fill="var(--color-brand)" fillOpacity={0.12}
           stroke="var(--color-brand)" strokeWidth={1.5} curve="natural" />
    <!-- Anomaly accent dots -->
    {#if anomalyData.length > 0}
      <Dot data={anomalyData} x="idx" y="frame_count"
           fill="var(--color-brand)" r={3}
           stroke="var(--color-brand)" strokeWidth={1} strokeOpacity={0.5} />
    {/if}
    </Plot>
  {/if}
</div>
