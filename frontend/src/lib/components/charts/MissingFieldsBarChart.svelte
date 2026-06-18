<script lang="ts">
  import ResponsivePlot from './ResponsivePlot.svelte';
  import { BarX } from 'svelteplot';
  import { BRAND } from '$lib/chart-theme';

  let { data = [] } = $props<{
    data: any[];
  }>();
</script>

<ResponsivePlot height={280}
  x={{ label: 'Count', labelAnchor: 'center', grid: true, nice: true }}
  y={{ type: 'band', label: false, domain: data.map((d: any) => d.field) }}
  marginLeft={80} marginRight={20} marginTop={10} marginBottom={40}>
  <BarX data={data} x="count" y="field" fill={BRAND} fillOpacity={0.8} />
</ResponsivePlot>
