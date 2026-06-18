<script lang="ts">
  import ResponsivePlot from './ResponsivePlot.svelte';
  import { AreaY, Line } from 'svelteplot';
  import { BRAND } from '$lib/chart-theme';

  let { data = [] } = $props<{
    data: any[];
  }>();

  let plotData = $derived(data.map((d: any) => ({
    ...d,
    dateObj: new Date(d.date)
  })));
</script>

<ResponsivePlot height={240}
  x={{ type: 'time', label: false, grid: true }}
  y={{ label: 'Frames Received', grid: true, nice: true }}
  marginTop={30} marginRight={20} marginBottom={40} marginLeft={60}>
  <AreaY data={plotData} x="dateObj" y="frame_count" fill={BRAND} fillOpacity={0.2} />
  <Line data={plotData} x="dateObj" y="frame_count" stroke={BRAND} strokeWidth={2} />
</ResponsivePlot>
