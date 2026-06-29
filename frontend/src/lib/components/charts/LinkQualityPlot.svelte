<script lang="ts">
  import ResponsivePlot from './ResponsivePlot.svelte';
  import { AreaY, Line } from 'svelteplot';
  import { SERIES_AMBER as AMBER } from '$lib/chart-theme';

  let { data = [], height = 200 } = $props<{
    data: any[];
    height?: number;
  }>();

  let plotData = $derived(data.map((d: any) => ({
    ...d,
    dateObj: new Date(d.date)
  })));
</script>

<ResponsivePlot {height}
  x={{ type: 'time', label: false, grid: true }}
  y={{ label: 'Dropped Suspects', grid: true, nice: true }}
  marginTop={30} marginRight={20} marginBottom={40} marginLeft={60}>
  <AreaY data={plotData} x="dateObj" y="dropped_suspects" fill={AMBER} fillOpacity={0.5} />
  <Line data={plotData} x="dateObj" y="dropped_suspects" stroke={AMBER} strokeWidth={2} />
</ResponsivePlot>
