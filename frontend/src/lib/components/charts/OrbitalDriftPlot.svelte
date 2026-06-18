<script lang="ts">
  import ResponsivePlot from './ResponsivePlot.svelte';
  import { AreaY, Line } from 'svelteplot';
  import { SERIES_AMBER as AMBER } from '$lib/chart-theme';

  let { data = [] } = $props<{
    data: any[];
  }>();

  let plotData = $derived(data
    .map((d: any) => ({ ...d, dateObj: new Date(d.date) }))
    .filter((d: any) => d.temp_panel_z_mean !== null)
  );
</script>

<ResponsivePlot height={340}
  x={{ type: 'time', label: false, grid: true }}
  y={{ label: 'Temperature (°C)', grid: true, nice: true }}
  marginTop={30} marginRight={20} marginBottom={40} marginLeft={60}>
  <AreaY data={plotData} 
         x="dateObj" 
         y1={(d: any) => d.temp_panel_z_mean - d.temp_panel_z_std} 
         y2={(d: any) => d.temp_panel_z_mean + d.temp_panel_z_std} 
         fill={AMBER} fillOpacity={0.15} />
  <Line data={plotData} 
        x="dateObj" y="temp_panel_z_mean" stroke={AMBER} strokeWidth={2} />
</ResponsivePlot>
