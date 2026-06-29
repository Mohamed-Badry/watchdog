<script lang="ts">
  import ResponsivePlot from './ResponsivePlot.svelte';
  import { AreaY, Line } from 'svelteplot';
  import { SERIES_CURRENT as BLUE } from '$lib/chart-theme';

  let { data = [], height = 340 } = $props<{
    data: any[];
    height?: number;
  }>();

  let plotData = $derived(data
    .map((d: any) => ({ ...d, dateObj: new Date(d.date) }))
    .filter((d: any) => d.batt_voltage_mean !== null)
  );
</script>

<ResponsivePlot {height}
  x={{ type: 'time', label: false, grid: true }}
  y={{ label: 'Voltage (V)', grid: true, nice: true }}
  marginTop={30} marginRight={20} marginBottom={40} marginLeft={60}>
  <AreaY data={plotData} 
         x="dateObj" 
         y1={(d: any) => d.batt_voltage_mean - d.batt_voltage_std} 
         y2={(d: any) => d.batt_voltage_mean + d.batt_voltage_std} 
         fill={BLUE} fillOpacity={0.15} />
  <Line data={plotData} 
        x="dateObj" y="batt_voltage_mean" stroke={BLUE} strokeWidth={2} />
</ResponsivePlot>
