<script lang="ts">
  import ResponsivePlot from './ResponsivePlot.svelte';
  import { BarX } from 'svelteplot';
  import { BRAND } from '$lib/chart-theme';

  let { data = [], height = 280 } = $props<{
    data: any[];
    height?: number;
  }>();

  let formattedData = $derived(data.map((d: any) => ({
    ...d,
    field: d.field
      .replace('missing_', '')
      .replace('temp_obc', 'OBC Temperature')
      .replace('temp_batt_a', 'Battery A Temp')
      .replace('temp_batt_b', 'Battery B Temp')
      .replace('temp_panel_z', 'Panel Z Temp')
      .replace('batt_voltage', 'Battery Voltage')
      .replace('batt_current', 'Battery Current')
      .replace('power_consumption', 'Power Consumption')
      .replace('_', ' ')
      .replace(/\b\w/g, (c: string) => c.toUpperCase())
  })));
</script>

<ResponsivePlot {height}
  x={{ label: 'Count', labelAnchor: 'center', grid: true, nice: true }}
  y={{ type: 'band', label: false, domain: formattedData.map((d: any) => d.field) }}
  marginLeft={110} marginRight={20} marginTop={10} marginBottom={40}>
  <BarX data={formattedData} x="count" y="field" fill={BRAND} fillOpacity={0.8} />
</ResponsivePlot>
