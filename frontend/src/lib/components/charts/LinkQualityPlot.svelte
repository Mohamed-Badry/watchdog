<script lang="ts">
  import { Plot, AreaY, Line } from 'svelteplot';
  import { SERIES_AMBER as AMBER } from '$lib/chart-theme';

  let { data = [] } = $props<{
    data: any[];
  }>();

  let plotData = $derived(data.map((d: any) => ({
    ...d,
    dateObj: new Date(d.date)
  })));
</script>

<Plot height={200}
  x={{ type: 'time', label: false, grid: true }}
  y={{ label: 'Dropped Suspects', grid: true, nice: true }}
  marginTop={30} marginRight={20} marginBottom={40} marginLeft={60}>
  <AreaY data={plotData} x="dateObj" y="dropped_suspects" fill={AMBER} fillOpacity={0.5} />
  <Line data={plotData} x="dateObj" y="dropped_suspects" stroke={AMBER} strokeWidth={2} />
</Plot>
