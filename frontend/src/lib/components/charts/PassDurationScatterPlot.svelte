<script lang="ts">
  import ResponsivePlot from './ResponsivePlot.svelte';
  import { Dot } from 'svelteplot';
  import { SERIES_CURRENT as BLUE } from '$lib/chart-theme';

  let { data = [] } = $props<{
    data: any[];
  }>();
</script>

<ResponsivePlot height={200}
  x={{ label: 'Pass Duration (seconds)', labelAnchor: 'center', grid: true, nice: true }}
  y={{ label: 'Frames Decoded', grid: true, nice: true }}
  marginTop={30} marginRight={20} marginBottom={40} marginLeft={60}>
  <Dot data={data} x="duration_sec" y="frame_count" fill={BLUE} r={3} fillOpacity={0.6} />
</ResponsivePlot>
