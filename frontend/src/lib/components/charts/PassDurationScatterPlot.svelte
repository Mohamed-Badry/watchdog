<script lang="ts">
  import { Plot, Dot } from 'svelteplot';
  import { SERIES_CURRENT as BLUE } from '$lib/chart-theme';

  let { data = [] } = $props<{
    data: any[];
  }>();
</script>

<Plot height={200}
  x={{ label: 'Pass Duration (seconds)', grid: true, nice: true }}
  y={{ label: 'Frames Decoded', grid: true, nice: true }}
  marginTop={30} marginRight={20} marginBottom={40} marginLeft={60}>
  <Dot data={data} x="duration_sec" y="frame_count" fill={BLUE} r={3} fillOpacity={0.6} />
</Plot>
