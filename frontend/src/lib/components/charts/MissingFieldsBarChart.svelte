<script lang="ts">
  import { Plot, BarX } from 'svelteplot';
  import { BRAND } from '$lib/chart-theme';

  let { data = [] } = $props<{
    data: any[];
  }>();
</script>

<Plot height={280}
  x={{ label: 'Count', grid: true, nice: true }}
  y={{ type: 'band', label: false, domain: data.map((d: any) => d.field) }}
  marginLeft={150} marginRight={20} marginTop={10} marginBottom={40}>
  <BarX data={data} x="count" y="field" fill={BRAND} fillOpacity={0.8} />
</Plot>
