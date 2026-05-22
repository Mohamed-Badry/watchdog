<script lang="ts">
  import { Plot, BarY } from 'svelteplot';

  let { actual = {}, expected = {} } = $props<{ actual: Record<string, number>, expected: Record<string, number> }>();

  const ACTUAL_LBL = 'Actual';
  const EXPECTED_LBL = 'Expected';
  const ACTUAL_COLOR = '#b13a4b'; // Brand red (anomaly)
  const EXPECTED_COLOR = '#4f7fb5'; // Muted blue (baseline)

  let sortedFeatures = $derived(
    Object.keys(expected).sort((a, b) => Math.abs(actual[b] - expected[b]) - Math.abs(actual[a] - expected[a]))
  );

  let data = $derived(() => {
     const pts = [];
     for (const f of sortedFeatures) {
        pts.push({ feature: f, type: ACTUAL_LBL, value: Number(actual[f]) });
        pts.push({ feature: f, type: EXPECTED_LBL, value: Number(expected[f]) });
     }
     return pts;
  });

  const actualInset = { left: 16, right: 4 };
  const expectedInset = { left: 4, right: 16 };
</script>

<div class="w-full">
  {#if Object.keys(expected).length > 0}
    <Plot height={240}
      x={{ type: 'band', label: false, domain: sortedFeatures }}
      y={{ label: 'Physical Value', grid: true, nice: true }}
      color={{ domain: [ACTUAL_LBL, EXPECTED_LBL], range: [ACTUAL_COLOR, EXPECTED_COLOR] }}
      marginTop={12} marginRight={10} marginBottom={40} marginLeft={44}>
      <BarY data={data()} x="feature" y1={0} y2="value" fill="type"
            fillOpacity={0.85}
            insetLeft={(d) => d.type === EXPECTED_LBL ? expectedInset.left : actualInset.left}
            insetRight={(d) => d.type === EXPECTED_LBL ? expectedInset.right : actualInset.right}
            borderRadius={{ topLeft: 2, topRight: 2 }} />
    </Plot>
    
    <div class="mt-1 flex items-center justify-center gap-6 text-[0.65rem] uppercase tracking-wider font-semibold text-ink-3">
      <span class="flex items-center gap-1.5">
        <span class="inline-block h-2.5 w-2.5 rounded-sm" style="background: {EXPECTED_COLOR}; opacity: 0.85"></span>
        Expected
      </span>
      <span class="flex items-center gap-1.5">
        <span class="inline-block h-2.5 w-2.5 rounded-sm" style="background: {ACTUAL_COLOR}; opacity: 0.85"></span>
        Actual
      </span>
    </div>
  {:else}
    <div class="flex h-[240px] items-center justify-center text-sm text-ink-3">No data available.</div>
  {/if}
</div>
