<script lang="ts">
  import ResponsivePlot from './ResponsivePlot.svelte';
  /**
   * Feature Correlation Matrix — Annotated heatmap
   * Reproduces docs/figures/correlation_heatmap.png
   *
   * 5×5 matrix showing Pearson correlation between golden features.
   * Uses Cell mark for the grid, Text for annotations.
   */
  import { Cell, Text } from 'svelteplot';
  import { pearsonCorrelation } from '$lib/data/statistics';

  type FrameFeatures = Record<string, number | null>;

  let { frames = [] } = $props<{ frames: FrameFeatures[] }>();

  const FEATURES = [
    'batt_voltage',
    'batt_current',
    'temp_batt_a',
    'temp_batt_b',
    'temp_panel_z',
  ] as const;

  const LABELS: Record<string, string> = {
    batt_voltage: 'Voltage',
    batt_current: 'Current',
    temp_batt_a: 'T Batt A',
    temp_batt_b: 'T Batt B',
    temp_panel_z: 'T Panel Z',
  };



  let cells = $derived(() => {
    const result: { x: string; y: string; r: number }[] = [];
    for (const f1 of FEATURES) {
      for (const f2 of FEATURES) {
        if (f1 === f2) {
          result.push({ x: LABELS[f1], y: LABELS[f2], r: 1 });
          continue;
        }
        const paired = frames
          .map((f: FrameFeatures) => [f[f1], f[f2]] as [number | null, number | null])
          .filter((p: [number | null, number | null]): p is [number, number] => p[0] != null && p[1] != null && !isNaN(p[0]) && !isNaN(p[1]));
        const r = pearsonCorrelation(paired.map((p: [number, number]) => p[0]), paired.map((p: [number, number]) => p[1]));
        result.push({ x: LABELS[f1], y: LABELS[f2], r: Math.round(r * 100) / 100 });
      }
    }
    return result;
  });
</script>

<div class="h-full w-full">
  <ResponsivePlot
    height={380}
    square={true}
    x={{ type: 'band', label: false, domain: FEATURES.map(f => LABELS[f]) }}
    y={{ type: 'band', label: false, domain: [...FEATURES].reverse().map(f => LABELS[f]) }}
    color={{ type: 'linear', domain: [-1, 1], scheme: 'magma', label: 'Correlation' }}
    marginTop={8}
    marginRight={8}
    marginBottom={60}
    marginLeft={72}
  >
    <Cell data={cells()} x="x" y="y" fill="r"
          fillOpacity={d => d.x === d.y ? 0.15 : 0.85}
          stroke="var(--color-border)" strokeWidth={1} />

    <!-- Annotate with correlation values (skip diagonal) -->
    <Text data={cells().filter(d => d.x !== d.y)} x="x" y="y"
          text={d => d.r.toFixed(2)}
          fill={d => d.r > 0.3 ? '#18181b' : 'white'}
          fontSize={12} fontWeight="600"
          textAnchor="middle" dy={4} />
  </ResponsivePlot>
</div>
