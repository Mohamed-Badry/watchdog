<script lang="ts">
  /**
   * Feature Contribution — Per-feature reconstruction error by fault type
   * Reproduces docs/figures/ae_feature_contribution.png
   * Static benchmark data from Python analysis.
   */
  import { Plot, BarY } from 'svelteplot';

  const FEATURES = ['batt_voltage', 'batt_current', 't_batt_a', 't_batt_b', 't_panel_z'];

  const faults = [
    { name: 'Sensor Stuck', data: [
      ...FEATURES.map((f, i) => ({ feature: f, type: 'Normal', error: [0.25, 0.35, 0.10, 0.12, 0.40][i] })),
      ...FEATURES.map((f, i) => ({ feature: f, type: 'Sensor Stuck', error: [0.47, 0.53, 0.10, 0.10, 0.30][i] })),
    ]},
    { name: 'Panel Failure', data: [
      ...FEATURES.map((f, i) => ({ feature: f, type: 'Normal', error: [0.25, 0.35, 0.10, 0.12, 0.40][i] })),
      ...FEATURES.map((f, i) => ({ feature: f, type: 'Panel Failure', error: [0.82, 1.47, 0.11, 0.12, 0.69][i] })),
    ]},
    { name: 'Thermal Runaway', data: [
      ...FEATURES.map((f, i) => ({ feature: f, type: 'Normal', error: [0.25, 0.35, 0.10, 0.12, 0.40][i] })),
      ...FEATURES.map((f, i) => ({ feature: f, type: 'Thermal Runaway', error: [0.72, 0.52, 0.74, 0.34, 0.33][i] })),
    ]},
  ];
</script>

<div class="grid gap-6 md:grid-cols-3">
  {#each faults as fault}
    <div>
      <p class="mb-2 text-center text-xs font-semibold text-ink-2">{fault.name}</p>
      <Plot height={220}
        x={{ type: 'band', label: false, tickRotate: -35 }}
        y={{ label: '|Reconstruction Error|', grid: true }}
        color={{ domain: ['Normal', fault.name], range: ['#6c7a96', '#b12142'] }}
        marginTop={12} marginRight={8} marginBottom={56} marginLeft={40}>
        <BarY data={fault.data} x="feature" y="error" fill="type"
              fillOpacity={0.75} />
      </Plot>
    </div>
  {/each}
</div>

<div class="mt-3 flex items-center justify-center gap-6 text-[0.65rem] text-ink-2">
  <span class="flex items-center gap-1.5">
    <span class="inline-block h-2.5 w-2.5 rounded-sm" style="background: #6c7a96; opacity: 0.75"></span>
    Normal Baseline
  </span>
  <span class="flex items-center gap-1.5">
    <span class="inline-block h-2.5 w-2.5 rounded-sm bg-brand opacity-75"></span>
    Injected Fault
  </span>
</div>
