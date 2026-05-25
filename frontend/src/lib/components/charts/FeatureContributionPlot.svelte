<script lang="ts">
  /**
   * Feature Contribution — Per-feature reconstruction error by fault type
   * Reproduces docs/figures/ae_feature_contribution.png
   * Static benchmark data from Python analysis.
   */
  import { Plot, BarY } from 'svelteplot';
  import { ML_NORMAL, ML_FAULT } from '$lib/chart-theme';

  const NORMAL = 'Normal baseline';
  const FAULT = 'Injected fault';
  const NORMAL_COLOR = ML_NORMAL;
  const FAULT_COLOR = ML_FAULT;

  const FEATURES = [
    { key: 'batt_voltage', label: 'Voltage' },
    { key: 'batt_current', label: 'Current' },
    { key: 't_batt_a', label: 'Batt A' },
    { key: 't_batt_b', label: 'Batt B' },
    { key: 't_panel_z', label: 'Panel Z' },
  ];

  const featureLabels = FEATURES.map((feature) => feature.label);

  const faults = [
    { name: 'Sensor Stuck', data: [
      ...FEATURES.map((f, i) => ({ feature: f.label, type: NORMAL, error: [0.25, 0.35, 0.10, 0.12, 0.40][i] })),
      ...FEATURES.map((f, i) => ({ feature: f.label, type: FAULT, error: [0.47, 0.53, 0.10, 0.10, 0.30][i] })),
    ]},
    { name: 'Panel Failure', data: [
      ...FEATURES.map((f, i) => ({ feature: f.label, type: NORMAL, error: [0.25, 0.35, 0.10, 0.12, 0.40][i] })),
      ...FEATURES.map((f, i) => ({ feature: f.label, type: FAULT, error: [0.82, 1.47, 0.11, 0.12, 0.69][i] })),
    ]},
    { name: 'Thermal Runaway', data: [
      ...FEATURES.map((f, i) => ({ feature: f.label, type: NORMAL, error: [0.25, 0.35, 0.10, 0.12, 0.40][i] })),
      ...FEATURES.map((f, i) => ({ feature: f.label, type: FAULT, error: [0.72, 0.52, 0.74, 0.34, 0.33][i] })),
    ]},
  ];

  const normalBarInset = { left: 3, right: 18 };
  const faultBarInset = { left: 18, right: 3 };
</script>

<div class="grid gap-6 md:grid-cols-3">
  {#each faults as fault}
    <div>
      <p class="mb-2 text-center text-xs font-semibold text-ink-2">{fault.name}</p>
      <Plot height={238}
        x={{ type: 'band', label: false, domain: featureLabels }}
        y={{ label: false, grid: true, nice: true }}
        color={{ domain: [NORMAL, FAULT], scheme: [NORMAL_COLOR, FAULT_COLOR] }}
        marginTop={12} marginRight={10} marginBottom={42} marginLeft={44}>
        <BarY data={fault.data} x="feature" y1={0} y2="error" fill="type"
              fillOpacity={0.82}
              insetLeft={(d) => d.type === NORMAL ? normalBarInset.left : faultBarInset.left}
              insetRight={(d) => d.type === NORMAL ? normalBarInset.right : faultBarInset.right}
              borderRadius={{ topLeft: 2, topRight: 2 }} />
      </Plot>
    </div>
  {/each}
</div>

<div class="mt-3 flex items-center justify-center gap-6 text-[0.65rem] text-ink-2">
  <span class="flex items-center gap-1.5">
    <span class="inline-block h-2.5 w-2.5 rounded-sm" style="background: {NORMAL_COLOR}; opacity: 0.82"></span>
    Normal Baseline
  </span>
  <span class="flex items-center gap-1.5">
    <span class="inline-block h-2.5 w-2.5 rounded-sm" style="background: {FAULT_COLOR}; opacity: 0.82"></span>
    Injected Fault
  </span>
</div>
