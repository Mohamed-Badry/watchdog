<script lang="ts">
  import { env } from '$env/dynamic/public';
  import type { PageData } from './$types';
  import { untrack } from 'svelte';

  import EclipseScatterPlot from '$lib/components/charts/EclipseScatterPlot.svelte';
  import CorrelationHeatmap from '$lib/components/charts/CorrelationHeatmap.svelte';
  import TimeGapHistogram from '$lib/components/charts/TimeGapHistogram.svelte';
  import MacroHealthPlot from '$lib/components/charts/MacroHealthPlot.svelte';
  import FeatureDistributionGrid from '$lib/components/charts/FeatureDistributionGrid.svelte';

  let { data }: { data: PageData } = $props();

  let satellites = $derived(data.satellites || []);
  let error = $derived(data.error);

  let noradId = $state<string>('all');
  let dataLimit = $state<number>(1000); // Increased limit for better macro trends

  let loading = $state(false);
  let telemetryFrames = $state<any[]>([]);

  type FeatureMap = Record<string, unknown>;
  type NormalizedFeatures = Record<string, number | null> & {
    batt_voltage: number | null;
    batt_current: number | null;
    batt_a_voltage: number | null;
    batt_b_voltage: number | null;
    temp_batt_a: number | null;
    temp_batt_b: number | null;
    temp_panel_z: number | null;
  };

  function toFiniteNumber(value: unknown): number | null {
    if (typeof value === 'number') return Number.isFinite(value) ? value : null;
    if (typeof value === 'string' && value.trim() !== '') {
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : null;
    }
    return null;
  }

  function featureNumber(features: FeatureMap, keys: string[]): number | null {
    for (const key of keys) {
      const value = toFiniteNumber(features[key]);
      if (value !== null) return value;
    }
    return null;
  }

  function batteryVoltage(features: FeatureMap): number | null {
    const combined = toFiniteNumber(features.batt_voltage);
    if (combined !== null) return combined;

    const batteryA = toFiniteNumber(features.batt_a_voltage);
    const batteryB = toFiniteNumber(features.batt_b_voltage);
    if (batteryA !== null && batteryB !== null) return (batteryA + batteryB) / 2;
    return batteryA ?? batteryB;
  }

  function normalizeFeatures(features: FeatureMap): NormalizedFeatures {
    return {
      batt_voltage: batteryVoltage(features),
      batt_current: featureNumber(features, ['batt_current']),
      batt_a_voltage: featureNumber(features, ['batt_a_voltage']),
      batt_b_voltage: featureNumber(features, ['batt_b_voltage']),
      temp_batt_a: featureNumber(features, ['temp_batt_a']),
      temp_batt_b: featureNumber(features, ['temp_batt_b']),
      temp_panel_z: featureNumber(features, ['temp_panel_z']),
    };
  }

  async function fetchTelemetry() {
    loading = true;
    const apiUrl = typeof window !== 'undefined' ? (env.PUBLIC_API_URL || 'http://127.0.0.1:8000') : 'http://backend:8000';
    let url = `${apiUrl}/api/telemetry/recent?limit=${dataLimit}`;
    if (noradId !== 'all') {
      url += `&norad_id=${noradId}`;
    }
    try {
      const res = await fetch(url);
      if (res.ok) {
        const json = await res.json();
        telemetryFrames = json.frames || [];
      } else {
        console.error(`Failed to fetch telemetry: ${res.status}`);
        telemetryFrames = [];
      }
    } catch (e) {
      console.error(e);
      telemetryFrames = [];
    } finally {
      loading = false;
    }
  }

  // Derive feature arrays from frames
  let featureFrames = $derived(
    telemetryFrames
      .filter((f: any): f is { features: FeatureMap } => !!f.features && typeof f.features === 'object')
      .map((f: { features: FeatureMap }) => normalizeFeatures(f.features))
  );

  let eclipseFrames = $derived(
    featureFrames
      .filter(
        (f: NormalizedFeatures): f is NormalizedFeatures & {
          temp_panel_z: number;
          batt_current: number;
          batt_voltage: number;
        } => f.temp_panel_z !== null && f.batt_current !== null && f.batt_voltage !== null
      )
      .map((f) => ({
        temp_panel_z: f.temp_panel_z,
        batt_current: f.batt_current,
        batt_voltage: f.batt_voltage,
      }))
  );

  let macroFrames = $derived(
    telemetryFrames
      .filter((f: any): f is { timestamp: string; features: FeatureMap } => !!f.timestamp && !!f.features && typeof f.features === 'object')
      .map((f: any) => ({
        timestamp: f.timestamp,
        batt_voltage: batteryVoltage(f.features),
        temp_batt_a: featureNumber(f.features, ['temp_batt_a']),
        temp_panel_z: featureNumber(f.features, ['temp_panel_z']),
      }))
  );

  let timestamps = $derived(
    telemetryFrames
      .filter((f: any) => f.timestamp)
      .map((f: any) => f.timestamp)
  );
</script>

<div class="mx-auto w-full pb-24">
  <header class="space-y-4 mb-12 animate-in fade-in slide-in-from-bottom-4 duration-500 ease-out">
    <p class="text-xs font-semibold uppercase tracking-[0.2em] text-muted">Analysis Report</p>
    <h1 class="text-4xl font-bold tracking-tight text-ink sm:text-5xl">Exploratory Data Analysis</h1>
    <p class="text-lg leading-8 text-ink-2 max-w-4xl">
      Physics-driven validation of raw telemetry data. This notebook-style report breaks down
      the fundamental orbital mechanics and multivariate correlations used to engineer the 
      golden features for our autoencoder models.
    </p>
  </header>

  {#if error}
    <div class="rounded-xl border border-brand/50 bg-brand/10 p-6 text-brand">
      <h2 class="text-lg font-semibold">Connection Error</h2>
      <p class="mt-2 text-sm">{error}</p>
    </div>
  {:else}
    <!-- Controls (Minimal, inline) -->
    <div class="mb-16 flex flex-wrap items-center gap-4 rounded-2xl border border-border bg-surface/50 p-4">
      <div class="flex items-center gap-3">
        <label for="sat-select" class="text-xs font-semibold uppercase tracking-wider text-ink-3">Target Profile</label>
        <select id="sat-select" bind:value={noradId} class="rounded-lg border border-border bg-panel px-3 py-1.5 text-sm text-ink outline-none transition hover:border-brand">
          <option value="all">All Satellites (Global View)</option>
          {#each satellites as sat}
            <option value={sat.norad_id.toString()}>{sat.name} ({sat.norad_id})</option>
          {/each}
        </select>
      </div>
      
      <div class="flex items-center gap-3 ml-4 border-l border-border pl-4">
        <label for="data-limit" class="text-xs font-semibold uppercase tracking-wider text-ink-3">Dataset Size</label>
        <select id="data-limit" bind:value={dataLimit} class="rounded-lg border border-border bg-panel px-3 py-1.5 text-sm text-ink outline-none transition hover:border-brand">
          <option value={1000}>1,000 frames</option>
          <option value={5000}>5,000 frames</option>
          <option value={10000}>10,000 frames (Full Macro)</option>
        </select>
      </div>

      <div class="ml-4 border-l border-border pl-4">
        <button 
          onclick={fetchTelemetry}
          disabled={loading}
          class="flex items-center justify-center rounded-lg bg-brand px-4 py-1.5 text-sm font-semibold text-white shadow-sm transition hover:bg-brand/90 disabled:opacity-50"
        >
          {loading ? 'Fetching...' : 'Fetch Data'}
        </button>
      </div>

      {#if loading}
        <div class="h-4 w-4 animate-spin rounded-full border-2 border-surface border-t-brand ml-auto"></div>
      {:else if telemetryFrames.length > 0}
        <div class="ml-auto text-xs text-ink-3">
          Analyzing <span class="font-mono font-medium text-ink">{telemetryFrames.length.toLocaleString()}</span> frames
        </div>
      {/if}
    </div>

    {#if !loading && telemetryFrames.length === 0}
      <div class="rounded-2xl border border-border border-dashed py-16 text-center text-ink-3">
        No telemetry data found for this profile.
      </div>
    {:else if telemetryFrames.length > 0}
      
      <div class="space-y-32">
        
        <!-- SECTION 1: Pipeline Audit & Distributions -->
        <section class="grid gap-12 xl:grid-cols-[1fr_2fr] items-start">
          <div class="prose max-w-none xl:sticky xl:top-24">
            <h2 class="text-2xl font-bold tracking-tight text-ink border-b border-border pb-4">1. Data Engineering & Sanity Checks</h2>
            <p>
              Before engaging in unsupervised anomaly detection, the integrity of the data pipeline (`raw → interim → processed`) must be strictly validated. The backend ingestion service connects to the MQTT broker, buffers Kaitai-decoded frames, and performs SI-unit normalization to create the "Golden Features." We actively filter out known physical impossibilities early (e.g. 8V battery spikes) because they represent communication corruptions, not physical hardware anomalies.
            </p>
            <div class="my-6 rounded-xl border border-border bg-surface/50 p-5">
              <h4 class="mt-0 mb-2 text-sm font-semibold uppercase tracking-wider text-ink-3">Unit Conversion Audit</h4>
              <ul class="m-0 space-y-2 text-sm">
                <li class="m-0"><code class="bg-panel px-1 py-0.5 rounded text-xs border border-border">batt_voltage</code>: mV → V (divisor 1000.0)</li>
                <li class="m-0"><code class="bg-panel px-1 py-0.5 rounded text-xs border border-border">batt_current</code>: mA → A (divisor 1000.0)</li>
                <li class="m-0"><code class="bg-panel px-1 py-0.5 rounded text-xs border border-border">temp_panel_z</code>: °C (direct pass-through)</li>
              </ul>
            </div>
            <p>
              <strong>Observation:</strong> Clean, Gaussian-like or strict bimodal distributions on the right prove the parsers are correctly aligning byte boundaries. Spikes at exactly 0.0 or extreme outliers indicate parsing misalignments.
            </p>
          </div>
          <div class="flex flex-col gap-6">
            <div class="rounded-2xl border border-border bg-panel p-6 shadow-sm">
              <h3 class="mt-0 mb-6 flex items-center gap-3 text-sm font-semibold uppercase tracking-widest text-ink-3">
                <span class="inline-block h-3 w-1 rounded-sm bg-brand"></span>
                Feature Distributions
              </h3>
              <FeatureDistributionGrid frames={featureFrames} />
            </div>
          </div>
        </section>

        <!-- SECTION 2: Macro Trends & Eclipse Cycle -->
        <section class="grid gap-12 xl:grid-cols-[1fr_2fr] items-start">
          <div class="prose max-w-none xl:sticky xl:top-24">
            <h2 class="text-2xl font-bold tracking-tight text-ink border-b border-border pb-4">2. Deep-Dive Exploratory Data Analysis</h2>
            
            <h3>Long-Term Macro Trends</h3>
            <p>
              Over long datasets (e.g., spanning 7+ months), we observe extreme seasonality in satellite thermodynamics and charge cycles. The baselines shift massively over the year due to Beta angle drift and solar proximity.
            </p>

            <h3>The Bimodality Challenge (Day vs. Eclipse)</h3>
            <p>
              The most prominent and predictable cycle for any Low Earth Orbit (LEO) satellite is the <strong>Day/Night (Eclipse) cycle</strong>. 
              By plotting the temperature of the outward-facing solar panels against the battery charging current, we see a strict bimodal distribution representing the two physical states:
            </p>
            <ul>
              <li><strong>Sunlight (Day):</strong> High panel temperatures. Positive battery current (charging).</li>
              <li><strong>Eclipse (Night):</strong> Low panel temperatures. Negative battery current (discharging to run payloads).</li>
            </ul>
            <div class="mt-6 rounded-lg bg-surface/60 p-4 border border-border/50">
              <p class="m-0 text-sm leading-relaxed text-ink-2">
                <strong class="text-ink">Non-Linear Warning:</strong> The transition into eclipse structurally uncouples features. A static threshold logic gate would either trigger 1,000 false positives every orbit, or be so loose that it misses catastrophic component failures. This necessitates non-linear anomaly models.
              </p>
            </div>
          </div>
          <div class="flex flex-col gap-8">
            <div class="rounded-2xl border border-border bg-panel p-6 shadow-sm">
              <h3 class="mt-0 mb-6 flex items-center gap-3 text-sm font-semibold uppercase tracking-widest text-ink-3">
                <span class="inline-block h-3 w-1 rounded-sm bg-brand"></span>
                Macro-Scale Health (Voltage & Temp)
              </h3>
              <MacroHealthPlot frames={macroFrames} />
            </div>

            <div class="rounded-2xl border border-border bg-panel p-6 shadow-sm">
              <h3 class="mt-0 mb-6 flex items-center gap-3 text-sm font-semibold uppercase tracking-widest text-ink-3">
                <span class="inline-block h-3 w-1 rounded-sm bg-brand"></span>
                Solar Panel Temp vs. Battery Current
              </h3>
              <div class="h-[500px]">
                <EclipseScatterPlot frames={eclipseFrames} />
              </div>
              <p class="mt-4 mb-0 text-xs leading-relaxed text-ink-3 border-t border-border/50 pt-4">
                <strong>Diagnostic Heuristic:</strong> Data points falling into the top-left quadrant (positive charging current while panels are freezing cold) violate orbital physics and immediately flag a sensor fault, a decoding alignment error, or an EPS anomaly.
              </p>
            </div>
          </div>
        </section>

        <!-- SECTION 3: Feature Correlation -->
        <section class="grid gap-12 xl:grid-cols-[1fr_2fr] items-start">
          <div class="prose max-w-none xl:sticky xl:top-24">
            <h2 class="text-2xl font-bold tracking-tight text-ink border-b border-border pb-4">3. Multivariate Feature Correlation</h2>
            <p>
              To build a robust Autoencoder, we must map how the golden features interact across different subsystems. 
              Highly correlated features (e.g., Battery A Voltage and Battery B Voltage) compress efficiently into a lower-dimensional latent space. Conversely, uncorrelated features require independent, weighted representation in the neural network architecture.
            </p>
            <p>
              The heatmap visualizes the Pearson correlation coefficients across the parsed continuous variables. 
              Darker red intersections indicate a strong positive linear relationship, while dark blue indicates inverse relationships.
            </p>
            <div class="mt-6 rounded-lg bg-surface/60 p-4 border border-border/50">
              <p class="m-0 text-sm leading-relaxed text-ink-2">
                <strong class="text-ink">Model Implication:</strong> The Autoencoder exploits these known physical correlations. If a live telemetry frame breaks an established correlation (e.g., Voltage drops but Current stays flat), the reconstruction error spikes, generating an anomaly score.
              </p>
            </div>
          </div>
          <div class="flex flex-col gap-6">
            <div class="rounded-2xl border border-border bg-panel p-6 shadow-sm">
              <h3 class="mt-0 mb-6 flex items-center gap-3 text-sm font-semibold uppercase tracking-widest text-ink-3">
                <span class="inline-block h-3 w-1 rounded-sm bg-brand"></span>
                Pearson Correlation Matrix
              </h3>
              <div>
                <CorrelationHeatmap frames={featureFrames} />
              </div>
            </div>
          </div>
        </section>

        <!-- SECTION 4: Edge Discontinuity -->
        <section class="grid gap-12 xl:grid-cols-[1fr_2fr] items-start">
          <div class="prose max-w-none xl:sticky xl:top-24">
            <h2 class="text-2xl font-bold tracking-tight text-ink border-b border-border pb-4">4. The Edge Discontinuity</h2>
            <p>
              Unlike datacenter server metrics, edge station inference is strictly constrained by Line-Of-Sight passes. The data is entirely discontinuous.
            </p>
            <ul>
              <li>Median gap <strong>within</strong> a pass: ~10-15s</li>
              <li>Median gap <strong>between</strong> passes: ~10 hours</li>
            </ul>
            <div class="mt-6 rounded-lg bg-brand/5 p-4 border border-brand/20">
              <p class="m-0 text-sm leading-relaxed text-ink-2">
                <strong class="text-brand">Why LSTMs / Transformers fail here:</strong> Rolling history expires between passes. Attempting to use the frame from "10 hours ago" breaks physics predictions. We require <strong>Stateless Ensembles</strong> (evaluating each frame in a vacuum).
              </p>
            </div>
          </div>
          <div class="flex flex-col gap-6">
            <div class="rounded-2xl border border-border bg-panel p-6 shadow-sm">
              <h3 class="mt-0 mb-6 flex items-center gap-3 text-sm font-semibold uppercase tracking-widest text-ink-3">
                <span class="inline-block h-3 w-1 rounded-sm bg-brand"></span>
                Intra-Pass Time Gaps
              </h3>
              <div>
                 <TimeGapHistogram {timestamps} />
              </div>
            </div>
          </div>
        </section>

      </div>

    {/if}
  {/if}
</div>
