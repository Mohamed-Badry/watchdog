<script lang="ts">
  import { env } from '$env/dynamic/public';
  import type { PageData } from './$types';
  import { Plot, AreaY, Line, BarX, BarY, Dot } from 'svelteplot';

  let { data }: { data: PageData } = $props();

  let error = $derived(data.error);
  let analytics = $derived(data.analytics);

  let activeTab = $state<'throughput' | 'quality' | 'health'>('throughput');

  const BRAND = '#b12142';
  const BLUE = '#3b82f6';
  const EMERALD = '#10b981';
  const AMBER = '#f59e0b';

  // Parse strings to Dates for svelteplot time scales
  let throughputParsed = $derived((analytics?.throughput_30d || []).map((d: any) => ({
    ...d,
    dateObj: new Date(d.date),
    dateStr: new Date(d.date).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
  })));

  let healthParsed = $derived((analytics?.macro_health || []).map((d: any) => ({
    ...d,
    dateObj: new Date(d.date)
  })));
</script>

<section class="flex flex-col h-full min-h-0 gap-5 animate-in fade-in slide-in-from-bottom-4 duration-500 ease-out">
  <div class="flex-none space-y-1">
    <p class="text-xs font-semibold uppercase tracking-[0.2em] text-muted">Deep Dive</p>
    <h1 class="text-3xl font-semibold tracking-tight text-ink">Analytics</h1>
  </div>

  {#if error}
    <div class="flex-none rounded-xl border border-brand/50 bg-brand/10 p-4 text-sm text-brand">
      {error}
    </div>
  {:else if !analytics}
    <div class="flex flex-1 items-center justify-center">
      <div class="h-6 w-6 animate-spin rounded-full border-2 border-surface border-t-brand"></div>
    </div>
  {:else}
    <!-- Tabs -->
    <div class="flex-none flex items-center gap-2 border-b border-border">
      <button 
        class="px-4 py-2 text-sm font-medium transition-colors border-b-2 {activeTab === 'throughput' ? 'border-brand text-brand' : 'border-transparent text-ink-3 hover:text-ink hover:border-border'}"
        onclick={() => activeTab = 'throughput'}
      >
        Network Throughput
      </button>
      <button 
        class="px-4 py-2 text-sm font-medium transition-colors border-b-2 {activeTab === 'quality' ? 'border-brand text-brand' : 'border-transparent text-ink-3 hover:text-ink hover:border-border'}"
        onclick={() => activeTab = 'quality'}
      >
        Data Quality
      </button>
      <button 
        class="px-4 py-2 text-sm font-medium transition-colors border-b-2 {activeTab === 'health' ? 'border-brand text-brand' : 'border-transparent text-ink-3 hover:text-ink hover:border-border'}"
        onclick={() => activeTab = 'health'}
      >
        Hardware Health
      </button>
    </div>

    <!-- Main Content Area - Strict height to prevent scrolling -->
    <div class="flex-1 min-h-0">
      
      {#if activeTab === 'throughput'}
        <div class="flex flex-col h-full gap-5">
          <!-- Top Row: Full width Area Chart -->
          <div class="flex flex-col flex-1 min-h-0 rounded-2xl border border-border bg-panel p-5 shadow-sm">
            <h3 class="flex-none mt-0 mb-2 flex items-center gap-3 text-sm font-semibold uppercase tracking-widest text-ink-3">
              <span class="inline-block h-3 w-1 rounded-sm bg-brand"></span>
              30-Day Frame Volume
            </h3>
            <div class="flex-1 min-h-0 w-full">
              <Plot height={240}
                x={{ type: 'time', label: false, grid: true }}
                y={{ label: 'Frames Received', grid: true, nice: true }}
                marginTop={30} marginRight={20} marginBottom={40} marginLeft={60}>
                <AreaY data={throughputParsed} x="dateObj" y="frame_count" fill={BRAND} fillOpacity={0.2} />
                <Line data={throughputParsed} x="dateObj" y="frame_count" stroke={BRAND} strokeWidth={2} />
              </Plot>
            </div>
          </div>

          <!-- Bottom Row: Split Grid -->
          <div class="flex-1 min-h-0 grid grid-cols-1 xl:grid-cols-2 gap-5">
            <div class="flex flex-col rounded-2xl border border-border bg-panel p-5 shadow-sm">
              <h3 class="flex-none mt-0 mb-1 flex items-center gap-3 text-sm font-semibold uppercase tracking-widest text-ink-3">
                <span class="inline-block h-3 w-1 rounded-sm" style="background: {BLUE}"></span>
                Pass Duration vs Frames
              </h3>
              <p class="flex-none text-[10px] text-ink-3 mb-2">Correlation between Line-of-Sight time and payload yield.</p>
              <div class="flex-1 min-h-0 w-full">
                <Plot height={200}
                  x={{ label: 'Pass Duration (seconds)', grid: true, nice: true }}
                  y={{ label: 'Frames Decoded', grid: true, nice: true }}
                  marginTop={30} marginRight={20} marginBottom={40} marginLeft={60}>
                  <Dot data={analytics.pass_metrics} x="duration_sec" y="frame_count" fill={BLUE} r={3} fillOpacity={0.6} />
                </Plot>
              </div>
            </div>

            <div class="flex flex-col rounded-2xl border border-border bg-panel p-5 shadow-sm">
              <h3 class="flex-none mt-0 mb-1 flex items-center gap-3 text-sm font-semibold uppercase tracking-widest text-ink-3">
                <span class="inline-block h-3 w-1 rounded-sm" style="background: {AMBER}"></span>
                Link Quality Degradation
              </h3>
              <p class="flex-none text-[10px] text-ink-3 mb-2">Daily count of dropped packet suspects over 30 days.</p>
              <div class="flex-1 min-h-0 w-full">
                <!-- Switched to AreaY on time scale instead of BarY to fix band mismatch and match visual style -->
                <Plot height={200}
                  x={{ type: 'time', label: false, grid: true }}
                  y={{ label: 'Dropped Suspects', grid: true, nice: true }}
                  marginTop={30} marginRight={20} marginBottom={40} marginLeft={60}>
                  <AreaY data={throughputParsed} x="dateObj" y="dropped_suspects" fill={AMBER} fillOpacity={0.5} />
                  <Line data={throughputParsed} x="dateObj" y="dropped_suspects" stroke={AMBER} strokeWidth={2} />
                </Plot>
              </div>
            </div>
          </div>
        </div>

      {:else if activeTab === 'quality'}
        <div class="flex flex-col h-full gap-5">
          <div class="flex flex-col flex-1 min-h-0 rounded-2xl border border-border bg-panel p-8 shadow-sm">
            <h3 class="flex-none mt-0 mb-8 flex items-center justify-center gap-3 text-sm font-semibold uppercase tracking-widest text-ink-3">
              <span class="inline-block h-3 w-1 rounded-sm bg-brand"></span>
              Parser Integrity Audit
            </h3>
            
            <div class="flex-1 min-h-0 flex flex-col items-center justify-center max-w-4xl mx-auto w-full gap-12">
              <p class="text-base text-center leading-relaxed text-ink-2 max-w-2xl">
                The Kaitai Struct decoders map the raw hex byte-stream into meaningful physics variables. 
                A <span class="font-bold text-emerald-500">Complete</span> frame parsed all required fields. 
                A <span class="font-bold text-brand">Partial</span> frame suffered transmission corruption, resulting in missing telemetry values.
              </p>
              
              <div class="flex w-full gap-6 justify-center">
                <div class="bg-surface/50 border border-border rounded-2xl p-6 w-64 text-center shadow-inner">
                  <p class="text-xs font-semibold uppercase tracking-wider text-ink-3 mb-3">Complete</p>
                  <p class="text-4xl font-bold tracking-tight text-emerald-500">{analytics.quality.complete_frames.toLocaleString()}</p>
                </div>
                <div class="bg-surface/50 border border-border rounded-2xl p-6 w-64 text-center shadow-inner">
                  <p class="text-xs font-semibold uppercase tracking-wider text-ink-3 mb-3">Partial</p>
                  <p class="text-4xl font-bold tracking-tight text-brand">{analytics.quality.partial_frames.toLocaleString()}</p>
                </div>
              </div>
              
              {#if analytics.quality.missing_fields.length > 0}
                <div class="w-full mt-4">
                  <h4 class="text-xs font-semibold uppercase tracking-wider text-ink-3 mb-6 text-center">Top Missing Fields</h4>
                  <Plot height={280}
                    x={{ label: 'Count', grid: true, nice: true }}
                    y={{ type: 'band', label: false, domain: analytics.quality.missing_fields.map((d: any) => d.field) }}
                    marginLeft={150} marginRight={20} marginTop={10} marginBottom={40}>
                    <BarX data={analytics.quality.missing_fields} x="count" y="field" fill={BRAND} fillOpacity={0.8} />
                  </Plot>
                </div>
              {:else}
                <div class="w-full max-w-lg mt-4 py-8 flex flex-col items-center justify-center text-emerald-500 border border-dashed border-emerald-500/30 bg-emerald-500/5 rounded-2xl">
                  <span class="text-lg font-semibold mb-1">100% Data Integrity</span>
                  <span class="text-sm opacity-80">No missing fields recorded. All packets parsed perfectly.</span>
                </div>
              {/if}
            </div>
          </div>
        </div>

      {:else if activeTab === 'health'}
        <div class="grid grid-cols-1 xl:grid-cols-2 gap-5 h-full">
          <div class="flex flex-col h-full rounded-2xl border border-border bg-panel p-6 shadow-sm">
            <h3 class="flex-none mt-0 mb-2 flex items-center gap-3 text-sm font-semibold uppercase tracking-widest text-ink-3">
              <span class="inline-block h-3 w-1 rounded-sm bg-brand"></span>
              Macro Battery Voltage Trends (180 Days)
            </h3>
            <p class="flex-none text-xs leading-relaxed text-ink-3 mb-4">
              Long-term voltage decay and seasonal charge capacity shifts. Shaded area represents ±1 standard deviation of daily variance.
            </p>
            <div class="flex-1 min-h-0 w-full">
              <Plot height={340}
                x={{ type: 'time', label: false, grid: true }}
                y={{ label: 'Voltage (V)', grid: true, nice: true, domain: [3.5, 4.3] }}
                marginTop={30} marginRight={20} marginBottom={40} marginLeft={60}>
                <!-- Variance band -->
                <AreaY data={healthParsed.filter((d: any) => d.batt_voltage_mean !== null)} 
                       x="dateObj" 
                       y1={(d: any) => d.batt_voltage_mean - d.batt_voltage_std} 
                       y2={(d: any) => d.batt_voltage_mean + d.batt_voltage_std} 
                       fill={BLUE} fillOpacity={0.15} />
                <!-- Mean line -->
                <Line data={healthParsed.filter((d: any) => d.batt_voltage_mean !== null)} 
                      x="dateObj" y="batt_voltage_mean" stroke={BLUE} strokeWidth={2} />
              </Plot>
            </div>
          </div>

          <div class="flex flex-col h-full rounded-2xl border border-border bg-panel p-6 shadow-sm">
            <h3 class="flex-none mt-0 mb-2 flex items-center gap-3 text-sm font-semibold uppercase tracking-widest text-ink-3">
              <span class="inline-block h-3 w-1 rounded-sm" style="background: {AMBER}"></span>
              Thermodynamic Orbital Drift (180 Days)
            </h3>
            <p class="flex-none text-xs leading-relaxed text-ink-3 mb-4">
              Daily average Solar Panel Z temperatures. This massive seasonal shift is driven by the satellite's changing Beta angle relative to the sun.
            </p>
            <div class="flex-1 min-h-0 w-full">
              <Plot height={340}
                x={{ type: 'time', label: false, grid: true }}
                y={{ label: 'Temperature (°C)', grid: true, nice: true }}
                marginTop={30} marginRight={20} marginBottom={40} marginLeft={60}>
                <!-- Variance band -->
                <AreaY data={healthParsed.filter((d: any) => d.temp_panel_z_mean !== null)} 
                       x="dateObj" 
                       y1={(d: any) => d.temp_panel_z_mean - d.temp_panel_z_std} 
                       y2={(d: any) => d.temp_panel_z_mean + d.temp_panel_z_std} 
                       fill={AMBER} fillOpacity={0.15} />
                <!-- Mean line -->
                <Line data={healthParsed.filter((d: any) => d.temp_panel_z_mean !== null)} 
                      x="dateObj" y="temp_panel_z_mean" stroke={AMBER} strokeWidth={2} />
              </Plot>
            </div>
          </div>
        </div>
      {/if}

    </div>
  {/if}
</section>