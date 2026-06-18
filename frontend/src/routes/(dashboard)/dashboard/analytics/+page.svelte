<script lang="ts">
  import { env } from '$env/dynamic/public';
  import type { PageData } from './$types';
  import ThroughputVolumePlot from '$lib/components/charts/ThroughputVolumePlot.svelte';
  import PassDurationScatterPlot from '$lib/components/charts/PassDurationScatterPlot.svelte';
  import LinkQualityPlot from '$lib/components/charts/LinkQualityPlot.svelte';
  import MissingFieldsBarChart from '$lib/components/charts/MissingFieldsBarChart.svelte';
  import MacroVoltageTrendPlot from '$lib/components/charts/MacroVoltageTrendPlot.svelte';
  import OrbitalDriftPlot from '$lib/components/charts/OrbitalDriftPlot.svelte';
  import { SERIES_CURRENT as BLUE, SERIES_AMBER as AMBER } from '$lib/chart-theme';

  let { data }: { data: PageData } = $props();

  let error = $derived(data.error);
  let analytics = $derived(data.analytics);

  let activeTab = $state<'throughput' | 'quality' | 'health'>('throughput');
</script>

<section class="flex flex-col gap-6 animate-in fade-in slide-in-from-bottom-4 duration-500 ease-out">
  <div class="space-y-1">
    <p class="text-xs font-semibold uppercase tracking-[0.2em] text-muted">Deep Dive</p>
    <h1 class="text-3xl font-semibold tracking-tight text-ink">Analytics</h1>
  </div>

  {#if error}
    <div class="rounded-xl border border-brand/50 bg-brand/10 p-4 text-sm text-brand">
      {error}
    </div>
  {:else if !analytics}
    <div class="flex items-center justify-center p-20">
      <div class="h-8 w-8 animate-spin rounded-full border-2 border-surface border-t-brand"></div>
    </div>
  {:else}
    <!-- Tabs -->
    <div class="flex items-center gap-2 border-b border-border">
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

    <!-- Main Content Area - No strict height constraints, allowing natural flow -->
    <div class="w-full">
      
      {#if activeTab === 'throughput'}
        <div class="flex flex-col gap-6">
          <!-- Top Row: Full width Area Chart -->
          <div class="flex flex-col rounded-[1.25rem] border border-border bg-panel p-6 shadow-sm">
            <h3 class="mt-0 mb-4 flex items-center gap-3 text-sm font-semibold uppercase tracking-widest text-ink-3">
              <span class="inline-block h-3 w-1 rounded-sm bg-brand"></span>
              30-Day Frame Volume
            </h3>
            <div class="w-full">
              <ThroughputVolumePlot data={analytics.throughput_30d || []} />
            </div>
          </div>

          <!-- Bottom Row: Split Grid -->
          <div class="grid grid-cols-1 xl:grid-cols-2 gap-6">
            <div class="flex flex-col rounded-[1.25rem] border border-border bg-panel p-6 shadow-sm">
              <h3 class="mt-0 mb-1 flex items-center gap-3 text-sm font-semibold uppercase tracking-widest text-ink-3">
                <span class="inline-block h-3 w-1 rounded-sm" style="background: {BLUE}"></span>
                Pass Duration vs Frames
              </h3>
              <p class="text-[10px] text-ink-3 mb-4">Correlation between Line-of-Sight time and payload yield.</p>
              <div class="w-full">
                <PassDurationScatterPlot data={analytics.pass_metrics} />
              </div>
            </div>

            <div class="flex flex-col rounded-[1.25rem] border border-border bg-panel p-6 shadow-sm">
              <h3 class="mt-0 mb-1 flex items-center gap-3 text-sm font-semibold uppercase tracking-widest text-ink-3">
                <span class="inline-block h-3 w-1 rounded-sm" style="background: {AMBER}"></span>
                Link Quality Degradation
              </h3>
              <p class="text-[10px] text-ink-3 mb-4">Daily count of dropped packet suspects over 30 days.</p>
              <div class="w-full">
                <LinkQualityPlot data={analytics.throughput_30d || []} />
              </div>
            </div>
          </div>
        </div>

      {:else if activeTab === 'quality'}
        <div class="flex flex-col gap-6">
          
          <div class="grid grid-cols-1 xl:grid-cols-3 gap-6">
            <!-- Info Card -->
            <div class="xl:col-span-1 rounded-[1.25rem] border border-border bg-panel p-6 shadow-sm flex flex-col justify-center">
              <h3 class="mt-0 mb-4 flex items-center gap-3 text-sm font-semibold uppercase tracking-widest text-ink-3">
                <span class="inline-block h-3 w-1 rounded-sm bg-brand"></span>
                Parser Integrity
              </h3>
              <p class="text-sm leading-relaxed text-ink-2">
                The Kaitai Struct decoders map the raw hex byte-stream into meaningful physics variables. 
                A <span class="font-bold text-emerald-500">Complete</span> frame parsed all required fields. 
                A <span class="font-bold text-warning">Partial</span> frame suffered transmission corruption, resulting in missing telemetry values.
              </p>
            </div>

            <!-- Stats -->
            <div class="xl:col-span-2 grid grid-cols-1 sm:grid-cols-2 gap-6">
              <div class="rounded-[1.25rem] border border-emerald-500/20 bg-emerald-500/5 p-6 shadow-sm flex flex-col justify-center items-center relative overflow-hidden group transition-all hover:border-emerald-500/40">
                <div class="absolute -inset-2 bg-gradient-to-tr from-emerald-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                <p class="text-xs font-semibold uppercase tracking-wider text-emerald-500 mb-2">Complete Frames</p>
                <p class="text-3xl sm:text-5xl font-bold tracking-tight text-emerald-500 drop-shadow-[0_0_12px_rgba(16,185,129,0.3)]">{analytics.quality.complete_frames.toLocaleString()}</p>
              </div>
              <div class="rounded-[1.25rem] border border-warning/20 bg-warning/5 p-6 shadow-sm flex flex-col justify-center items-center relative overflow-hidden group transition-all hover:border-warning/40">
                <div class="absolute -inset-2 bg-gradient-to-tr from-warning/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                <p class="text-xs font-semibold uppercase tracking-wider text-warning mb-2">Partial Frames</p>
                <p class="text-3xl sm:text-5xl font-bold tracking-tight text-warning drop-shadow-[0_0_12px_rgba(245,158,11,0.3)]">{analytics.quality.partial_frames.toLocaleString()}</p>
              </div>
            </div>
          </div>

          <!-- Chart Card -->
          <div class="rounded-[1.25rem] border border-border bg-panel p-6 shadow-sm">
            <h3 class="mt-0 mb-6 flex items-center gap-3 text-sm font-semibold uppercase tracking-widest text-ink-3">
              <span class="inline-block h-3 w-1 rounded-sm bg-warning"></span>
              Top Missing Fields
            </h3>
            
            {#if analytics.quality.missing_fields.length > 0}
              <div class="w-full">
                <MissingFieldsBarChart data={analytics.quality.missing_fields} />
              </div>
            {:else}
              <div class="w-full py-12 flex flex-col items-center justify-center text-emerald-500 border border-dashed border-emerald-500/30 bg-emerald-500/5 rounded-[1.25rem]">
                <span class="text-xl font-semibold mb-2">100% Data Integrity</span>
                <span class="text-sm opacity-80">No missing fields recorded. All packets parsed perfectly.</span>
              </div>
            {/if}
          </div>

        </div>

      {:else if activeTab === 'health'}
        <div class="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <div class="flex flex-col rounded-[1.25rem] border border-border bg-panel p-6 shadow-sm">
            <h3 class="mt-0 mb-2 flex items-center gap-3 text-sm font-semibold uppercase tracking-widest text-ink-3">
              <span class="inline-block h-3 w-1 rounded-sm bg-brand"></span>
              Macro Battery Voltage Trends (180 Days)
            </h3>
            <p class="text-xs leading-relaxed text-ink-3 mb-6">
              Long-term voltage decay and seasonal charge capacity shifts. Shaded area represents ±1 standard deviation of daily variance.
            </p>
            <div class="w-full">
              <MacroVoltageTrendPlot data={analytics.macro_health || []} />
            </div>
          </div>

          <div class="flex flex-col rounded-[1.25rem] border border-border bg-panel p-6 shadow-sm">
            <h3 class="mt-0 mb-2 flex items-center gap-3 text-sm font-semibold uppercase tracking-widest text-ink-3">
              <span class="inline-block h-3 w-1 rounded-sm" style="background: {AMBER}"></span>
              Thermodynamic Orbital Drift (180 Days)
            </h3>
            <p class="text-xs leading-relaxed text-ink-3 mb-6">
              Daily average Solar Panel Z temperatures. This massive seasonal shift is driven by the satellite's changing Beta angle relative to the sun.
            </p>
            <div class="w-full">
              <OrbitalDriftPlot data={analytics.macro_health || []} />
            </div>
          </div>
        </div>
      {/if}

    </div>
  {/if}
</section>