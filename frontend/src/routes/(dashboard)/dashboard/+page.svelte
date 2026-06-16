<script lang="ts">
  import type { PageData } from './$types';
  import { onMount, onDestroy } from 'svelte';
  import { getApiUrl } from '$lib/api';
  import type { DashboardSummary } from '$lib/types/api';
  import SparklinePlot from '$lib/components/charts/SparklinePlot.svelte';
  import { Satellite, Radio, AlertTriangle, Globe } from 'lucide-svelte';

  let { data }: { data: PageData } = $props();

  let summary = $state<DashboardSummary | null>(data.summary);
  let error = $state<string | undefined>(data.error);

  function getSeverityColor(score: number | undefined | null) {
    if (!score) return 'var(--color-info)'; // Default
    if (score >= 0.7) return 'var(--color-critical)'; // Rose Red
    if (score >= 0.45) return 'var(--color-warning)'; // Amber
    if (score >= 0.3) return 'var(--color-brand)'; // Amethyst
    return 'var(--color-info)'; // Blue
  }

  $effect(() => {
    summary = data.summary;
    error = data.error;
  });

  // ── WebSocket with exponential-backoff reconnection ────────────────────────
  let ws: WebSocket | null = null;
  let retryDelay = 1000;
  let retryTimer: ReturnType<typeof setTimeout> | null = null;
  let destroyed = false;

  function connectWs() {
    if (destroyed) return;

    const apiUrl = getApiUrl();
    const wsUrl = apiUrl.replace(/^http/, 'ws') + '/api/ws/dashboard';

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      retryDelay = 1000; // Reset backoff on successful connect
    };

    ws.onmessage = (event) => {
      try {
        summary = JSON.parse(event.data);
        error = undefined;
      } catch (e: unknown) {
        console.error('Failed to parse websocket message', e);
      }
    };

    ws.onerror = () => {
      // onerror always fires before onclose, just let onclose handle reconnect
    };

    ws.onclose = () => {
      if (destroyed) return;
      retryTimer = setTimeout(() => {
        retryDelay = Math.min(retryDelay * 2, 30_000);
        connectWs();
      }, retryDelay);
    };
  }

  onMount(() => {
    connectWs();
  });

  onDestroy(() => {
    destroyed = true;
    if (retryTimer) clearTimeout(retryTimer);
    if (ws) ws.close();
  });

  // ── Stat card config (lucide icons instead of emojis) ─────────────────────
  const statCards = [
    { label: 'Active Satellites', key: 'satellite_count' as const, icon: Satellite },
    { label: 'Total Frames',     key: 'frame_count' as const,     icon: Radio },
    { label: 'Anomalies Detected', key: 'anomaly_count' as const, icon: AlertTriangle },
    { label: 'Total Passes',     key: 'pass_count' as const,      icon: Globe },
  ];
</script>

<svelte:head>
  <title>Dashboard — Watchdog</title>
</svelte:head>

{#if error}
  <div class="rounded-xl border border-brand/50 bg-brand/10 p-6 text-brand">
    <h2 class="text-lg font-semibold">Connection Error</h2>
    <p class="mt-2 text-sm">{error}</p>
  </div>
{:else if summary}
  <section class="flex flex-col h-full min-h-0 gap-5">
    <div class="flex-none space-y-1">
      <p class="text-xs font-semibold uppercase tracking-[0.2em] text-muted">System Overview</p>
      <h1 class="text-3xl font-semibold tracking-tight text-ink">Dashboard Home</h1>
    </div>

    <!-- Totals -->
    <div class="flex-none grid gap-4 md:grid-cols-4">
      {#each statCards as stat}
        {@const Icon = stat.icon}
        {@const value = summary.totals[stat.key]}
        <article class="group flex items-center justify-between rounded-[1.25rem] border border-border bg-panel p-5 shadow-panel backdrop-blur transition-all duration-300 hover:-translate-y-1 hover:border-brand/30 hover:shadow-md">
          <div class="flex flex-col">
            <p class="text-xs font-semibold uppercase tracking-wider text-ink-3">{stat.label}</p>
            <p class="mt-1 text-3xl font-bold tracking-tight text-brand">{typeof value === 'number' ? value.toLocaleString() : value}</p>
          </div>
          <div class="flex h-12 w-12 items-center justify-center rounded-full bg-brand/10 text-brand shadow-inner transition-colors group-hover:bg-brand group-hover:text-white">
            <Icon class="size-6" />
          </div>
        </article>
      {/each}
    </div>

    <!-- Main Grid Layout -->
    <div class="flex-1 min-h-0 grid gap-6 xl:grid-cols-[1fr_2fr]">

      <!-- Left Col (Component Health & Active Profiles) -->
      <div class="flex flex-col gap-6 min-h-0">
        <!-- Service Status -->
        <div class="flex flex-col rounded-[1.25rem] border border-border bg-panel shadow-panel backdrop-blur flex-none overflow-hidden">
          <div class="bg-surface/35 p-4 border-b border-border shrink-0">
            <h2 class="text-sm font-semibold uppercase tracking-[0.16em] text-ink-3">Component Health</h2>
          </div>
          <div class="flex flex-col gap-3 p-4 shrink-0">
            {#each summary.service_status as component}
              <div class="group flex items-start justify-between rounded-xl border border-border/50 bg-surface/30 p-4 transition-all hover:border-brand/30 hover:bg-surface/60">
                <div class="flex flex-col gap-1.5">
                  <span class="text-sm font-semibold capitalize text-ink transition-colors group-hover:text-brand">{component.name.replace('_', ' ')}</span>
                  <p class="text-[11px] font-medium leading-relaxed text-ink-3">{component.detail}</p>
                </div>
                <div class="flex shrink-0 items-center gap-2 rounded-full border border-border/50 bg-panel px-2.5 py-1">
                  <span class="h-2 w-2 rounded-full {component.status === 'online' ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.4)]' : 'bg-brand shadow-[0_0_8px_rgba(177,33,66,0.4)]'}"></span>
                  <span class="text-[10px] font-semibold uppercase tracking-wider {component.status === 'online' ? 'text-emerald-500' : 'text-brand'}">{component.status}</span>
                </div>
              </div>
            {/each}
          </div>
        </div>

        <!-- Active Profiles Table -->
        <div class="flex flex-col flex-1 min-h-0 rounded-[1.25rem] border border-border bg-panel shadow-panel backdrop-blur">
          <div class="bg-surface/35 p-4 border-b border-border shrink-0">
            <h2 class="text-sm font-semibold uppercase tracking-[0.16em] text-ink-3">Active Profiles</h2>
          </div>
          <div class="flex-1 min-h-0 overflow-y-auto relative">
            <table class="w-full text-left text-xs">
              <thead class="sticky top-0 bg-surface/90 backdrop-blur text-ink-3 shadow-sm z-10">
                <tr>
                  <th class="px-4 py-3 font-medium uppercase tracking-wider">Satellite</th>
                  <th class="px-4 py-3 font-medium uppercase tracking-wider">Decoder</th>
                  <th class="px-4 py-3 font-medium uppercase tracking-wider">Model Status</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-border">
                {#each summary.active_satellites as sat}
                  <tr class="transition-colors hover:bg-surface/80">
                    <td class="px-4 py-3">
                      <span class="font-semibold text-ink">{sat.name}</span>
                      <span class="ml-2 rounded border border-border bg-surface px-1.5 py-0.5 text-[9px] text-ink-3">NORAD {sat.norad_id}</span>
                    </td>
                    <td class="px-4 py-3 text-ink-2">{sat.decoder || 'Generic'}</td>
                    <td class="px-4 py-3">
                      <span class="inline-flex items-center gap-1 rounded bg-panel border border-border px-1.5 py-0.5 text-[10px] font-medium {sat.model.status === 'ready' ? 'text-emerald-500' : 'text-brand'}">
                        <span class="h-1.5 w-1.5 rounded-full {sat.model.status === 'ready' ? 'bg-emerald-500' : 'bg-brand'}"></span>
                        {sat.model.status}
                      </span>
                    </td>
                  </tr>
                {/each}
                {#if summary.active_satellites.length === 0}
                  <tr>
                    <td colspan="3" class="px-4 py-8 text-center text-ink-3">No satellites active.</td>
                  </tr>
                {/if}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- Right Col (Throughput & Anomalies) -->
      <div class="flex flex-col gap-6 min-h-0">
        <!-- Throughput Sparkline -->
        {#if summary.throughput_buckets && summary.throughput_buckets.length > 0}
          <div class="flex-none chart-card border border-border rounded-[1.25rem] bg-panel p-4 shadow-sm backdrop-blur">
            <div class="flex items-center justify-between mb-2">
              <h2 class="text-sm font-semibold uppercase tracking-[0.16em] text-ink-3">Throughput (24h)</h2>
              <div class="text-right">
                <span class="text-xl font-semibold tracking-tight text-ink">
                  {summary.throughput_buckets.reduce((s, b) => s + b.frame_count, 0).toLocaleString()}
                </span>
                <span class="text-[10px] uppercase tracking-wider text-ink-3 ml-1">total frames</span>
              </div>
            </div>
            <div class="h-16 w-full">
              <SparklinePlot data={summary.throughput_buckets} width={900} height={60} />
            </div>
          </div>
        {/if}

        <!-- Recent Anomalies -->
        <div class="flex flex-col flex-1 min-h-0 rounded-[1.25rem] border border-border bg-panel shadow-panel backdrop-blur overflow-hidden">
          <div class="bg-surface/35 p-4 border-b border-border shrink-0 flex items-center justify-between">
            <h2 class="text-sm font-semibold uppercase tracking-[0.16em] text-ink-3">Recent Anomalies</h2>
            <span class="text-xs font-mono text-ink-3">{summary.recent_anomalies.length} recorded</span>
          </div>
          <div class="flex-1 min-h-0 overflow-y-auto p-5">
            {#if summary.recent_anomalies.length === 0}
              <div class="flex h-full items-center justify-center p-6 text-center text-sm text-ink-3">
                No recent anomalies detected.
              </div>
            {:else}
              <div class="grid grid-cols-1 md:grid-cols-2 2xl:grid-cols-3 gap-4">
                {#each summary.recent_anomalies as anomaly}
                  {@const severityHex = getSeverityColor(anomaly.score)}
                  <a href="/dashboard/ml" class="group relative overflow-hidden rounded-xl border bg-surface/20 p-5 transition-all duration-300 hover:-translate-y-1 hover:shadow-md" style="border-color: {severityHex};">
                    <div class="absolute inset-0 opacity-5" style="background-color: {severityHex};"></div>
                    <div class="relative flex items-center justify-between mb-4">
                      <span class="rounded-md px-2 py-0.5 text-[10px] font-bold tracking-widest border" style="color: {severityHex}; border-color: {severityHex}; background-color: color-mix(in srgb, {severityHex} 15%, transparent);">NORAD {anomaly.norad_id}</span>
                      <span class="text-[10px] font-medium text-ink-3 bg-surface/50 px-2 py-0.5 rounded border border-border/50">{new Date(anomaly.timestamp).toLocaleTimeString()}</span>
                    </div>
                    <div class="relative flex items-baseline justify-between mt-auto">
                      <div class="flex flex-col">
                        <span class="text-3xl font-bold tracking-tight text-ink transition-colors group-hover:text-[var(--hover-color)]" style="--hover-color: {severityHex};">{anomaly.score?.toFixed(2) || 'N/A'}</span>
                        <span class="text-[9px] font-semibold uppercase tracking-wider text-ink-3">Reconstruction Score</span>
                      </div>
                      <span class="text-xs font-medium text-ink-3">{new Date(anomaly.timestamp).toLocaleDateString()}</span>
                    </div>
                  </a>
                {/each}
              </div>
            {/if}
          </div>
        </div>
      </div>
    </div>
  </section>
{:else}
  <div class="flex h-full items-center justify-center">
    <div class="relative flex h-12 w-12 items-center justify-center">
      <div class="absolute inset-0 rounded-full border-4 border-surface border-t-brand animate-spin"></div>
      <div class="h-2 w-2 rounded-full bg-brand"></div>
    </div>
  </div>
{/if}