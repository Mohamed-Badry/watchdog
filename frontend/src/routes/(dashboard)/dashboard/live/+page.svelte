<script lang="ts">
  import { env } from "$env/dynamic/public";
  import type { PageData } from "./$types";
  import { untrack } from "svelte";

  import AnomalyTimelinePlot from "$lib/components/charts/AnomalyTimelinePlot.svelte";

  let { data }: { data: PageData } = $props();

  let satellites = $derived(data.satellites || []);
  let error = $derived(data.error);

  let noradId = $state<string>("all");
  let limit = $state<number>(25);
  let isLive = $state<boolean>(true);

  let frames = $state<any[]>([]);
  let loading = $state(false);
  let selectedTimestamp = $state<string | null>(null);

  async function fetchRecent() {
    loading = true;
    const apiUrl =
      typeof window !== "undefined"
        ? env.PUBLIC_API_URL || "http://127.0.0.1:8000"
        : "http://backend:8000";
    let url = `${apiUrl}/api/telemetry/recent?limit=${limit}`;
    if (noradId !== "all") {
      url += `&norad_id=${noradId}`;
    }
    try {
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        frames = data.frames || [];
      }
    } catch (e) {
      console.error("Failed to fetch recent telemetry", e);
    } finally {
      loading = false;
    }
  }

  // Effect to re-fetch when parameters change
  $effect(() => {
    // Track parameters
    noradId;
    limit;
    untrack(() => fetchRecent());
  });

  // Effect to handle live polling
  $effect(() => {
    if (!isLive) return;
    const interval = setInterval(() => {
      untrack(() => fetchRecent());
    }, 5000);
    return () => clearInterval(interval);
  });
</script>

<section class="flex h-full min-h-0 flex-col gap-4 animate-in fade-in slide-in-from-bottom-4 duration-500 ease-out">
  <!-- 1. HEADER: Title & Global Controls -->
  <header class="flex flex-none flex-wrap items-center justify-between gap-4">
    <div class="space-y-1">
      <p class="text-xs font-semibold uppercase tracking-[0.2em] text-muted">Real-Time Ingress</p>
      <h1 class="text-3xl font-semibold tracking-tight text-ink">Live Watcher</h1>
    </div>

    <div class="flex items-center gap-4">
      <!-- Satellite Filter -->
      <div class="flex items-center gap-2">
        <label for="live-sat-select" class="text-xs font-semibold uppercase tracking-wider text-ink-3">Sat Filter</label>
        <select
          id="live-sat-select"
          bind:value={noradId}
          class="rounded-lg border border-border bg-surface px-3 py-1.5 text-sm text-ink outline-none transition hover:border-brand focus:border-brand"
        >
          <option value="all">All</option>
          {#each satellites as sat}
            <option value={sat.norad_id.toString()}>{sat.norad_id}</option>
          {/each}
        </select>
      </div>

      <!-- Feed Limit -->
      <div class="flex items-center gap-2">
        <label for="live-feed-size" class="text-xs font-semibold uppercase tracking-wider text-ink-3">Limit</label>
        <select
          id="live-feed-size"
          bind:value={limit}
          class="rounded-lg border border-border bg-surface px-3 py-1.5 text-sm text-ink outline-none transition hover:border-brand focus:border-brand"
        >
          <option value={10}>10</option>
          <option value={25}>25</option>
          <option value={50}>50</option>
        </select>
      </div>

      <!-- Live Sync Toggle -->
      <button
        onclick={() => (isLive = !isLive)}
        class="flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-semibold transition-all hover:scale-105 {isLive
          ? 'border-emerald-500/50 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 shadow-sm shadow-emerald-500/10'
          : 'border-border bg-surface text-ink-3 hover:border-brand hover:text-brand'}"
      >
        <span class="relative flex h-2 w-2">
          {#if isLive}
            <span class="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75"></span>
          {/if}
          <span class="relative inline-flex h-2 w-2 rounded-full {isLive ? 'bg-emerald-500' : 'bg-ink-3'}"></span>
        </span>
        {isLive ? "Sync Active" : "Paused"}
      </button>
    </div>
  </header>

  <!-- 2. MAIN CONTENT: Error, Feed, or Placeholder -->
  {#if error}
    <div class="flex-none rounded-xl border border-brand/50 bg-brand/10 p-4 text-sm text-brand">
      {error}
    </div>
  {:else}
    <!-- TELEMETRY FEED (Scrollable Area) -->
    <div class="flex min-h-0 flex-1 flex-col rounded-[1.25rem] border border-border bg-panel shadow-panel backdrop-blur">
      <div class="shrink-0 border-b border-border bg-surface/35 p-3">
        <h2 class="text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-3">Telemetry Feed</h2>
      </div>

      <div class="flex-1 overflow-y-auto p-4 min-h-0">
        {#if loading && frames.length === 0}
          <div class="flex h-full items-center justify-center py-12">
            <div class="h-8 w-8 animate-spin rounded-full border-2 border-surface border-t-brand"></div>
          </div>
        {:else if frames.length === 0}
          <div class="rounded-lg border border-border border-dashed p-12 text-center text-sm text-ink-3">
            No telemetry packets received for the current filter.
          </div>
        {:else}
          <div class="grid grid-cols-1 gap-5 xl:grid-cols-2">
            {#each frames as frame (frame.timestamp + frame.norad_id)}
              <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
              <article 
                class="group relative flex flex-col gap-2 overflow-hidden rounded-xl border p-4 transition-all hover:border-brand/30 cursor-pointer {selectedTimestamp === frame.timestamp ? 'border-highlight bg-highlight/5 shadow-md shadow-highlight/10' : 'border-border bg-surface/50'}"
                onclick={() => selectedTimestamp = selectedTimestamp === frame.timestamp ? null : frame.timestamp}
              >
                <!-- Status Indicator Bar -->
                <div class="absolute inset-y-0 left-0 w-1 {frame.model?.is_anomaly ? 'bg-brand shadow-[0_0_8px_rgba(177,33,66,0.8)]' : 'bg-emerald-500/50'}"></div>

                <!-- Card Header: NORAD, Quality, Timestamp, Score -->
                <div class="flex items-start justify-between gap-3 pl-2">
                  <div class="space-y-1.5">
                    <div class="flex items-center gap-2">
                      <span class="rounded border border-border bg-panel px-2 py-1 font-mono text-xs font-bold text-ink-3">NORAD {frame.norad_id}</span>
                      <span class="text-xs font-medium {frame.quality?.frame_is_complete ? 'text-emerald-500' : 'text-brand'}">
                        {frame.quality?.frame_is_complete ? "Complete" : "Partial"}
                      </span>
                    </div>
                    <p class="font-medium text-base text-ink">{new Date(frame.timestamp).toLocaleTimeString()}</p>
                  </div>

                  <div class="text-right">
                    <p class="text-xs font-semibold uppercase tracking-wider text-ink-3">Anomaly Score</p>
                    <p class="text-2xl font-bold tracking-tight {frame.model?.is_anomaly ? 'text-brand' : 'text-ink'}">
                      {frame.model?.anomaly_score !== null ? frame.model.anomaly_score.toFixed(2) : "-"}
                    </p>
                  </div>
                </div>

                <!-- Feature Grid (Top 4) -->
                {#if frame.features}
                  {@const fKeys = Object.keys(frame.features).slice(0, 4)}
                  <div class="mt-2 grid grid-cols-4 gap-3 border-t border-border/50 pl-2 pt-3">
                    {#each fKeys as key}
                      <div class="flex flex-col min-w-0">
                        <span class="truncate text-xs font-semibold uppercase tracking-wider text-ink-3" title={key}>{key.replace(/_/g, " ")}</span>
                        <span class="font-mono text-base text-ink">
                          {frame.features[key] !== null ? Number(frame.features[key]).toFixed(2) : "-"}
                        </span>
                      </div>
                    {/each}
                  </div>
                {/if}
              </article>
            {/each}
          </div>
        {/if}
      </div>
    </div>

    <!-- ANOMALY TIMELINE (Pinned to Bottom) -->
    {#if frames.length > 0}
      {@const timelineFrames = frames
        .filter((f: any) => f.model?.anomaly_score != null)
        .map((f: any) => ({
          timestamp: f.timestamp,
          anomaly_score: f.model.anomaly_score,
          is_anomaly: f.model.is_anomaly,
        }))}
      
      {#if timelineFrames.length > 0}
        <div class="flex-none chart-card overflow-hidden">
          <p class="chart-card-title text-[11px] mb-2">Anomaly Score History</p>
          <div class="h-48 pb-2">
            <AnomalyTimelinePlot
              frames={timelineFrames}
              threshold={frames[0]?.model?.threshold ?? 0.3}
              {selectedTimestamp}
            />
          </div>
        </div>
      {/if}
    {/if}
  {/if}
</section>
