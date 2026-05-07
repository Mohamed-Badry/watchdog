<script lang="ts">
  import "../app.css";
  import { page } from "$app/stores";
  import AntennaBackground from "$lib/components/AntennaBackground.svelte";

  let { children } = $props();

  const nav = [
    { href: "/", label: "Overview" },
    { href: "/dashboard", label: "Dashboard" },
    { href: "/team", label: "Team" },
  ];

  // ── Theme toggle ──────────────────────────────────────────────────────────
  // Dark is the default (no class on <html>).  Adding class="light" switches.
  let isLight = $state(false);

  function toggleTheme() {
    isLight = !isLight;
    document.documentElement.classList.toggle("light", isLight);
  }

  // ── Antenna Theme Configuration ───────────────────────────────────────────
  //
  // Dark mode (mix-blend-mode: screen):
  //   Uses lighter colors. The shader's additive output creates a holographic glow.
  // Light mode (mix-blend-mode: multiply):
  //   Uses darker colors. The shader output creates precise shadows/darkening.
  let antennaColorOff = $derived(isLight ? "#a0b0c0" : "#1c2a3e");
  let antennaColorOn = $derived(isLight ? "#3a5068" : "#7eb8da");
  let antennaBeamColor = $derived(isLight ? "#8a1833" : "#B12142");

  // Interaction Geometry
  let antennaMaxDist = $derived(isLight ? 375 : 300);
  let antennaBeamWidth = $derived(isLight ? 125 : 150);
  let antennaSignalFadeScale = $derived(isLight ? 2.0 : 1.0);
  let antennaBaseFadeScale = $derived(isLight ? 1.8 : 1.0); // Linear multiplier (e.g., 1.5 = fades out 50% further away)

  // Show the antenna field on landing + team, not on dashboard
  const showAntennaBg = $derived(!$page.url.pathname.startsWith("/dashboard"));
</script>

<svelte:head>
  <title>Project Watchdog</title>
  <meta
    name="description"
    content="Project Watchdog is a Bun-powered SvelteKit interface for real-time amateur satellite telemetry monitoring."
  />
</svelte:head>

{#if showAntennaBg}
  <AntennaBackground
    lightMode={isLight}
    colorOff={antennaColorOff}
    colorOn={antennaColorOn}
    beamColor={antennaBeamColor}
    maxDist={antennaMaxDist}
    beamWidth={antennaBeamWidth}
    signalFadeScale={antennaSignalFadeScale}
    antennaFadeScale={antennaBaseFadeScale}
  />
{/if}
<div class="min-h-screen">
  <div
    class="mx-auto flex min-h-screen w-full max-w-6xl flex-col px-6 py-6 sm:px-8 lg:px-10"
  >
    <header
      class="mb-10 rounded-[2rem] border border-border bg-panel px-5 py-4 shadow-panel backdrop-blur sm:px-6"
    >
      <div
        class="flex flex-col gap-4 md:flex-row md:items-center md:justify-between"
      >
        <div>
          <a
            class="text-lg font-semibold tracking-[0.18em] text-muted uppercase"
            href="/"
          >
            Project Watchdog
          </a>
          <p class="mt-1 max-w-xl text-sm text-ink-3">
            A low-latency operator shell for live telemetry, anomaly scoring,
            and replay-driven validation.
          </p>
        </div>

        <nav class="flex flex-wrap items-center gap-2 text-sm font-medium">
          {#each nav as item}
            <a
              class="rounded-full border border-border px-4 py-2 text-ink-2 transition hover:border-brand hover:bg-surface hover:text-brand"
              href={item.href}
            >
              {item.label}
            </a>
          {/each}

          <button
            onclick={toggleTheme}
            class="ml-1 rounded-full border border-border px-3 py-2 text-xs text-ink-3 transition hover:border-brand hover:text-brand"
            title="Toggle light / dark mode"
          >
            {isLight ? "☽" : "☀"}
          </button>
        </nav>
      </div>
    </header>

    <main class="flex-1">
      {@render children()}
    </main>
  </div>
</div>
