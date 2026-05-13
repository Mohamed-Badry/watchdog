<script lang="ts">
  type StationLocation = {
    id?: string;
    label: string;
    lat: number;
    lon: number;
    elevationM: number;
  };

  type TrackPoint = {
    lat: number;
    lon: number;
    elevation?: number;
    time?: string | Date;
  };

  let {
    location,
    presets = [],
    selectedTrack = [],
    onLocationChange,
  } = $props<{
    location: StationLocation;
    presets?: StationLocation[];
    selectedTrack?: TrackPoint[];
    onLocationChange?: (location: StationLocation) => void;
  }>();

  const mapWidth = 720;
  const mapHeight = 360;
  const longitudeTicks = [-120, -60, 0, 60, 120];
  const latitudeTicks = [-60, -30, 0, 30, 60];

  function clamp(value: number, min: number, max: number) {
    return Math.min(Math.max(value, min), max);
  }

  function roundCoord(value: number) {
    return Math.round(value * 10000) / 10000;
  }

  function project(lat: number, lon: number) {
    return {
      x: ((lon + 180) / 360) * mapWidth,
      y: ((90 - lat) / 180) * mapHeight,
    };
  }

  function segmentPoints(points: TrackPoint[]) {
    const cleanPoints = points.filter((point) => Number.isFinite(point.lat) && Number.isFinite(point.lon));
    const segments: { x: number; y: number; point: TrackPoint }[][] = [];

    for (const point of cleanPoints) {
      const projected = { ...project(point.lat, point.lon), point };
      const currentSegment = segments.at(-1);
      const previous = currentSegment?.at(-1);

      if (!currentSegment || (previous && Math.abs(projected.x - previous.x) > mapWidth / 2)) {
        segments.push([projected]);
      } else {
        currentSegment.push(projected);
      }
    }

    return segments;
  }

  function pointsAttribute(points: { x: number; y: number }[]) {
    return points.map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(' ');
  }

  function chooseLocation(nextLocation: StationLocation) {
    onLocationChange?.({
      ...nextLocation,
      lat: roundCoord(nextLocation.lat),
      lon: roundCoord(nextLocation.lon),
      elevationM: Number.isFinite(nextLocation.elevationM) ? nextLocation.elevationM : 0,
    });
  }

  function handleMapClick(event: MouseEvent) {
    const svg = event.currentTarget as SVGSVGElement;
    const rect = svg.getBoundingClientRect();
    const x = clamp((event.clientX - rect.left) / rect.width, 0, 1);
    const y = clamp((event.clientY - rect.top) / rect.height, 0, 1);

    chooseLocation({
      id: 'custom',
      label: 'Custom Ground Station',
      lat: roundCoord(90 - y * 180),
      lon: roundCoord(x * 360 - 180),
      elevationM: location.elevationM,
    });
  }

  function handleKeydown(event: KeyboardEvent) {
    const step = event.shiftKey ? 5 : 1;
    let nextLat = location.lat;
    let nextLon = location.lon;

    if (event.key === 'ArrowUp') nextLat += step;
    else if (event.key === 'ArrowDown') nextLat -= step;
    else if (event.key === 'ArrowLeft') nextLon -= step;
    else if (event.key === 'ArrowRight') nextLon += step;
    else return;

    event.preventDefault();
    chooseLocation({
      ...location,
      id: 'custom',
      label: 'Custom Ground Station',
      lat: clamp(nextLat, -90, 90),
      lon: clamp(nextLon, -180, 180),
    });
  }

  function handlePresetClick(event: MouseEvent, preset: StationLocation) {
    event.stopPropagation();
    chooseLocation(preset);
  }

  let selectedPoint = $derived(project(location.lat, location.lon));
  let trackSegments = $derived(segmentPoints(selectedTrack ?? []));
</script>

<div class="space-y-3">
  <div class="overflow-hidden rounded-2xl border border-border bg-surface/60">
    <!-- svelte-ignore a11y_click_events_have_key_events -->
    <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
    <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <svg
      class="block aspect-[2/1] w-full cursor-crosshair touch-none select-none"
      viewBox={`0 0 ${mapWidth} ${mapHeight}`}
      role="application"
      tabindex="0"
      aria-label="Ground station location map"
      onclick={handleMapClick}
      onkeydown={handleKeydown}
    >
      <rect width={mapWidth} height={mapHeight} fill="rgba(15, 23, 42, 0.18)" />
      <rect x="0" y="0" width={mapWidth} height={mapHeight} fill="url(#ops-map-grid)" opacity="0.45" />

      <defs>
        <pattern id="ops-map-grid" width="72" height="36" patternUnits="userSpaceOnUse">
          <path d="M 72 0 L 0 0 0 36" fill="none" stroke="var(--color-border)" stroke-width="1" />
        </pattern>
        <filter id="station-glow" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {#each longitudeTicks as lonTick}
        {@const x = project(0, lonTick).x}
        <line x1={x} x2={x} y1="0" y2={mapHeight} stroke="var(--color-border)" stroke-width="1" opacity="0.6" />
      {/each}
      {#each latitudeTicks as latTick}
        {@const y = project(latTick, 0).y}
        <line x1="0" x2={mapWidth} y1={y} y2={y} stroke="var(--color-border)" stroke-width="1" opacity="0.6" />
      {/each}

      <g fill="var(--color-ink-3)" opacity="0.38">
        <path d="M86 118 C104 76 151 62 190 78 C224 92 243 126 232 163 C219 207 182 216 158 199 C133 181 113 190 91 169 C74 152 74 135 86 118 Z" />
        <path d="M197 211 C226 204 251 224 254 262 C257 303 227 331 203 321 C177 309 180 273 191 249 C197 236 187 225 197 211 Z" />
        <path d="M316 94 C352 66 415 70 452 98 C489 127 492 169 456 185 C423 200 382 185 349 192 C316 199 288 176 288 143 C288 123 300 106 316 94 Z" />
        <path d="M366 190 C391 184 421 205 426 237 C432 275 409 313 377 304 C349 296 338 263 348 232 C354 214 350 198 366 190 Z" />
        <path d="M462 98 C506 73 566 88 596 128 C624 165 605 206 562 207 C529 208 507 188 477 188 C444 187 431 151 445 124 C449 114 455 105 462 98 Z" />
        <path d="M575 217 C608 200 653 216 665 251 C678 288 651 318 617 311 C587 305 568 276 574 245 C576 236 570 225 575 217 Z" />
        <path d="M277 314 C327 299 386 300 444 311 C503 322 582 315 631 332 C559 351 417 357 302 346 C247 341 223 327 277 314 Z" />
      </g>

      {#if trackSegments.length > 0}
        <g fill="none" stroke-linecap="round" stroke-linejoin="round">
          {#each trackSegments as segment}
            {#if segment.length > 1}
              <polyline
                points={pointsAttribute(segment)}
                stroke="var(--color-ok)"
                stroke-width="3"
                stroke-opacity="0.88"
              />
              <polyline
                points={pointsAttribute(segment)}
                stroke="var(--color-brand)"
                stroke-width="1.2"
                stroke-opacity="0.95"
                stroke-dasharray="6 5"
              />
            {/if}
          {/each}
        </g>
      {/if}

      {#each presets as preset}
        {@const point = project(preset.lat, preset.lon)}
        <!-- svelte-ignore a11y_click_events_have_key_events -->
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <g class="cursor-pointer" onclick={(event) => handlePresetClick(event, preset)}>
          <circle cx={point.x} cy={point.y} r="8" fill="var(--color-paper)" stroke="var(--color-ink-3)" stroke-width="1.5" opacity="0.9" />
          <circle cx={point.x} cy={point.y} r="3.5" fill="var(--color-ink-3)" />
          <title>{preset.label}</title>
        </g>
      {/each}

      <g filter="url(#station-glow)">
        <circle cx={selectedPoint.x} cy={selectedPoint.y} r="13" fill="var(--color-brand)" opacity="0.22" />
        <circle cx={selectedPoint.x} cy={selectedPoint.y} r="7" fill="var(--color-brand)" stroke="var(--color-paper)" stroke-width="2" />
      </g>
    </svg>
  </div>

  <div class="flex flex-wrap items-center justify-between gap-3 text-xs text-ink-3">
    <span class="font-medium text-ink-2">{location.label}</span>
    <span class="font-mono">
      {location.lat.toFixed(4)} lat, {location.lon.toFixed(4)} lon, {Math.round(location.elevationM)} m
    </span>
  </div>
</div>
