<script lang="ts">
  import { Plot } from 'svelteplot';
  
  let { children, height, square = false, ...rest } = $props<{
    children?: import('svelte').Snippet;
    height?: number;
    square?: boolean;
    [key: string]: any;
  }>();
  
  let innerWidth = $state(0);
  let width = $state(0);
  
  // Calculate a proportional height for narrow screens to prevent vertical stretching,
  // unless 'square' is true, which forces a 1:1 aspect ratio.
  let computedHeight = $derived.by(() => {
    if (width === 0) return height;
    
    if (square) {
      return width;
    }
    
    // Only scale down if the actual screen is narrow (mobile/tablet),
    // not just because the container is narrow (e.g., CSS grid on desktop).
    if (height && innerWidth > 0 && innerWidth < 768) {
      // Scale down proportionally based on container width
      return Math.max(150, Math.round(height * (width / 768)));
    }
    return height;
  });

  let computedWidth = $derived.by(() => {
    if (width === 0) return 0;
    
    if (square) {
      return width;
    }
    
    return width;
  });
</script>

<svelte:window bind:innerWidth={innerWidth} />

<div class="w-full h-full min-w-0 flex items-center justify-center" bind:clientWidth={width}>
  {#if width > 0}
    <Plot width={computedWidth} height={computedHeight} {...rest}>
      {@render children?.()}
    </Plot>
  {:else}
    <!-- Placeholder to maintain height while measuring -->
    <div style="height: {height ?? 300}px; width: 100%;"></div>
  {/if}
</div>
