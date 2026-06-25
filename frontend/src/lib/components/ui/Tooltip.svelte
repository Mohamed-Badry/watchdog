<script lang="ts">
  import { Info } from "lucide-svelte";
  
  let { text, class: className = "", align = "center" } = $props<{ 
    text: string; 
    class?: string;
    align?: "left" | "center" | "right";
  }>();
</script>

<!-- A pure CSS tooltip using group-hover. No JS state required. -->
<div class="group relative flex items-center {className}">
  <Info class="size-3 text-ink-3 hover:text-brand transition-colors cursor-pointer" />
  
  <!-- Tooltip Bubble (Desktop) -->
  <div class="hidden sm:block pointer-events-none invisible absolute top-full z-50 mt-2 opacity-0 transition-all duration-200 group-hover:visible group-hover:opacity-100
    {align === 'center' ? 'left-1/2 -translate-x-1/2' : ''}
    {align === 'left' ? 'left-0' : ''}
    {align === 'right' ? 'right-0' : ''}
  ">
    <div class="w-max max-w-[250px] rounded-lg border border-border bg-panel px-3 py-2 text-[11px] leading-relaxed text-ink shadow-lg backdrop-blur-md">
      {text}
      <!-- Triangle Pointer -->
      <div class="absolute -top-1 h-2 w-2 rotate-45 border-l border-t border-border bg-panel
        {align === 'center' ? 'left-1/2 -translate-x-1/2' : ''}
        {align === 'left' ? 'left-3' : ''}
        {align === 'right' ? 'right-3' : ''}
      "></div>
    </div>
  </div>

  <!-- Tooltip Bubble (Mobile Toast Style) -->
  <div class="sm:hidden pointer-events-none invisible fixed bottom-24 left-1/2 -translate-x-1/2 z-[100] w-[90vw] max-w-sm opacity-0 transition-all duration-300 group-hover:visible group-hover:opacity-100 group-active:visible group-active:opacity-100">
    <div class="rounded-xl border border-brand/30 bg-panel/95 px-4 py-3 text-xs font-medium leading-relaxed text-ink shadow-[0_10px_40px_rgba(0,0,0,0.3)] backdrop-blur-xl text-center">
      <span class="text-brand font-bold mr-1">Info:</span> {text}
    </div>
  </div>
</div>