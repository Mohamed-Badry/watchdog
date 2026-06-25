<script lang="ts">
  import { ChevronDown } from "lucide-svelte";
  import { fly, fade } from "svelte/transition";
  import { onMount } from "svelte";

  let {
    id = "",
    value = $bindable(),
    options = [],
    class: className = "",
    labelClass = "",
    onchange,
  } = $props<{
    id?: string;
    value: string | number;
    options: { value: string | number; label: string }[];
    class?: string;
    labelClass?: string;
    onchange?: (value: string | number) => void;
  }>();

  let isOpen = $state(false);
  let selectContainer: HTMLElement;

  function toggle() {
    isOpen = !isOpen;
  }

  function selectOption(optValue: string | number) {
    value = optValue;
    isOpen = false;
    if (onchange) {
      onchange(optValue);
    }
  }

  function handleClickOutside(event: MouseEvent) {
    if (isOpen && selectContainer && !selectContainer.contains(event.target as Node)) {
      isOpen = false;
    }
  }

  onMount(() => {
    document.addEventListener("click", handleClickOutside);
    return () => {
      document.removeEventListener("click", handleClickOutside);
    };
  });

  let selectedLabel = $derived(options.find((o: {value: string | number, label: string}) => o.value == value)?.label ?? value);
</script>

<div class="relative {isOpen ? 'z-50' : 'z-10'}" bind:this={selectContainer}>
  <button
    type="button"
    {id}
    onclick={toggle}
    class="flex items-center justify-between gap-2 {className} cursor-pointer group"
  >
    <span class="truncate {labelClass}">{selectedLabel}</span>
    <ChevronDown
      class="size-4 shrink-0 text-ink-3 transition-transform duration-300 group-hover:text-brand {isOpen ? 'rotate-180 text-brand' : ''}"
    />
  </button>

  {#if isOpen}
    <div
      in:fly={{ y: -10, duration: 200, opacity: 0 }}
      out:fade={{ duration: 150 }}
      class="absolute left-0 top-full z-50 mt-1 w-full min-w-fit overflow-hidden rounded-xl border border-brand/30 bg-panel/95 backdrop-blur-xl shadow-[0_10px_40px_rgba(0,0,0,0.3)]"
    >
      <div class="flex max-h-60 flex-col overflow-y-auto py-1 custom-scrollbar">
        {#each options as option}
          <button
            type="button"
            onclick={() => selectOption(option.value)}
            class="relative flex w-full items-center px-4 py-2.5 text-left text-sm transition-all duration-200 group/btn
              {value == option.value
              ? 'bg-brand/10 text-brand font-semibold pl-5'
              : 'text-ink-2 hover:bg-brand/5 hover:text-brand hover:pl-5'}"
          >
            <div class="absolute left-0 top-0 bottom-0 w-1 bg-brand transition-transform duration-200 origin-left
              {value == option.value ? 'scale-x-100' : 'scale-x-0 group-hover/btn:scale-x-100 opacity-50'}"></div>
            <span class="relative">{option.label}</span>
          </button>
        {/each}
      </div>
    </div>
  {/if}
</div>

<style>
  .custom-scrollbar::-webkit-scrollbar {
    width: 4px;
  }
  .custom-scrollbar::-webkit-scrollbar-track {
    background: transparent;
  }
  .custom-scrollbar::-webkit-scrollbar-thumb {
    background: rgba(139, 92, 246, 0.3);
    border-radius: 4px;
  }
  .custom-scrollbar::-webkit-scrollbar-thumb:hover {
    background: rgba(139, 92, 246, 0.5);
  }
</style>
