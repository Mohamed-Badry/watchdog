<script lang="ts">
  import { page } from '$app/stores';
</script>

<div class="flex min-h-[70vh] flex-col items-center justify-center p-6 text-center">
  <div class="rounded-[2rem] border border-brand/20 bg-surface/50 p-10 backdrop-blur-md max-w-xl">
    <div class="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-full bg-brand/10 text-brand">
      <svg class="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
    </div>
    
    <h1 class="text-4xl font-bold tracking-tight text-ink">
      {$page.status} Error
    </h1>
    
    <p class="mt-4 text-lg text-ink-2">
      {$page.error?.message || 'An unexpected internal server error occurred.'}
    </p>

    <div class="mt-8">
      <a href="/" class="inline-flex items-center justify-center rounded-xl bg-brand px-6 py-3 text-sm font-semibold text-white shadow-lg transition hover:bg-brand/90 hover:shadow-brand/20 focus:outline-none focus:ring-2 focus:ring-brand focus:ring-offset-2 focus:ring-offset-surface">
        Return to Dashboard
      </a>
    </div>
  </div>
</div>