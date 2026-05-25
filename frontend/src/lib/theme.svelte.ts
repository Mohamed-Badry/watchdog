import { browser } from '$app/environment';

class Theme {
  isLight = $state(false);

  constructor() {
    if (browser) {
      // Read from the DOM class that was already set by the blocking script in app.html.
      // This avoids a hydration mismatch where Svelte state says "dark" but DOM says "light".
      this.isLight = document.documentElement.classList.contains('light');
      
      // Automatically sync DOM when state changes
      $effect.root(() => {
        $effect(() => {
          if (this.isLight) {
            document.documentElement.classList.add('light');
            localStorage.setItem('theme', 'light');
          } else {
            document.documentElement.classList.remove('light');
            localStorage.setItem('theme', 'dark');
          }
        });
      });
    }
  }

  toggle() {
    this.isLight = !this.isLight;
  }
}

export const themeState = new Theme();

export function toggleTheme() {
  themeState.toggle();
}
