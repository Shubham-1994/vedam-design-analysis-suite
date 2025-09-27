<script lang="ts">
  import { onMount } from 'svelte';
  import Header from './lib/components/Header.svelte';
  import UploadSection from './lib/components/UploadSection.svelte';
  import AnalysisResults from './lib/components/AnalysisResults.svelte';
  import AnalysisDashboard from './lib/components/AnalysisDashboard.svelte';
  import LoadingSpinner from './lib/components/LoadingSpinner.svelte';
  import { analysisStore } from './lib/stores/analysisStore';

  let currentView: 'upload' | 'loading' | 'results' | 'dashboard' = 'upload';

  onMount(() => {
    // Subscribe to analysis store changes
    analysisStore.subscribe((state) => {
      if (state.status === 'idle') {
        currentView = 'upload';
      } else if (state.status === 'uploading' || state.status === 'analyzing') {
        currentView = 'loading';
      } else if (state.status === 'completed') {
        currentView = 'results';
      } else if (state.status === 'error') {
        currentView = 'upload';
      }
    });

    // Listen for dashboard navigation events
    document.addEventListener('viewAnalysis', () => {
      currentView = 'results';
    });
    
    // Listen for home navigation events
    document.addEventListener('goHome', () => {
      currentView = 'upload';
    });
  });

  function handleNewAnalysis() {
    analysisStore.reset();
    currentView = 'upload';
  }

  function showDashboard() {
    currentView = 'dashboard';
  }
</script>

<div class="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
  <Header on:showDashboard={showDashboard} />
  
  <main class="container mx-auto px-4 py-8">
    {#if currentView === 'upload'}
      <div class="animate-fade-in">
        <UploadSection />
      </div>
    {:else if currentView === 'loading'}
      <div class="animate-fade-in">
        <LoadingSpinner />
      </div>
    {:else if currentView === 'results'}
      <div class="animate-fade-in">
        <AnalysisResults on:newAnalysis={handleNewAnalysis} />
      </div>
    {:else if currentView === 'dashboard'}
      <div class="animate-fade-in">
        <AnalysisDashboard />
      </div>
    {/if}
  </main>
</div>

<style>
  :global(html) {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }
</style>
