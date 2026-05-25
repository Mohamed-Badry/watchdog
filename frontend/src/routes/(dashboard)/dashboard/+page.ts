import type { PageLoad } from './$types';
import { apiFetch } from '$lib/api';
import type { DashboardSummary } from '$lib/types/api';

export const load: PageLoad = async ({ fetch }) => {
    try {
        const [summary, systemStatus] = await Promise.all([
            apiFetch<DashboardSummary>('/api/dashboard/summary', undefined, fetch),
            apiFetch('/api/status', undefined, fetch),
        ]);

        return { summary, systemStatus };
    } catch (e) {
        console.error('Error fetching dashboard data:', e);
        return {
            summary: null,
            systemStatus: null,
            error: 'Could not connect to the backend API.',
        };
    }
};
