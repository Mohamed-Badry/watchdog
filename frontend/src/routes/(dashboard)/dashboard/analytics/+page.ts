import type { PageLoad } from './$types';
import { apiFetch, fetchSatellites } from '$lib/api';

export const load: PageLoad = async ({ fetch }) => {
    try {
        const [satsData, analytics] = await Promise.all([
            fetchSatellites(fetch),
            apiFetch<any>('/api/analytics', undefined, fetch)
        ]);

        return {
            satellites: satsData.satellites,
            analytics
        };
    } catch (e: any) {
        console.error("Error fetching analytics data:", e);
        return {
            satellites: [],
            analytics: null,
            error: e.message || "Could not connect to the backend API."
        };
    }
};