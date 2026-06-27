import type { PageLoad } from './$types';
import { apiFetch, fetchSatellites } from '$lib/api';

export const load: PageLoad = async ({ fetch, url }) => {
    try {
        const noradId = url.searchParams.get('norad_id') || '43880';
        const query = noradId !== 'all' ? `?norad_id=${noradId}` : '';
        
        const [satsData, analytics] = await Promise.all([
            fetchSatellites(fetch),
            apiFetch<any>(`/api/analytics${query}`, undefined, fetch)
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