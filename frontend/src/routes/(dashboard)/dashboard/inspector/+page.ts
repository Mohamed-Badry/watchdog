import type { PageLoad } from './$types';
import { fetchSatellites } from '$lib/api';

export const load: PageLoad = async ({ fetch }) => {
    try {
        const { satellites } = await fetchSatellites(fetch);
        return { satellites };
    } catch (e) {
        console.error('Error fetching inspector data:', e);
        return { satellites: [], error: 'Could not connect to the backend API.' };
    }
};