import type { PageLoad } from './$types';
import { env } from '$env/dynamic/public';

export const load: PageLoad = async ({ fetch }) => {
    const apiUrl = typeof window !== 'undefined' ? (env.PUBLIC_API_URL || 'http://127.0.0.1:8000') : 'http://backend:8000';
    
    try {
        const [satsRes, analyticsRes] = await Promise.all([
            fetch(`${apiUrl}/api/satellites`),
            fetch(`${apiUrl}/api/analytics`)
        ]);
        
        let satellites = [];
        let analytics = null;
        
        if (satsRes.ok) {
            const data = await satsRes.json();
            satellites = data.satellites;
        }
        
        if (analyticsRes.ok) {
            analytics = await analyticsRes.json();
        }

        return {
            satellites,
            analytics
        };
    } catch (e) {
        console.error("Error fetching analytics data:", e);
        return {
            satellites: [],
            analytics: null,
            error: "Could not connect to the backend API."
        };
    }
};