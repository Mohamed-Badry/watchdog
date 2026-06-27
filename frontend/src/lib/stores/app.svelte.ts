import { fetchSatellites } from '$lib/api';

export class AppStore {
    satellites = $state<{ name: string; norad_id: number }[]>([]);
    activeSatellite = $state<string>('');
    connectionStatus = $state<'connected' | 'disconnected' | 'connecting'>('disconnected');

    constructor() {}

    async loadSatellites() {
        const data = await fetchSatellites();
        this.satellites = data.satellites;
        if (this.satellites.length > 0 && !this.activeSatellite) {
            const uwe4 = this.satellites.find(s => s.norad_id === 43880);
            this.activeSatellite = uwe4 ? '43880' : this.satellites[0].norad_id.toString();
        }
    }

    setActiveSatellite(id: string) {
        this.activeSatellite = id;
    }

    setConnectionStatus(status: 'connected' | 'disconnected' | 'connecting') {
        this.connectionStatus = status;
    }
}

export const appStore = new AppStore();
