// leaflet.markercluster and leaflet.heat are classic plugins that extend a global `L`.
// Importing this module first (before the plugins) makes that global available.
import L from 'leaflet'

;(window as unknown as { L: typeof L }).L = L
export default L
