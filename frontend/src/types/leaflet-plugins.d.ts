declare module 'leaflet.heat'

import 'leaflet'
declare module 'leaflet' {
  function heatLayer(latlngs: [number, number, number?][], options?: Record<string, unknown>): Layer & {
    setLatLngs(latlngs: [number, number, number?][]): void
  }
}
