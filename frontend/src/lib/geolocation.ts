import { useState } from 'react'

export interface BrowserFix { lat: number; lon: number; accuracy: number }

/** Wraps navigator.geolocation. Coordinates only exist if the user grants permission. */
export function useBrowserLocation() {
  const [fix, setFix] = useState<BrowserFix | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [pending, setPending] = useState(false)

  const request = () =>
    new Promise<BrowserFix | null>((resolve) => {
      if (!('geolocation' in navigator)) {
        setError('Geolocation is not supported by this browser')
        resolve(null)
        return
      }
      setPending(true)
      setError(null)
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          const f = { lat: pos.coords.latitude, lon: pos.coords.longitude, accuracy: pos.coords.accuracy }
          setFix(f)
          setPending(false)
          resolve(f)
        },
        (err) => {
          setError(err.code === err.PERMISSION_DENIED ? 'Location permission denied' : err.message)
          setPending(false)
          resolve(null)
        },
        { enableHighAccuracy: true, timeout: 15000, maximumAge: 30000 },
      )
    })

  return { fix, error, pending, request, clear: () => setFix(null) }
}
