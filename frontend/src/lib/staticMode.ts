/** Static (GitHub Pages) build: the model runs in the browser and data lives in IndexedDB. */
export const STATIC = import.meta.env.VITE_STATIC === '1'

const BASE = import.meta.env.BASE_URL.replace(/\/$/, '')

/** Prefix site-relative asset URLs (/files/..., /data/...) with the deployment base path. */
export function asset(url: string | null | undefined): string | undefined {
  if (!url) return undefined
  if (url.startsWith('data:') || url.startsWith('blob:') || /^https?:/.test(url)) return url
  return url.startsWith('/') ? `${BASE}${url}` : url
}
