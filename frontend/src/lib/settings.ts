/** Per-browser preferences (localStorage). Defaults match the backend defaults. */
export interface Settings {
  defaultConf: number
  preferExif: boolean
  saveResults: boolean
}

const KEY = 'rdd-yolo-settings'
export const DEFAULTS: Settings = { defaultConf: 0.25, preferExif: true, saveResults: true }

export function loadSettings(): Settings {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(KEY) ?? '{}') }
  } catch {
    return { ...DEFAULTS }
  }
}

export function saveSettings(s: Settings) {
  try { localStorage.setItem(KEY, JSON.stringify(s)) } catch { /* storage unavailable: keep defaults */ }
}
