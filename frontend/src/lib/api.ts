import { useCallback, useEffect, useState } from 'react'
import { STATIC } from './staticMode'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function parse<T>(res: Response): Promise<T> {
  if (res.status === 204) return undefined as T
  const text = await res.text()
  let body: any = null
  try { body = text ? JSON.parse(text) : null } catch { body = text }
  if (!res.ok) {
    const detail = body?.detail
    const msg = typeof detail === 'string' ? detail : Array.isArray(detail)
      ? detail.map((d: any) => `${d.loc?.slice(-1)[0] ?? ''}: ${d.msg}`).join('; ')
      : `HTTP ${res.status}`
    throw new ApiError(res.status, msg)
  }
  return body as T
}

export async function apiGet<T>(path: string, params?: Record<string, string | number | boolean | undefined | null>): Promise<T> {
  const qs = params
    ? '?' + new URLSearchParams(Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== '')
        .map(([k, v]) => [k, String(v)])).toString()
    : ''
  if (STATIC) return (await (await import('./staticApi')).staticRequest('GET', `${path}${qs}`)) as T
  return parse<T>(await fetch(`/api${path}${qs}`))
}

export async function apiSend<T>(path: string, method: 'POST' | 'PATCH' | 'DELETE', body?: FormData | object): Promise<T> {
  if (STATIC) return (await (await import('./staticApi')).staticRequest(method, path, body)) as T
  const init: RequestInit = { method }
  if (body instanceof FormData) init.body = body
  else if (body) {
    init.body = JSON.stringify(body)
    init.headers = { 'Content-Type': 'application/json' }
  }
  return parse<T>(await fetch(`/api${path}`, init))
}

/** Minimal data-fetching hook with loading / error / reload. */
export function useApi<T>(path: string | null, params?: Record<string, any>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState<boolean>(!!path)
  const key = JSON.stringify(params ?? {})

  const load = useCallback(async () => {
    if (!path) return
    setLoading(true)
    try {
      setData(await apiGet<T>(path, params))
      setError(null)
    } catch (e: any) {
      setError(e.message ?? 'Request failed')
    } finally {
      setLoading(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, key, ...deps])

  useEffect(() => { load() }, [load])
  return { data, error, loading, reload: load, setData }
}
