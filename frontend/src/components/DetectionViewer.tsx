import { useState } from 'react'
import { CLASS_COLORS } from '../lib/format'
import type { DetectionResult } from '../lib/types'

/** Renders an image with SVG bounding boxes (interactive: hover highlights a box). */
export default function DetectionViewer({ src, width, height, detections, highlight, onHover }: {
  src: string; width: number; height: number; detections: DetectionResult[]
  highlight?: number | null; onHover?: (i: number | null) => void
}) {
  const [local, setLocal] = useState<number | null>(null)
  const active = highlight ?? local
  const stroke = Math.max(2, Math.round(Math.min(width, height) / 250))
  const font = Math.max(12, Math.round(Math.min(width, height) / 40))
  return (
    <div className="relative w-full overflow-hidden rounded-lg bg-slate-900">
      <img src={src} alt="Analysed road" className="block h-auto w-full" />
      <svg viewBox={`0 0 ${width} ${height}`} className="absolute inset-0 h-full w-full">
        {detections.map((d, i) => {
          const [x1, y1, x2, y2] = d.bbox
          const color = CLASS_COLORS[d.class_code]
          const dim = active !== null && active !== i
          const label = `${d.class_code} ${(d.confidence * 100).toFixed(0)}%`
          return (
            <g key={i} opacity={dim ? 0.3 : 1} onMouseEnter={() => { setLocal(i); onHover?.(i) }}
              onMouseLeave={() => { setLocal(null); onHover?.(null) }} style={{ cursor: 'pointer' }}>
              <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill={active === i ? `${color}22` : 'transparent'}
                stroke={color} strokeWidth={active === i ? stroke * 1.6 : stroke} rx={2} />
              <rect x={x1} y={Math.max(0, y1 - font * 1.4)} width={label.length * font * 0.62 + 8} height={font * 1.4} fill={color} rx={2} />
              <text x={x1 + 4} y={Math.max(font * 1.05, y1 - font * 0.35)} fontSize={font} fill="#fff" fontWeight={600}>{label}</text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
