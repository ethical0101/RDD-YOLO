export type ClassCode = 'D00' | 'D10' | 'D20' | 'D40'
export type SeverityLevel = 'LOW' | 'MEDIUM' | 'HIGH'
export type LocationSource = 'browser' | 'exif' | 'manual' | 'route'

export interface DetectionResult {
  class_id: number
  class_code: ClassCode
  class_name: string
  confidence: number
  bbox: [number, number, number, number]
  bbox_norm: [number, number, number, number]
  severity: SeverityLevel
  severity_score: number
  severity_detail: Record<string, number | string>
}

export interface ImageInferenceResponse {
  detections: DetectionResult[]
  inference_ms: number
  width: number
  height: number
  model_version: string
  location: { lat: number; lon: number; source: LocationSource; accuracy_m: number | null } | null
  saved: boolean
  inference_id: number | null
  annotated_url?: string
  image_url?: string
  annotated_base64?: string
  exif_gps: { lat: number; lon: number } | null
}

export interface StoredDetection {
  id: number
  inference_id: number
  kind: string | null
  class_code: ClassCode
  class_name: string
  confidence: number
  bbox: number[]
  severity: SeverityLevel
  severity_score: number
  severity_detail: Record<string, number> | null
  latitude: number | null
  longitude: number | null
  location_source: LocationSource | null
  frame_index: number | null
  video_time_s: number | null
  crop_url: string | null
  image_url: string | null
  model_version: string
  created_at: string
}

export interface Inference {
  id: number
  kind: string
  status: string
  progress: number
  error: string | null
  source_filename: string | null
  image_url: string | null
  annotated_url: string | null
  video_url: string | null
  output_video_url: string | null
  width: number | null
  height: number | null
  latitude: number | null
  longitude: number | null
  location_source: LocationSource | null
  model_version: string
  conf_threshold: number
  inference_ms: number | null
  num_detections: number
  extra: Record<string, any> | null
  created_at: string
  detections?: StoredDetection[]
}

export interface Stats {
  total_detections: number
  total_inferences: number
  inferences_by_kind: Record<string, number>
  geolocated_detections: number
  per_class: Record<ClassCode, { name: string; count: number; avg_confidence: number | null; severity: Record<SeverityLevel, number> }>
  average_confidence: number | null
  severity: Record<SeverityLevel, number>
  confidence_histogram: { bin: string; count: number }[]
  over_time: ({ date: string; total: number } & Record<ClassCode, number>)[]
}

export interface Health {
  status: string
  model_loaded: boolean
  model_version: string
  model_error: string | null
  database: string
  cuda_available: boolean
  version: string
}

export interface MetricSet { precision: number; recall: number; f1: number; mAP50: number; mAP50_95: number }

export interface ExperimentSummary {
  name: string
  title: string | null
  experiment: string | null
  status: string
  epochs_completed: number
  epochs_requested: number | null
  train_time_hours: number | null
  best_val: (MetricSet & { epoch: number }) | null
  test_metrics: MetricSet | null
  has_weights: boolean
}

export interface Plot { name: string; url: string }

export interface EvalMetrics {
  overall: MetricSet
  per_class: Record<string, MetricSet & { name: string; instances?: number }>
  parameters: number
  gflops: number | null
  model_size_mb: number
  speed_benchmark: { fps_end_to_end: number; fps_inference_only: number; inference_ms_mean: number; end_to_end_ms_mean: number; device: string; images: number }
  images: number
  gpu: string
  evaluated_utc: string
}

export interface ExperimentDetail {
  name: string
  run_info: Record<string, any> | null
  history: Record<string, number>[]
  plots: Plot[]
  evaluations: Record<string, { metrics: EvalMetrics | null; plots: Plot[] }>
}
