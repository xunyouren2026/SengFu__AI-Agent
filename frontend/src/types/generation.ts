import type { DateTime } from './common';

export interface TTSRequest {
  text: string;
  voice_id?: string;
  language?: string;
  rate?: number;
  pitch?: number;
  engine?: string;
  output_format?: string;
}

export interface TTSResponse {
  success: boolean;
  audio_path: string;
  duration: number;
  format: string;
  voice_id: string;
}

export interface ImageGenerationRequest {
  prompt: string;
  negative_prompt?: string;
  width?: number;
  height?: number;
  num_inference_steps?: number;
  guidance_scale?: number;
  num_images?: number;
  seed?: number;
  engine?: string;
  model?: string;
}

export interface ImageGenerationResponse {
  success: boolean;
  images: string[];
  seed?: number;
  inference_time: number;
  model: string;
}

export interface VideoGenerationRequest {
  prompt: string;
  negative_prompt?: string;
  width?: number;
  height?: number;
  num_frames?: number;
  fps?: number;
  num_inference_steps?: number;
  guidance_scale?: number;
  seed?: number;
  engine?: string;
}

export interface VideoGenerationResponse {
  success: boolean;
  video_path: string;
  duration: number;
  num_frames: number;
  fps: number;
  inference_time: number;
}

export interface AudioGenerationRequest {
  prompt: string;
  negative_prompt?: string;
  duration?: number;
  num_inference_steps?: number;
  guidance_scale?: number;
  seed?: number;
  engine?: string;
}

export interface AudioGenerationResponse {
  success: boolean;
  audio_path: string;
  duration: number;
  sample_rate: number;
  inference_time: number;
}

export interface ThreeDGenerationRequest {
  prompt?: string;
  num_inference_steps?: number;
  guidance_scale?: number;
  seed?: number;
  engine?: string;
  remove_background?: boolean;
}

export interface ThreeDGenerationResponse {
  success: boolean;
  model_path: string;
  format: string;
  vertices: number;
  faces: number;
  inference_time: number;
}

export interface GenerationTask {
  id: string;
  type: string;
  status: string;
  progress: number;
  result?: Record<string, unknown>;
  error?: string;
  created_at: DateTime;
  completed_at?: DateTime;
}

export interface GenerationStats {
  total_tasks: number;
  completed_tasks: number;
  failed_tasks: number;
  running_tasks: number;
  avg_inference_time: number;
}
