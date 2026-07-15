import type { PaginatedResponse, BaseEntity, DateTime } from './common';

export type TrainingJobStatus = 'pending' | 'queued' | 'running' | 'paused' | 'completed' | 'failed' | 'stopped' | 'cancelled';
export type TrainingModelType = 'transformer' | 'cnn' | 'rnn' | 'lstm' | 'gru' | 'bert' | 'gpt' | 'vision' | 'multimodal' | 'custom';
export type OptimizerType = 'adam' | 'adamw' | 'sgd' | 'rmsprop' | 'adagrad' | 'adadelta' | 'adamax' | 'lamb';
export type SchedulerType = 'constant' | 'linear' | 'cosine' | 'cosine_with_restarts' | 'polynomial' | 'exponential' | 'reduce_on_plateau';
export type CheckpointStatus = 'available' | 'restoring' | 'archived' | 'corrupted';
export type DatasetType = 'text' | 'image' | 'audio' | 'video' | 'multimodal' | 'tabular' | 'custom';
export type DatasetFormat = 'json' | 'jsonl' | 'csv' | 'parquet' | 'tfrecord' | 'hdf5' | 'custom';
export type SearchStrategy = 'grid' | 'random' | 'bayesian' | 'hyperband' | 'evolutionary';

export interface TrainingConfig {
  epochs: number;
  batch_size: number;
  learning_rate: number;
  weight_decay: number;
  warmup_steps: number;
  max_grad_norm: number;
  save_steps: number;
  eval_steps: number;
  logging_steps: number;
  optimizer: OptimizerType;
  scheduler: SchedulerType;
  fp16: boolean;
  bf16: boolean;
  gradient_accumulation_steps: number;
  max_seq_length?: number;
  custom_params: Record<string, unknown>;
}

export interface ResourceConfig {
  gpu_count: number;
  gpu_type?: string;
  cpu_count: number;
  memory_gb: number;
  storage_gb: number;
  distributed: boolean;
  nodes: number;
}

export interface TrainingJobCreateRequest {
  name: string;
  description?: string;
  model_type: TrainingModelType;
  base_model?: string;
  dataset_id: string;
  config: TrainingConfig;
  resources: ResourceConfig;
  tags?: string[];
}

export interface TrainingProgress {
  current_epoch: number;
  total_epochs: number;
  current_step: number;
  total_steps: number;
  progress_percent: number;
  estimated_remaining_seconds?: number;
}

export interface TrainingMetrics {
  loss: number;
  learning_rate: number;
  throughput_samples_per_sec: number;
  gpu_utilization_percent?: number;
  memory_used_gb?: number;
  custom_metrics: Record<string, number>;
}

export interface TrainingJob extends BaseEntity {
  id: string;
  name: string;
  description?: string;
  model_type: TrainingModelType;
  base_model?: string;
  dataset_id: string;
  status: TrainingJobStatus;
  config: TrainingConfig;
  resources: ResourceConfig;
  tags: string[];
  progress: TrainingProgress;
  current_metrics: TrainingMetrics;
  best_metrics?: Record<string, number>;
  output_path?: string;
  error_message?: string;
  started_at?: DateTime;
  completed_at?: DateTime;
  created_by?: string;
}

export interface TrainingLogEntry {
  id: string;
  job_id: string;
  level: string;
  message: string;
  step?: number;
  epoch?: number;
  metrics?: Record<string, number>;
  timestamp: DateTime;
}

export interface Checkpoint {
  id: string;
  job_id: string;
  job_name: string;
  step: number;
  epoch: number;
  metrics: Record<string, unknown>;
  path: string;
  size_mb: number;
  status: CheckpointStatus;
  created_at: DateTime;
}

export interface Dataset {
  id: string;
  name: string;
  description?: string;
  dataset_type: DatasetType;
  format: DatasetFormat;
  size_bytes: number;
  num_samples: number;
  num_features?: number;
  schema?: Record<string, unknown>;
  tags: string[];
  storage_path: string;
  created_by?: string;
}

export interface DatasetUploadRequest {
  name: string;
  description?: string;
  dataset_type: DatasetType;
  format: DatasetFormat;
  tags?: string[];
}

export interface HyperparameterSearchRequest {
  name: string;
  model_type: TrainingModelType;
  dataset_id: string;
  search_space: Record<string, unknown>;
  strategy?: SearchStrategy;
  max_trials?: number;
  metric?: string;
  direction?: string;
  resources: ResourceConfig;
}

export interface TrialResult {
  trial_id: string;
  params: Record<string, unknown>;
  metrics: Record<string, number>;
  status: string;
  duration_seconds: number;
}

export interface HyperparameterSearch {
  id: string;
  name: string;
  status: TrainingJobStatus;
  strategy: SearchStrategy;
  current_trial: number;
  max_trials: number;
  best_trial?: TrialResult;
  trials: TrialResult[];
  created_at: DateTime;
}

export type TrainingJobListResponse = PaginatedResponse<TrainingJob>;
export type DatasetListResponse = PaginatedResponse<Dataset>;
export type CheckpointListResponse = PaginatedResponse<Checkpoint>;
