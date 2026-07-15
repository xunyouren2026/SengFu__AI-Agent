/**
 * 训练算法API集成
 * 
 * 集成RLHF、DPO、PPO、爆发动作等训练算法
 */

import { apiClient } from './client';

// =============================================================================
// RLHF API
// =============================================================================

export interface RLHFStartRequest {
  model_name: string;
  dataset_path: string;
  beta?: number;
  lr?: number;
  epochs?: number;
  batch_size?: number;
}

export interface RLHFStatusResponse {
  job_id: string;
  status: string;
  progress: number;
  loss: number;
  kl_div: number;
  reward: number;
  eta_seconds: number;
}

export async function rlhfStart(request: RLHFStartRequest) {
  const response = await apiClient.post('/training-algo/rlhf/start', request);
  return response.data;
}

export async function rlhfStatus(jobId: string) {
  const response = await apiClient.get(`/training-algo/rlhf/${jobId}/status`);
  return response.data;
}

export async function rlhfStop(jobId: string) {
  const response = await apiClient.post(`/training-algo/rlhf/${jobId}/stop`);
  return response.data;
}

// =============================================================================
// DPO API
// =============================================================================

export interface DPOStartRequest {
  model_name: string;
  dataset_path: string;
  beta?: number;
  lr?: number;
  epochs?: number;
  variant?: string;
}

export interface DPOStatusResponse {
  job_id: string;
  status: string;
  progress: number;
  loss: number;
  accuracy: number;
  eta_seconds: number;
}

export async function dpoStart(request: DPOStartRequest) {
  const response = await apiClient.post('/training-algo/dpo/start', request);
  return response.data;
}

export async function dpoStatus(jobId: string) {
  const response = await apiClient.get(`/training-algo/dpo/${jobId}/status`);
  return response.data;
}

// =============================================================================
// PPO API
// =============================================================================

export interface PPOStartRequest {
  env_name: string;
  model_name: string;
  lr?: number;
  gamma?: number;
  gae_lambda?: number;
  clip_epsilon?: number;
  epochs?: number;
}

export interface PPOStatusResponse {
  job_id: string;
  status: string;
  episode: number;
  reward_mean: number;
  reward_std: number;
  value_loss: number;
  policy_loss: number;
}

export async function ppoStart(request: PPOStartRequest) {
  const response = await apiClient.post('/training-algo/ppo/start', request);
  return response.data;
}

export async function ppoStatus(jobId: string) {
  const response = await apiClient.get(`/training-algo/ppo/${jobId}/status`);
  return response.data;
}

// =============================================================================
// Burst Action API
// =============================================================================

export interface BurstExecuteRequest {
  depression_score: number;
  action_type?: string;
  context?: Record<string, any>;
}

export interface BurstActionResponse {
  action: string;
  effect: string;
  new_depression: number;
  success: boolean;
}

export async function burstExecute(request: BurstExecuteRequest) {
  const response = await apiClient.post('/training-algo/burst/execute', request);
  return response.data;
}

export async function burstActions() {
  const response = await apiClient.get('/training-algo/burst/actions');
  return response.data;
}

// =============================================================================
// Distillation API
// =============================================================================

export interface DistillStartRequest {
  teacher_model: string;
  student_model: string;
  dataset_path: string;
  temperature?: number;
  alpha?: number;
  lr?: number;
}

export interface DistillStatusResponse {
  job_id: string;
  status: string;
  progress: number;
  ce_loss: number;
  kl_loss: number;
  total_loss: number;
}

export async function distillStart(request: DistillStartRequest) {
  const response = await apiClient.post('/training-algo/distill/start', request);
  return response.data;
}

export async function distillStatus(jobId: string) {
  const response = await apiClient.get(`/training-algo/distill/${jobId}/status`);
  return response.data;
}

// =============================================================================
// Job Management
// =============================================================================

export async function listTrainingJobs(status?: string) {
  const params = status ? { status } : {};
  const response = await apiClient.get('/training-algo/jobs', { params });
  return response.data;
}

export async function deleteTrainingJob(jobId: string) {
  const response = await apiClient.delete(`/training-algo/jobs/${jobId}`);
  return response.data;
}

// =============================================================================
// Health
// =============================================================================

export async function trainingHealth() {
  const response = await apiClient.get('/training-algo/health');
  return response.data;
}

// =============================================================================
// 导出
// =============================================================================

export const trainingAPI = {
  // RLHF
  rlhfStart,
  rlhfStatus,
  rlhfStop,
  
  // DPO
  dpoStart,
  dpoStatus,
  
  // PPO
  ppoStart,
  ppoStatus,
  
  // Burst
  burstExecute,
  burstActions,
  
  // Distillation
  distillStart,
  distillStatus,
  
  // Jobs
  listTrainingJobs,
  deleteTrainingJob,
  
  // Health
  trainingHealth,
};

export default trainingAPI;
