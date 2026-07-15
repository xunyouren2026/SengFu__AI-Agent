/**
 * 胜复学算法API集成
 * 
 * 集成天衡/Pendulum AGI框架的核心算法
 */

import { apiClient } from './client';

// =============================================================================
// Swing Engine API
// =============================================================================

export interface SwingExecuteRequest {
  observation: any;
  training?: boolean;
  force_exploration?: boolean;
}

export interface SwingActionResponse {
  action: string;
  confidence: number;
  expert_id: string;
  timestamp: string;
}

export async function swingExecute(request: SwingExecuteRequest) {
  const response = await apiClient.post('/algorithms/swing/execute', request);
  return response.data;
}

export async function swingStats() {
  const response = await apiClient.get('/algorithms/swing/stats');
  return response.data;
}

// =============================================================================
// Reflexion API
// =============================================================================

export interface ReflectionRequest {
  task_id: string;
  task_description: string;
  outcome: string;
  evaluation: number;
  use_llm?: boolean;
  tags?: string[];
}

export interface ReflectionResponse {
  reflection_id: string;
  summary: string;
  reflection_type: string;
  priority: string;
  timestamp: string;
}

export async function reflexionThink(request: ReflectionRequest) {
  const response = await apiClient.post('/algorithms/reflection/think', request);
  return response.data;
}

export async function reflexionHistory(task_id?: string, limit: number = 10) {
  const params = task_id ? { task_id, limit } : { limit };
  const response = await apiClient.get('/algorithms/reflection/history', { params });
  return response.data;
}

// =============================================================================
// Confidence & Halt API
// =============================================================================

export interface ConfidenceRequest {
  observation: any;
  prediction: any;
  actual?: any;
}

export interface ConfidenceResponse {
  confidence: number;
  is_reliable: boolean;
  temperature_scaled: number;
  entropy: number;
}

export async function calculateConfidence(request: ConfidenceRequest) {
  const response = await apiClient.post('/algorithms/confidence/calc', request);
  return response.data;
}

export interface HaltRequest {
  observations: any[];
  threshold?: number;
}

export interface HaltResponse {
  should_halt: boolean;
  halt_reason: string;
  confidence: number;
  iterations: number;
}

export async function checkHalt(request: HaltRequest) {
  const response = await apiClient.post('/algorithms/halt/check', request);
  return response.data;
}

// =============================================================================
// Balance & Depression API
// =============================================================================

export interface BalanceAdjustRequest {
  metric: string;
  value: number;
  target: number;
}

export interface BalanceResponse {
  depression_level: string;
  depression_score: number;
  adjustment_needed: number;
  recommendation: string;
  trigger_burst: boolean;
}

export async function balanceAdjust(request: BalanceAdjustRequest) {
  const response = await apiClient.post('/algorithms/balance/adjust', request);
  return response.data;
}

export async function balanceStatus() {
  const response = await apiClient.get('/algorithms/balance/status');
  return response.data;
}

// =============================================================================
// Goal & Trigger API
// =============================================================================

export interface GoalValidateRequest {
  goal: string;
  current_state?: Record<string, any>;
}

export interface GoalResponse {
  is_valid: boolean;
  safety_level: string;
  constraints_satisfied: boolean;
  estimated_difficulty: number;
}

export async function goalValidate(request: GoalValidateRequest) {
  const response = await apiClient.post('/algorithms/goal/validate', request);
  return response.data;
}

export interface BurstTriggerRequest {
  depression_score: number;
  context?: Record<string, any>;
  action_type?: string;
}

export interface BurstActionResponse {
  action_performed: string;
  effect: string;
  new_state: Record<string, any>;
}

export async function triggerBurst(request: BurstTriggerRequest) {
  const response = await apiClient.post('/algorithms/trigger/burst', request);
  return response.data;
}

// =============================================================================
// Intrinsic Motivation API
// =============================================================================

export interface MotivationRequest {
  current_state: any;
  next_state: any;
  motivation_type: string;
  extra_info?: Record<string, any>;
}

export interface MotivationResponse {
  intrinsic_reward: number;
  motivation_type: string;
  novelty_score: number;
  exploration_bonus: number;
}

export async function calculateMotivation(request: MotivationRequest) {
  const response = await apiClient.post('/algorithms/motivation/calc', request);
  return response.data;
}

// =============================================================================
// MoE API
// =============================================================================

export interface MoERouteRequest {
  input_data: any;
  available_experts?: string[];
  temperature?: number;
}

export interface MoERouteResponse {
  selected_expert: string;
  routing_weights: Record<string, number>;
  confidence: number;
}

export async function moeRoute(request: MoERouteRequest) {
  const response = await apiClient.post('/algorithms/moe/route', request);
  return response.data;
}

export async function expertsList() {
  const response = await apiClient.get('/algorithms/experts/list');
  return response.data;
}

// =============================================================================
// Memory API
// =============================================================================

export interface MemoryRequest {
  key: string;
  value: any;
  memory_type?: string;
}

export interface MemoryResponse {
  success: boolean;
  memory_id: string;
  retrieval_results?: any[];
}

export async function memoryHot(request: MemoryRequest) {
  const response = await apiClient.post('/algorithms/memory/hot', request);
  return response.data;
}

export async function memoryWarm(request: MemoryRequest) {
  const response = await apiClient.post('/algorithms/memory/warm', request);
  return response.data;
}

export async function memoryCold(request: MemoryRequest) {
  const response = await apiClient.post('/algorithms/memory/cold', request);
  return response.data;
}

// =============================================================================
// Stats API
// =============================================================================

export async function algorithmsStats() {
  const response = await apiClient.get('/algorithms/stats');
  return response.data;
}

export async function algorithmsHealth() {
  const response = await apiClient.get('/algorithms/health');
  return response.data;
}

// =============================================================================
// 胜复学闭环 - 组合调用
// =============================================================================

export interface PendulumCycleRequest {
  observation: any;
  goal: string;
  current_depression?: number;
}

export interface PendulumCycleResponse {
  action: string;
  confidence: number;
  depression_score: number;
  should_halt: boolean;
  intrinsic_reward: number;
  reflection_summary?: string;
}

/**
 * 执行完整的胜复学闭环
 * 
 * 1. 计算郁值
 * 2. 检查是否停止
 * 3. 执行Swing动作
 * 4. 计算内在奖励
 * 5. 生成反思
 */
export async function pendulumCycle(request: PendulumCycleRequest): Promise<PendulumCycleResponse> {
  const results: any = {};
  
  // 1. 郁值检测 (Balance)
  const balanceResult = await balanceAdjust({
    metric: 'error_rate',
    value: request.current_depression || 0.3,
    target: 0.1
  });
  results.depression_score = balanceResult.data?.depression_score || 0.3;
  
  // 2. 停止检测 (Halt)
  const haltResult = await checkHalt({
    observations: [request.observation],
    threshold: 0.5
  });
  results.should_halt = haltResult.data?.should_halt || false;
  
  // 3. 执行Swing动作
  const swingResult = await swingExecute({
    observation: request.observation,
    training: true
  });
  results.action = swingResult.data?.action || 'continue';
  results.confidence = swingResult.data?.confidence || 0.5;
  
  // 4. 内在动机
  const motivationResult = await calculateMotivation({
    current_state: request.observation,
    next_state: { action: results.action },
    motivation_type: 'novelty'
  });
  results.intrinsic_reward = motivationResult.data?.intrinsic_reward || 0.1;
  
  return {
    action: results.action,
    confidence: results.confidence,
    depression_score: results.depression_score,
    should_halt: results.should_halt,
    intrinsic_reward: results.intrinsic_reward
  };
}

// =============================================================================
// 导出所有API
// =============================================================================

export const algorithmsAPI = {
  // Swing
  swingExecute,
  swingStats,
  
  // Reflexion
  reflexionThink,
  reflexionHistory,
  
  // Confidence & Halt
  calculateConfidence,
  checkHalt,
  
  // Balance & Depression
  balanceAdjust,
  balanceStatus,
  
  // Goal & Trigger
  goalValidate,
  triggerBurst,
  
  // Motivation
  calculateMotivation,
  
  // MoE
  moeRoute,
  expertsList,
  
  // Memory
  memoryHot,
  memoryWarm,
  memoryCold,
  
  // Stats
  algorithmsStats,
  algorithmsHealth,
  
  // Cycle
  pendulumCycle,
};

export default algorithmsAPI;
