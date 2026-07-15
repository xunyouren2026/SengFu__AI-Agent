import apiClient from './client';
import { wrapListResponse } from './utils';
import type { CognitiveStateSchema, StateHistoryItem, ReflectionTriggerRequest, ReflectionResponse, ReflectionRecord, MemoryEntry, MemorySearchRequest, MemorySearchResponse, MemoryForgetRequest, EmotionStateSchema, Goal, GoalCreateRequest, GoalUpdateRequest, CognitiveMetric, MemoryListResponse, ReflectionListResponse, GoalListResponse } from '../types/cognitive';

export const cognitiveApi = {
  getState: () =>
    apiClient.get<{ current_state: CognitiveStateSchema; previous_state?: CognitiveStateSchema; state_duration_seconds: number }>('/cognitive/state'),
  getStateHistory: (params?: { limit?: number; hours?: number }) =>
    apiClient.get<{ history: StateHistoryItem[]; total_transitions: number }>('/cognitive/state/history', { params }),
  triggerReflection: (data: ReflectionTriggerRequest) =>
    apiClient.post<ReflectionResponse>('/cognitive/reflection', data),
  getReflections: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get<ReflectionListResponse>('/cognitive/reflections', { params });
    return wrapListResponse<ReflectionRecord>(response);
  },
  getMemory: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get<MemoryListResponse>('/cognitive/memory', { params });
    return wrapListResponse<MemoryEntry>(response);
  },
  searchMemory: (data: MemorySearchRequest) =>
    apiClient.post<MemorySearchResponse>('/cognitive/memory/search', data),
  forgetMemory: (data: MemoryForgetRequest) =>
    apiClient.post('/cognitive/memory/forget', data),
  getEmotions: (params?: { hours?: number }) =>
    apiClient.get<{ current_emotion: EmotionStateSchema; emotion_history: { emotion: string; intensity: number; timestamp: string }[] }>('/cognitive/emotions', { params }),
  getGoals: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get<GoalListResponse>('/cognitive/goals', { params });
    return wrapListResponse<Goal>(response);
  },
  createGoal: (data: GoalCreateRequest) =>
    apiClient.post<Goal>('/cognitive/goals', data),
  updateGoal: (id: string, data: GoalUpdateRequest) =>
    apiClient.put<Goal>(`/cognitive/goals/${id}`, data),
  deleteGoal: (id: string) =>
    apiClient.delete(`/cognitive/goals/${id}`),
  getMetrics: (params?: { period?: string }) =>
    apiClient.get<{ overall_score: number; metrics: CognitiveMetric[] }>('/cognitive/metrics', { params }),
  getWebSocketUrl: () =>
    `${apiClient.defaults.baseURL?.replace('/api/v1', '') || ''}/api/v1/cognitive/ws/state`,
};
