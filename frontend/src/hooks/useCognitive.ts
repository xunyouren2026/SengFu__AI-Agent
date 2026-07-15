import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { cognitiveApi } from '../api/cognitive';
import type { ReflectionTriggerRequest, MemorySearchRequest, GoalCreateRequest, GoalUpdateRequest } from '../types/cognitive';

export const cognitiveKeys = {
  all: ['cognitive'] as const,
  state: () => [...cognitiveKeys.all, 'state'] as const,
  memory: () => [...cognitiveKeys.all, 'memory'] as const,
  goals: () => [...cognitiveKeys.all, 'goals'] as const,
  reflections: () => [...cognitiveKeys.all, 'reflections'] as const,
};

export const useCognitiveState = () =>
  useQuery({ queryKey: cognitiveKeys.state(), queryFn: () => cognitiveApi.getState(), refetchInterval: 10000 });

export const useMemory = (params?: Record<string, unknown>) =>
  useQuery({ queryKey: [...cognitiveKeys.memory(), params], queryFn: () => cognitiveApi.getMemory(params) });

export const useSearchMemory = () =>
  useMutation({ mutationFn: (data: MemorySearchRequest) => cognitiveApi.searchMemory(data) });

export const useTriggerReflection = () =>
  useMutation({ mutationFn: (data: ReflectionTriggerRequest) => cognitiveApi.triggerReflection(data) });

export const useGoals = (params?: Record<string, unknown>) =>
  useQuery({ queryKey: [...cognitiveKeys.goals(), params], queryFn: () => cognitiveApi.getGoals(params) });

export const useCreateGoal = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: (data: GoalCreateRequest) => cognitiveApi.createGoal(data), onSuccess: () => qc.invalidateQueries({ queryKey: cognitiveKeys.goals() }) });
};

export const useUpdateGoal = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: ({ id, data }: { id: string; data: GoalUpdateRequest }) => cognitiveApi.updateGoal(id, data), onSuccess: () => qc.invalidateQueries({ queryKey: cognitiveKeys.goals() }) });
};

export const useDeleteGoal = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: (id: string) => cognitiveApi.deleteGoal(id), onSuccess: () => qc.invalidateQueries({ queryKey: cognitiveKeys.goals() }) });
};

export const useCognitiveMetrics = (period?: string) =>
  useQuery({ queryKey: [...cognitiveKeys.all, 'metrics', period], queryFn: () => cognitiveApi.getMetrics({ period }) });
