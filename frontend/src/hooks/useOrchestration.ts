import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { orchestrationApi } from '../api/orchestration';
import type { StrategyCreateRequest, StrategyUpdateRequest, RoutingRuleCreateRequest, RoutingRuleUpdateRequest, LoadBalancerCreateRequest } from '../types/orchestration';

export const orchestrationKeys = {
  all: ['orchestration'] as const,
  strategies: () => [...orchestrationKeys.all, 'strategies'] as const,
  routingRules: () => [...orchestrationKeys.all, 'routing-rules'] as const,
};

export const useStrategies = (params?: Record<string, unknown>) =>
  useQuery({ queryKey: [...orchestrationKeys.strategies(), params], queryFn: () => orchestrationApi.getStrategies(params), staleTime: 5000 });

export const useCreateStrategy = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: (data: StrategyCreateRequest) => orchestrationApi.createStrategy(data), onSuccess: () => qc.invalidateQueries({ queryKey: orchestrationKeys.strategies() }) });
};

export const useUpdateStrategy = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: ({ id, data }: { id: string; data: StrategyUpdateRequest }) => orchestrationApi.updateStrategy(id, data), onSuccess: () => qc.invalidateQueries({ queryKey: orchestrationKeys.strategies() }) });
};

export const useDeleteStrategy = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: (id: string) => orchestrationApi.deleteStrategy(id), onSuccess: () => qc.invalidateQueries({ queryKey: orchestrationKeys.strategies() }) });
};

export const useRoutingRules = (params?: Record<string, unknown>) =>
  useQuery({ queryKey: [...orchestrationKeys.routingRules(), params], queryFn: () => orchestrationApi.getRoutingRules(params) });

export const useCreateRoutingRule = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: (data: RoutingRuleCreateRequest) => orchestrationApi.createRoutingRule(data), onSuccess: () => qc.invalidateQueries({ queryKey: orchestrationKeys.routingRules() }) });
};

export const useDeleteRoutingRule = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: (id: string) => orchestrationApi.deleteRoutingRule(id), onSuccess: () => qc.invalidateQueries({ queryKey: orchestrationKeys.routingRules() }) });
};

export const useLoadBalancers = () =>
  useQuery({ queryKey: [...orchestrationKeys.all, 'load-balancers'], queryFn: () => orchestrationApi.getLoadBalancers() });

export const useCreateLoadBalancer = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: (data: LoadBalancerCreateRequest) => orchestrationApi.createLoadBalancer(data), onSuccess: () => qc.invalidateQueries({ queryKey: [...orchestrationKeys.all, 'load-balancers'] }) });
};

export const useCircuitBreakers = (params?: Record<string, unknown>) =>
  useQuery({ queryKey: [...orchestrationKeys.all, 'circuit-breakers', params], queryFn: () => orchestrationApi.getCircuitBreakers(params) });

export const useOrchestrationMetrics = (period?: string) =>
  useQuery({ queryKey: [...orchestrationKeys.all, 'metrics', period], queryFn: () => orchestrationApi.getMetrics({ period }) });
