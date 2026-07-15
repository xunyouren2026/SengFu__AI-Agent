import apiClient from './client';
import { wrapListResponse } from './utils';
import type { Strategy, StrategyCreateRequest, StrategyUpdateRequest, RoutingRule, RoutingRuleCreateRequest, RoutingRuleUpdateRequest, LoadBalancer, LoadBalancerCreateRequest, CircuitBreaker, StrategyListResponse, RoutingRuleListResponse } from '../types/orchestration';

export const orchestrationApi = {
  getStrategies: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get<StrategyListResponse>('/orchestration/strategies', { params });
    return wrapListResponse<Strategy>(response);
  },
  createStrategy: (data: StrategyCreateRequest) =>
    apiClient.post<Strategy>('/orchestration/strategies', data),
  getStrategy: (id: string) =>
    apiClient.get<Strategy>(`/orchestration/strategies/${id}`),
  updateStrategy: (id: string, data: StrategyUpdateRequest) =>
    apiClient.put<Strategy>(`/orchestration/strategies/${id}`, data),
  deleteStrategy: (id: string) =>
    apiClient.delete(`/orchestration/strategies/${id}`),
  enableStrategy: (id: string) =>
    apiClient.post<Strategy>(`/orchestration/strategies/${id}/enable`),
  disableStrategy: (id: string) =>
    apiClient.post<Strategy>(`/orchestration/strategies/${id}/disable`),
  getRoutingRules: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get<RoutingRuleListResponse>('/orchestration/routing-rules', { params });
    return wrapListResponse<RoutingRule>(response);
  },
  createRoutingRule: (data: RoutingRuleCreateRequest) =>
    apiClient.post<RoutingRule>('/orchestration/routing-rules', data),
  updateRoutingRule: (id: string, data: RoutingRuleUpdateRequest) =>
    apiClient.put<RoutingRule>(`/orchestration/routing-rules/${id}`, data),
  deleteRoutingRule: (id: string) =>
    apiClient.delete(`/orchestration/routing-rules/${id}`),
  getLoadBalancers: async () => {
    const response = await apiClient.get('/orchestration/load-balancers');
    return wrapListResponse<LoadBalancer>(response);
  },
  createLoadBalancer: (data: LoadBalancerCreateRequest) =>
    apiClient.post<LoadBalancer>('/orchestration/load-balancers', data),
  getCircuitBreakers: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get('/orchestration/circuit-breakers', { params });
    return wrapListResponse<CircuitBreaker>(response);
  },
  resetCircuitBreaker: (id: string) =>
    apiClient.post<CircuitBreaker>(`/orchestration/circuit-breakers/${id}/reset`),
  getMetrics: (params?: { period?: string }) =>
    apiClient.get('/orchestration/metrics', { params }),
  testRouting: (data: { message: string; context?: Record<string, unknown> }) =>
    apiClient.post('/orchestration/routing/test', data),
};
