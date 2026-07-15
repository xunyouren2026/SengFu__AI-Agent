import type { PaginatedResponse, BaseEntity, DateTime } from './common';

export type StrategyType = 'fallback' | 'load_balance' | 'routing' | 'cost_optimize' | 'quality_optimize' | 'custom';
export type StrategyStatus = 'active' | 'inactive' | 'error' | 'draft';
export type RoutingRuleType = 'model_priority' | 'cost_based' | 'latency_based' | 'quality_based' | 'capability_match' | 'content_based' | 'time_based' | 'user_based';
export type LoadBalanceAlgorithm = 'round_robin' | 'weighted_round_robin' | 'least_connections' | 'least_latency' | 'random' | 'hash';
export type CircuitBreakerState = 'closed' | 'open' | 'half_open';

export interface StrategyConfig {
  primary_model?: string;
  fallback_models: string[];
  timeout_seconds: number;
  retry_attempts: number;
  retry_delay_seconds: number;
  custom_params: Record<string, unknown>;
}

export interface StrategyCreateRequest {
  name: string;
  description?: string;
  type: StrategyType;
  config: StrategyConfig;
  priority?: number;
  tags?: string[];
}

export interface Strategy extends BaseEntity {
  id: string;
  name: string;
  description?: string;
  type: StrategyType;
  config: StrategyConfig;
  priority: number;
  tags: string[];
  status: StrategyStatus;
  is_active: boolean;
  execution_count: number;
  success_count: number;
  error_count: number;
  avg_execution_time_ms: number;
  created_by?: string;
}

export interface RoutingCondition {
  type: string;
  operator: string;
  value: unknown;
  field?: string;
}

export interface RoutingAction {
  target_model: string;
  weight?: number;
  priority?: number;
  transform_config?: Record<string, unknown>;
}

export interface RoutingRuleCreateRequest {
  name: string;
  description?: string;
  type: RoutingRuleType;
  conditions: RoutingCondition[];
  actions: RoutingAction[];
  priority?: number;
  enabled?: boolean;
  tags?: string[];
}

export interface RoutingRule extends BaseEntity {
  id: string;
  name: string;
  description?: string;
  type: RoutingRuleType;
  conditions: RoutingCondition[];
  actions: RoutingAction[];
  priority: number;
  enabled: boolean;
  tags: string[];
  match_count: number;
  last_matched_at?: DateTime;
}

export interface LoadBalancerBackend {
  model_id: string;
  weight: number;
  max_connections: number;
  current_connections: number;
  is_healthy: boolean;
}

export interface LoadBalancerCreateRequest {
  name: string;
  description?: string;
  algorithm?: LoadBalanceAlgorithm;
  backends: LoadBalancerBackend[];
  health_check_interval?: number;
  health_check_timeout?: number;
  enabled?: boolean;
}

export interface LoadBalancer extends BaseEntity {
  id: string;
  name: string;
  description?: string;
  algorithm: LoadBalanceAlgorithm;
  backends: LoadBalancerBackend[];
  health_check_interval: number;
  timeout: number;
  enabled: boolean;
  total_requests: number;
  active_requests: number;
}

export interface CircuitBreakerConfig {
  failure_threshold: number;
  success_threshold: number;
  timeout_seconds: number;
  half_open_max_calls: number;
}

export interface CircuitBreaker {
  id: string;
  model_id: string;
  model_name: string;
  state: CircuitBreakerState;
  config: CircuitBreakerConfig;
  failure_count: number;
  success_count: number;
  last_failure_at?: DateTime;
  last_success_at?: DateTime;
  opened_at?: DateTime;
  next_retry_at?: DateTime;
}

export type StrategyListResponse = PaginatedResponse<Strategy>;
export type RoutingRuleListResponse = PaginatedResponse<RoutingRule>;
