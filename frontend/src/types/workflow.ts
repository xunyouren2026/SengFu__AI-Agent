import type { PaginatedResponse, BaseEntity, DateTime } from './common';

export type WorkflowStatus = 'draft' | 'published' | 'archived' | 'deprecated';
export type WorkflowExecutionStatus = 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled' | 'timeout';
export type NodeType = 'start' | 'end' | 'llm' | 'prompt' | 'condition' | 'loop' | 'parallel' | 'wait' | 'webhook' | 'api' | 'code' | 'transform' | 'filter' | 'aggregate' | 'memory' | 'tool' | 'human' | 'notification';
export type NodeExecutionStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped' | 'cancelled';
export type TriggerType = 'manual' | 'scheduled' | 'webhook' | 'event' | 'api';

export interface Position {
  x: number;
  y: number;
}

export interface NodeConfig {
  model_id?: string;
  prompt_template?: string;
  temperature?: number;
  max_tokens?: number;
  timeout?: number;
  retry_count?: number;
  condition?: string;
  code?: string;
  api_endpoint?: string;
  http_method?: string;
  headers?: Record<string, string>;
  custom_config?: Record<string, unknown>;
}

export interface WorkflowNode {
  id: string;
  type: NodeType;
  name: string;
  description?: string;
  position: Position;
  config: NodeConfig;
  inputs: string[];
  outputs: string[];
  enabled: boolean;
}

export interface WorkflowEdge {
  id: string;
  source: string;
  target: string;
  source_handle?: string;
  target_handle?: string;
  condition?: string;
  label?: string;
  enabled: boolean;
}

export interface WorkflowVariable {
  name: string;
  type?: string;
  default_value?: unknown;
  description?: string;
  required?: boolean;
}

export interface WorkflowTrigger {
  type: TriggerType;
  config: Record<string, unknown>;
  enabled: boolean;
}

export interface WorkflowCreateRequest {
  name: string;
  description?: string;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  variables?: WorkflowVariable[];
  triggers?: WorkflowTrigger[];
  tags?: string[];
  category?: string;
}

export interface WorkflowUpdateRequest {
  name?: string;
  description?: string;
  nodes?: WorkflowNode[];
  edges?: WorkflowEdge[];
  variables?: WorkflowVariable[];
  triggers?: WorkflowTrigger[];
  tags?: string[];
  category?: string;
}

export interface Workflow extends BaseEntity {
  id: string;
  name: string;
  description?: string;
  status: WorkflowStatus;
  version: string;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  variables: WorkflowVariable[];
  triggers: WorkflowTrigger[];
  tags: string[];
  category: string;
  execution_count: number;
  success_count: number;
  failure_count: number;
  avg_execution_time_ms: number;
  published_at?: DateTime;
  created_by?: string;
}

export interface WorkflowExecuteRequest {
  variables?: Record<string, unknown>;
  async_execution?: boolean;
  timeout?: number;
  priority?: number;
  callback_url?: string;
}

export interface NodeExecutionResult {
  node_id: string;
  node_name: string;
  status: NodeExecutionStatus;
  started_at?: DateTime;
  completed_at?: DateTime;
  duration_ms?: number;
  output?: unknown;
  error?: string;
  logs: string[];
}

export interface WorkflowExecutionDetail {
  execution_id: string;
  workflow_id: string;
  workflow_name: string;
  status: WorkflowExecutionStatus;
  started_at: DateTime;
  completed_at?: DateTime;
  duration_ms?: number;
  variables: Record<string, unknown>;
  results: Record<string, unknown>;
  node_results: NodeExecutionResult[];
  current_node_id?: string;
  progress_percent: number;
  error_message?: string;
}

export interface WorkflowTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  tags: string[];
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  variables: WorkflowVariable[];
  icon?: string;
  difficulty: string;
  estimated_setup_time_minutes: number;
}

export interface NodeTypeInfo {
  type: NodeType;
  name: string;
  description: string;
  icon: string;
  category: string;
  inputs: string[];
  outputs: string[];
  config_schema: Record<string, unknown>;
  default_config: Record<string, unknown>;
}

export type WorkflowListResponse = PaginatedResponse<Workflow>;
export type WorkflowExecutionListResponse = PaginatedResponse<WorkflowExecutionDetail>;
