import type { PaginatedResponse, BaseEntity, DateTime } from './common';

export type AgentStatus = 'active' | 'inactive' | 'error' | 'pending' | 'training' | 'maintenance' | 'suspended';
export type AgentCapability = 'research' | 'analysis' | 'writing' | 'coding' | 'planning' | 'communication' | 'decision_making' | 'learning' | 'memory' | 'tool_use' | 'collaboration' | 'negotiation';
export type AgentRole = 'leader' | 'member' | 'observer' | 'coordinator' | 'specialist';
export type AllianceStatus = 'active' | 'inactive' | 'dissolved' | 'pending' | 'recruiting';
export type AllianceType = 'cooperative' | 'competitive' | 'hybrid' | 'hierarchical' | 'flat';
export type DebateStatus = 'pending' | 'active' | 'paused' | 'completed' | 'cancelled';
export type DebateType = 'proposition' | 'comparison' | 'brainstorming' | 'decision' | 'analysis';
export type TaskStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'paused';
export type TaskPriority = 'low' | 'medium' | 'high' | 'critical';
export type LogLevel = 'debug' | 'info' | 'warning' | 'error' | 'critical';

export interface AgentConfig {
  max_concurrent_tasks: number;
  task_timeout_seconds: number;
  auto_recovery: boolean;
  learning_enabled: boolean;
  collaboration_enabled: boolean;
  memory_enabled: boolean;
  tool_access: string[];
  custom_params: Record<string, unknown>;
}

export interface AgentMetrics {
  total_tasks: number;
  successful_tasks: number;
  failed_tasks: number;
  success_rate: number;
  avg_task_duration_ms: number;
  total_tokens_used: number;
  reputation_score: number;
}

export interface AgentCreateRequest {
  name: string;
  description?: string;
  capabilities: AgentCapability[];
  personality_id?: string;
  model_id?: string;
  config: AgentConfig;
  tags: string[];
  avatar_url?: string;
}

export interface AgentUpdateRequest {
  name?: string;
  description?: string;
  capabilities?: AgentCapability[];
  personality_id?: string;
  model_id?: string;
  config?: Partial<AgentConfig>;
  tags?: string[];
  avatar_url?: string;
}

export interface Agent extends BaseEntity {
  id: string;
  name: string;
  description?: string;
  capabilities: AgentCapability[];
  personality_id?: string;
  model_id?: string;
  config: AgentConfig;
  tags: string[];
  avatar_url?: string;
  status: AgentStatus;
  metrics: AgentMetrics;
  current_alliance_id?: string;
  current_role?: AgentRole;
  last_active_at?: DateTime;
  created_by?: string;
}

export interface AgentLogEntry {
  id: string;
  agent_id: string;
  level: LogLevel;
  message: string;
  source: string;
  metadata: Record<string, unknown>;
  timestamp: DateTime;
}

export interface TaskExecutionRequest {
  task_type: string;
  task_input: Record<string, unknown>;
  priority?: TaskPriority;
  timeout_seconds?: number;
  callback_url?: string;
  metadata?: Record<string, unknown>;
}

export interface TaskExecutionResponse {
  task_id: string;
  agent_id: string;
  status: TaskStatus;
  result?: Record<string, unknown>;
  started_at: DateTime;
  completed_at?: DateTime;
  duration_ms?: number;
  tokens_used?: number;
}

export interface TaskHistoryItem {
  id: string;
  task_type: string;
  status: TaskStatus;
  priority: TaskPriority;
  result_summary?: string;
  started_at: DateTime;
  completed_at?: DateTime;
  duration_ms?: number;
  tokens_used?: number;
}

export interface AllianceMember {
  agent_id: string;
  agent_name: string;
  role: AgentRole;
  joined_at: DateTime;
  contribution_score: number;
  is_active: boolean;
}

export interface AllianceCreateRequest {
  name: string;
  description?: string;
  alliance_type?: AllianceType;
  max_members?: number;
  founder_id: string;
  goals?: string[];
  rules?: string[];
  tags?: string[];
}

export interface Alliance extends BaseEntity {
  id: string;
  name: string;
  description?: string;
  alliance_type: AllianceType;
  status: AllianceStatus;
  max_members: number;
  current_members: number;
  founder_id: string;
  members: AllianceMember[];
  goals: string[];
  rules: string[];
  tags: string[];
  performance_score: number;
}

export interface DebateParticipant {
  agent_id: string;
  agent_name: string;
  stance: string;
  arguments: string[];
  votes_received: number;
}

export interface DebateCreateRequest {
  title: string;
  description?: string;
  debate_type?: DebateType;
  topic: string;
  participant_ids: string[];
  duration_minutes?: number;
  rules?: string[];
}

export interface Debate extends BaseEntity {
  id: string;
  title: string;
  description?: string;
  debate_type: DebateType;
  topic: string;
  status: DebateStatus;
  participants: DebateParticipant[];
  winner_id?: string;
  duration_minutes: number;
  started_at?: DateTime;
  ended_at?: DateTime;
  total_votes: number;
  rules: string[];
  created_by?: string;
}

export type AgentListResponse = PaginatedResponse<Agent>;
export type AllianceListResponse = PaginatedResponse<Alliance>;
export type DebateListResponse = PaginatedResponse<Debate>;
export type TaskHistoryResponse = PaginatedResponse<TaskHistoryItem>;
