import type { PaginatedResponse, DateTime } from './common';

export type CognitiveState = 'focused' | 'reflective' | 'learning' | 'resting' | 'adapting' | 'creative' | 'analyzing';
export type ReflectionDepth = 'surface' | 'moderate' | 'deep';
export type ReflectionType = 'self' | 'decision' | 'interaction' | 'learning' | 'error' | 'goal';
export type MemoryType = 'episodic' | 'semantic' | 'procedural' | 'working' | 'long_term';
export type MemoryPriority = 'critical' | 'high' | 'medium' | 'low';
export type EmotionType = 'joy' | 'sadness' | 'anger' | 'fear' | 'surprise' | 'disgust' | 'trust' | 'anticipation' | 'neutral';
export type GoalStatus = 'pending' | 'active' | 'paused' | 'completed' | 'failed' | 'cancelled';
export type GoalPriority = 'critical' | 'high' | 'medium' | 'low';
export type MetricType = 'attention' | 'memory' | 'reasoning' | 'creativity' | 'learning_rate' | 'adaptability';

export interface CognitiveStateSchema {
  state: CognitiveState;
  confidence: number;
  intensity: number;
  context: Record<string, unknown>;
  since: DateTime;
}

export interface StateHistoryItem {
  state: CognitiveState;
  confidence: number;
  intensity: number;
  timestamp: DateTime;
  trigger?: string;
}

export interface ReflectionTriggerRequest {
  topic: string;
  depth?: ReflectionDepth;
  reflection_type?: ReflectionType;
  context?: Record<string, unknown>;
}

export interface ReflectionInsight {
  insight: string;
  confidence: number;
  category: string;
  action_items: string[];
}

export interface ReflectionResponse {
  reflection_id: string;
  topic: string;
  depth: ReflectionDepth;
  reflection_type: ReflectionType;
  insights: ReflectionInsight[];
  summary: string;
  started_at: DateTime;
  completed_at: DateTime;
  duration_ms: number;
}

export interface ReflectionRecord {
  id: string;
  topic: string;
  depth: ReflectionDepth;
  reflection_type: ReflectionType;
  insight_count: number;
  summary: string;
  created_at: DateTime;
}

export interface MemoryEntry {
  id: string;
  content: string;
  memory_type: MemoryType;
  priority: MemoryPriority;
  importance_score: number;
  associations: string[];
  metadata: Record<string, unknown>;
  created_at: DateTime;
  last_accessed_at?: DateTime;
  access_count: number;
}

export interface MemorySearchRequest {
  query: string;
  memory_types?: MemoryType[];
  min_importance?: number;
  max_results?: number;
  include_associated?: boolean;
}

export interface MemorySearchResult {
  memory: MemoryEntry;
  relevance_score: number;
  matched_keywords: string[];
}

export interface MemoryForgetRequest {
  memory_ids: string[];
  reason?: string;
  permanent?: boolean;
}

export interface EmotionStateSchema {
  emotion: EmotionType;
  intensity: number;
  valence: number;
  arousal: number;
  trigger?: string;
  since: DateTime;
}

export interface Goal {
  id: string;
  title: string;
  description?: string;
  status: GoalStatus;
  priority: GoalPriority;
  progress: number;
  parent_id?: string;
  sub_goals: string[];
  deadline?: DateTime;
  metadata: Record<string, unknown>;
  created_at: DateTime;
  updated_at: DateTime;
  completed_at?: DateTime;
}

export interface GoalCreateRequest {
  title: string;
  description?: string;
  priority?: GoalPriority;
  parent_id?: string;
  deadline?: DateTime;
  metadata?: Record<string, unknown>;
}

export interface GoalUpdateRequest {
  title?: string;
  description?: string;
  status?: GoalStatus;
  priority?: GoalPriority;
  progress?: number;
  deadline?: DateTime;
  metadata?: Record<string, unknown>;
}

export interface CognitiveMetric {
  metric_type: MetricType;
  value: number;
  baseline: number;
  trend: string;
  history: Record<string, unknown>[];
  last_updated: DateTime;
}

export type MemoryListResponse = PaginatedResponse<MemoryEntry>;
export type ReflectionListResponse = PaginatedResponse<ReflectionRecord>;
export type GoalListResponse = PaginatedResponse<Goal>;
