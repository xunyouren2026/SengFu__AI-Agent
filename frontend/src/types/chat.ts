import type { BaseEntity, PaginatedResponse, DateTime } from './common';

export interface ConversationCreate {
  title?: string;
  model_name?: string;
  system_prompt?: string;
  description?: string;
  tags?: string[];
  category?: string;
}

export interface ConversationUpdate {
  title?: string;
  description?: string;
  tags?: string[];
  category?: string;
  is_archived?: boolean;
  is_pinned?: boolean;
}

export interface MessageCreate {
  content: string;
  role?: string;
  parent_id?: number;
  attachments?: Record<string, unknown>[];
}

export interface MessageUpdate {
  content?: string;
}

export interface MessageRate {
  rating: number;
  feedback?: string;
}

export interface Conversation extends BaseEntity {
  id: string;
  user_id: string;
  title: string;
  model_name?: string;
  description?: string;
  is_archived: boolean;
  is_pinned: boolean;
  is_bookmarked: boolean;
  tags: string[];
  category?: string;
  summary?: string;
  message_count: number;
  total_tokens: number;
  total_cost: number;
  last_message_at?: DateTime;
}

export interface Message extends BaseEntity {
  id: string;
  conversation_id: string;
  role: string;
  content: string;
  model_name?: string;
  model_provider?: string;
  parent_id?: number;
  is_edited: boolean;
  is_deleted: boolean;
  is_pinned: boolean;
  is_flagged: boolean;
  attachments: Record<string, unknown>[];
  images: string[];
  tool_calls?: Record<string, unknown>[];
  tool_results?: Record<string, unknown>[];
  rating?: number;
  feedback?: string;
  cost: number;
  latency_ms?: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface ConversationStats {
  message_count: number;
  user_message_count: number;
  assistant_message_count: number;
  total_tokens: number;
  total_cost: number;
  average_latency_ms: number;
  first_message_at?: DateTime;
  last_message_at?: DateTime;
  duration_minutes: number;
}

export interface StreamChunk {
  chunk: string;
  index: number;
  total: number;
  done: boolean;
}

export type ConversationListResponse = PaginatedResponse<Conversation>;
export type MessageListResponse = PaginatedResponse<Message>;
