import type { PaginatedResponse, BaseEntity, DateTime } from './common';

export type AlertSeverity = 'critical' | 'high' | 'medium' | 'low' | 'info';
export type AlertStatus = 'active' | 'acknowledged' | 'resolved' | 'suppressed';
export type UserStatus = 'active' | 'inactive' | 'suspended' | 'pending';
export type BackupStatus = 'pending' | 'in_progress' | 'completed' | 'failed' | 'restoring';
export type MaintenanceStatus = 'idle' | 'in_progress' | 'scheduled' | 'completed' | 'failed';
export type PermissionScope = 'system' | 'user' | 'api' | 'resource';

export interface SystemMetrics {
  cpu_usage_percent: number;
  memory_usage_percent: number;
  disk_usage_percent: number;
  network_in_bytes: number;
  network_out_bytes: number;
  gpu_usage_percent?: number;
  gpu_memory_percent?: number;
  active_connections: number;
  uptime_seconds: number;
  timestamp: DateTime;
}

export interface LogEntry {
  id: string;
  timestamp: DateTime;
  level: string;
  source: string;
  message: string;
  context: Record<string, unknown>;
  trace_id?: string;
  user_id?: string;
}

export interface Alert {
  id: string;
  name: string;
  description: string;
  severity: AlertSeverity;
  status: AlertStatus;
  source: string;
  metric_name?: string;
  threshold?: number;
  current_value?: number;
  acknowledged_by?: string;
  acknowledged_at?: DateTime;
  resolved_at?: DateTime;
  created_at: DateTime;
  tags: string[];
}

export interface UserInfo {
  id: string;
  username: string;
  email?: string;
  full_name?: string;
  avatar_url?: string;
  status: UserStatus;
  roles: string[];
  permissions: string[];
  last_login_at: DateTime;
  created_at: DateTime;
  updated_at: DateTime;
  is_superuser: boolean;
}

export interface UserCreateRequest {
  username: string;
  email?: string;
  full_name?: string;
  password: string;
  roles?: string[];
  is_superuser?: boolean;
}

export interface UserUpdateRequest {
  email?: string;
  full_name?: string;
  status?: UserStatus;
  roles?: string[];
  is_superuser?: boolean;
}

export interface RoleInfo {
  id: string;
  name: string;
  description?: string;
  permissions: string[];
  user_count: number;
  is_system: boolean;
  created_at: DateTime;
  updated_at: DateTime;
}

export interface APIKeyInfo {
  id: string;
  name: string;
  key_preview: string;
  permissions: string[];
  last_used_at?: DateTime;
  expires_at?: DateTime;
  created_at: DateTime;
  created_by: string;
  is_active: boolean;
}

export interface APIKeyCreateRequest {
  name: string;
  permissions: string[];
  expires_in_days?: number;
}

export interface BackupInfo {
  id: string;
  name: string;
  description: string;
  status: BackupStatus;
  size_bytes: number;
  size_formatted: string;
  includes: string[];
  created_at: DateTime;
  created_by: string;
  expires_at?: DateTime;
  checksum?: string;
}

export interface BackupCreateRequest {
  name?: string;
  description?: string;
  include_config?: boolean;
  include_data?: boolean;
  include_logs?: boolean;
  encrypt?: boolean;
  retention_days?: number;
}

export interface GPUInfo {
  id: string;
  index: number;
  name: string;
  vendor: string;
  memory_total_gb: number;
  memory_used_gb: number;
  memory_free_gb: number;
  utilization_percent: number;
  temperature_celsius: number;
  power_draw_watts: number;
  power_limit_watts: number;
  clock_speed_mhz: number;
  driver_version: string;
  compute_capability?: string;
  processes: Record<string, unknown>[];
}

export interface HardwareInfo {
  cpu_count: number;
  cpu_model: string;
  memory_total_gb: string;
  disk_total_gb: string;
  platform: string;
  python_version: string;
  gpu_count: number;
  gpus: GPUInfo[];
  cuda_version?: string;
  cudnn_version?: string;
}

export interface SettingCategory {
  id: string;
  name: string;
  description?: string;
  icon?: string;
  order: number;
}

export interface SettingValue {
  key: string;
  value: unknown;
  type: string;
  category: string;
  label: string;
  description?: string;
  default_value?: unknown;
  options?: Record<string, unknown>[];
  min_value?: number;
  max_value?: number;
  is_secret?: boolean;
  requires_restart?: boolean;
}

export interface SystemSettings {
  settings: Record<string, unknown>;
  categories: SettingCategory[];
  schema: SettingValue[];
  last_modified?: DateTime;
}

export interface HelpDocInfo {
  id: string;
  title: string;
  category: string;
  summary?: string;
  tags?: string[];
  version: string;
  last_updated: DateTime;
  author?: string;
}

export interface HelpDocContent extends HelpDocInfo {
  content: string;
  related_docs: string[];
  attachments: Record<string, unknown>[];
}

export interface FAQItem {
  id: string;
  question: string;
  answer: string;
  category: string;
  helpful_count: number;
  not_helpful_count: number;
  tags: string[];
  last_updated: DateTime;
}

export interface FeedbackRequest {
  type: string;
  subject: string;
  content: string;
  email?: string;
  attachments?: string[];
  context?: Record<string, unknown>;
}

export interface ShortcutInfo {
  id: string;
  action: string;
  description: string;
  key_combination: string;
  context?: string;
  platform?: string;
}

export interface ChangelogEntry {
  version: string;
  release_date: DateTime;
  changes: string[];
  breaking_changes: string[];
  bug_fixes: string[];
  improvements: string[];
  contributors: string[];
}

export interface DashboardWidget {
  id: string;
  type: string;
  title: string;
  config: Record<string, unknown>;
  position: Record<string, number>;
  size: Record<string, number>;
}

export interface DashboardConfig extends BaseEntity {
  id: string;
  name: string;
  description?: string;
  is_default: boolean;
  widgets: DashboardWidget[];
  layout: string;
  refresh_interval: number;
}

export type UserListResponse = PaginatedResponse<UserInfo>;
export type AlertListResponse = PaginatedResponse<Alert>;
export type LogListResponse = PaginatedResponse<LogEntry>;
export type BackupListResponse = PaginatedResponse<BackupInfo>;
export type HelpDocListResponse = PaginatedResponse<HelpDocInfo>;
export type FAQListResponse = PaginatedResponse<FAQItem>;
