import type { PaginatedResponse, BaseEntity, DateTime } from './common';

// Physics Engine
export type SimulationType = 'molecular_dynamics' | 'quantum_chemistry' | 'fluid_dynamics' | 'structural_mechanics' | 'thermal_analysis' | 'electromagnetic';
export type SimulationStatus = 'pending' | 'setup' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';

export interface SimulationConfig {
  time_step: number;
  total_steps: number;
  temperature?: number;
  pressure?: number;
  ensemble: string;
  constraints: string[];
}

export interface SimulationCreateRequest {
  name: string;
  type: SimulationType;
  description?: string;
  config: SimulationConfig;
  input_files: string[];
  molecule_ids?: string[];
}

export interface Simulation extends BaseEntity {
  id: string;
  name: string;
  type: SimulationType;
  status: SimulationStatus;
  description?: string;
  config: SimulationConfig;
  progress: number;
  current_step: number;
  started_at?: DateTime;
  completed_at?: DateTime;
  results_url?: string;
  logs_url?: string;
}

// Computer Use
export type ComputerAction = 'click' | 'double_click' | 'right_click' | 'drag' | 'scroll' | 'type' | 'keypress' | 'navigate';
export type RecordingStatus = 'idle' | 'recording' | 'paused' | 'saving';

export interface ScreenshotResponse {
  image_url: string;
  image_data?: string;
  width: number;
  height: number;
  format: string;
  timestamp: DateTime;
}

export interface ClickRequest {
  x: number;
  y: number;
  button?: string;
  clicks?: number;
  selector?: string;
}

export interface TypeRequest {
  text: string;
  selector?: string;
  delay?: number;
  clear_first?: boolean;
  submit?: boolean;
}

export interface ScrollRequest {
  direction?: string;
  amount?: number;
  selector?: string;
  smooth?: boolean;
}

export interface NavigateRequest {
  url: string;
  wait_until?: string;
  timeout?: number;
  referer?: string;
}

export interface RecordingConfig {
  fps: number;
  resolution: string;
  format: string;
  quality: string;
  audio: boolean;
}

export interface Recording {
  recording_id: string;
  status: RecordingStatus;
  config: RecordingConfig;
  started_at?: DateTime;
  duration: number;
  file_size: number;
  file_url?: string;
}

// Security
export type FirewallAction = 'allow' | 'deny' | 'log' | 'alert';
export type ThreatLevel = 'low' | 'medium' | 'high' | 'critical';
export type ScanType = 'vulnerability' | 'malware' | 'penetration' | 'compliance' | 'prompt_injection';

export interface FirewallRule extends BaseEntity {
  id: string;
  name: string;
  description?: string;
  source_ip?: string;
  destination_ip?: string;
  port?: string;
  protocol?: string;
  action: FirewallAction;
  priority: number;
  enabled: boolean;
  expires_at?: DateTime;
}

export interface SecurityScanResponse {
  scan_id: string;
  scan_type: ScanType;
  status: string;
  target: string;
  progress: number;
  started_at: DateTime;
  completed_at?: DateTime;
  findings_count: number;
  findings: Record<string, unknown>[];
  report_url?: string;
}

export interface PromptGuardTestResponse {
  test_id: string;
  prompt: string;
  passed: boolean;
  risk_score: number;
  detected_issues: Record<string, unknown>[];
  recommendations: string[];
  explanation?: string;
}

// Federated Learning
export type NodeStatus = 'online' | 'offline' | 'training' | 'syncing' | 'error';
export type AggregationStrategy = 'fedavg' | 'fedprox' | 'fedopt' | 'scaffold';

export interface FederatedNode extends BaseEntity {
  id: string;
  name: string;
  endpoint: string;
  status: NodeStatus;
  public_key?: string;
  capabilities: string[];
  dataset_size?: number;
  last_seen: DateTime;
  contribution_score: number;
}

export interface FederatedNodeRegisterRequest {
  name: string;
  endpoint: string;
  public_key?: string;
  capabilities?: string[];
  dataset_info?: Record<string, unknown>;
}

export interface TrainingRound {
  id: string;
  round_number: number;
  status: string;
  participating_nodes: string[];
  aggregation_strategy: AggregationStrategy;
  started_at: DateTime;
  completed_at?: DateTime;
  global_model_version?: string;
  metrics: Record<string, unknown>;
}

export interface AggregationRequest {
  node_updates: string[];
  strategy?: AggregationStrategy;
  weights?: Record<string, number>;
  options?: Record<string, unknown>;
}

export interface ContributionStats {
  node_id: string;
  node_name: string;
  total_rounds: number;
  successful_rounds: number;
  data_samples_contributed: number;
  computation_hours: number;
  reward_tokens: number;
  reputation_score: number;
  last_contribution_at?: DateTime;
}

// RAG / Knowledge Retrieval
export type DocumentType = 'pdf' | 'word' | 'text' | 'markdown' | 'html' | 'code' | 'json';
export type SearchStrategy = 'semantic' | 'keyword' | 'hybrid' | 'vector';

export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  type: DocumentType;
  size: number;
  status: string;
  chunks_count?: number;
  uploaded_at: DateTime;
  processing_started_at?: DateTime;
}

export interface DocumentInfo {
  id: string;
  filename: string;
  type: DocumentType;
  size: number;
  status: string;
  chunks_count: number;
  metadata: Record<string, unknown>;
  tags: string[];
  uploaded_at: DateTime;
  processed_at?: DateTime;
}

export interface RAGSearchRequest {
  query: string;
  strategy?: SearchStrategy;
  top_k?: number;
  filters?: Record<string, unknown>;
  document_ids?: string[];
  min_score?: number;
}

export interface RAGSearchResult {
  document_id: string;
  document_name: string;
  chunk_index: number;
  content: string;
  score: number;
  metadata: Record<string, unknown>;
}

export interface RAGSearchResponse {
  query: string;
  results: RAGSearchResult[];
  total_results: number;
  search_time_ms: number;
  strategy_used: SearchStrategy;
}

export interface KnowledgeBaseInfo {
  id: string;
  name: string;
  description?: string;
  document_count: number;
  total_chunks: number;
  embedding_model: string;
  created_at: DateTime;
  updated_at: DateTime;
  owner_id: string;
}

// Data Pipeline
export type DatasetFormat = 'csv' | 'json' | 'parquet' | 'excel' | 'sql' | 'hdf5';
export type PipelineStatus = 'draft' | 'active' | 'running' | 'paused' | 'error' | 'completed';

export interface DatasetInfo {
  id: string;
  name: string;
  description?: string;
  format: DatasetFormat;
  size: number;
  rows?: number;
  columns?: number;
  schema?: Record<string, unknown>;
  tags: string[];
  uploaded_at: DateTime;
  processed_at?: DateTime;
  status: string;
}

export interface PipelineStep {
  id: string;
  name: string;
  type: string;
  config: Record<string, unknown>;
  inputs: string[];
  outputs: string[];
  dependencies: string[];
}

export interface PipelineCreateRequest {
  name: string;
  description?: string;
  steps: PipelineStep[];
  schedule?: string;
  trigger_events?: string[];
}

export interface Pipeline extends BaseEntity {
  id: string;
  name: string;
  description?: string;
  status: PipelineStatus;
  steps: PipelineStep[];
  schedule?: string;
  last_run_at?: DateTime;
  last_run_status?: string;
  run_count: number;
}

// Value Alignment
export type PrincipleCategory = 'safety' | 'fairness' | 'transparency' | 'privacy' | 'accountability';
export type AlignmentTestResult = 'passed' | 'failed' | 'warning' | 'pending';

export interface AlignmentPrinciple extends BaseEntity {
  id: string;
  name: string;
  description: string;
  category: PrincipleCategory;
  priority: number;
  rules: string[];
  examples: { input: string; output: string }[];
  is_active: boolean;
}

export interface AlignmentTestResponse {
  test_id: string;
  status: string;
  overall_score: number;
  results: {
    case_id: string;
    input_text: string;
    expected_behavior: string;
    actual_response: string;
    result: AlignmentTestResult;
    score: number;
    violations: string[];
  }[];
  principles_evaluated: string[];
  started_at: DateTime;
  completed_at?: DateTime;
}

export interface AlignmentReport {
  id: string;
  test_id: string;
  generated_at: DateTime;
  summary: string;
  overall_score: number;
  category_scores: Record<string, number>;
  findings: Record<string, unknown>[];
  recommendations: string[];
  report_url?: string;
}

// Robot Control
export type RobotType = 'arm' | 'mobile' | 'humanoid' | 'drone' | 'wheelchair';
export type RobotStatus = 'idle' | 'busy' | 'error' | 'offline' | 'emergency';

export interface RobotInfo {
  id: string;
  name: string;
  type: RobotType;
  model: string;
  manufacturer: string;
  status: RobotStatus;
  capabilities: string[];
  battery_level?: number;
  current_position?: Record<string, number>;
  connected_at?: DateTime;
  last_heartbeat?: DateTime;
}

export interface RobotMoveRequest {
  movement_type: string;
  target_position?: Record<string, number>;
  relative_movement?: Record<string, number>;
  speed?: number;
  acceleration?: number;
  coordinate_system?: string;
}

export interface RobotStatusResponse {
  robot_id: string;
  status: RobotStatus;
  battery_level?: number;
  current_task?: string;
  current_position?: Record<string, number>;
  joint_states?: Record<string, unknown>;
  sensor_data?: Record<string, unknown>;
  errors: string[];
  timestamp: DateTime;
}

// Channel (advanced)
export type ChannelType = 'email' | 'slack' | 'discord' | 'telegram' | 'wechat' | 'webhook' | 'sms';
export type ChannelStatusType = 'active' | 'inactive' | 'error' | 'pending';

export interface ChannelConfig {
  webhook_url?: string;
  api_key?: string;
  api_secret?: string;
  channel_id?: string;
  settings: Record<string, unknown>;
}

export interface ChannelCreateRequest {
  name: string;
  type: ChannelType;
  description?: string;
  config: ChannelConfig;
  is_default?: boolean;
}

export interface ChannelResponse extends BaseEntity {
  id: string;
  name: string;
  type: ChannelType;
  description?: string;
  status: ChannelStatusType;
  config: ChannelConfig;
  is_default: boolean;
  message_count: number;
  last_message_at?: DateTime;
}

// Plugin (advanced)
export type PluginCategory = 'generation' | 'integration' | 'automation' | 'analytics' | 'security' | 'custom';
export type PluginStatusType = 'installed' | 'enabled' | 'disabled' | 'error' | 'updating';

export interface PluginInfo {
  id: string;
  name: string;
  description: string;
  version: string;
  author: string;
  category: PluginCategory;
  status: PluginStatusType;
  icon_url?: string;
  readme_url?: string;
  config_schema?: Record<string, unknown>;
  permissions: string[];
  installed_at?: DateTime;
  updated_at?: DateTime;
}

export interface PluginMarketplaceItem {
  id: string;
  name: string;
  description: string;
  version: string;
  author: string;
  category: PluginCategory;
  rating: number;
  download_count: number;
  icon_url?: string;
  screenshots: string[];
  tags: string[];
  price: number;
  is_official: boolean;
}

// Personality (advanced)
export type PersonalityTrait = 'openness' | 'conscientiousness' | 'extraversion' | 'agreeableness' | 'neuroticism';
export type CommunicationStyle = 'formal' | 'casual' | 'technical' | 'friendly' | 'professional' | 'humorous';

export interface PersonalityTraits {
  openness: number;
  conscientiousness: number;
  extraversion: number;
  agreeableness: number;
  neuroticism: number;
}

export interface PersonalityCreateRequest {
  name: string;
  description?: string;
  avatar_url?: string;
  system_prompt: string;
  traits: PersonalityTraits;
  communication_style?: CommunicationStyle;
  response_templates?: string[];
  knowledge_domains?: string[];
  forbidden_topics?: string[];
  example_conversations?: { user: string; assistant: string }[];
}

export interface PersonalityResponse extends BaseEntity {
  id: string;
  name: string;
  description?: string;
  avatar_url?: string;
  system_prompt: string;
  traits: PersonalityTraits;
  communication_style: CommunicationStyle;
  is_active: boolean;
  usage_count: number;
  created_by: string;
}

export type SimulationListResponse = PaginatedResponse<Simulation>;
export type FederatedNodeListResponse = PaginatedResponse<FederatedNode>;
export type DocumentListResponse = PaginatedResponse<DocumentInfo>;
export type PipelineListResponse = PaginatedResponse<Pipeline>;
export type ChannelListResponse = PaginatedResponse<ChannelResponse>;
export type PluginListResponse = PaginatedResponse<PluginInfo>;
export type PersonalityListResponse = PaginatedResponse<PersonalityResponse>;
