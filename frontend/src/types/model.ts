// 模型能力类型
export type ModelCapability = 
  | 'chat' 
  | 'completion' 
  | 'embedding' 
  | 'image_generation' 
  | 'image_understanding'
  | 'audio_transcription'
  | 'audio_generation'
  | 'code'
  | 'coding'
  | 'function_calling'
  | 'json_mode'
  | 'vision'
  | 'tools'
  | 'reasoning'
  | 'creative_writing'
  | 'analysis'
  | 'math';

// 模型状态
export type ModelStatus = 'active' | 'inactive' | 'error' | 'loading';

// 模型类型
export type ModelType = 'llm' | 'embedding' | 'image' | 'audio' | 'multimodal';

// 模型接口
export interface Model {
  id: string;
  name: string;
  provider: string;
  type: ModelType;
  status: ModelStatus;
  capabilities: ModelCapability[];
  config?: Record<string, any>;
  created_at?: string;
  updated_at?: string;
}

// 模型列表响应
export interface ModelsResponse {
  models: Model[];
  total: number;
}

// 模型测试请求
export interface TestModelRequest {
  prompt: string;
  temperature?: number;
  max_tokens?: number;
}

// 模型测试响应
export interface TestModelResponse {
  response: string;
  latency: number;
  tokens_used: number;
}

// 模型配置
export interface ModelConfig {
  temperature: number;
  max_tokens: number;
  top_p: number;
  frequency_penalty: number;
  presence_penalty: number;
}
