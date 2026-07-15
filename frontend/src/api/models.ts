import apiClient from './client';
import { wrapListResponse } from './utils';
import type { Model, ModelsResponse, TestModelRequest, TestModelResponse, ModelConfig } from '../types/model';

// 获取模型列表
export const getModels = async (): Promise<Model[]> => {
  const response = await apiClient.get<ModelsResponse>('/models');
  // 拦截器已返回 response.data.data，可能是数组或包含 models 字段的对象
  if (Array.isArray(response)) return response;
  return (response as any).models || [];
};

// 获取模型详情
export const getModel = async (id: string): Promise<Model> => {
  return apiClient.get<Model>(`/models/${id}`);
};

// 创建模型
export const createModel = async (data: Partial<Model>): Promise<Model> => {
  return apiClient.post<Model>('/models', data);
};

// 更新模型
export const updateModel = async (id: string, data: Partial<Model>): Promise<Model> => {
  return apiClient.put<Model>(`/models/${id}`, data);
};

// 删除模型
export const deleteModel = async (id: string): Promise<void> => {
  return apiClient.delete(`/models/${id}`);
};

// 启用模型
export const enableModel = async (id: string): Promise<Model> => {
  return apiClient.post<Model>(`/models/${id}/enable`);
};

// 禁用模型
export const disableModel = async (id: string): Promise<Model> => {
  return apiClient.post<Model>(`/models/${id}/disable`);
};

// 测试模型
export const testModel = async (id: string, data: TestModelRequest): Promise<TestModelResponse> => {
  return apiClient.post<TestModelResponse>(`/models/${id}/test`, data);
};

// 获取模型配置
export const getModelConfig = async (id: string): Promise<ModelConfig> => {
  return apiClient.get<ModelConfig>(`/models/${id}/config`);
};

// 更新模型配置
export const updateModelConfig = async (id: string, config: ModelConfig): Promise<ModelConfig> => {
  return apiClient.put<ModelConfig>(`/models/${id}/config`, config);
};
