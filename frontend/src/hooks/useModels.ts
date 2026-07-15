import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getModels, getModel, createModel, updateModel, deleteModel, enableModel, disableModel, testModel, getModelConfig, updateModelConfig } from '../api/models';
import type { Model, TestModelRequest, ModelConfig } from '../types/model';

// Query Keys
export const modelKeys = {
  all: ['models'] as const,
  lists: () => [...modelKeys.all, 'list'] as const,
  list: (filters: string) => [...modelKeys.lists(), { filters }] as const,
  details: () => [...modelKeys.all, 'detail'] as const,
  detail: (id: string) => [...modelKeys.details(), id] as const,
};

// 获取模型列表
export const useModels = () => {
  return useQuery({
    queryKey: modelKeys.lists(),
    queryFn: getModels,
    staleTime: 5000, // 5秒内不重新请求
    refetchInterval: 30000, // 30秒自动刷新
  });
};

// 获取模型详情
export const useModel = (id: string) => {
  return useQuery({
    queryKey: modelKeys.detail(id),
    queryFn: () => getModel(id),
    enabled: !!id, // id存在时才请求
  });
};

// 创建模型
export const useCreateModel = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: createModel,
    onSuccess: () => {
      // 创建成功后刷新列表
      queryClient.invalidateQueries({ queryKey: modelKeys.lists() });
    },
  });
};

// 更新模型
export const useUpdateModel = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: Partial<Model> }) =>
      updateModel(id, data),
    onSuccess: (_, variables) => {
      // 更新成功后刷新该模型详情和列表
      queryClient.invalidateQueries({ queryKey: modelKeys.detail(variables.id) });
      queryClient.invalidateQueries({ queryKey: modelKeys.lists() });
    },
  });
};

// 删除模型
export const useDeleteModel = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: deleteModel,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: modelKeys.lists() });
    },
  });
};

// 启用/禁用模型
export const useToggleModel = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, enabled }: { id: string; enabled: boolean }) =>
      enabled ? enableModel(id) : disableModel(id),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: modelKeys.detail(variables.id) });
      queryClient.invalidateQueries({ queryKey: modelKeys.lists() });
    },
  });
};

// 测试模型
export const useTestModel = () => {
  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: TestModelRequest }) =>
      testModel(id, data),
  });
};

// 获取模型配置
export const useModelConfig = (id: string) => {
  return useQuery({
    queryKey: [...modelKeys.detail(id), 'config'],
    queryFn: () => getModelConfig(id),
    enabled: !!id,
  });
};

// 更新模型配置
export const useUpdateModelConfig = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, config }: { id: string; config: ModelConfig }) =>
      updateModelConfig(id, config),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: [...modelKeys.detail(variables.id), 'config'] });
    },
  });
};
