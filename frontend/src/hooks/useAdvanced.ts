import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { advancedApi } from '../api/advanced';
import type { SimulationCreateRequest, FederatedNodeRegisterRequest, RAGSearchRequest, PipelineCreateRequest, ChannelCreateRequest, PersonalityCreateRequest } from '../types/advanced';

export const advancedKeys = {
  all: ['advanced'] as const,
  simulations: () => [...advancedKeys.all, 'simulations'] as const,
  federated: () => [...advancedKeys.all, 'federated'] as const,
  rag: () => [...advancedKeys.all, 'rag'] as const,
  pipelines: () => [...advancedKeys.all, 'pipelines'] as const,
  channels: () => [...advancedKeys.all, 'channels'] as const,
  plugins: () => [...advancedKeys.all, 'plugins'] as const,
  personalities: () => [...advancedKeys.all, 'personalities'] as const,
  robots: () => [...advancedKeys.all, 'robots'] as const,
  security: () => [...advancedKeys.all, 'security'] as const,
};

export const useSimulations = (params?: Record<string, unknown>) =>
  useQuery({ queryKey: [...advancedKeys.simulations(), params], queryFn: () => advancedApi.getSimulations(params) });

export const useCreateSimulation = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: (data: SimulationCreateRequest) => advancedApi.createSimulation(data), onSuccess: () => qc.invalidateQueries({ queryKey: advancedKeys.simulations() }) });
};

export const useFederatedNodes = (params?: Record<string, unknown>) =>
  useQuery({ queryKey: [...advancedKeys.federated(), params], queryFn: () => advancedApi.getNodes(params), refetchInterval: 30000 });

export const useRegisterNode = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: (data: FederatedNodeRegisterRequest) => advancedApi.registerNode(data), onSuccess: () => qc.invalidateQueries({ queryKey: advancedKeys.federated() }) });
};

export const useDocuments = (params?: Record<string, unknown>) =>
  useQuery({ queryKey: [...advancedKeys.rag(), 'documents', params], queryFn: () => advancedApi.getDocuments(params) });

export const useRAGSearch = () =>
  useMutation({ mutationFn: (data: RAGSearchRequest) => advancedApi.search(data) });

export const useKnowledgeBases = () =>
  useQuery({ queryKey: [...advancedKeys.rag(), 'knowledge-bases'], queryFn: () => advancedApi.getKnowledgeBases() });

export const usePipelines = (params?: Record<string, unknown>) =>
  useQuery({ queryKey: [...advancedKeys.pipelines(), params], queryFn: () => advancedApi.getPipelines(params) });

export const useCreatePipeline = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: (data: PipelineCreateRequest) => advancedApi.createPipeline(data), onSuccess: () => qc.invalidateQueries({ queryKey: advancedKeys.pipelines() }) });
};

export const useChannels = (params?: Record<string, unknown>) =>
  useQuery({ queryKey: [...advancedKeys.channels(), params], queryFn: () => advancedApi.getChannels(params) });

export const useCreateChannel = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: (data: ChannelCreateRequest) => advancedApi.createChannel(data), onSuccess: () => qc.invalidateQueries({ queryKey: advancedKeys.channels() }) });
};

export const useDeleteChannel = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: (id: string) => advancedApi.deleteChannel(id), onSuccess: () => qc.invalidateQueries({ queryKey: advancedKeys.channels() }) });
};

export const usePlugins = (params?: Record<string, unknown>) =>
  useQuery({ queryKey: [...advancedKeys.plugins(), params], queryFn: () => advancedApi.getPlugins(params) });

export const useInstallPlugin = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: (data: { source: string; version?: string }) => advancedApi.installPlugin(data), onSuccess: () => qc.invalidateQueries({ queryKey: advancedKeys.plugins() }) });
};

export const usePluginMarketplace = (params?: Record<string, unknown>) =>
  useQuery({ queryKey: [...advancedKeys.plugins(), 'marketplace', params], queryFn: () => advancedApi.getPluginMarketplace(params) });

export const usePersonalities = (params?: Record<string, unknown>) =>
  useQuery({ queryKey: [...advancedKeys.personalities(), params], queryFn: () => advancedApi.getPersonalities(params) });

export const useCreatePersonality = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: (data: PersonalityCreateRequest) => advancedApi.createPersonality(data), onSuccess: () => qc.invalidateQueries({ queryKey: advancedKeys.personalities() }) });
};

export const useRobots = () =>
  useQuery({ queryKey: advancedKeys.robots(), queryFn: () => advancedApi.getRobots(), refetchInterval: 10000 });
