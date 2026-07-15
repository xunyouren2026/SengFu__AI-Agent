import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { agentsApi } from '../api/agents';
import type { AgentCreateRequest, AgentUpdateRequest, TaskExecutionRequest, AllianceCreateRequest, DebateCreateRequest } from '../types/agent';

export const agentKeys = {
  all: ['agents'] as const,
  lists: () => [...agentKeys.all, 'list'] as const,
  list: (params?: Record<string, unknown>) => [...agentKeys.lists(), params] as const,
  details: () => [...agentKeys.all, 'detail'] as const,
  detail: (id: string) => [...agentKeys.details(), id] as const,
  alliances: () => [...agentKeys.all, 'alliances'] as const,
  debates: () => [...agentKeys.all, 'debates'] as const,
};

export const useAgents = (params?: Record<string, unknown>) =>
  useQuery({ queryKey: agentKeys.list(params), queryFn: () => agentsApi.getAgents(params), staleTime: 5000 });

export const useAgent = (id: string) =>
  useQuery({ queryKey: agentKeys.detail(id), queryFn: () => agentsApi.getAgent(id), enabled: !!id });

export const useCreateAgent = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: (data: AgentCreateRequest) => agentsApi.createAgent(data), onSuccess: () => qc.invalidateQueries({ queryKey: agentKeys.lists() }) });
};

export const useUpdateAgent = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: ({ id, data }: { id: string; data: AgentUpdateRequest }) => agentsApi.updateAgent(id, data), onSuccess: (_, { id }) => { qc.invalidateQueries({ queryKey: agentKeys.lists() }); qc.invalidateQueries({ queryKey: agentKeys.detail(id) }); } });
};

export const useDeleteAgent = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: (id: string) => agentsApi.deleteAgent(id), onSuccess: () => qc.invalidateQueries({ queryKey: agentKeys.lists() }) });
};

export const useExecuteTask = () =>
  useMutation({ mutationFn: ({ id, data }: { id: string; data: TaskExecutionRequest }) => agentsApi.executeTask(id, data) });

export const useAlliances = (params?: Record<string, unknown>) =>
  useQuery({ queryKey: [...agentKeys.alliances(), params], queryFn: () => agentsApi.getAlliances(params) });

export const useCreateAlliance = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: (data: AllianceCreateRequest) => agentsApi.createAlliance(data), onSuccess: () => qc.invalidateQueries({ queryKey: agentKeys.alliances() }) });
};

export const useDebates = (params?: Record<string, unknown>) =>
  useQuery({ queryKey: [...agentKeys.debates(), params], queryFn: () => agentsApi.getDebates(params) });

export const useCreateDebate = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: (data: DebateCreateRequest) => agentsApi.createDebate(data), onSuccess: () => qc.invalidateQueries({ queryKey: agentKeys.debates() }) });
};
