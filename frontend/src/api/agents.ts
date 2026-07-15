import apiClient from './client';
import { wrapListResponse } from './utils';
import type { Agent, AgentCreateRequest, AgentUpdateRequest, AgentLogEntry, TaskExecutionRequest, TaskExecutionResponse, TaskHistoryItem, Alliance, AllianceCreateRequest, Debate, DebateCreateRequest, AgentListResponse, AllianceListResponse, DebateListResponse, TaskHistoryResponse } from '../types/agent';

export const agentsApi = {
  getAgents: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get<AgentListResponse>('/agents', { params });
    return wrapListResponse<Agent>(response);
  },
  createAgent: (data: AgentCreateRequest) =>
    apiClient.post<Agent>('/agents', data),
  getAgent: (id: string) =>
    apiClient.get<Agent>(`/agents/${id}`),
  updateAgent: (id: string, data: AgentUpdateRequest) =>
    apiClient.put<Agent>(`/agents/${id}`, data),
  deleteAgent: (id: string) =>
    apiClient.delete(`/agents/${id}`),
  activateAgent: (id: string) =>
    apiClient.post<Agent>(`/agents/${id}/activate`),
  deactivateAgent: (id: string) =>
    apiClient.post<Agent>(`/agents/${id}/deactivate`),
  getAgentLogs: (id: string, params?: Record<string, unknown>) =>
    apiClient.get(`/agents/${id}/logs`, { params }),
  executeTask: (id: string, data: TaskExecutionRequest) =>
    apiClient.post<TaskExecutionResponse>(`/agents/${id}/execute`, data),
  getTaskHistory: async (id: string, params?: Record<string, unknown>) => {
    const response = await apiClient.get<TaskHistoryResponse>(`/agents/${id}/tasks`, { params });
    return wrapListResponse<TaskHistoryItem>(response);
  },
  getAlliances: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get<AllianceListResponse>('/agents/alliances', { params });
    return wrapListResponse<Alliance>(response);
  },
  createAlliance: (data: AllianceCreateRequest) =>
    apiClient.post<Alliance>('/agents/alliances', data),
  getAlliance: (id: string) =>
    apiClient.get<Alliance>(`/agents/alliances/${id}`),
  joinAlliance: (id: string, data: { agent_id: string; role?: string }) =>
    apiClient.post<Alliance>(`/agents/alliances/${id}/join`, data),
  leaveAlliance: (id: string, data: { agent_id: string }) =>
    apiClient.post(`/agents/alliances/${id}/leave`, data),
  disbandAlliance: (id: string) =>
    apiClient.post(`/agents/alliances/${id}/disband`),
  getDebates: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get<DebateListResponse>('/agents/debates', { params });
    return wrapListResponse<Debate>(response);
  },
  createDebate: (data: DebateCreateRequest) =>
    apiClient.post<Debate>(`/agents/debates`, data),
  getDebate: (id: string) =>
    apiClient.get<Debate>(`/agents/debates/${id}`),
  voteDebate: (id: string, data: { voter_id: string; participant_id: string }) =>
    apiClient.post(`/agents/debates/${id}/vote`, data),
  getMarketplace: (params?: Record<string, unknown>) =>
    apiClient.get('/agents/marketplace', { params }),
};
