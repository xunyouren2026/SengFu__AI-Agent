import apiClient from './client';
import { wrapListResponse } from './utils';
import type { Workflow, WorkflowCreateRequest, WorkflowUpdateRequest, WorkflowExecuteRequest, WorkflowExecutionDetail, WorkflowTemplate, NodeTypeInfo, WorkflowListResponse, WorkflowExecutionListResponse } from '../types/workflow';

export const workflowsApi = {
  getWorkflows: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get<WorkflowListResponse>('/workflows', { params });
    return wrapListResponse<Workflow>(response);
  },
  createWorkflow: (data: WorkflowCreateRequest) =>
    apiClient.post<Workflow>('/workflows', data),
  getWorkflow: (id: string) =>
    apiClient.get<Workflow>(`/workflows/${id}`),
  updateWorkflow: (id: string, data: WorkflowUpdateRequest) =>
    apiClient.put<Workflow>(`/workflows/${id}`, data),
  deleteWorkflow: (id: string) =>
    apiClient.delete(`/workflows/${id}`),
  executeWorkflow: (id: string, data: WorkflowExecuteRequest) =>
    apiClient.post(`/workflows/${id}/execute`, data),
  getWorkflowExecutions: async (id: string, params?: Record<string, unknown>) => {
    const response = await apiClient.get<WorkflowExecutionListResponse>(`/workflows/${id}/executions`, { params });
    return wrapListResponse<WorkflowExecutionDetail>(response);
  },
  getExecutionDetail: (executionId: string) =>
    apiClient.get<WorkflowExecutionDetail>(`/workflows/executions/${executionId}`),
  cancelExecution: (executionId: string) =>
    apiClient.post(`/workflows/executions/${executionId}/cancel`),
  getTemplates: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get('/workflows/templates', { params });
    // 页面使用 templates?.templates 访问
    if (Array.isArray(response)) {
      return { templates: response };
    }
    if (response && typeof response === 'object' && 'templates' in response) {
      return response;
    }
    return { templates: Array.isArray(response) ? response : [response] };
  },
  cloneWorkflow: (id: string) =>
    apiClient.post<Workflow>(`/workflows/${id}/clone`),
  publishWorkflow: (id: string) =>
    apiClient.post<Workflow>(`/workflows/${id}/publish`),
  unpublishWorkflow: (id: string) =>
    apiClient.post<Workflow>(`/workflows/${id}/unpublish`),
  getNodeTypes: () =>
    apiClient.get('/workflows/nodes/types'),
  getWebSocketUrl: (id: string) =>
    `${apiClient.defaults.baseURL?.replace('/api/v1', '') || ''}/api/v1/workflows/ws/${id}/execute`,
};
