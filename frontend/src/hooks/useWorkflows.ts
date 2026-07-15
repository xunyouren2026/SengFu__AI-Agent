import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { workflowsApi } from '../api/workflows';
import type { WorkflowCreateRequest, WorkflowUpdateRequest, WorkflowExecuteRequest } from '../types/workflow';

export const workflowKeys = {
  all: ['workflows'] as const,
  lists: () => [...workflowKeys.all, 'list'] as const,
  list: (params?: Record<string, unknown>) => [...workflowKeys.lists(), params] as const,
  detail: (id: string) => [...workflowKeys.all, 'detail', id] as const,
  executions: (id: string) => [...workflowKeys.all, 'executions', id] as const,
};

export const useWorkflows = (params?: Record<string, unknown>) =>
  useQuery({ queryKey: workflowKeys.list(params), queryFn: () => workflowsApi.getWorkflows(params), staleTime: 5000 });

export const useWorkflow = (id: string) =>
  useQuery({ queryKey: workflowKeys.detail(id), queryFn: () => workflowsApi.getWorkflow(id), enabled: !!id });

export const useCreateWorkflow = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: (data: WorkflowCreateRequest) => workflowsApi.createWorkflow(data), onSuccess: () => qc.invalidateQueries({ queryKey: workflowKeys.lists() }) });
};

export const useUpdateWorkflow = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: ({ id, data }: { id: string; data: WorkflowUpdateRequest }) => workflowsApi.updateWorkflow(id, data), onSuccess: (_, { id }) => { qc.invalidateQueries({ queryKey: workflowKeys.lists() }); qc.invalidateQueries({ queryKey: workflowKeys.detail(id) }); } });
};

export const useDeleteWorkflow = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: (id: string) => workflowsApi.deleteWorkflow(id), onSuccess: () => qc.invalidateQueries({ queryKey: workflowKeys.lists() }) });
};

export const useExecuteWorkflow = () =>
  useMutation({ mutationFn: ({ id, data }: { id: string; data: WorkflowExecuteRequest }) => workflowsApi.executeWorkflow(id, data) });

export const useWorkflowExecutions = (id: string) =>
  useQuery({ queryKey: workflowKeys.executions(id), queryFn: () => workflowsApi.getWorkflowExecutions(id), enabled: !!id });

export const useWorkflowTemplates = (params?: Record<string, unknown>) =>
  useQuery({ queryKey: [...workflowKeys.all, 'templates', params], queryFn: () => workflowsApi.getTemplates(params) });
