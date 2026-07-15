import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { systemApi } from '../api/system';
import type { UserCreateRequest, UserUpdateRequest, APIKeyCreateRequest, BackupCreateRequest, FeedbackRequest } from '../types/system';

export const systemKeys = {
  all: ['system'] as const,
  metrics: () => [...systemKeys.all, 'metrics'] as const,
  settings: () => [...systemKeys.all, 'settings'] as const,
  users: () => [...systemKeys.all, 'users'] as const,
  alerts: () => [...systemKeys.all, 'alerts'] as const,
  backups: () => [...systemKeys.all, 'backups'] as const,
  notifications: () => [...systemKeys.all, 'notifications'] as const,
  hardware: () => [...systemKeys.all, 'hardware'] as const,
};

export const useSystemMetrics = () =>
  useQuery({ queryKey: systemKeys.metrics(), queryFn: () => systemApi.getMetrics(), refetchInterval: 30000 });

export const useSettings = () =>
  useQuery({ queryKey: systemKeys.settings(), queryFn: () => systemApi.getSettings(), staleTime: 60000 });

export const useUpdateSettings = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: (data: { settings: Record<string, unknown> }) => systemApi.updateSettings(data), onSuccess: () => qc.invalidateQueries({ queryKey: systemKeys.settings() }) });
};

export const useUsers = (params?: Record<string, unknown>) =>
  useQuery({ queryKey: [...systemKeys.users(), params], queryFn: () => systemApi.getUsers(params) });

export const useCreateUser = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: (data: UserCreateRequest) => systemApi.createUser(data), onSuccess: () => qc.invalidateQueries({ queryKey: systemKeys.users() }) });
};

export const useDeleteUser = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: (id: string) => systemApi.deleteUser(id), onSuccess: () => qc.invalidateQueries({ queryKey: systemKeys.users() }) });
};

export const useAlerts = (params?: Record<string, unknown>) =>
  useQuery({ queryKey: [...systemKeys.alerts(), params], queryFn: () => systemApi.getAlerts(params), refetchInterval: 30000 });

export const useBackups = (params?: Record<string, unknown>) =>
  useQuery({ queryKey: [...systemKeys.backups(), params], queryFn: () => systemApi.getBackups(params) });

export const useCreateBackup = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: (data: BackupCreateRequest) => systemApi.createBackup(data), onSuccess: () => qc.invalidateQueries({ queryKey: systemKeys.backups() }) });
};

export const useNotifications = (params?: Record<string, unknown>) =>
  useQuery({ queryKey: [...systemKeys.notifications(), params], queryFn: () => systemApi.getNotifications(params), refetchInterval: 60000 });

export const useHardwareInfo = () =>
  useQuery({ queryKey: systemKeys.hardware(), queryFn: () => systemApi.getHardwareInfo(), staleTime: 60000 });

export const useHelpDocs = (params?: Record<string, unknown>) =>
  useQuery({ queryKey: [...systemKeys.all, 'help-docs', params], queryFn: () => systemApi.getHelpDocs(params) });

export const useFAQs = (params?: Record<string, unknown>) =>
  useQuery({ queryKey: [...systemKeys.all, 'faqs', params], queryFn: () => systemApi.getFAQs(params) });

export const useSubmitFeedback = () =>
  useMutation({ mutationFn: (data: FeedbackRequest) => systemApi.submitFeedback(data) });
