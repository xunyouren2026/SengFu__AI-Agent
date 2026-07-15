import apiClient from './client';
import { wrapListResponse } from './utils';
import type { SystemMetrics, LogEntry, Alert, UserInfo, UserCreateRequest, UserUpdateRequest, RoleInfo, APIKeyInfo, APIKeyCreateRequest, BackupInfo, BackupCreateRequest, GPUInfo, HardwareInfo, SystemSettings, HelpDocInfo, HelpDocContent, FAQItem, FeedbackRequest, ShortcutInfo, ChangelogEntry, DashboardConfig, UserListResponse, AlertListResponse, LogListResponse, BackupListResponse, HelpDocListResponse, FAQListResponse } from '../types/system';

export const systemApi = {
  // Metrics & Monitoring
  getMetrics: () =>
    apiClient.get<SystemMetrics>('/system/metrics'),
  getRealtimeMetrics: () =>
    apiClient.get('/system/metrics/realtime'),
  getLogs: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get<LogListResponse>('/system/logs', { params });
    return wrapListResponse<LogEntry>(response);
  },
  exportLogs: (data: Record<string, unknown>) =>
    apiClient.post('/system/logs/export', data),
  getAlerts: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get<AlertListResponse>('/system/alerts', { params });
    return wrapListResponse<Alert>(response);
  },
  acknowledgeAlert: (id: string, data?: { comment?: string }) =>
    apiClient.post(`/system/alerts/${id}/acknowledge`, data),
  resolveAlert: (id: string) =>
    apiClient.post(`/system/alerts/${id}/resolve`),
  // Hardware
  getHardwareInfo: () =>
    apiClient.get<HardwareInfo>('/hardware/info'),
  getGPUs: () =>
    apiClient.get('/hardware/gpus'),
  getGPUUtilization: (gpuId: string) =>
    apiClient.get(`/hardware/gpus/${gpuId}/utilization`),
  // Settings
  getSettings: () =>
    apiClient.get<SystemSettings>('/system/settings'),
  updateSettings: (data: { settings: Record<string, unknown> }) =>
    apiClient.put('/system/settings', data),
  getSettingCategories: () =>
    apiClient.get('/system/settings/categories'),
  // Users
  getUsers: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get<UserListResponse>('/system/users', { params });
    return wrapListResponse<UserInfo>(response);
  },
  createUser: (data: UserCreateRequest) =>
    apiClient.post<UserInfo>('/system/users', data),
  updateUser: (id: string, data: UserUpdateRequest) =>
    apiClient.put<UserInfo>(`/system/users/${id}`, data),
  deleteUser: (id: string) =>
    apiClient.delete(`/system/users/${id}`),
  resetUserPassword: (id: string, data: { new_password: string }) =>
    apiClient.post(`/system/users/${id}/reset-password`, data),
  // Roles
  getRoles: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get('/system/roles', { params });
    return wrapListResponse<RoleInfo>(response);
  },
  createRole: (data: { name: string; description?: string; permissions: string[] }) =>
    apiClient.post('/system/roles', data),
  updateRole: (id: string, data: Record<string, unknown>) =>
    apiClient.put(`/system/roles/${id}`, data),
  deleteRole: (id: string) =>
    apiClient.delete(`/system/roles/${id}`),
  // API Keys
  getApiKeys: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get('/system/api-keys', { params });
    return wrapListResponse<APIKeyInfo>(response);
  },
  createApiKey: (data: APIKeyCreateRequest) =>
    apiClient.post('/system/api-keys', data),
  deleteApiKey: (id: string) =>
    apiClient.delete(`/system/api-keys/${id}`),
  regenerateApiKey: (id: string) =>
    apiClient.post(`/system/api-keys/${id}/regenerate`),
  // Backups
  getBackups: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get<BackupListResponse>('/system/backups', { params });
    return wrapListResponse<BackupInfo>(response);
  },
  createBackup: (data: BackupCreateRequest) =>
    apiClient.post<BackupInfo>('/system/backups', data),
  restoreBackup: (id: string) =>
    apiClient.post(`/system/backups/${id}/restore`),
  downloadBackup: (id: string) =>
    apiClient.get(`/system/backups/${id}/download`),
  deleteBackup: (id: string) =>
    apiClient.delete(`/system/backups/${id}`),
  // Help
  getHelpDocs: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get<HelpDocListResponse>('/system/help/docs', { params });
    return wrapListResponse<HelpDocInfo>(response);
  },
  getHelpDoc: (id: string) =>
    apiClient.get<HelpDocContent>(`/system/help/docs/${id}`),
  searchHelpDocs: async (q: string) => {
    const response = await apiClient.get<HelpDocListResponse>('/system/help/docs/search', { params: { q } });
    return wrapListResponse<HelpDocInfo>(response);
  },
  getFAQs: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get<FAQListResponse>('/system/help/faqs', { params });
    return wrapListResponse<FAQItem>(response);
  },
  submitFeedback: (data: FeedbackRequest) =>
    apiClient.post('/system/help/feedback', data),
  getShortcuts: () =>
    apiClient.get('/system/help/shortcuts'),
  getChangelog: (params?: Record<string, unknown>) =>
    apiClient.get('/system/help/changelog', { params }),
  // Dashboard
  getDashboards: (params?: Record<string, unknown>) =>
    apiClient.get('/system/dashboards', { params }),
  updateDashboard: (id: string, data: Record<string, unknown>) =>
    apiClient.put(`/system/dashboards/${id}`, data),
  // Maintenance
  getMaintenanceStatus: () =>
    apiClient.get('/system/maintenance/status'),
  clearCache: () =>
    apiClient.post('/system/maintenance/clear-cache'),
  optimizeDatabase: () =>
    apiClient.post('/system/maintenance/optimize-db'),
  restartService: () =>
    apiClient.post('/system/maintenance/restart'),
  // Files
  getFiles: (params?: Record<string, unknown>) =>
    apiClient.get('/system/files', { params }),
  uploadFile: (file: File, path?: string) => {
    const formData = new FormData();
    formData.append('file', file);
    if (path) formData.append('path', path);
    return apiClient.post('/system/files/upload', formData, { headers: { 'Content-Type': 'multipart/form-data' } });
  },
  deleteFile: (path: string) =>
    apiClient.delete('/system/files', { params: { path } }),
  // Knowledge base
  getKnowledgeBases: (params?: Record<string, unknown>) =>
    apiClient.get('/system/knowledge-bases', { params }),
  // Notifications
  getNotifications: (params?: Record<string, unknown>) =>
    apiClient.get('/system/notifications', { params }),
  markNotificationRead: (id: string) =>
    apiClient.post(`/system/notifications/${id}/read`),
  markAllNotificationsRead: () =>
    apiClient.post('/system/notifications/read-all'),
};
