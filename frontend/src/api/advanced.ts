import apiClient from './client';
import { wrapListResponse } from './utils';
import type { Simulation, SimulationCreateRequest, ScreenshotResponse, ClickRequest, TypeRequest, ScrollRequest, NavigateRequest, Recording, FirewallRule, SecurityScanResponse, PromptGuardTestResponse, FederatedNode, FederatedNodeRegisterRequest, TrainingRound, AggregationRequest, DocumentUploadResponse, DocumentInfo, RAGSearchRequest, RAGSearchResponse, KnowledgeBaseInfo, DatasetInfo, Pipeline, PipelineCreateRequest, AlignmentPrinciple, AlignmentTestResponse, AlignmentReport, RobotInfo, RobotMoveRequest, RobotStatusResponse, ChannelResponse, ChannelCreateRequest, PluginInfo, PluginMarketplaceItem, PersonalityResponse, PersonalityCreateRequest, SimulationListResponse, FederatedNodeListResponse, DocumentListResponse, PipelineListResponse, ChannelListResponse, PluginListResponse, PersonalityListResponse } from '../types/advanced';

export const advancedApi = {
  // Physics
  getSimulations: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get<SimulationListResponse>('/physics/simulations', { params });
    return wrapListResponse<Simulation>(response);
  },
  createSimulation: (data: SimulationCreateRequest) =>
    apiClient.post<Simulation>('/physics/simulations', data),
  getSimulation: (id: string) =>
    apiClient.get<Simulation>(`/physics/simulations/${id}`),
  startSimulation: (id: string) =>
    apiClient.post(`/physics/simulations/${id}/start`),
  stopSimulation: (id: string) =>
    apiClient.post(`/physics/simulations/${id}/stop`),
  deleteSimulation: (id: string) =>
    apiClient.delete(`/physics/simulations/${id}`),
  // Computer Use
  screenshot: (params?: Record<string, unknown>) =>
    apiClient.post<ScreenshotResponse>('/computer/screenshot', params),
  click: (data: ClickRequest) =>
    apiClient.post('/computer/click', data),
  type: (data: TypeRequest) =>
    apiClient.post('/computer/type', data),
  scroll: (data: ScrollRequest) =>
    apiClient.post('/computer/scroll', data),
  navigate: (data: NavigateRequest) =>
    apiClient.post('/computer/navigate', data),
  keypress: (data: { key: string; modifiers?: string[] }) =>
    apiClient.post('/computer/keypress', data),
  startRecording: (config?: Record<string, unknown>) =>
    apiClient.post<Recording>('/computer/recording/start', config),
  stopRecording: (recordingId: string) =>
    apiClient.post<Recording>(`/computer/recording/${recordingId}/stop`),
  // Security
  getFirewallRules: async () => {
    const response = await apiClient.get('/security/firewall/rules');
    return wrapListResponse<FirewallRule>(response);
  },
  createFirewallRule: (data: Record<string, unknown>) =>
    apiClient.post<FirewallRule>('/security/firewall/rules', data),
  deleteFirewallRule: (id: string) =>
    apiClient.delete(`/security/firewall/rules/${id}`),
  getAuditLogs: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get('/security/audit-logs', { params });
    return wrapListResponse(response);
  },
  startSecurityScan: (data: { scan_type: string; target: string }) =>
    apiClient.post<SecurityScanResponse>('/security/scans', data),
  getScanResult: (scanId: string) =>
    apiClient.get<SecurityScanResponse>(`/security/scans/${scanId}`),
  testPromptGuard: (data: { prompt: string; test_categories?: string[] }) =>
    apiClient.post<PromptGuardTestResponse>('/security/prompt-guard/test', data),
  getThreats: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get('/security/threats', { params });
    return wrapListResponse(response);
  },
  // Federated Learning
  getNodes: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get<FederatedNodeListResponse>('/federated/nodes', { params });
    return wrapListResponse<FederatedNode>(response);
  },
  registerNode: (data: FederatedNodeRegisterRequest) =>
    apiClient.post<FederatedNode>('/federated/nodes', data),
  getNode: (id: string) =>
    apiClient.get<FederatedNode>(`/federated/nodes/${id}`),
  deleteNode: (id: string) =>
    apiClient.delete(`/federated/nodes/${id}`),
  getTrainingRounds: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get('/federated/rounds', { params });
    return wrapListResponse<TrainingRound>(response);
  },
  aggregate: (data: AggregationRequest) =>
    apiClient.post('/federated/aggregate', data),
  getContributionStats: () =>
    apiClient.get('/federated/contributions'),
  // RAG
  uploadDocument: (file: File, knowledgeBaseId?: string) => {
    const formData = new FormData();
    formData.append('file', file);
    if (knowledgeBaseId) formData.append('knowledge_base_id', knowledgeBaseId);
    return apiClient.post<DocumentUploadResponse>('/rag/documents/upload', formData, { headers: { 'Content-Type': 'multipart/form-data' } });
  },
  getDocuments: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get<DocumentListResponse>('/rag/documents', { params });
    return wrapListResponse<DocumentInfo>(response);
  },
  deleteDocument: (id: string) =>
    apiClient.delete(`/rag/documents/${id}`),
  search: (data: RAGSearchRequest) =>
    apiClient.post<RAGSearchResponse>('/rag/search', data),
  getKnowledgeBases: async () => {
    const response = await apiClient.get<KnowledgeBaseInfo[]>('/rag/knowledge-bases');
    // 页面直接使用数组
    if (Array.isArray(response)) return response;
    if (response && typeof response === 'object' && 'data' in response) return (response as any).data;
    return [response];
  },
  // Data Pipeline
  getDatasets: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get('/data-pipeline/datasets', { params });
    return wrapListResponse<DatasetInfo>(response);
  },
  uploadDataset: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return apiClient.post('/data-pipeline/datasets/upload', formData, { headers: { 'Content-Type': 'multipart/form-data' } });
  },
  deleteDataset: (id: string) =>
    apiClient.delete(`/data-pipeline/datasets/${id}`),
  getPipelines: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get<PipelineListResponse>('/data-pipeline/pipelines', { params });
    return wrapListResponse<Pipeline>(response);
  },
  createPipeline: (data: PipelineCreateRequest) =>
    apiClient.post<Pipeline>('/data-pipeline/pipelines', data),
  getPipeline: (id: string) =>
    apiClient.get<Pipeline>(`/data-pipeline/pipelines/${id}`),
  // Alignment
  getPrinciples: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get('/alignment/principles', { params });
    return wrapListResponse<AlignmentPrinciple>(response);
  },
  createPrinciple: (data: Record<string, unknown>) =>
    apiClient.post<AlignmentPrinciple>('/alignment/principles', data),
  deletePrinciple: (id: string) =>
    apiClient.delete(`/alignment/principles/${id}`),
  runAlignmentTest: (data: Record<string, unknown>) =>
    apiClient.post<AlignmentTestResponse>('/alignment/tests', data),
  getTestResult: (testId: string) =>
    apiClient.get<AlignmentTestResponse>(`/alignment/tests/${testId}`),
  getReport: (testId: string) =>
    apiClient.get<AlignmentReport>(`/alignment/tests/${testId}/report`),
  // Robot
  getRobots: async () => {
    const response = await apiClient.get<RobotInfo[]>('/robot/robots');
    // 页面直接使用数组
    if (Array.isArray(response)) return response;
    if (response && typeof response === 'object' && 'data' in response) return (response as any).data;
    return [response];
  },
  getRobotStatus: (id: string) =>
    apiClient.get<RobotStatusResponse>(`/robot/robots/${id}/status`),
  moveRobot: (id: string, data: RobotMoveRequest) =>
    apiClient.post(`/robot/robots/${id}/move`, data),
  stopRobot: (id: string) =>
    apiClient.post(`/robot/robots/${id}/stop`),
  // Channels (advanced)
  getChannels: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get<ChannelListResponse>('/channels', { params });
    return wrapListResponse<ChannelResponse>(response);
  },
  createChannel: (data: ChannelCreateRequest) =>
    apiClient.post<ChannelResponse>('/channels', data),
  updateChannel: (id: string, data: Record<string, unknown>) =>
    apiClient.put<ChannelResponse>(`/channels/${id}`, data),
  deleteChannel: (id: string) =>
    apiClient.delete(`/channels/${id}`),
  testChannel: (id: string, data: { test_message: string }) =>
    apiClient.post(`/channels/${id}/test`, data),
  // Plugins (advanced)
  getPlugins: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get<PluginListResponse>('/plugins', { params });
    return wrapListResponse<PluginInfo>(response);
  },
  installPlugin: (data: { source: string; version?: string; config?: Record<string, unknown> }) =>
    apiClient.post<PluginInfo>('/plugins/install', data),
  uninstallPlugin: (id: string) =>
    apiClient.post(`/plugins/${id}/uninstall`),
  enablePlugin: (id: string) =>
    apiClient.post<PluginInfo>(`/plugins/${id}/enable`),
  disablePlugin: (id: string) =>
    apiClient.post<PluginInfo>(`/plugins/${id}/disable`),
  configurePlugin: (id: string, config: Record<string, unknown>) =>
    apiClient.post<PluginInfo>(`/plugins/${id}/config`, { config }),
  getPluginMarketplace: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get<PluginMarketplaceItem[]>('/plugins/marketplace', { params });
    // 页面直接使用数组
    if (Array.isArray(response)) return response;
    if (response && typeof response === 'object' && 'data' in response) return (response as any).data;
    return [response];
  },
  // Personality (advanced)
  getPersonalities: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get<PersonalityListResponse>('/personality', { params });
    return wrapListResponse<PersonalityResponse>(response);
  },
  createPersonality: (data: PersonalityCreateRequest) =>
    apiClient.post<PersonalityResponse>('/personality', data),
  getPersonality: (id: string) =>
    apiClient.get<PersonalityResponse>(`/personality/${id}`),
  updatePersonality: (id: string, data: Record<string, unknown>) =>
    apiClient.put<PersonalityResponse>(`/personality/${id}`, data),
  deletePersonality: (id: string) =>
    apiClient.delete(`/personality/${id}`),
  applyPersonality: (id: string, data: { target_type: string; target_id?: string }) =>
    apiClient.post(`/personality/${id}/apply`, data),
  clonePersonality: (id: string, newName: string) =>
    apiClient.post<PersonalityResponse>(`/personality/${id}/clone`, null, { params: { new_name: newName } }),
  exportPersonality: (id: string, format?: string) =>
    apiClient.get(`/personality/${id}/export`, { params: { format } }),
};
