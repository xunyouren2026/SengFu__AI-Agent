/**
 * 多智能体算法API集成
 */

import { apiClient } from './client';

// =============================================================================
// Alliance API
// =============================================================================

export interface AllianceCreateRequest {
  name: string;
  description: string;
  goals?: string[];
  max_members?: number;
}

export interface AllianceJoinRequest {
  agent_id: string;
  contribution: string;
}

export interface AllianceVoteRequest {
  voter_id: string;
  vote: 'approve' | 'reject' | 'abstain';
  reason?: string;
}

export async function createAlliance(request: AllianceCreateRequest) {
  const response = await apiClient.post('/multiagent/alliance/create', request);
  return response.data;
}

export async function joinAlliance(allianceId: string, request: AllianceJoinRequest) {
  const response = await apiClient.post(`/multiagent/alliance/${allianceId}/join`, request);
  return response.data;
}

export async function allianceVote(allianceId: string, request: AllianceVoteRequest) {
  const response = await apiClient.post(`/multiagent/alliance/${allianceId}/vote`, request);
  return response.data;
}

export async function listAlliances() {
  const response = await apiClient.get('/multiagent/alliance/list');
  return response.data;
}

// =============================================================================
// Debate API
// =============================================================================

export interface DebateStartRequest {
  topic: string;
  participants: string[];
  debate_type?: string;
}

export interface ArgumentSubmitRequest {
  participant_id: string;
  content: string;
  evidence?: string[];
}

export async function startDebate(request: DebateStartRequest) {
  const response = await apiClient.post('/multiagent/debate/start', request);
  return response.data;
}

export async function submitArgument(debateId: string, request: ArgumentSubmitRequest) {
  const response = await apiClient.post(`/multiagent/debate/${debateId}/argue`, request);
  return response.data;
}

export async function getDebateResult(debateId: string) {
  const response = await apiClient.get(`/multiagent/debate/${debateId}/result`);
  return response.data;
}

export async function listDebates() {
  const response = await apiClient.get('/multiagent/debate/list');
  return response.data;
}

// =============================================================================
// Reputation API
// =============================================================================

export interface ReputationUpdateRequest {
  agent_id: string;
  delta: number;
  reason: string;
}

export async function getReputation(agentId: string) {
  const response = await apiClient.get(`/multiagent/reputation/${agentId}`);
  return response.data;
}

export async function updateReputation(request: ReputationUpdateRequest) {
  const response = await apiClient.post('/multiagent/reputation/update', request);
  return response.data;
}

export async function reputationLeaderboard(limit: number = 10) {
  const response = await apiClient.get('/multiagent/reputation/leaderboard', { params: { limit } });
  return response.data;
}

// =============================================================================
// Market API
// =============================================================================

export interface BidRequest {
  agent_id: string;
  resource: string;
  amount: number;
  price: number;
}

export async function submitBid(request: BidRequest) {
  const response = await apiClient.post('/multiagent/market/bid', request);
  return response.data;
}

export async function getMarketOrders(resource?: string) {
  const params = resource ? { resource } : {};
  const response = await apiClient.get('/multiagent/market/orders', { params });
  return response.data;
}

// =============================================================================
// Health
// =============================================================================

export async function multiagentHealth() {
  const response = await apiClient.get('/multiagent/health');
  return response.data;
}

// =============================================================================
// 导出
// =============================================================================

export const multiagentAPI = {
  createAlliance,
  joinAlliance,
  allianceVote,
  listAlliances,
  startDebate,
  submitArgument,
  getDebateResult,
  listDebates,
  getReputation,
  updateReputation,
  reputationLeaderboard,
  submitBid,
  getMarketOrders,
  multiagentHealth,
};

export default multiagentAPI;
