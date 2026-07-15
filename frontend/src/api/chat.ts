import apiClient from './client';
import { wrapListResponse } from './utils';
import type { Conversation, ConversationCreate, ConversationUpdate, Message, MessageCreate, MessageUpdate, MessageRate, ConversationStats, ConversationListResponse, MessageListResponse } from '../types/chat';

export const chatApi = {
  getConversations: async (params?: { status?: string; search?: string; page?: number; page_size?: number }) => {
    const response = await apiClient.get<ConversationListResponse>('/chat/conversations', { params });
    return wrapListResponse<Conversation>(response);
  },
  createConversation: (data: ConversationCreate) =>
    apiClient.post<Conversation>('/chat/conversations', data),
  getConversation: (id: number) =>
    apiClient.get<Conversation>(`/chat/conversations/${id}`),
  updateConversation: (id: number, data: ConversationUpdate) =>
    apiClient.put<Conversation>(`/chat/conversations/${id}`, data),
  deleteConversation: (id: number) =>
    apiClient.delete(`/chat/conversations/${id}`),
  getMessages: async (conversationId: number, params?: { page?: number; page_size?: number }) => {
    const response = await apiClient.get<MessageListResponse>(`/chat/conversations/${conversationId}/messages`, { params });
    return wrapListResponse<Message>(response);
  },
  sendMessage: (conversationId: number, data: MessageCreate) =>
    apiClient.post<Message>(`/chat/conversations/${conversationId}/messages`, data),
  streamMessage: (conversationId: number, data: MessageCreate) =>
    apiClient.post(`/chat/conversations/${conversationId}/stream`, data, { responseType: 'text' }),
  updateMessage: (messageId: number, data: MessageUpdate) =>
    apiClient.put<Message>(`/chat/messages/${messageId}`, data),
  deleteMessage: (messageId: number) =>
    apiClient.delete(`/chat/messages/${messageId}`),
  rateMessage: (messageId: number, data: MessageRate) =>
    apiClient.post(`/chat/messages/${messageId}/rate`, data),
  clearConversation: (conversationId: number) =>
    apiClient.post(`/chat/conversations/${conversationId}/clear`),
  exportConversation: (conversationId: number, format?: string) =>
    apiClient.post(`/chat/conversations/${conversationId}/export`, null, { params: { format } }),
  forkConversation: (conversationId: number) =>
    apiClient.post<Conversation>(`/chat/conversations/${conversationId}/fork`),
  getConversationStats: (conversationId: number) =>
    apiClient.get<ConversationStats>(`/chat/conversations/${conversationId}/stats`),
  regenerateMessage: (conversationId: number, messageId: number) =>
    apiClient.post<Message>(`/chat/conversations/${conversationId}/regenerate`, null, { params: { message_id: messageId } }),
  batchDeleteConversations: (ids: number[]) =>
    apiClient.post('/chat/conversations/batch-delete', ids),
  searchConversations: async (q: string) => {
    const response = await apiClient.get<ConversationListResponse>('/chat/conversations/search', { params: { q } });
    return wrapListResponse<Conversation>(response);
  },
  ragQuery: (query: string, options?: Record<string, unknown>) =>
    apiClient.post('/chat/rag/query', { query, ...options }),
  getChatWebSocketUrl: (conversationId: number) =>
    `${apiClient.defaults.baseURL?.replace('/api/v1', '') || ''}/api/v1/chat/ws/chat/${conversationId}`,
};
