import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { chatApi } from '../api/chat';
import type { ConversationCreate, ConversationUpdate, MessageCreate, MessageUpdate, MessageRate } from '../types/chat';

export const chatKeys = {
  all: ['chat'] as const,
  conversations: () => [...chatKeys.all, 'conversations'] as const,
  conversation: (id: number) => [...chatKeys.conversations(), id] as const,
  messages: (conversationId: number) => [...chatKeys.all, 'messages', conversationId] as const,
};

export const useConversations = (params?: { status?: string; search?: string; page?: number; page_size?: number }) =>
  useQuery({ queryKey: [...chatKeys.conversations(), params], queryFn: () => chatApi.getConversations(params), staleTime: 5000 });

export const useConversation = (id: number) =>
  useQuery({ queryKey: chatKeys.conversation(id), queryFn: () => chatApi.getConversation(id), enabled: !!id });

export const useMessages = (conversationId: number, params?: { page?: number; page_size?: number }) =>
  useQuery({ queryKey: chatKeys.messages(conversationId), queryFn: () => chatApi.getMessages(conversationId, params), enabled: !!conversationId, staleTime: 3000 });

export const useCreateConversation = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: (data: ConversationCreate) => chatApi.createConversation(data), onSuccess: () => qc.invalidateQueries({ queryKey: chatKeys.conversations() }) });
};

export const useUpdateConversation = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: ({ id, data }: { id: number; data: ConversationUpdate }) => chatApi.updateConversation(id, data), onSuccess: (_, { id }) => { qc.invalidateQueries({ queryKey: chatKeys.conversations() }); qc.invalidateQueries({ queryKey: chatKeys.conversation(id) }); } });
};

export const useDeleteConversation = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: (id: number) => chatApi.deleteConversation(id), onSuccess: () => qc.invalidateQueries({ queryKey: chatKeys.conversations() }) });
};

export const useSendMessage = () => {
  const qc = useQueryClient();
  return useMutation({ mutationFn: ({ conversationId, data }: { conversationId: number; data: MessageCreate }) => chatApi.sendMessage(conversationId, data), onSuccess: (_, { conversationId }) => { qc.invalidateQueries({ queryKey: chatKeys.messages(conversationId) }); qc.invalidateQueries({ queryKey: chatKeys.conversation(conversationId) }); } });
};

export const useRateMessage = () =>
  useMutation({ mutationFn: ({ messageId, data }: { messageId: number; data: MessageRate }) => chatApi.rateMessage(messageId, data) });

export const useConversationStats = (conversationId: number) =>
  useQuery({ queryKey: [...chatKeys.conversation(conversationId), 'stats'], queryFn: () => chatApi.getConversationStats(conversationId), enabled: !!conversationId });
