import { useState, useRef, useEffect } from 'react';
import { Send, Plus, Search, Settings, MoreVertical, Copy, RefreshCw, Star, Trash2, MessageSquare, Bot, User, Paperclip, Mic, Sparkles } from 'lucide-react';
import { Card, Button, Input, Badge, Modal, Select, EmptyState, Loading, Tabs, TabPanel } from '../components/common';
import { useConversations, useMessages, useCreateConversation, useSendMessage, useDeleteConversation } from '../hooks/useChat';
import { useModels } from '../hooks/useModels';

export default function Chat() {
  const [selectedConversation, setSelectedConversation] = useState<number | null>(null);
  const [newMessage, setNewMessage] = useState('');
  const [showNewChat, setShowNewChat] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { data: conversations, isLoading: convLoading } = useConversations();
  const { data: messages, isLoading: msgLoading } = useMessages(selectedConversation || 0, { page_size: 50 });
  const { data: models } = useModels();
  const sendMessageMutation = useSendMessage();
  const createConvMutation = useCreateConversation();
  const deleteConvMutation = useDeleteConversation();

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = () => {
    if (!newMessage.trim() || !selectedConversation) return;
    sendMessageMutation.mutate({ conversationId: selectedConversation, data: { content: newMessage.trim() } }, { onSuccess: () => setNewMessage('') });
  };

  const handleNewChat = () => {
    createConvMutation.mutate({ title: '新对话' }, {
      onSuccess: (conv) => { setSelectedConversation((conv as any).id); setShowNewChat(false); }
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  return (
    <div className="flex h-[calc(100vh-7rem)] gap-4 -m-6">
      {/* Conversation Sidebar */}
      <Card className="w-72 flex flex-col shrink-0 rounded-none border-0 border-r border-gray-200 dark:border-dark-600">
        <div className="p-3 border-b border-gray-100 dark:border-dark-600">
          <div className="flex items-center justify-between mb-2">
            <h2 className="font-semibold text-gray-900 dark:text-white text-sm">对话列表</h2>
            <Button size="sm" leftIcon={<Plus size={14} />} onClick={handleNewChat}>新建</Button>
          </div>
          <Input placeholder={'"搜索对话..."'} size="sm" value={searchQuery} onChange={e => setSearchQuery(e.target.value)} leftIcon={<Search size={14} />} />
        </div>
        <div className="flex-1 overflow-y-auto">
          {convLoading ? <Loading text={'"加载中..."'} /> :
            (conversations?.data || []).length === 0 ? <EmptyState icon={<MessageSquare size={32} />} title={'"暂无对话"'} description={'"开始你的第一次 AI 对话"'} /> :
              (conversations?.data || []).map((conv: any) => (
                <button key={conv.id} onClick={() => setSelectedConversation(conv.id)}
                  className={`w-full text-left px-3 py-2.5 border-b border-gray-50 dark:border-dark-700 hover:bg-gray-50 dark:hover:bg-dark-700 transition-colors ${selectedConversation === conv.id ? 'bg-primary-50 dark:bg-primary-900/20 border-l-2 border-l-primary-500' : ''}`}>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-900 dark:text-white truncate">{conv.title}</span>
                    <button onClick={e => { e.stopPropagation(); deleteConvMutation.mutate(conv.id); }} className="p-1 rounded hover:bg-gray-200 dark:hover:bg-dark-600 opacity-0 group-hover:opacity-100"><Trash2 size={12} className="text-gray-400" /></button>
                  </div>
                  <p className="text-xs text-gray-500 mt-0.5">{conv.message_count} 条消息 · {conv.model_name || '默认模型'}</p>
                </button>
              ))}
        </div>
      </Card>

      {/* Chat Area */}
      <div className="flex-1 flex flex-col bg-white dark:bg-dark-800 rounded-xl border border-gray-200 dark:border-dark-600">
        {/* Chat Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-dark-600">
          <div className="flex items-center gap-2">
            <Bot size={20} className="text-primary-500" />
            <span className="font-medium text-gray-900 dark:text-white">{(conversations?.data || []).find((c: any) => c.id === selectedConversation)?.title || '选择一个对话'}</span>
          </div>
          <div className="flex items-center gap-2">
            <Select options={models?.map(m => ({ label: m.name, value: m.id })) || []} className="w-40" size="sm" />
            <Button variant="ghost" size="sm"><Settings size={16} /></Button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {!selectedConversation ? (
            <EmptyState icon={<Sparkles size={48} />} title={'"开始新对话"'} description={'"选择左侧对话或创建新对话"'} action={<Button leftIcon={<Plus size={16} />} onClick={handleNewChat}>新建对话</Button>} />
          ) : msgLoading ? <Loading text={'"加载消息..."'} /> :
            (messages?.data || []).map((msg: any) => (
              <div key={msg.id} className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${msg.role === 'user' ? 'bg-primary-500' : 'bg-gray-200 dark:bg-dark-600'}`}>
                  {msg.role === 'user' ? <User size={16} className="text-white" /> : <Bot size={16} className="text-gray-600 dark:text-gray-300" />}
                </div>
                <div className={`max-w-[70%] rounded-2xl px-4 py-2.5 ${msg.role === 'user' ? 'bg-primary-500 text-white' : 'bg-gray-100 dark:bg-dark-700 text-gray-900 dark:text-white'}`}>
                  <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                  {msg.total_tokens > 0 && <p className="text-xs mt-1 opacity-60">{msg.total_tokens} tokens · {msg.latency_ms ? `${msg.latency_ms}ms` : ''}</p>}
                </div>
              </div>
            ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-4 border-t border-gray-200 dark:border-dark-600">
          <div className="flex items-end gap-2">
            <Button variant="ghost" size="sm"><Paperclip size={18} /></Button>
            <div className="flex-1">
              <textarea value={newMessage} onChange={e => setNewMessage(e.target.value)} onKeyDown={handleKeyDown} placeholder={'"输入消息... (Enter 发送, Shift+Enter 换行)"'} rows={1} className="w-full px-4 py-2.5 rounded-xl border border-gray-200 dark:border-dark-600 bg-gray-50 dark:bg-dark-700 text-sm text-gray-900 dark:text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-primary-500 resize-none" style={{ minHeight: '42px', maxHeight: '120px' }} />
            </div>
            <Button variant="ghost" size="sm"><Mic size={18} /></Button>
            <Button onClick={handleSend} disabled={!newMessage.trim() || !selectedConversation} leftIcon={<Send size={16} />}>发送</Button>
          </div>
        </div>
      </div>
    </div>
  );
}
