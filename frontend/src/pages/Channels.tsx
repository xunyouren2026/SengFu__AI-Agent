import { useState } from 'react';
import { MessageSquare, Plus, Trash2, Edit, TestTube, CheckCircle, XCircle, AlertTriangle, Settings, Mail, Hash, Send, Bell, Webhook } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Badge, Input, Select, Textarea, Tabs, TabPanel, EmptyState, Loading, StatsCard, Modal, Table } from '../components/common';
import { useChannels, useCreateChannel, useDeleteChannel } from '../hooks/useAdvanced';

export default function Channels() {
  const [activeTab, setActiveTab] = useState('channels');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showTestModal, setShowTestModal] = useState(false);
  const [testChannelId, setTestChannelId] = useState('');
  const [testMessage, setTestMessage] = useState('');
  const [newChannel, setNewChannel] = useState({ name: '', type: 'webhook' as string, description: '', webhook_url: '', api_key: '', is_default: false });

  const { data: channelsData, isLoading } = useChannels();
  const createChannel = useCreateChannel();
  const deleteChannel = useDeleteChannel();

  const channels = (channelsData?.data || []) as any[];

  const handleCreate = () => {
    if (!newChannel.name.trim()) return;
    createChannel.mutate({
      name: newChannel.name,
      type: newChannel.type as any,
      description: newChannel.description || undefined,
      config: {
        webhook_url: newChannel.webhook_url || undefined,
        api_key: newChannel.api_key || undefined,
        settings: {},
      },
      is_default: newChannel.is_default,
    });
    setShowCreateModal(false);
    setNewChannel({ name: '', type: 'webhook', description: '', webhook_url: '', api_key: '', is_default: false });
  };

  const handleDelete = (id: string) => {
    if (confirm('确定要删除此渠道吗？')) {
      deleteChannel.mutate(id);
    }
  };

  const handleTest = (id: string) => {
    setTestChannelId(id);
    setTestMessage('');
    setShowTestModal(true);
  };

  const handleSendTest = () => {
    if (!testMessage.trim()) return;
    // Would call testChannel API
    setShowTestModal(false);
  };

  const typeLabel: Record<string, string> = {
    email: '邮件', slack: 'Slack', discord: 'Discord', telegram: 'Telegram', wechat: '微信', webhook: 'Webhook', sms: '短信',
  };
  const typeIcon: Record<string, any> = {
    email: <Mail size={16} />, slack: <Hash size={16} />, discord: <MessageSquare size={16} />, telegram: <Send size={16} />, wechat: <MessageSquare size={16} />, webhook: <Webhook size={16} />, sms: <Bell size={16} />,
  };
  const statusLabel: Record<string, string> = { active: '活跃', inactive: '未激活', error: '错误', pending: '待配置' };
  const statusVariant: Record<string, any> = { active: 'success', inactive: 'default', error: 'danger', pending: 'warning' };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">渠道管理</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">多渠道消息通知与集成管理</p>
        </div>
        <Button leftIcon={<Plus size={16} />} onClick={() => setShowCreateModal(true)}>创建渠道</Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatsCard title={'"渠道总数"'} value={channels.length} icon={<MessageSquare size={24} />} color="blue" />
        <StatsCard title={'"活跃渠道"'} value={channels.filter((c: any) => c.status === 'active').length} icon={<CheckCircle size={24} />} color="green" />
        <StatsCard title={'"错误渠道"'} value={channels.filter((c: any) => c.status === 'error').length} icon={<AlertTriangle size={24} />} color="red" />
        <StatsCard title={'"总消息数"'} value={channels.reduce((sum: number, c: any) => sum + (c.message_count || 0), 0)} icon={<Send size={24} />} color="purple" />
      </div>

      <Card>
        <CardHeader><CardTitle>渠道列表</CardTitle></CardHeader>
        <CardContent>
          {isLoading ? (
            <Loading />
          ) : channels.length === 0 ? (
            <EmptyState icon={<MessageSquare size={32} />} title={'"暂无渠道"'} description={'"创建渠道以接收消息通知"'} action={<Button leftIcon={<Plus size={16} />} onClick={() => setShowCreateModal(true)}>创建渠道</Button>} />
          ) : (
            <Table
              columns={[
                { key: 'name', title: '名称', render: (val: any, row: any) => (
                  <div className="flex items-center gap-2">
                    {typeIcon[row.type]}
                    <div>
                      <span className="font-medium text-gray-900 dark:text-white">{val}</span>
                      {row.is_default && <Badge variant="primary" size="sm" className="ml-2">默认</Badge>}
                    </div>
                  </div>
                )},
                { key: 'type', title: '类型', render: (val: any) => <Badge variant="outline">{typeLabel[val] || val}</Badge> },
                { key: 'status', title: '状态', render: (val: any) => <Badge variant={statusVariant[val]} dot>{statusLabel[val] || val}</Badge> },
                { key: 'message_count', title: '消息数' },
                { key: 'last_message_at', title: '最后消息' },
                { key: 'actions', title: '操作', render: (_: any, row: any) => (
                  <div className="flex gap-1">
                    <Button variant="ghost" size="sm" leftIcon={<Settings size={14} />}>配置</Button>
                    <Button variant="ghost" size="sm" leftIcon={<TestTube size={14} />} onClick={() => handleTest(row.id)}>测试</Button>
                    <Button variant="ghost" size="sm" leftIcon={<Edit size={14} />}>编辑</Button>
                    <Button variant="ghost" size="sm" className="text-red-500" leftIcon={<Trash2 size={14} />} onClick={() => handleDelete(row.id)}>删除</Button>
                  </div>
                )},
              ]}
              data={channels}
            />
          )}
        </CardContent>
      </Card>

      <Modal isOpen={showCreateModal} onClose={() => setShowCreateModal(false)} title={'"创建渠道"'} size="md" footer={
        <div className="flex gap-2">
          <Button variant="secondary" onClick={() => setShowCreateModal(false)}>取消</Button>
          <Button onClick={handleCreate} isLoading={createChannel.isPending}>创建</Button>
        </div>
      }>
        <div className="space-y-4">
          <Input label={'"渠道名称"'} placeholder={'"输入渠道名称"'} value={newChannel.name} onChange={e => setNewChannel({ ...newChannel, name: e.target.value })} />
          <Select label={'"渠道类型"'} value={newChannel.type} onChange={e => setNewChannel({ ...newChannel, type: e.target.value })} options={[
            { label: 'Webhook', value: 'webhook' },
            { label: '邮件', value: 'email' },
            { label: 'Slack', value: 'slack' },
            { label: 'Discord', value: 'discord' },
            { label: 'Telegram', value: 'telegram' },
            { label: '微信', value: 'wechat' },
            { label: '短信', value: 'sms' },
          ]} />
          <Textarea label={'"描述"'} placeholder={'"渠道描述..."'} value={newChannel.description} onChange={e => setNewChannel({ ...newChannel, description: e.target.value })} rows={2} />
          {(newChannel.type === 'webhook' || newChannel.type === 'slack' || newChannel.type === 'discord') && (
            <Input label="Webhook URL" placeholder="https://hooks.example.com/..." value={newChannel.webhook_url} onChange={e => setNewChannel({ ...newChannel, webhook_url: e.target.value })} />
          )}
          {(newChannel.type === 'telegram' || newChannel.type === 'wechat') && (
            <Input label="API Key" placeholder={'"输入 API Key"'} type="password" value={newChannel.api_key} onChange={e => setNewChannel({ ...newChannel, api_key: e.target.value })} />
          )}
          <div className="flex items-center gap-2">
            <input type="checkbox" id="is_default" checked={newChannel.is_default} onChange={e => setNewChannel({ ...newChannel, is_default: e.target.checked })} className="rounded" />
            <label htmlFor="is_default" className="text-sm text-gray-700 dark:text-gray-300">设为默认渠道</label>
          </div>
        </div>
      </Modal>

      <Modal isOpen={showTestModal} onClose={() => setShowTestModal(false)} title={'"测试渠道"'} size="md" footer={
        <div className="flex gap-2">
          <Button variant="secondary" onClick={() => setShowTestModal(false)}>取消</Button>
          <Button onClick={handleSendTest}>发送测试</Button>
        </div>
      }>
        <div className="space-y-4">
          <Textarea label={'"测试消息"'} placeholder={'"输入测试消息内容..."'} value={testMessage} onChange={e => setTestMessage(e.target.value)} rows={3} />
          <p className="text-sm text-gray-500 dark:text-gray-400">消息将发送到所选渠道进行测试</p>
        </div>
      </Modal>
    </div>
  );
}
