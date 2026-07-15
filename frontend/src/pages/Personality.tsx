import { useState } from 'react';
import { User, Plus, Trash2, Edit, TestTube, Copy, Download, Star, MessageSquare, Settings, Palette, Brain } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Badge, Input, Select, Textarea, Tabs, TabPanel, EmptyState, Loading, StatsCard, Modal, Table } from '../components/common';
import { usePersonalities, useCreatePersonality } from '../hooks/useAdvanced';

export default function Personality() {
  const [activeTab, setActiveTab] = useState('personalities');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showTestModal, setShowTestModal] = useState(false);
  const [testPersonalityId, setTestPersonalityId] = useState('');
  const [testMessage, setTestMessage] = useState('');
  const [testResponse, setTestResponse] = useState('');
  const [newPersonality, setNewPersonality] = useState({
    name: '',
    description: '',
    system_prompt: '',
    communication_style: 'professional' as string,
    openness: 70,
    conscientiousness: 70,
    extraversion: 50,
    agreeableness: 70,
    neuroticism: 30,
  });

  const { data: personalitiesData, isLoading } = usePersonalities();
  const createPersonality = useCreatePersonality();

  const personalities = (personalitiesData?.data || []) as any[];

  const handleCreate = () => {
    if (!newPersonality.name.trim() || !newPersonality.system_prompt.trim()) return;
    createPersonality.mutate({
      name: newPersonality.name,
      description: newPersonality.description || undefined,
      system_prompt: newPersonality.system_prompt,
      traits: {
        openness: newPersonality.openness / 100,
        conscientiousness: newPersonality.conscientiousness / 100,
        extraversion: newPersonality.extraversion / 100,
        agreeableness: newPersonality.agreeableness / 100,
        neuroticism: newPersonality.neuroticism / 100,
      },
      communication_style: newPersonality.communication_style as any,
    });
    setShowCreateModal(false);
    setNewPersonality({ name: '', description: '', system_prompt: '', communication_style: 'professional', openness: 70, conscientiousness: 70, extraversion: 50, agreeableness: 70, neuroticism: 30 });
  };

  const handleTest = (id: string) => {
    setTestPersonalityId(id);
    setTestMessage('');
    setTestResponse('');
    setShowTestModal(true);
  };

  const handleSendTest = () => {
    if (!testMessage.trim()) return;
    setTestResponse('这是使用该人格生成的模拟回复。在实际环境中，将调用 AI 模型生成真实的个性化回复。');
  };

  const styleLabel: Record<string, string> = { formal: '正式', casual: '随意', technical: '技术', friendly: '友好', professional: '专业', humorous: '幽默' };
  const traitLabels: Record<string, string> = { openness: '开放性', conscientiousness: '尽责性', extraversion: '外向性', agreeableness: '宜人性', neuroticism: '神经质' };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">人格引擎</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">AI 人格配置与个性化对话管理</p>
        </div>
        <Button leftIcon={<Plus size={16} />} onClick={() => setShowCreateModal(true)}>创建人格</Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatsCard title={'"人格总数"'} value={personalities.length} icon={<Palette size={24} />} color="blue" />
        <StatsCard title={'"活跃人格"'} value={personalities.filter((p: any) => p.is_active).length} icon={<Star size={24} />} color="green" />
        <StatsCard title={'"总使用次数"'} value={personalities.reduce((sum: number, p: any) => sum + (p.usage_count || 0), 0)} icon={<MessageSquare size={24} />} color="purple" />
        <StatsCard title={'"平均特质分"'} value="65" icon={<Brain size={24} />} color="indigo" />
      </div>

      <Card>
        <CardHeader><CardTitle>人格列表</CardTitle></CardHeader>
        <CardContent>
          {isLoading ? (
            <Loading />
          ) : personalities.length === 0 ? (
            <EmptyState icon={<User size={32} />} title={'"暂无人格配置"'} description={'"创建人格以实现个性化对话"'} action={<Button leftIcon={<Plus size={16} />} onClick={() => setShowCreateModal(true)}>创建人格</Button>} />
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {personalities.map((p: any) => (
                <Card key={p.id} hoverable>
                  <CardContent>
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary-500 to-purple-500 flex items-center justify-center text-white font-bold">
                          {p.name?.charAt(0) || '?'}
                        </div>
                        <div>
                          <div className="flex items-center gap-1">
                            <p className="font-medium text-gray-900 dark:text-white">{p.name}</p>
                            {p.is_active && <Badge variant="success" size="sm">活跃</Badge>}
                          </div>
                          <p className="text-xs text-gray-500">{styleLabel[p.communication_style] || p.communication_style}</p>
                        </div>
                      </div>
                    </div>
                    {p.description && <p className="text-sm text-gray-500 dark:text-gray-400 mb-3 line-clamp-2">{p.description}</p>}
                    {p.traits && (
                      <div className="space-y-1 mb-3">
                        {Object.entries(p.traits).map(([key, val]: [string, any]) => (
                          <div key={key} className="flex items-center gap-2">
                            <span className="text-xs text-gray-500 w-16">{traitLabels[key]}</span>
                            <div className="flex-1 h-1.5 bg-gray-100 dark:bg-dark-600 rounded-full overflow-hidden">
                              <div className="h-full bg-primary-500 rounded-full" style={{ width: `${(val as number) * 100}%` }} />
                            </div>
                            <span className="text-xs text-gray-400 w-8">{((val as number) * 100).toFixed(0)}</span>
                          </div>
                        ))}
                      </div>
                    )}
                    <div className="flex items-center gap-2 text-xs text-gray-400 mb-3">
                      <span>使用 {p.usage_count || 0} 次</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button variant="primary" size="sm" leftIcon={<TestTube size={14} />} onClick={() => handleTest(p.id)}>测试</Button>
                      <Button variant="ghost" size="sm" leftIcon={<Edit size={14} />}>编辑</Button>
                      <Button variant="ghost" size="sm" leftIcon={<Copy size={14} />}>克隆</Button>
                      <Button variant="ghost" size="sm" leftIcon={<Download size={14} />}>导出</Button>
                      <Button variant="ghost" size="sm" className="text-red-500" leftIcon={<Trash2 size={14} />}>删除</Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <Modal isOpen={showCreateModal} onClose={() => setShowCreateModal(false)} title={'"创建人格"'} size="lg" footer={
        <div className="flex gap-2">
          <Button variant="secondary" onClick={() => setShowCreateModal(false)}>取消</Button>
          <Button onClick={handleCreate} isLoading={createPersonality.isPending}>创建</Button>
        </div>
      }>
        <div className="space-y-4">
          <Input label={'"人格名称"'} placeholder={'"输入人格名称"'} value={newPersonality.name} onChange={e => setNewPersonality({ ...newPersonality, name: e.target.value })} />
          <Textarea label={'"描述"'} placeholder={'"描述该人格的特点..."'} value={newPersonality.description} onChange={e => setNewPersonality({ ...newPersonality, description: e.target.value })} rows={2} />
          <Select label={'"沟通风格"'} value={newPersonality.communication_style} onChange={e => setNewPersonality({ ...newPersonality, communication_style: e.target.value })} options={[
            { label: '专业', value: 'professional' },
            { label: '友好', value: 'friendly' },
            { label: '正式', value: 'formal' },
            { label: '随意', value: 'casual' },
            { label: '技术', value: 'technical' },
            { label: '幽默', value: 'humorous' },
          ]} />
          <Textarea label={'"系统提示词"'} placeholder={'"定义该人格的行为准则和回复风格..."'} value={newPersonality.system_prompt} onChange={e => setNewPersonality({ ...newPersonality, system_prompt: e.target.value })} rows={4} />
          <div>
            <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">人格特质 (0-100)</p>
            <div className="space-y-3">
              {Object.entries(traitLabels).map(([key, label]) => (
                <div key={key} className="flex items-center gap-3">
                  <span className="text-sm text-gray-500 w-16">{label}</span>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={newPersonality[key as keyof typeof newPersonality] as number}
                    onChange={e => setNewPersonality({ ...newPersonality, [key]: Number(e.target.value) })}
                    className="flex-1"
                  />
                  <span className="text-sm text-gray-700 dark:text-gray-300 w-8">{newPersonality[key as keyof typeof newPersonality] as number}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Modal>

      <Modal isOpen={showTestModal} onClose={() => setShowTestModal(false)} title={'"测试人格"'} size="lg" footer={
        <div className="flex gap-2">
          <Button variant="secondary" onClick={() => setShowTestModal(false)}>关闭</Button>
        </div>
      }>
        <div className="space-y-4">
          <Textarea label={'"测试消息"'} placeholder={'"输入测试消息..."'} value={testMessage} onChange={e => setTestMessage(e.target.value)} rows={3} />
          <Button leftIcon={<MessageSquare size={16} />} onClick={handleSendTest} disabled={!testMessage.trim()}>发送</Button>
          {testResponse && (
            <div className="p-4 bg-gray-50 dark:bg-dark-700 rounded-lg">
              <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">AI 回复:</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">{testResponse}</p>
            </div>
          )}
        </div>
      </Modal>
    </div>
  );
}
