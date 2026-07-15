import { useState } from 'react';
import { Plus, Users, Swords, Trophy, Play, Trash2, Settings, Brain, Target, Network } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Badge, Modal, Input, Select, Tabs, TabPanel, EmptyState, Loading, Table, StatsCard } from '../components/common';
import { useAgents, useCreateAgent, useDeleteAgent, useAlliances, useDebates } from '../hooks/useAgents';

export default function MultiAgent() {
  const [activeTab, setActiveTab] = useState('agents');
  const [createOpen, setCreateOpen] = useState(false);
  const { data: agents, isLoading } = useAgents();
  const { data: alliances } = useAlliances();
  const { data: debates } = useDebates();
  const createAgent = useCreateAgent();
  const deleteAgent = useDeleteAgent();

  const statusColors: Record<string, 'success' | 'warning' | 'danger' | 'info'> = { active: 'success', inactive: 'warning', error: 'danger', pending: 'info', training: 'info' };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div><h1 className="text-2xl font-bold text-gray-900 dark:text-white">多智能体</h1><p className="text-sm text-gray-500 mt-1">管理和协调多个 AI 智能体协作</p></div>
        <Button leftIcon={<Plus size={16} />} onClick={() => setCreateOpen(true)}>创建智能体</Button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatsCard title={'"智能体总数"'} value={agents?.data?.length || 0} icon={<Users size={24} />} color="blue" />
        <StatsCard title={'"活跃智能体"'} value={agents?.data?.filter((a: any) => a.status === 'active').length || 0} icon={<Brain size={24} />} color="green" />
        <StatsCard title={'"联盟数"'} value={alliances?.data?.length || 0} icon={<Network size={24} />} color="purple" />
        <StatsCard title={'"辩论数"'} value={debates?.data?.length || 0} icon={<Swords size={24} />} color="indigo" />
      </div>
      <Tabs tabs={[{ key: 'agents', label: '智能体' }, { key: 'alliances', label: '联盟管理' }, { key: 'debates', label: '辩论系统' }, { key: 'marketplace', label: '市场' }]} activeKey={activeTab} onChange={setActiveTab} />
      {activeTab === 'agents' && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {isLoading ? <Loading /> : (agents?.data || []).length === 0 ? <EmptyState icon={<Users size={48} />} title={'"暂无智能体"'} /> :
            (agents?.data || []).map((agent: any) => (
              <Card key={agent.id} hoverable variant="bordered">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary-400 to-primary-600 flex items-center justify-center text-white font-bold">{agent.name[0]}</div>
                    <div><h3 className="font-medium text-gray-900 dark:text-white">{agent.name}</h3><p className="text-xs text-gray-500">{agent.description?.slice(0, 50)}</p></div>
                  </div>
                  <Badge variant={statusColors[agent.status] || 'default'} dot>{agent.status}</Badge>
                </div>
                <div className="flex flex-wrap gap-1 mb-3">{agent.capabilities?.map((cap: string) => <Badge key={cap} variant="outline" size="sm">{cap}</Badge>)}</div>
                <div className="grid grid-cols-3 gap-2 text-center text-xs mb-3">
                  <div className="p-2 rounded bg-gray-50 dark:bg-dark-700"><p className="font-bold text-gray-900 dark:text-white">{agent.metrics?.total_tasks || 0}</p><p className="text-gray-500">任务</p></div>
                  <div className="p-2 rounded bg-gray-50 dark:bg-dark-700"><p className="font-bold text-green-500">{(agent.metrics?.success_rate * 100 || 0).toFixed(0)}%</p><p className="text-gray-500">成功率</p></div>
                  <div className="p-2 rounded bg-gray-50 dark:bg-dark-700"><p className="font-bold text-primary-500">{agent.metrics?.reputation_score || 0}</p><p className="text-gray-500">声誉</p></div>
                </div>
                <div className="flex items-center gap-2 pt-3 border-t border-gray-100 dark:border-dark-600">
                  <Button variant="ghost" size="sm" leftIcon={<Play size={14} />}>执行</Button>
                  <Button variant="ghost" size="sm"><Settings size={14} /></Button>
                  <Button variant="ghost" size="sm" onClick={() => deleteAgent.mutate(agent.id)} className="text-red-500"><Trash2 size={14} /></Button>
                </div>
              </Card>
            ))}
        </div>
      )}
      {activeTab === 'alliances' && <Card><CardContent><EmptyState icon={<Network size={48} />} title={'"暂无联盟"'} description={'"创建智能体联盟以实现协作"'} /></CardContent></Card>}
      {activeTab === 'debates' && <Card><CardContent><EmptyState icon={<Swords size={48} />} title={'"暂无辩论"'} description={'"创建多智能体辩论以获取不同视角"'} /></CardContent></Card>}
      {activeTab === 'marketplace' && <Card><CardContent><EmptyState icon={<Trophy size={48} />} title={'"智能体市场"'} description={'"浏览和下载社区智能体"'} /></CardContent></Card>}
      <Modal isOpen={createOpen} onClose={() => setCreateOpen(false)} title={'"创建智能体"'} size="lg" footer={<><Button variant="ghost" onClick={() => setCreateOpen(false)}>取消</Button><Button onClick={() => setCreateOpen(false)}>创建</Button></>}>
        <div className="space-y-4">
          <Input label={'"智能体名称"'} placeholder={'"输入名称"'} />
          <Input label={'"描述"'} placeholder={'"描述智能体的功能和特点"'} />
          <Select label={'"能力"'} options={[{ label: '研究', value: 'research' }, { label: '分析', value: 'analysis' }, { label: '编程', value: 'coding' }, { label: '写作', value: 'writing' }, { label: '规划', value: 'planning' }]} />
          <Input label={'"关联模型ID"'} placeholder={'"可选"'} />
        </div>
      </Modal>
    </div>
  );
}
