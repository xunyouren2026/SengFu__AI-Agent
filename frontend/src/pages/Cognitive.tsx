import { useState } from 'react';
import { Brain, MemoryStick, Lightbulb, Target, Heart, BarChart3, RefreshCw, Plus, Trash2, Search } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Badge, Modal, Input, Select, Tabs, TabPanel, EmptyState, Loading, StatsCard } from '../components/common';
import { useCognitiveState, useMemory, useGoals, useCreateGoal, useTriggerReflection, useCognitiveMetrics } from '../hooks/useCognitive';

export default function Cognitive() {
  const [activeTab, setActiveTab] = useState('state');
  const [reflectionOpen, setReflectionOpen] = useState(false);
  const [reflectionTopic, setReflectionTopic] = useState('');
  const { data: state } = useCognitiveState();
  const { data: memory } = useMemory();
  const { data: goals } = useGoals();
  const { data: metrics } = useCognitiveMetrics('24h');
  const triggerReflection = useTriggerReflection();
  const createGoal = useCreateGoal();

  const stateColors: Record<string, string> = { focused: 'bg-blue-500', reflective: 'bg-purple-500', learning: 'bg-green-500', resting: 'bg-gray-400', adapting: 'bg-yellow-500', creative: 'bg-pink-500', analyzing: 'bg-indigo-500' };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div><h1 className="text-2xl font-bold text-gray-900 dark:text-white">认知系统</h1><p className="text-sm text-gray-500 mt-1">监控和管理 AI 认知状态、记忆和目标</p></div>
        <Button leftIcon={<Lightbulb size={16} />} onClick={() => setReflectionOpen(true)}>触发反思</Button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatsCard title={'"当前状态"'} value={(state as any)?.current_state?.state || 'N/A'} icon={<Brain size={24} />} color="blue" />
        <StatsCard title={'"记忆条目"'} value={memory?.total || 0} icon={<MemoryStick size={24} />} color="green" />
        <StatsCard title={'"活跃目标"'} value={goals?.data?.filter((g: any) => g.status === 'active').length || 0} icon={<Target size={24} />} color="purple" />
        <StatsCard title={'"认知评分"'} value={metrics?.overall_score?.toFixed(2) || '0.50'} icon={<BarChart3 size={24} />} color="indigo" />
      </div>
      <Tabs tabs={[{ key: 'state', label: '认知状态' }, { key: 'memory', label: '记忆管理' }, { key: 'goals', label: '目标管理' }, { key: 'emotions', label: '情绪分析' }, { key: 'metrics', label: '认知指标' }]} activeKey={activeTab} onChange={setActiveTab} />
      {activeTab === 'state' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card><CardHeader><CardTitle>当前认知状态</CardTitle></CardHeader><CardContent>
            <div className="text-center py-8">
              <div className={`w-24 h-24 rounded-full ${stateColors[(state as any)?.current_state?.state || 'resting']} mx-auto mb-4 flex items-center justify-center`}><Brain size={40} className="text-white" /></div>
              <h3 className="text-xl font-bold text-gray-900 dark:text-white capitalize">{(state as any)?.current_state?.state || 'resting'}</h3>
              <p className="text-sm text-gray-500 mt-1">置信度: {((state as any)?.current_state?.confidence * 100 || 0).toFixed(0)}% · 强度: {((state as any)?.current_state?.intensity * 100 || 0).toFixed(0)}%</p>
              <p className="text-sm text-gray-400 mt-2">持续时间: {(state as any)?.state_duration_seconds || 0}s</p>
            </div>
          </CardContent></Card>
          <Card><CardHeader><CardTitle>认知指标</CardTitle></CardHeader><CardContent>
            <div className="space-y-4">
              {(metrics?.metrics || []).map((m: any) => (
                <div key={m.metric_type} className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400 capitalize">{m.metric_type}</span>
                  <div className="flex items-center gap-2"><div className="w-32 h-2 bg-gray-100 dark:bg-dark-600 rounded-full overflow-hidden"><div className="h-full bg-primary-500 rounded-full" style={{ width: `${m.value * 100}%` }} /></div><span className="text-sm font-medium w-12 text-right">{(m.value * 100).toFixed(0)}%</span></div>
                </div>
              ))}
            </div>
          </CardContent></Card>
        </div>
      )}
      {activeTab === 'memory' && <Card><CardContent><EmptyState icon={<MemoryStick size={48} />} title={'"记忆管理"'} description={'"搜索、管理和遗忘 AI 记忆条目"'} action={<Button leftIcon={<Search size={16} />}>搜索记忆</Button>} /></CardContent></Card>}
      {activeTab === 'goals' && <Card><CardContent><EmptyState icon={<Target size={48} />} title={'"目标管理"'} description={'"创建和管理 AI 的目标和任务"'} action={<Button leftIcon={<Plus size={16} />} onClick={() => createGoal.mutate({ title: '新目标' })}>创建目标</Button>} /></CardContent></Card>}
      {activeTab === 'emotions' && <Card><CardContent><EmptyState icon={<Heart size={48} />} title={'"情绪分析"'} description={'"查看 AI 的情绪状态和历史变化"'} /></CardContent></Card>}
      {activeTab === 'metrics' && <Card><CardContent><EmptyState icon={<BarChart3 size={48} />} title={'"认知指标"'} description={'"查看详细的认知能力评估指标"'} /></CardContent></Card>}
      <Modal isOpen={reflectionOpen} onClose={() => setReflectionOpen(false)} title={'"触发反思"'} footer={<><Button variant="ghost" onClick={() => setReflectionOpen(false)}>取消</Button><Button onClick={() => { triggerReflection.mutate({ topic: reflectionTopic }); setReflectionOpen(false); setReflectionTopic(''); }} isLoading={triggerReflection.isPending}>开始反思</Button></>}>
        <div className="space-y-4">
          <Input label={'"反思主题"'} placeholder={'"输入反思的主题或问题"'} value={reflectionTopic} onChange={e => setReflectionTopic(e.target.value)} />
          <Select label={'"反思深度"'} options={[{ label: '表面', value: 'surface' }, { label: '中等', value: 'moderate' }, { label: '深度', value: 'deep' }]} />
          <Select label={'"反思类型"'} options={[{ label: '自我反思', value: 'self' }, { label: '决策反思', value: 'decision' }, { label: '交互反思', value: 'interaction' }, { label: '学习反思', value: 'learning' }]} />
        </div>
      </Modal>
    </div>
  );
}
