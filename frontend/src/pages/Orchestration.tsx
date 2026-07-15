import { useState } from 'react';
import { Plus, RefreshCw, Play, Pause, Trash2, Settings, Route, Shield, BarChart3 } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Badge, Modal, Input, Select, Tabs, TabPanel, EmptyState, Loading, Table, StatsCard } from '../components/common';
import { useStrategies, useCreateStrategy, useDeleteStrategy, useRoutingRules, useCreateRoutingRule, useLoadBalancers, useOrchestrationMetrics } from '../hooks/useOrchestration';

export default function Orchestration() {
  const [activeTab, setActiveTab] = useState('strategies');
  const [createOpen, setCreateOpen] = useState(false);
  const { data: strategies, isLoading } = useStrategies();
  const { data: rules } = useRoutingRules();
  const { data: loadBalancers } = useLoadBalancers();
  const { data: metrics } = useOrchestrationMetrics('24h');
  const createStrategy = useCreateStrategy();
  const deleteStrategy = useDeleteStrategy();
  const createRule = useCreateRoutingRule();

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div><h1 className="text-2xl font-bold text-gray-900 dark:text-white">模型编排</h1><p className="text-sm text-gray-500 mt-1">配置模型路由策略、负载均衡和熔断器</p></div>
        <Button leftIcon={<Plus size={16} />} onClick={() => setCreateOpen(true)}>创建策略</Button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatsCard title={'"总策略数"'} value={strategies?.data?.length || 0} icon={<Route size={24} />} color="blue" />
        <StatsCard title={'"活跃策略"'} value={strategies?.data?.filter((s: any) => s.is_active).length || 0} icon={<Play size={24} />} color="green" />
        <StatsCard title={'"路由规则"'} value={rules?.data?.length || 0} icon={<Shield size={24} />} color="purple" />
        <StatsCard title={'"负载均衡器"'} value={(loadBalancers as any)?.data?.length || 0} icon={<BarChart3 size={24} />} color="indigo" />
      </div>
      <Tabs tabs={[{ key: 'strategies', label: '策略管理' }, { key: 'rules', label: '路由规则' }, { key: 'loadbalancers', label: '负载均衡' }, { key: 'circuitbreakers', label: '熔断器' }]} activeKey={activeTab} onChange={setActiveTab} />
      {activeTab === 'strategies' && (
        <Card padding="none">
          {isLoading ? <Loading /> : (strategies?.data || []).length === 0 ? <EmptyState icon={<Route size={48} />} title={'"暂无策略"'} /> : (
            <Table columns={[
              { key: 'name', title: '策略名称', render: (_, row) => <span className="font-medium">{row.name}</span> },
              { key: 'type', title: '类型', render: (v) => <Badge variant="info">{v}</Badge> },
              { key: 'status', title: '状态', render: (_, row) => <Badge variant={row.is_active ? 'success' : 'warning'} dot>{row.is_active ? '活跃' : '未激活'}</Badge> },
              { key: 'execution_count', title: '执行次数' },
              { key: 'avg_execution_time_ms', title: '平均延迟', render: (v) => `${(v as number)?.toFixed(0)}ms` },
              { key: 'actions', title: '操作', width: '150px', render: (_, row) => (<div className="flex gap-1"><Button variant="ghost" size="sm"><Settings size={14} /></Button><Button variant="ghost" size="sm" onClick={() => deleteStrategy.mutate(row.id)} className="text-red-500"><Trash2 size={14} /></Button></div>) },
            ]} data={(strategies?.data || []) as any[]} />
          )}
        </Card>
      )}
      {activeTab === 'rules' && (
        <Card padding="none">
          {(rules?.data || []).length === 0 ? <EmptyState icon={<Shield size={48} />} title={'"暂无路由规则"'} /> : (
            <Table columns={[{ key: 'name', title: '规则名称' }, { key: 'type', title: '类型', render: (v) => <Badge variant="info">{v}</Badge> }, { key: 'priority', title: '优先级' }, { key: 'enabled', title: '状态', render: (v) => <Badge variant={v ? 'success' : 'warning'}>{v ? '启用' : '禁用'}</Badge> }, { key: 'match_count', title: '匹配次数' }]} data={(rules?.data || []) as any[]} />
          )}
        </Card>
      )}
      {activeTab === 'loadbalancers' && <Card><CardContent><EmptyState icon={<BarChart3 size={48} />} title={'"负载均衡配置"'} description={'"配置模型请求的负载均衡策略"'} /></CardContent></Card>}
      {activeTab === 'circuitbreakers' && <Card><CardContent><EmptyState icon={<Shield size={48} />} title={'"熔断器配置"'} description={'"配置模型调用的熔断保护策略"'} /></CardContent></Card>}
      <Modal isOpen={createOpen} onClose={() => setCreateOpen(false)} title={'"创建策略"'} footer={<><Button variant="ghost" onClick={() => setCreateOpen(false)}>取消</Button><Button onClick={() => { createStrategy.mutate({ name: '新策略', type: 'fallback', config: { primary_model: '', fallback_models: [], timeout_seconds: 30, retry_attempts: 3, retry_delay_seconds: 1, custom_params: {} } }); setCreateOpen(false); }}>创建</Button></>}>
        <div className="space-y-4">
          <Input label={'"策略名称"'} placeholder={'"输入策略名称"'} />
          <Select label={'"策略类型"'} options={[{ label: '故障转移', value: 'fallback' }, { label: '负载均衡', value: 'load_balance' }, { label: '路由', value: 'routing' }, { label: '成本优化', value: 'cost_optimize' }, { label: '质量优化', value: 'quality_optimize' }]} />
          <Input label={'"主模型"'} placeholder={'"模型ID"'} />
          <Input label={'"超时时间(秒)"'} type="number" defaultValue="30" />
        </div>
      </Modal>
    </div>
  );
}
