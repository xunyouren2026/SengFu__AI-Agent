import { useState } from 'react';
import { Plus, Play, Pause, Copy, Trash2, Eye, Code, GitBranch, Box, ArrowRight, Filter, Zap } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Badge, Modal, Input, Select, Tabs, TabPanel, EmptyState, Loading, Table, StatsCard } from '../components/common';
import { useWorkflows, useCreateWorkflow, useDeleteWorkflow, useExecuteWorkflow, useWorkflowTemplates } from '../hooks/useWorkflows';

export default function Workflow() {
  const [activeTab, setActiveTab] = useState('workflows');
  const [createOpen, setCreateOpen] = useState(false);
  const { data: workflows, isLoading } = useWorkflows();
  const { data: templates } = useWorkflowTemplates();
  const createWorkflow = useCreateWorkflow();
  const deleteWorkflow = useDeleteWorkflow();
  const executeWorkflow = useExecuteWorkflow();

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div><h1 className="text-2xl font-bold text-gray-900 dark:text-white">工作流</h1><p className="text-sm text-gray-500 mt-1">设计和管理工作流自动化</p></div>
        <Button leftIcon={<Plus size={16} />} onClick={() => setCreateOpen(true)}>创建工作流</Button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatsCard title={'"总工作流"'} value={workflows?.data?.length || 0} icon={<GitBranch size={24} />} color="blue" />
        <StatsCard title={'"已发布"'} value={workflows?.data?.filter((w: any) => w.status === 'published').length || 0} icon={<Play size={24} />} color="green" />
        <StatsCard title={'"草稿"'} value={workflows?.data?.filter((w: any) => w.status === 'draft').length || 0} icon={<Code size={24} />} color="yellow" />
        <StatsCard title={'"总执行次数"'} value={workflows?.data?.reduce((sum: number, w: any) => sum + w.execution_count, 0) || 0} icon={<Zap size={24} />} color="purple" />
      </div>
      <Tabs tabs={[{ key: 'workflows', label: '我的工作流' }, { key: 'templates', label: '模板库' }, { key: 'executions', label: '执行历史' }]} activeKey={activeTab} onChange={setActiveTab} />
      {activeTab === 'workflows' && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {isLoading ? <Loading /> : (workflows?.data || []).length === 0 ? <EmptyState icon={<GitBranch size={48} />} title={'"暂无工作流"'} description={'"创建你的第一个自动化工作流"'} action={<Button leftIcon={<Plus size={16} />} onClick={() => setCreateOpen(true)}>创建工作流</Button>} /> :
            (workflows?.data || []).map((wf: any) => (
              <Card key={wf.id} hoverable variant="bordered">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-2"><GitBranch size={18} className="text-primary-500" /><h3 className="font-medium text-gray-900 dark:text-white">{wf.name}</h3></div>
                  <Badge variant={wf.status === 'published' ? 'success' : wf.status === 'draft' ? 'warning' : 'default'}>{wf.status === 'published' ? '已发布' : wf.status === 'draft' ? '草稿' : wf.status}</Badge>
                </div>
                <p className="text-sm text-gray-500 mb-3 line-clamp-2">{wf.description || '无描述'}</p>
                <div className="flex items-center gap-4 text-xs text-gray-400 mb-3">
                  <span>{wf.nodes?.length || 0} 节点</span><span>{wf.execution_count} 次执行</span><span>成功率 {(wf.execution_count ? (wf.success_count / wf.execution_count * 100).toFixed(0) : 0)}%</span>
                </div>
                <div className="flex items-center gap-2 pt-3 border-t border-gray-100 dark:border-dark-600">
                  <Button variant="ghost" size="sm" onClick={() => executeWorkflow.mutate({ id: wf.id, variables: {} })} leftIcon={<Play size={14} />}>执行</Button>
                  <Button variant="ghost" size="sm"><Eye size={14} /></Button>
                  <Button variant="ghost" size="sm"><Copy size={14} /></Button>
                  <Button variant="ghost" size="sm" onClick={() => deleteWorkflow.mutate(wf.id)} className="text-red-500"><Trash2 size={14} /></Button>
                </div>
              </Card>
            ))}
        </div>
      )}
      {activeTab === 'templates' && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {(templates?.templates || []).map((tpl: any) => (
            <Card key={tpl.id} hoverable variant="bordered">
              <div className="flex items-center gap-2 mb-2"><Box size={18} className="text-purple-500" /><h3 className="font-medium">{tpl.name}</h3></div>
              <p className="text-sm text-gray-500 mb-2">{tpl.description}</p>
              <div className="flex items-center gap-2"><Badge variant="info">{tpl.category}</Badge><Badge variant="outline">{tpl.difficulty}</Badge></div>
            </Card>
          ))}
        </div>
      )}
      {activeTab === 'executions' && <Card><CardContent><EmptyState icon={<Play size={48} />} title={'"暂无执行记录"'} /></CardContent></Card>}
      <Modal isOpen={createOpen} onClose={() => setCreateOpen(false)} title={'"创建工作流"'} size="lg" footer={<><Button variant="ghost" onClick={() => setCreateOpen(false)}>取消</Button><Button onClick={() => setCreateOpen(false)}>创建</Button></>}>
        <div className="space-y-4"><Input label={'"工作流名称"'} placeholder={'"输入名称"'} /><Input label={'"描述"'} placeholder={'"输入描述"'} /><Select label={'"分类"'} options={[{ label: '通用', value: 'general' }, { label: '数据处理', value: 'data' }, { label: 'AI 推理', value: 'ai' }, { label: '自动化', value: 'automation' }]} /></div>
      </Modal>
    </div>
  );
}
