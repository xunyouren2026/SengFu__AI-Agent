import { useState } from 'react';
import { Plus, Play, Pause, Square, Trash2, Settings, Upload, Database, BarChart3, Zap, HardDrive, Clock, CheckCircle } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Badge, Modal, Input, Select, Textarea, Tabs, TabPanel, EmptyState, Loading, Table, StatsCard } from '../components/common';
import { useTrainingJobs, useCreateTrainingJob, useDeleteTrainingJob, useStartTraining, usePauseTraining, useStopTraining, useDatasets, useCheckpoints } from '../hooks/useTraining';

export default function Training() {
  const [activeTab, setActiveTab] = useState('jobs');
  const [createOpen, setCreateOpen] = useState(false);
  const { data: jobs, isLoading } = useTrainingJobs();
  const { data: datasets } = useDatasets();
  const { data: checkpoints } = useCheckpoints();
  const createJob = useCreateTrainingJob();
  const deleteJob = useDeleteTrainingJob();
  const startTraining = useStartTraining();
  const pauseTraining = usePauseTraining();
  const stopTraining = useStopTraining();

  const statusColors: Record<string, 'success' | 'warning' | 'danger' | 'info' | 'default'> = { running: 'info', completed: 'success', paused: 'warning', failed: 'danger', pending: 'default', queued: 'default', stopped: 'warning', cancelled: 'default' };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div><h1 className="text-2xl font-bold text-gray-900 dark:text-white">训练中心</h1><p className="text-sm text-gray-500 mt-1">管理模型训练任务、数据集和检查点</p></div>
        <Button leftIcon={<Plus size={16} />} onClick={() => setCreateOpen(true)}>创建训练任务</Button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatsCard title={'"训练任务"'} value={jobs?.data?.length || 0} icon={<Zap size={24} />} color="blue" />
        <StatsCard title={'"运行中"'} value={jobs?.data?.filter((j: any) => j.status === 'running').length || 0} icon={<Play size={24} />} color="green" />
        <StatsCard title={'"数据集"'} value={datasets?.data?.length || 0} icon={<Database size={24} />} color="purple" />
        <StatsCard title={'"检查点"'} value={checkpoints?.data?.length || 0} icon={<HardDrive size={24} />} color="indigo" />
      </div>
      <Tabs tabs={[{ key: 'jobs', label: '训练任务' }, { key: 'datasets', label: '数据集' }, { key: 'checkpoints', label: '检查点' }, { key: 'hyperparam', label: '超参搜索' }]} activeKey={activeTab} onChange={setActiveTab} />
      {activeTab === 'jobs' && (
        <Card padding="none">
          {isLoading ? <Loading /> : (jobs?.data || []).length === 0 ? <EmptyState icon={<Zap size={48} />} title={'"暂无训练任务"'} /> : (
            <Table columns={[
              { key: 'name', title: '任务名称', render: (_, row) => <span className="font-medium">{row.name}</span> },
              { key: 'model_type', title: '模型类型', render: (v) => <Badge variant="info">{v}</Badge> },
              { key: 'status', title: '状态', render: (v) => <Badge variant={statusColors[v as string] || 'default'} dot>{v}</Badge> },
              { key: 'progress', title: '进度', render: (_, row) => (
                <div className="flex items-center gap-2"><div className="w-24 h-2 bg-gray-100 dark:bg-dark-600 rounded-full overflow-hidden"><div className="h-full bg-primary-500 rounded-full transition-all" style={{ width: `${row.progress?.progress_percent || 0}%` }} /></div><span className="text-xs">{row.progress?.progress_percent?.toFixed(1) || 0}%</span></div>
              )},
              { key: 'current_metrics', title: 'Loss', render: (v) => (v as any)?.loss?.toFixed(4) || '-' },
              { key: 'actions', title: '操作', width: '200px', render: (_, row) => (
                <div className="flex gap-1">
                  {row.status === 'pending' || row.status === 'queued' ? <Button variant="ghost" size="sm" onClick={() => startTraining.mutate(row.id)} leftIcon={<Play size={14} />}>开始</Button> : null}
                  {row.status === 'running' ? <Button variant="ghost" size="sm" onClick={() => pauseTraining.mutate(row.id)} leftIcon={<Pause size={14} />}>暂停</Button> : null}
                  {row.status === 'paused' ? <Button variant="ghost" size="sm" onClick={() => startTraining.mutate(row.id)} leftIcon={<Play size={14} />}>继续</Button> : null}
                  {row.status === 'running' || row.status === 'paused' ? <Button variant="ghost" size="sm" onClick={() => stopTraining.mutate(row.id)} leftIcon={<Square size={14} />}>停止</Button> : null}
                  <Button variant="ghost" size="sm" onClick={() => deleteJob.mutate(row.id)} className="text-red-500"><Trash2 size={14} /></Button>
                </div>
              )},
            ]} data={(jobs?.data || []) as any[]} />
          )}
        </Card>
      )}
      {activeTab === 'datasets' && <Card><CardContent><EmptyState icon={<Database size={48} />} title={'"数据集管理"'} description={'"上传和管理训练数据集"'} action={<Button leftIcon={<Upload size={16} />}>上传数据集</Button>} /></CardContent></Card>}
      {activeTab === 'checkpoints' && <Card><CardContent><EmptyState icon={<HardDrive size={48} />} title={'"检查点管理"'} description={'"查看和恢复训练检查点"'} /></CardContent></Card>}
      {activeTab === 'hyperparam' && <Card><CardContent><EmptyState icon={<BarChart3 size={48} />} title={'"超参数搜索"'} description={'"自动搜索最优超参数配置"'} /></CardContent></Card>}
      <Modal isOpen={createOpen} onClose={() => setCreateOpen(false)} title={'"创建训练任务"'} size="lg" footer={<><Button variant="ghost" onClick={() => setCreateOpen(false)}>取消</Button><Button onClick={() => setCreateOpen(false)}>创建</Button></>}>
        <div className="space-y-4">
          <Input label={'"任务名称"'} placeholder={'"输入训练任务名称"'} />
          <Select label={'"模型类型"'} options={[{ label: 'Transformer', value: 'transformer' }, { label: 'GPT', value: 'gpt' }, { label: 'BERT', value: 'bert' }, { label: 'Vision', value: 'vision' }, { label: '多模态', value: 'multimodal' }]} />
          <Input label={'"基础模型"'} placeholder={'"可选，如 meta-llama/Llama-2-7b"'} />
          <Select label={'"数据集"'} options={(datasets?.data || []).map((d: any) => ({ label: d.name, value: d.id }))} placeholder={'"选择数据集"'} />
          <div className="grid grid-cols-2 gap-4">
            <Input label="Epochs" type="number" defaultValue="3" />
            <Input label="Batch Size" type="number" defaultValue="32" />
            <Input label="Learning Rate" type="number" defaultValue="0.00001" step="0.00001" />
            <Select label={'"优化器"'} options={[{ label: 'AdamW', value: 'adamw' }, { label: 'Adam', value: 'adam' }, { label: 'SGD', value: 'sgd' }]} />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <Input label={'"GPU 数量"'} type="number" defaultValue="1" />
            <Input label={'"内存(GB)"'} type="number" defaultValue="16" />
          </div>
        </div>
      </Modal>
    </div>
  );
}
