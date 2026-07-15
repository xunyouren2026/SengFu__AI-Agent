import { useState } from 'react';
import { Atom, Play, Square, Trash2, Plus, Clock, Activity, Cpu, CheckCircle, AlertTriangle, Settings, RotateCcw, FileText, BarChart3 } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, CardFooter, Button, Badge, Input, Select, Textarea, Tabs, TabPanel, EmptyState, Loading, StatsCard, Modal, Table } from '../components/common';
import { useSimulations, useCreateSimulation } from '../hooks/useAdvanced';

export default function PhysicsEngine() {
  const [activeTab, setActiveTab] = useState('simulations');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newSim, setNewSim] = useState({ name: '', type: 'molecular_dynamics' as string, description: '', time_step: 0.001, total_steps: 10000, ensemble: 'NVT', constraints: [] as string[] });

  const { data: simsData, isLoading } = useSimulations();
  const createSimulation = useCreateSimulation();

  const simulations = (simsData?.data || []) as any[];
  const runningCount = simulations.filter((s: any) => s.status === 'running').length;
  const completedCount = simulations.filter((s: any) => s.status === 'completed').length;
  const failedCount = simulations.filter((s: any) => s.status === 'failed').length;

  const handleCreate = () => {
    if (!newSim.name.trim()) return;
    createSimulation.mutate({
      name: newSim.name,
      type: newSim.type as any,
      description: newSim.description || undefined,
      config: {
        time_step: newSim.time_step,
        total_steps: newSim.total_steps,
        ensemble: newSim.ensemble,
        constraints: newSim.constraints,
      },
      input_files: [],
    });
    setShowCreateModal(false);
    setNewSim({ name: '', type: 'molecular_dynamics', description: '', time_step: 0.001, total_steps: 10000, ensemble: 'NVT', constraints: [] });
  };

  const statusLabel: Record<string, string> = {
    pending: '等待中', setup: '配置中', running: '运行中', paused: '已暂停', completed: '已完成', failed: '失败', cancelled: '已取消',
  };
  const statusVariant: Record<string, any> = {
    pending: 'info', setup: 'warning', running: 'primary', paused: 'warning', completed: 'success', failed: 'danger', cancelled: 'default',
  };
  const typeLabel: Record<string, string> = {
    molecular_dynamics: '分子动力学', quantum_chemistry: '量子化学', fluid_dynamics: '流体力学', structural_mechanics: '结构力学', thermal_analysis: '热分析', electromagnetic: '电磁学',
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">物理引擎</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">AI 驱动的物理模拟与计算</p>
        </div>
        <Button leftIcon={<Plus size={16} />} onClick={() => setShowCreateModal(true)}>新建模拟</Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatsCard title={'"总模拟数"'} value={simulations.length} icon={<Atom size={24} />} color="blue" />
        <StatsCard title={'"运行中"'} value={runningCount} icon={<Activity size={24} />} color="green" />
        <StatsCard title={'"已完成"'} value={completedCount} icon={<CheckCircle size={24} />} color="purple" />
        <StatsCard title={'"失败"'} value={failedCount} icon={<AlertTriangle size={24} />} color="red" />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>模拟任务列表</CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <Loading />
          ) : simulations.length === 0 ? (
            <EmptyState icon={<Atom size={32} />} title={'"暂无模拟任务"'} description={'"点击「新建模拟」创建你的第一个物理模拟"'} />
          ) : (
            <Table
              columns={[
                { key: 'name', title: '名称', render: (val: any) => <span className="font-medium text-gray-900 dark:text-white">{val}</span> },
                { key: 'type', title: '类型', render: (val: any) => <Badge variant="outline">{typeLabel[val] || val}</Badge> },
                { key: 'status', title: '状态', render: (val: any) => <Badge variant={statusVariant[val] || 'default'}>{statusLabel[val] || val}</Badge> },
                { key: 'progress', title: '进度', render: (val: any) => (
                  <div className="flex items-center gap-2">
                    <div className="w-20 h-2 bg-gray-100 dark:bg-dark-600 rounded-full overflow-hidden">
                      <div className="h-full bg-primary-500 rounded-full transition-all" style={{ width: `${val || 0}%` }} />
                    </div>
                    <span className="text-xs text-gray-500">{(val || 0).toFixed(1)}%</span>
                  </div>
                )},
                { key: 'current_step', title: '当前步数' },
                { key: 'actions', title: '操作', render: (_: any, row: any) => (
                  <div className="flex items-center gap-1">
                    {row.status === 'running' ? (
                      <Button variant="ghost" size="sm" leftIcon={<Square size={14} />}>停止</Button>
                    ) : row.status === 'paused' ? (
                      <Button variant="ghost" size="sm" leftIcon={<Play size={14} />}>继续</Button>
                    ) : (
                      <Button variant="ghost" size="sm" leftIcon={<Play size={14} />}>启动</Button>
                    )}
                    <Button variant="ghost" size="sm" leftIcon={<FileText size={14} />}>日志</Button>
                    <Button variant="ghost" size="sm" leftIcon={<Trash2 size={14} />} className="text-red-500">删除</Button>
                  </div>
                )},
              ]}
              data={simulations}
            />
          )}
        </CardContent>
      </Card>

      <Modal isOpen={showCreateModal} onClose={() => setShowCreateModal(false)} title={'"新建物理模拟"'} size="lg" footer={
        <div className="flex gap-2">
          <Button variant="secondary" onClick={() => setShowCreateModal(false)}>取消</Button>
          <Button onClick={handleCreate} isLoading={createSimulation.isPending}>创建</Button>
        </div>
      }>
        <div className="space-y-4">
          <Input label={'"模拟名称"'} placeholder={'"输入模拟名称"'} value={newSim.name} onChange={e => setNewSim({ ...newSim, name: e.target.value })} />
          <Select label={'"模拟类型"'} value={newSim.type} onChange={e => setNewSim({ ...newSim, type: e.target.value })} options={[
            { label: '分子动力学', value: 'molecular_dynamics' },
            { label: '量子化学', value: 'quantum_chemistry' },
            { label: '流体力学', value: 'fluid_dynamics' },
            { label: '结构力学', value: 'structural_mechanics' },
            { label: '热分析', value: 'thermal_analysis' },
            { label: '电磁学', value: 'electromagnetic' },
          ]} />
          <Textarea label={'"描述"'} placeholder={'"模拟描述..."'} value={newSim.description} onChange={e => setNewSim({ ...newSim, description: e.target.value })} rows={2} />
          <div className="grid grid-cols-2 gap-4">
            <Input label={'"时间步长"'} type="number" step="0.0001" value={newSim.time_step} onChange={e => setNewSim({ ...newSim, time_step: Number(e.target.value) })} />
            <Input label={'"总步数"'} type="number" value={newSim.total_steps} onChange={e => setNewSim({ ...newSim, total_steps: Number(e.target.value) })} />
            <Select label={'"系综"'} value={newSim.ensemble} onChange={e => setNewSim({ ...newSim, ensemble: e.target.value })} options={[
              { label: 'NVT', value: 'NVT' },
              { label: 'NPT', value: 'NPT' },
              { label: 'NVE', value: 'NVE' },
            ]} />
            <Input label={'"温度 (K)"'} type="number" placeholder={'"可选"'} />
          </div>
        </div>
      </Modal>
    </div>
  );
}
