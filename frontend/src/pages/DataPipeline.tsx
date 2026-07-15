import { useState, useRef } from 'react';
import { Database, Upload, Plus, Trash2, Play, Pause, Settings, GitBranch, FileSpreadsheet, Clock, CheckCircle, AlertTriangle, BarChart3, RefreshCw, Eye } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Badge, Input, Select, Textarea, Tabs, TabPanel, EmptyState, Loading, StatsCard, Modal, Table } from '../components/common';
import { usePipelines, useCreatePipeline } from '../hooks/useAdvanced';

export default function DataPipeline() {
  const [activeTab, setActiveTab] = useState('datasets');
  const [showPipelineModal, setShowPipelineModal] = useState(false);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [newPipeline, setNewPipeline] = useState({ name: '', description: '', steps: [] as any[], schedule: '' });

  const { data: pipelinesData, isLoading: pipelinesLoading } = usePipelines();
  const createPipeline = useCreatePipeline();

  const pipelines = (pipelinesData?.data || []) as any[];

  // Mock datasets
  const datasets = [
    { id: '1', name: '训练数据集 A', format: 'csv', size: 52428800, rows: 100000, columns: 25, status: 'ready', tags: ['训练', 'NLP'], uploaded_at: '2024-01-15 10:00:00' },
    { id: '2', name: '验证数据集 B', format: 'json', size: 10485760, rows: 20000, columns: 25, status: 'ready', tags: ['验证', 'NLP'], uploaded_at: '2024-01-14 08:00:00' },
    { id: '3', name: '图像标注数据', format: 'parquet', size: 209715200, rows: 50000, columns: 10, status: 'processing', tags: ['图像', '标注'], uploaded_at: '2024-01-13 12:00:00' },
  ];

  const handleCreatePipeline = () => {
    if (!newPipeline.name.trim()) return;
    createPipeline.mutate({
      name: newPipeline.name,
      description: newPipeline.description || undefined,
      steps: newPipeline.steps,
      schedule: newPipeline.schedule || undefined,
    });
    setShowPipelineModal(false);
    setNewPipeline({ name: '', description: '', steps: [], schedule: '' });
  };

  const handleUpload = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      console.log('Uploading dataset:', file.name);
    }
  };

  const statusLabel: Record<string, string> = { draft: '草稿', active: '活跃', running: '运行中', paused: '已暂停', error: '错误', completed: '已完成', ready: '就绪', processing: '处理中' };
  const statusVariant: Record<string, any> = { draft: 'default', active: 'success', running: 'primary', paused: 'warning', error: 'danger', completed: 'success', ready: 'success', processing: 'warning' };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">数据管道</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">数据集管理与处理管道</p>
        </div>
        <div className="flex gap-2">
          <Button leftIcon={<Upload size={16} />} onClick={handleUpload}>上传数据集</Button>
          <Button leftIcon={<Plus size={16} />} onClick={() => setShowPipelineModal(true)}>创建管道</Button>
        </div>
        <input ref={fileInputRef} type="file" className="hidden" onChange={handleFileChange} accept=".csv,.json,.parquet,.xlsx,.xls" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatsCard title={'"数据集"'} value={datasets.length} icon={<FileSpreadsheet size={24} />} color="blue" />
        <StatsCard title={'"管道"'} value={pipelines.length} icon={<GitBranch size={24} />} color="green" />
        <StatsCard title={'"运行中"'} value={pipelines.filter((p: any) => p.status === 'running').length} icon={<Play size={24} />} color="purple" />
        <StatsCard title={'"总数据量"'} value="273 MB" icon={<Database size={24} />} color="indigo" />
      </div>

      <Card>
        <CardContent className="pt-5">
          <Tabs
            tabs={[
              { key: 'datasets', label: '数据集', icon: <FileSpreadsheet size={16} /> },
              { key: 'pipelines', label: '处理管道', icon: <GitBranch size={16} /> },
              { key: 'visualization', label: '管道可视化', icon: <BarChart3 size={16} /> },
            ]}
            activeKey={activeTab}
            onChange={setActiveTab}
          />

          {activeTab === 'datasets' && (
            <TabPanel>
              <div className="flex justify-end mb-4">
                <Button leftIcon={<Upload size={16} />} onClick={handleUpload}>上传数据集</Button>
              </div>
              {datasets.length === 0 ? (
                <EmptyState icon={<Database size={32} />} title={'"暂无数据集"'} description={'"上传数据集以开始数据处理"'} />
              ) : (
                <Table
                  columns={[
                    { key: 'name', title: '名称', render: (val: any) => (
                      <div className="flex items-center gap-2">
                        <FileSpreadsheet size={16} className="text-gray-400" />
                        <span className="font-medium text-gray-900 dark:text-white">{val}</span>
                      </div>
                    )},
                    { key: 'format', title: '格式', render: (val: any) => <Badge variant="outline">{val?.toUpperCase()}</Badge> },
                    { key: 'size', title: '大小', render: (val: any) => val ? `${(val / (1024 * 1024)).toFixed(1)} MB` : '-' },
                    { key: 'rows', title: '行数', render: (val: any) => val?.toLocaleString() || '-' },
                    { key: 'columns', title: '列数' },
                    { key: 'tags', title: '标签', render: (val: any) => (
                      <div className="flex gap-1">{(val || []).map((tag: string, i: number) => <Badge key={i} variant="info" size="sm">{tag}</Badge>)}</div>
                    )},
                    { key: 'status', title: '状态', render: (val: any) => <Badge variant={statusVariant[val]}>{statusLabel[val] || val}</Badge> },
                    { key: 'uploaded_at', title: '上传时间' },
                    { key: 'actions', title: '操作', render: () => (
                      <div className="flex gap-1">
                        <Button variant="ghost" size="sm" leftIcon={<Eye size={14} />}>预览</Button>
                        <Button variant="ghost" size="sm" className="text-red-500" leftIcon={<Trash2 size={14} />}>删除</Button>
                      </div>
                    )},
                  ]}
                  data={datasets}
                />
              )}
            </TabPanel>
          )}

          {activeTab === 'pipelines' && (
            <TabPanel>
              <div className="flex justify-end mb-4">
                <Button leftIcon={<Plus size={16} />} onClick={() => setShowPipelineModal(true)}>创建管道</Button>
              </div>
              {pipelinesLoading ? (
                <Loading />
              ) : pipelines.length === 0 ? (
                <EmptyState icon={<GitBranch size={32} />} title={'"暂无处理管道"'} description={'"创建管道以自动化数据处理流程"'} />
              ) : (
                <Table
                  columns={[
                    { key: 'name', title: '名称', render: (val: any) => (
                      <div className="flex items-center gap-2">
                        <GitBranch size={16} className="text-gray-400" />
                        <span className="font-medium text-gray-900 dark:text-white">{val}</span>
                      </div>
                    )},
                    { key: 'status', title: '状态', render: (val: any) => <Badge variant={statusVariant[val]}>{statusLabel[val] || val}</Badge> },
                    { key: 'steps', title: '步骤数', render: (val: any) => val?.length || 0 },
                    { key: 'run_count', title: '运行次数' },
                    { key: 'last_run_at', title: '最后运行' },
                    { key: 'schedule', title: '调度', render: (val: any) => val ? <Badge variant="outline">{val}</Badge> : <span className="text-gray-400">手动</span> },
                    { key: 'actions', title: '操作', render: (_: any, row: any) => (
                      <div className="flex gap-1">
                        {row.status === 'running' ? (
                          <Button variant="ghost" size="sm" leftIcon={<Pause size={14} />}>暂停</Button>
                        ) : (
                          <Button variant="ghost" size="sm" leftIcon={<Play size={14} />}>运行</Button>
                        )}
                        <Button variant="ghost" size="sm" leftIcon={<Settings size={14} />}>配置</Button>
                        <Button variant="ghost" size="sm" className="text-red-500" leftIcon={<Trash2 size={14} />}>删除</Button>
                      </div>
                    )},
                  ]}
                  data={pipelines}
                />
              )}
            </TabPanel>
          )}

          {activeTab === 'visualization' && (
            <TabPanel>
              {pipelines.length === 0 ? (
                <EmptyState icon={<BarChart3 size={32} />} title={'"暂无管道可视化"'} description={'"创建管道后可查看处理流程图"'} />
              ) : (
                <div className="space-y-6">
                  {pipelines.map((pipeline: any) => (
                    <Card key={pipeline.id}>
                      <CardHeader>
                        <div className="flex items-center gap-2">
                          <CardTitle>{pipeline.name}</CardTitle>
                          <Badge variant={statusVariant[pipeline.status]}>{statusLabel[pipeline.status]}</Badge>
                        </div>
                      </CardHeader>
                      <CardContent>
                        {pipeline.steps && pipeline.steps.length > 0 ? (
                          <div className="flex items-center gap-2 overflow-x-auto pb-2">
                            {pipeline.steps.map((step: any, i: number) => (
                              <div key={step.id || i} className="flex items-center gap-2">
                                <div className="px-4 py-2 bg-gray-50 dark:bg-dark-700 rounded-lg border border-gray-200 dark:border-dark-600 min-w-[120px] text-center">
                                  <p className="text-sm font-medium text-gray-900 dark:text-white">{step.name}</p>
                                  <p className="text-xs text-gray-500">{step.type}</p>
                                </div>
                                {i < pipeline.steps.length - 1 && (
                                  <div className="flex items-center text-gray-400">
                                    <div className="w-8 h-0.5 bg-gray-300 dark:bg-dark-600" />
                                    <RefreshCw size={12} />
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        ) : (
                          <p className="text-sm text-gray-500 dark:text-gray-400">暂无步骤</p>
                        )}
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </TabPanel>
          )}
        </CardContent>
      </Card>

      <Modal isOpen={showPipelineModal} onClose={() => setShowPipelineModal(false)} title={'"创建处理管道"'} size="lg" footer={
        <div className="flex gap-2">
          <Button variant="secondary" onClick={() => setShowPipelineModal(false)}>取消</Button>
          <Button onClick={handleCreatePipeline} isLoading={createPipeline.isPending}>创建</Button>
        </div>
      }>
        <div className="space-y-4">
          <Input label={'"管道名称"'} placeholder={'"输入管道名称"'} value={newPipeline.name} onChange={e => setNewPipeline({ ...newPipeline, name: e.target.value })} />
          <Textarea label={'"描述"'} placeholder={'"管道描述..."'} value={newPipeline.description} onChange={e => setNewPipeline({ ...newPipeline, description: e.target.value })} rows={2} />
          <Input label={'"调度规则"'} placeholder={'"可选：Cron 表达式，如 0 0 * * *"'} value={newPipeline.schedule} onChange={e => setNewPipeline({ ...newPipeline, schedule: e.target.value })} hint={'"留空表示手动触发"'} />
          <div>
            <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">管道步骤</p>
            <div className="border-2 border-dashed border-gray-300 dark:border-dark-600 rounded-lg p-6 text-center">
              <Plus size={24} className="mx-auto text-gray-400 mb-2" />
              <p className="text-sm text-gray-500 dark:text-gray-400">点击添加处理步骤</p>
              <p className="text-xs text-gray-400 mt-1">支持数据清洗、转换、特征工程等步骤</p>
            </div>
          </div>
        </div>
      </Modal>
    </div>
  );
}
