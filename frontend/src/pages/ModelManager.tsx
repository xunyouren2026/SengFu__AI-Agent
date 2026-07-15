import { useState } from 'react';
import { Plus, Search, RefreshCw, Trash2, Edit, Power, PowerOff, TestTube, Star, Grid3X3, List, MoreVertical, Cpu, Zap, Eye } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Input, Badge, Modal, Select, EmptyState, Loading, Table, Pagination, StatsCard } from '../components/common';
import { useModels, useCreateModel, useUpdateModel, useDeleteModel, useToggleModel, useTestModel } from '../hooks/useModels';
import type { Model, ModelType, ModelStatus, ModelCapability } from '../types/model';

const statusColors: Record<string, 'success' | 'warning' | 'danger' | 'info'> = { active: 'success', inactive: 'warning', error: 'danger', loading: 'info' };
const statusLabels: Record<string, string> = { active: '运行中', inactive: '未激活', error: '错误', loading: '加载中' };
const typeLabels: Record<string, string> = { llm: '大语言模型', embedding: '嵌入模型', image: '图像模型', audio: '音频模型', multimodal: '多模态' };

export default function ModelManager() {
  const [search, setSearch] = useState('');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [page, setPage] = useState(1);
  const [createOpen, setCreateOpen] = useState(false);
  const [editModel, setEditModel] = useState<Model | null>(null);
  const [testModelId, setTestModelId] = useState<string | null>(null);
  const [testPrompt, setTestPrompt] = useState('');
  const [testResult, setTestResult] = useState<string>('');

  const { data: models, isLoading } = useModels();
  const createMutation = useCreateModel();
  const deleteMutation = useDeleteModel();
  const toggleMutation = useToggleModel();
  const testMutation = useTestModel();

  const filteredModels = (models || []).filter(m => m.name.toLowerCase().includes(search.toLowerCase()) || m.provider.toLowerCase().includes(search.toLowerCase()));
  const activeCount = (models || []).filter(m => m.status === 'active').length;
  const totalModels = models?.length || 0;

  const handleTest = () => {
    if (!testModelId || !testPrompt) return;
    testMutation.mutate({ id: testModelId, data: { prompt: testPrompt } }, {
      onSuccess: (res) => setTestResult((res as any).response || '测试成功'),
      onError: () => setTestResult('测试失败'),
    });
  };

  if (isLoading) return <Loading size="lg" text={'"加载模型列表..."'} />;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">模型管理</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">管理所有 AI 模型，查看状态和配置</p>
        </div>
        <Button leftIcon={<Plus size={16} />} onClick={() => setCreateOpen(true)}>添加模型</Button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatsCard title={'"总模型数"'} value={totalModels} icon={<Cpu size={24} />} color="blue" />
        <StatsCard title={'"活跃模型"'} value={activeCount} icon={<Zap size={24} />} color="green" />
        <StatsCard title={'"提供商"'} value={new Set((models || []).map(m => m.provider)).size} icon={<Eye size={24} />} color="purple" />
      </div>

      {/* Toolbar */}
      <Card padding="none">
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center gap-3">
            <Input placeholder={'"搜索模型..."'} value={search} onChange={e => setSearch(e.target.value)} leftIcon={<Search size={16} />} className="w-64" />
            <Select options={[{ label: '全部类型', value: '' }, { label: '大语言模型', value: 'llm' }, { label: '嵌入模型', value: 'embedding' }, { label: '图像模型', value: 'image' }]} className="w-36" />
          </div>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={() => setViewMode('grid')}><Grid3X3 size={16} /></Button>
            <Button variant="ghost" size="sm" onClick={() => setViewMode('list')}><List size={16} /></Button>
            <Button variant="ghost" size="sm" leftIcon={<RefreshCw size={14} />}>刷新</Button>
          </div>
        </div>
      </Card>

      {/* Models Grid/List */}
      {filteredModels.length === 0 ? (
        <EmptyState icon={<Cpu size={48} />} title={'"暂无模型"'} description={'点击"添加模型"按钮来添加第一个 AI 模型'} action={<Button leftIcon={<Plus size={16} />} onClick={() => setCreateOpen(true)}>添加模型</Button>} />
      ) : viewMode === 'grid' ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredModels.map(model => (
            <Card key={model.id} hoverable variant="bordered">
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center"><Cpu size={20} className="text-primary-500" /></div>
                  <div>
                    <h3 className="font-medium text-gray-900 dark:text-white">{model.name}</h3>
                    <p className="text-xs text-gray-500">{model.provider}</p>
                  </div>
                </div>
                <Badge variant={statusColors[model.status]} dot>{statusLabels[model.status]}</Badge>
              </div>
              <div className="flex flex-wrap gap-1 mb-3">
                <Badge variant="info" size="sm">{typeLabels[model.type] || model.type}</Badge>
                {model.capabilities?.slice(0, 3).map(cap => <Badge key={cap} variant="outline" size="sm">{cap}</Badge>)}
                {model.capabilities?.length > 3 && <Badge variant="outline" size="sm">+{model.capabilities.length - 3}</Badge>}
              </div>
              <div className="flex items-center gap-2 pt-3 border-t border-gray-100 dark:border-dark-600">
                <Button variant="ghost" size="sm" onClick={() => setEditModel(model)} leftIcon={<Edit size={14} />}>编辑</Button>
                <Button variant="ghost" size="sm" onClick={() => setTestModelId(model.id)} leftIcon={<TestTube size={14} />}>测试</Button>
                <Button variant="ghost" size="sm" onClick={() => toggleMutation.mutate(model.id)} leftIcon={model.status === 'active' ? <PowerOff size={14} /> : <Power size={14} />}>
                  {model.status === 'active' ? '禁用' : '启用'}
                </Button>
                <Button variant="ghost" size="sm" onClick={() => deleteMutation.mutate(model.id)} leftIcon={<Trash2 size={14} />} className="text-red-500 hover:text-red-600">删除</Button>
              </div>
            </Card>
          ))}
        </div>
      ) : (
        <Card padding="none">
          <Table columns={[
            { key: 'name', title: '模型名称', render: (_, row) => <div className="flex items-center gap-3"><Cpu size={16} className="text-primary-500" /><div><span className="font-medium">{row.name}</span><br /><span className="text-xs text-gray-400">{row.provider}</span></div></div> },
            { key: 'type', title: '类型', render: (v) => <Badge variant="info">{typeLabels[v as string] || v}</Badge> },
            { key: 'status', title: '状态', render: (v) => <Badge variant={statusColors[v as string]} dot>{statusLabels[v as string]}</Badge> },
            { key: 'capabilities', title: '能力', render: (v) => <div className="flex flex-wrap gap-1">{(v as string[]).slice(0, 3).map(c => <Badge key={c} variant="outline" size="sm">{c}</Badge>)}</div> },
            { key: 'actions', title: '操作', width: '200px', render: (_, row) => (
              <div className="flex items-center gap-1">
                <Button variant="ghost" size="sm" onClick={() => setEditModel(row)}><Edit size={14} /></Button>
                <Button variant="ghost" size="sm" onClick={() => setTestModelId(row.id)}><TestTube size={14} /></Button>
                <Button variant="ghost" size="sm" onClick={() => deleteMutation.mutate(row.id)} className="text-red-500"><Trash2 size={14} /></Button>
              </div>
            )},
          ]} data={filteredModels as any[]} />
        </Card>
      )}

      {/* Create/Edit Modal */}
      <Modal isOpen={createOpen || !!editModel} onClose={() => { setCreateOpen(false); setEditModel(null); }} title={editModel ? '编辑模型' : '添加模型'} size="lg" footer={<><Button variant="ghost" onClick={() => { setCreateOpen(false); setEditModel(null); }}>取消</Button><Button onClick={() => { setCreateOpen(false); setEditModel(null); }}>保存</Button></>}>
        <div className="space-y-4">
          <Input label={'"模型名称"'} placeholder={'"例如: gpt-4o"'} defaultValue={editModel?.name} />
          <Select label={'"提供商"'} options={[{ label: 'OpenAI', value: 'openai' }, { label: 'Anthropic', value: 'anthropic' }, { label: 'DeepSeek', value: 'deepseek' }, { label: '智谱AI', value: 'zhipu' }, { label: '本地模型', value: 'local' }]} defaultValue={editModel?.provider} />
          <Select label={'"类型"'} options={[{ label: '大语言模型', value: 'llm' }, { label: '嵌入模型', value: 'embedding' }, { label: '图像模型', value: 'image' }, { label: '音频模型', value: 'audio' }, { label: '多模态', value: 'multimodal' }]} defaultValue={editModel?.type} />
          <Input label={'"API 端点"'} placeholder="https://api.openai.com/v1" defaultValue={(editModel?.config as any)?.api_base} />
          <Input label={'"API 密钥"'} type="password" placeholder="sk-..." defaultValue={(editModel?.config as any)?.api_key} />
        </div>
      </Modal>

      {/* Test Modal */}
      <Modal isOpen={!!testModelId} onClose={() => { setTestModelId(null); setTestResult(''); setTestPrompt(''); }} title={'"模型测试"'} size="lg" footer={<><Button variant="ghost" onClick={() => { setTestModelId(null); setTestResult(''); }}>关闭</Button><Button onClick={handleTest} isLoading={testMutation.isPending}>发送测试</Button></>}>
        <div className="space-y-4">
          <Input label={'"测试提示词"'} placeholder={'"输入测试内容..."'} value={testPrompt} onChange={e => setTestPrompt(e.target.value)} />
          {testResult && (
            <div className="p-4 rounded-lg bg-gray-50 dark:bg-dark-700 border border-gray-200 dark:border-dark-600">
              <p className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">{testResult}</p>
            </div>
          )}
        </div>
      </Modal>
    </div>
  );
}
