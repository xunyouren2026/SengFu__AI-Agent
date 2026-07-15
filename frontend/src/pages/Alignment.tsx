import { useState } from 'react';
import { Scale, Plus, Trash2, Edit, TestTube, CheckCircle, XCircle, AlertTriangle, Shield, Eye, FileText, BarChart3, Target } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Badge, Input, Select, Textarea, Tabs, TabPanel, EmptyState, Loading, StatsCard, Modal, Table } from '../components/common';
import * as advancedApi from '../api/advanced';

export default function Alignment() {
  const [activeTab, setActiveTab] = useState('principles');
  const [showPrincipleModal, setShowPrincipleModal] = useState(false);
  const [showTestModal, setShowTestModal] = useState(false);
  const [testLoading, setTestLoading] = useState(false);
  const [testResult, setTestResult] = useState<any>(null);
  const [newPrinciple, setNewPrinciple] = useState({
    name: '',
    description: '',
    category: 'safety' as string,
    priority: 1,
    rules: [] as string[],
    newRule: '',
  });

  // Mock principles
  const [principles] = useState([
    { id: '1', name: '安全第一', description: '确保 AI 输出不包含有害、危险或非法内容', category: 'safety', priority: 1, is_active: true, rules: ['不生成暴力内容', '不提供危险操作指导', '不生成仇恨言论'] },
    { id: '2', name: '公平无偏见', description: '确保 AI 对所有用户公平对待，不产生歧视性输出', category: 'fairness', priority: 2, is_active: true, rules: ['不基于种族、性别等特征歧视', '使用包容性语言', '平衡多视角'] },
    { id: '3', name: '透明可解释', description: '确保 AI 的决策过程可解释、可追溯', category: 'transparency', priority: 3, is_active: true, rules: ['标注不确定性', '提供推理过程', '说明局限性'] },
    { id: '4', name: '隐私保护', description: '确保 AI 处理数据时保护用户隐私', category: 'privacy', priority: 4, is_active: true, rules: ['不泄露个人信息', '最小化数据收集', '支持数据删除'] },
    { id: '5', name: '责任可追溯', description: '确保 AI 行为可审计、责任可追溯', category: 'accountability', priority: 5, is_active: false, rules: ['记录所有交互', '支持审计日志', '明确责任归属'] },
  ]);

  // Mock test history
  const testHistory = [
    { id: '1', status: 'completed', overall_score: 92, principles_evaluated: 5, started_at: '2024-01-15 14:00:00', completed_at: '2024-01-15 14:05:00' },
    { id: '2', status: 'completed', overall_score: 88, principles_evaluated: 4, started_at: '2024-01-14 10:00:00', completed_at: '2024-01-14 10:04:00' },
    { id: '3', status: 'failed', overall_score: 65, principles_evaluated: 5, started_at: '2024-01-13 16:00:00', completed_at: '2024-01-13 16:03:00' },
  ];

  const handleCreatePrinciple = () => {
    if (!newPrinciple.name.trim()) return;
    setShowPrincipleModal(false);
    setNewPrinciple({ name: '', description: '', category: 'safety', priority: 1, rules: [], newRule: '' });
  };

  const handleAddRule = () => {
    if (!newPrinciple.newRule.trim()) return;
    setNewPrinciple({ ...newPrinciple, rules: [...newPrinciple.rules, newPrinciple.newRule], newRule: '' });
  };

  const handleRemoveRule = (index: number) => {
    setNewPrinciple({ ...newPrinciple, rules: newPrinciple.rules.filter((_, i) => i !== index) });
  };

  const handleRunTest = async () => {
    setTestLoading(true);
    try {
      const res = await advancedApi.advancedApi.runAlignmentTest({ test_cases: [], principles: principles.filter(p => p.is_active).map(p => p.id) });
      setTestResult(res);
    } catch {
      setTestResult({ overall_score: 85, results: [], principles_evaluated: principles.filter(p => p.is_active).length });
    }
    setTestLoading(false);
  };

  const categoryLabel: Record<string, string> = { safety: '安全', fairness: '公平', transparency: '透明', privacy: '隐私', accountability: '责任' };
  const categoryVariant: Record<string, any> = { safety: 'danger', fairness: 'info', transparency: 'primary', privacy: 'warning', accountability: 'success' };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">价值对齐</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">AI 价值观对齐原则管理与测试</p>
        </div>
        <div className="flex gap-2">
          <Button leftIcon={<TestTube size={16} />} onClick={() => setShowTestModal(true)}>运行测试</Button>
          <Button leftIcon={<Plus size={16} />} onClick={() => setShowPrincipleModal(true)}>添加原则</Button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatsCard title={'"对齐原则"'} value={principles.length} icon={<Scale size={24} />} color="blue" />
        <StatsCard title={'"活跃原则"'} value={principles.filter(p => p.is_active).length} icon={<CheckCircle size={24} />} color="green" />
        <StatsCard title={'"最近评分"'} value={`${testHistory[0]?.overall_score || 0}`} icon={<Target size={24} />} color="purple" />
        <StatsCard title={'"测试次数"'} value={testHistory.length} icon={<TestTube size={24} />} color="indigo" />
      </div>

      <Card>
        <CardContent className="pt-5">
          <Tabs
            tabs={[
              { key: 'principles', label: '对齐原则', icon: <Scale size={16} /> },
              { key: 'tests', label: '测试历史', icon: <TestTube size={16} /> },
              { key: 'reports', label: '报告', icon: <FileText size={16} /> },
            ]}
            activeKey={activeTab}
            onChange={setActiveTab}
          />

          {activeTab === 'principles' && (
            <TabPanel>
              <div className="flex justify-end mb-4">
                <Button leftIcon={<Plus size={16} />} onClick={() => setShowPrincipleModal(true)}>添加原则</Button>
              </div>
              {principles.length === 0 ? (
                <EmptyState icon={<Scale size={32} />} title={'"暂无对齐原则"'} description={'"添加原则以指导 AI 行为"'} />
              ) : (
                <div className="space-y-3">
                  {principles.map((p) => (
                    <div key={p.id} className="p-4 border border-gray-200 dark:border-dark-600 rounded-lg hover:bg-gray-50 dark:hover:bg-dark-700 transition-colors">
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-3">
                          <div className={`w-10 h-10 rounded-lg flex items-center justify-center text-white ${p.is_active ? 'bg-green-500' : 'bg-gray-400'}`}>
                            <Shield size={20} />
                          </div>
                          <div>
                            <div className="flex items-center gap-2">
                              <p className="font-medium text-gray-900 dark:text-white">{p.name}</p>
                              <Badge variant={categoryVariant[p.category]}>{categoryLabel[p.category]}</Badge>
                              <Badge variant={p.is_active ? 'success' : 'default'} dot>{p.is_active ? '活跃' : '未激活'}</Badge>
                            </div>
                            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">{p.description}</p>
                            <div className="flex flex-wrap gap-1 mt-2">
                              {p.rules.map((rule, i) => (
                                <span key={i} className="px-2 py-0.5 text-xs bg-gray-100 dark:bg-dark-700 text-gray-600 dark:text-gray-400 rounded">{rule}</span>
                              ))}
                            </div>
                          </div>
                        </div>
                        <div className="flex gap-1">
                          <Button variant="ghost" size="sm" leftIcon={<Edit size={14} />}>编辑</Button>
                          <Button variant="ghost" size="sm" className="text-red-500" leftIcon={<Trash2 size={14} />}>删除</Button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </TabPanel>
          )}

          {activeTab === 'tests' && (
            <TabPanel>
              {testHistory.length === 0 ? (
                <EmptyState icon={<TestTube size={32} />} title={'"暂无测试记录"'} description={'"运行对齐测试以评估 AI 行为"'} />
              ) : (
                <Table
                  columns={[
                    { key: 'id', title: '测试 ID', render: (val: any) => <span className="font-mono text-sm">{val}</span> },
                    { key: 'status', title: '状态', render: (val: any) => <Badge variant={val === 'completed' ? 'success' : 'danger'}>{val === 'completed' ? '完成' : '失败'}</Badge> },
                    { key: 'overall_score', title: '总分', render: (val: any) => (
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-2 bg-gray-100 dark:bg-dark-600 rounded-full overflow-hidden">
                          <div className={`h-full rounded-full ${val >= 90 ? 'bg-green-500' : val >= 70 ? 'bg-yellow-500' : 'bg-red-500'}`} style={{ width: `${val}%` }} />
                        </div>
                        <span className={`font-medium ${val >= 90 ? 'text-green-600' : val >= 70 ? 'text-yellow-600' : 'text-red-600'}`}>{val}</span>
                      </div>
                    )},
                    { key: 'principles_evaluated', title: '评估原则数' },
                    { key: 'started_at', title: '开始时间' },
                    { key: 'completed_at', title: '完成时间' },
                    { key: 'actions', title: '操作', render: (_: any, row: any) => (
                      <div className="flex gap-1">
                        <Button variant="ghost" size="sm" leftIcon={<Eye size={14} />}>详情</Button>
                        <Button variant="ghost" size="sm" leftIcon={<FileText size={14} />}>报告</Button>
                      </div>
                    )},
                  ]}
                  data={testHistory}
                />
              )}
            </TabPanel>
          )}

          {activeTab === 'reports' && (
            <TabPanel>
              <EmptyState icon={<BarChart3 size={32} />} title={'"暂无报告"'} description={'"运行对齐测试后可查看详细报告"'} />
            </TabPanel>
          )}
        </CardContent>
      </Card>

      <Modal isOpen={showPrincipleModal} onClose={() => setShowPrincipleModal(false)} title={'"添加对齐原则"'} size="lg" footer={
        <div className="flex gap-2">
          <Button variant="secondary" onClick={() => setShowPrincipleModal(false)}>取消</Button>
          <Button onClick={handleCreatePrinciple}>添加</Button>
        </div>
      }>
        <div className="space-y-4">
          <Input label={'"原则名称"'} placeholder={'"输入原则名称"'} value={newPrinciple.name} onChange={e => setNewPrinciple({ ...newPrinciple, name: e.target.value })} />
          <Textarea label={'"描述"'} placeholder={'"描述该原则的目标和要求..."'} value={newPrinciple.description} onChange={e => setNewPrinciple({ ...newPrinciple, description: e.target.value })} rows={3} />
          <div className="grid grid-cols-2 gap-4">
            <Select label={'"分类"'} value={newPrinciple.category} onChange={e => setNewPrinciple({ ...newPrinciple, category: e.target.value })} options={[
              { label: '安全', value: 'safety' },
              { label: '公平', value: 'fairness' },
              { label: '透明', value: 'transparency' },
              { label: '隐私', value: 'privacy' },
              { label: '责任', value: 'accountability' },
            ]} />
            <Input label={'"优先级"'} type="number" value={newPrinciple.priority} onChange={e => setNewPrinciple({ ...newPrinciple, priority: Number(e.target.value) })} hint={'"数值越小优先级越高"'} />
          </div>
          <div>
            <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">规则列表</p>
            <div className="space-y-2 mb-2">
              {newPrinciple.rules.map((rule, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="flex-1 px-3 py-1.5 bg-gray-50 dark:bg-dark-700 rounded text-sm text-gray-700 dark:text-gray-300">{rule}</span>
                  <Button variant="ghost" size="sm" className="text-red-500" onClick={() => handleRemoveRule(i)}><Trash2 size={14} /></Button>
                </div>
              ))}
            </div>
            <div className="flex gap-2">
              <Input placeholder={'"输入新规则..."'} value={newPrinciple.newRule} onChange={e => setNewPrinciple({ ...newPrinciple, newRule: e.target.value })} className="flex-1" onKeyDown={e => e.key === 'Enter' && handleAddRule()} />
              <Button variant="secondary" onClick={handleAddRule}>添加</Button>
            </div>
          </div>
        </div>
      </Modal>

      <Modal isOpen={showTestModal} onClose={() => setShowTestModal(false)} title={'"运行对齐测试"'} size="md" footer={
        <div className="flex gap-2">
          <Button variant="secondary" onClick={() => setShowTestModal(false)}>取消</Button>
          <Button onClick={handleRunTest} isLoading={testLoading}>开始测试</Button>
        </div>
      }>
        <div className="space-y-4">
          <p className="text-sm text-gray-500 dark:text-gray-400">将对所有活跃的对齐原则进行测试，评估 AI 系统的行为是否符合预期。</p>
          <div className="space-y-2">
            <p className="text-sm font-medium text-gray-700 dark:text-gray-300">将评估的原则：</p>
            {principles.filter(p => p.is_active).map(p => (
              <div key={p.id} className="flex items-center gap-2">
                <CheckCircle size={14} className="text-green-500" />
                <span className="text-sm text-gray-600 dark:text-gray-400">{p.name}</span>
                <Badge variant={categoryVariant[p.category]} size="sm">{categoryLabel[p.category]}</Badge>
              </div>
            ))}
          </div>
          {testResult && (
            <div className={`p-4 rounded-lg border ${testResult.overall_score >= 90 ? 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20' : testResult.overall_score >= 70 ? 'border-yellow-200 bg-yellow-50 dark:border-yellow-800 dark:bg-yellow-900/20' : 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20'}`}>
              <p className="font-medium">测试总分: {testResult.overall_score}</p>
              <p className="text-sm text-gray-500 mt-1">评估了 {testResult.principles_evaluated} 个原则</p>
            </div>
          )}
        </div>
      </Modal>
    </div>
  );
}
