import { useState } from 'react';
import { Network, Plus, Trash2, Server, Activity, Users, Trophy, Clock, BarChart3, RefreshCw, Wifi, WifiOff } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Badge, Input, Select, Textarea, Tabs, TabPanel, EmptyState, Loading, StatsCard, Modal, Table } from '../components/common';
import { useFederatedNodes, useRegisterNode } from '../hooks/useAdvanced';

export default function Federated() {
  const [activeTab, setActiveTab] = useState('nodes');
  const [showNodeModal, setShowNodeModal] = useState(false);
  const [newNode, setNewNode] = useState({ name: '', endpoint: '', public_key: '', capabilities: [] as string[] });

  const { data: nodesData, isLoading } = useFederatedNodes();
  const registerNode = useRegisterNode();

  const nodes = (nodesData?.data || []) as any[];

  const handleRegister = () => {
    if (!newNode.name.trim() || !newNode.endpoint.trim()) return;
    registerNode.mutate({
      name: newNode.name,
      endpoint: newNode.endpoint,
      public_key: newNode.public_key || undefined,
      capabilities: newNode.capabilities,
    });
    setShowNodeModal(false);
    setNewNode({ name: '', endpoint: '', public_key: '', capabilities: [] });
  };

  const statusLabel: Record<string, string> = { online: '在线', offline: '离线', training: '训练中', syncing: '同步中', error: '错误' };
  const statusVariant: Record<string, any> = { online: 'success', offline: 'default', training: 'primary', syncing: 'warning', error: 'danger' };

  // Mock training rounds
  const trainingRounds = [
    { id: '1', round_number: 12, status: 'completed', participating_nodes: ['node-1', 'node-2', 'node-3'], aggregation_strategy: 'fedavg', started_at: '2024-01-15 08:00:00', global_model_version: 'v1.12', metrics: { accuracy: 0.95, loss: 0.05 } },
    { id: '2', round_number: 11, status: 'completed', participating_nodes: ['node-1', 'node-2', 'node-3', 'node-4'], aggregation_strategy: 'fedavg', started_at: '2024-01-14 08:00:00', global_model_version: 'v1.11', metrics: { accuracy: 0.94, loss: 0.06 } },
    { id: '3', round_number: 10, status: 'completed', participating_nodes: ['node-1', 'node-2'], aggregation_strategy: 'fedprox', started_at: '2024-01-13 08:00:00', global_model_version: 'v1.10', metrics: { accuracy: 0.93, loss: 0.07 } },
  ];

  // Mock contribution stats
  const contributionStats = [
    { node_id: 'node-1', node_name: 'GPU 集群 A', total_rounds: 12, successful_rounds: 12, data_samples_contributed: 50000, computation_hours: 120, reward_tokens: 1200, reputation_score: 98 },
    { node_id: 'node-2', node_name: 'GPU 集群 B', total_rounds: 10, successful_rounds: 9, data_samples_contributed: 35000, computation_hours: 85, reward_tokens: 850, reputation_score: 92 },
    { node_id: 'node-3', node_name: '边缘节点 C', total_rounds: 8, successful_rounds: 7, data_samples_contributed: 20000, computation_hours: 40, reward_tokens: 400, reputation_score: 85 },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">联邦学习</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">分布式模型训练与聚合</p>
        </div>
        <Button leftIcon={<Plus size={16} />} onClick={() => setShowNodeModal(true)}>注册节点</Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatsCard title={'"节点总数"'} value={nodes.length} icon={<Server size={24} />} color="blue" />
        <StatsCard title={'"在线节点"'} value={nodes.filter((n: any) => n.status === 'online' || n.status === 'training').length} icon={<Wifi size={24} />} color="green" />
        <StatsCard title={'"训练轮次"'} value={trainingRounds.length} icon={<RefreshCw size={24} />} color="purple" />
        <StatsCard title={'"总数据样本"'} value="105K" icon={<BarChart3 size={24} />} color="indigo" />
      </div>

      <Card>
        <CardContent className="pt-5">
          <Tabs
            tabs={[
              { key: 'nodes', label: '节点管理', icon: <Server size={16} /> },
              { key: 'rounds', label: '训练轮次', icon: <RefreshCw size={16} /> },
              { key: 'contributions', label: '贡献统计', icon: <Trophy size={16} /> },
            ]}
            activeKey={activeTab}
            onChange={setActiveTab}
          />

          {activeTab === 'nodes' && (
            <TabPanel>
              {isLoading ? (
                <Loading />
              ) : nodes.length === 0 ? (
                <EmptyState icon={<Network size={32} />} title={'"暂无注册节点"'} description={'"注册节点以开始联邦学习训练"'} />
              ) : (
                <Table
                  columns={[
                    { key: 'name', title: '节点名称', render: (val: any) => <span className="font-medium text-gray-900 dark:text-white">{val}</span> },
                    { key: 'endpoint', title: '端点' },
                    { key: 'status', title: '状态', render: (val: any) => <Badge variant={statusVariant[val]} dot>{statusLabel[val] || val}</Badge> },
                    { key: 'dataset_size', title: '数据量', render: (val: any) => val ? `${(val / 1000).toFixed(0)}K` : '-' },
                    { key: 'contribution_score', title: '贡献分', render: (val: any) => <Badge variant={val >= 90 ? 'success' : val >= 70 ? 'warning' : 'danger'}>{val}</Badge> },
                    { key: 'last_seen', title: '最后活跃' },
                    { key: 'actions', title: '操作', render: () => (
                      <div className="flex gap-1">
                        <Button variant="ghost" size="sm">详情</Button>
                        <Button variant="ghost" size="sm" className="text-red-500" leftIcon={<Trash2 size={14} />}>移除</Button>
                      </div>
                    )},
                  ]}
                  data={nodes}
                />
              )}
            </TabPanel>
          )}

          {activeTab === 'rounds' && (
            <TabPanel>
              {trainingRounds.length === 0 ? (
                <EmptyState icon={<RefreshCw size={32} />} title={'"暂无训练轮次"'} />
              ) : (
                <Table
                  columns={[
                    { key: 'round_number', title: '轮次', render: (val: any) => <span className="font-medium text-gray-900 dark:text-white">#{val}</span> },
                    { key: 'status', title: '状态', render: (val: any) => <Badge variant={val === 'completed' ? 'success' : val === 'running' ? 'primary' : 'default'}>{val === 'completed' ? '已完成' : val === 'running' ? '进行中' : val}</Badge> },
                    { key: 'participating_nodes', title: '参与节点', render: (val: any) => <Badge variant="outline">{val?.length || 0} 个节点</Badge> },
                    { key: 'aggregation_strategy', title: '聚合策略', render: (val: any) => <Badge variant="info">{val}</Badge> },
                    { key: 'global_model_version', title: '模型版本' },
                    { key: 'started_at', title: '开始时间' },
                    { key: 'metrics', title: '准确率', render: (val: any) => val?.accuracy ? `${(val.accuracy * 100).toFixed(1)}%` : '-' },
                  ]}
                  data={trainingRounds}
                />
              )}
            </TabPanel>
          )}

          {activeTab === 'contributions' && (
            <TabPanel>
              {contributionStats.length === 0 ? (
                <EmptyState icon={<Trophy size={32} />} title={'"暂无贡献数据"'} />
              ) : (
                <Table
                  columns={[
                    { key: 'node_name', title: '节点名称', render: (val: any) => <span className="font-medium text-gray-900 dark:text-white">{val}</span> },
                    { key: 'total_rounds', title: '总轮次' },
                    { key: 'successful_rounds', title: '成功轮次', render: (val: any, row: any) => `${val}/${row.total_rounds}` },
                    { key: 'data_samples_contributed', title: '数据样本', render: (val: any) => `${(val / 1000).toFixed(0)}K` },
                    { key: 'computation_hours', title: '计算时长', render: (val: any) => `${val}h` },
                    { key: 'reward_tokens', title: '奖励代币', render: (val: any) => <span className="text-yellow-600 dark:text-yellow-400 font-medium">{val}</span> },
                    { key: 'reputation_score', title: '信誉分', render: (val: any) => <Badge variant={val >= 90 ? 'success' : val >= 70 ? 'warning' : 'danger'}>{val}</Badge> },
                  ]}
                  data={contributionStats}
                />
              )}
            </TabPanel>
          )}
        </CardContent>
      </Card>

      <Modal isOpen={showNodeModal} onClose={() => setShowNodeModal(false)} title={'"注册联邦节点"'} size="md" footer={
        <div className="flex gap-2">
          <Button variant="secondary" onClick={() => setShowNodeModal(false)}>取消</Button>
          <Button onClick={handleRegister} isLoading={registerNode.isPending}>注册</Button>
        </div>
      }>
        <div className="space-y-4">
          <Input label={'"节点名称"'} placeholder={'"输入节点名称"'} value={newNode.name} onChange={e => setNewNode({ ...newNode, name: e.target.value })} />
          <Input label={'"端点 URL"'} placeholder="https://node.example.com" value={newNode.endpoint} onChange={e => setNewNode({ ...newNode, endpoint: e.target.value })} />
          <Textarea label={'"公钥"'} placeholder={'"输入节点公钥（可选）"'} value={newNode.public_key} onChange={e => setNewNode({ ...newNode, public_key: e.target.value })} rows={3} />
          <Input label={'"数据集信息"'} placeholder={'"描述数据集规模和类型"'} />
        </div>
      </Modal>
    </div>
  );
}
