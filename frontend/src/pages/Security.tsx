import { useState } from 'react';
import { Shield, ShieldAlert, ShieldCheck, FileText, Search, Bug, Plus, Trash2, AlertTriangle, CheckCircle, Clock, Eye, Ban, AlertOctagon, Lock } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Badge, Input, Select, Textarea, Tabs, TabPanel, EmptyState, Loading, StatsCard, Modal, Table } from '../components/common';
import * as advancedApi from '../api/advanced';

export default function Security() {
  const [activeTab, setActiveTab] = useState('firewall');
  const [showRuleModal, setShowRuleModal] = useState(false);
  const [showScanModal, setShowScanModal] = useState(false);
  const [promptTest, setPromptTest] = useState('');
  const [promptResult, setPromptResult] = useState<any>(null);
  const [newRule, setNewRule] = useState({ name: '', source_ip: '', destination_ip: '', port: '', protocol: 'tcp', action: 'deny' as string, priority: 100 });
  const [scanTarget, setScanTarget] = useState('');
  const [scanType, setScanType] = useState('vulnerability');
  const [loading, setLoading] = useState(false);

  // Mock data for demo
  const [firewallRules] = useState([
    { id: '1', name: '阻止 SQL 注入', source_ip: '*', destination_ip: '*', port: '80,443', protocol: 'tcp', action: 'deny', priority: 1, enabled: true },
    { id: '2', name: '允许内部 API', source_ip: '10.0.0.0/8', destination_ip: '*', port: '8080', protocol: 'tcp', action: 'allow', priority: 10, enabled: true },
    { id: '3', name: '阻止 XSS 攻击', source_ip: '*', destination_ip: '*', port: '*', protocol: 'http', action: 'deny', priority: 2, enabled: true },
  ]);

  const [auditLogs] = useState([
    { id: '1', timestamp: '2024-01-15 10:30:00', action: 'prompt_guard_triggered', user: 'system', ip: '192.168.1.100', details: '检测到潜在注入攻击' },
    { id: '2', timestamp: '2024-01-15 10:25:00', action: 'firewall_blocked', user: 'anonymous', ip: '45.33.32.156', details: '防火墙规则阻止了请求' },
    { id: '3', timestamp: '2024-01-15 10:20:00', action: 'scan_completed', user: 'admin', ip: '127.0.0.1', details: '漏洞扫描完成，发现 2 个问题' },
  ]);

  const [threats] = useState([
    { id: '1', level: 'high', type: 'prompt_injection', description: '检测到多起提示注入尝试', source: '外部请求', detected_at: '2024-01-15 10:00:00', status: 'active' },
    { id: '2', level: 'medium', type: 'rate_limit', description: '异常高频请求', source: '192.168.1.50', detected_at: '2024-01-15 09:30:00', status: 'monitoring' },
    { id: '3', level: 'low', type: 'suspicious_pattern', description: '异常访问模式检测', source: '10.0.0.15', detected_at: '2024-01-15 09:00:00', status: 'resolved' },
  ]);

  const handleCreateRule = () => {
    setShowRuleModal(false);
    setNewRule({ name: '', source_ip: '', destination_ip: '', port: '', protocol: 'tcp', action: 'deny', priority: 100 });
  };

  const handleTestPrompt = async () => {
    if (!promptTest.trim()) return;
    setLoading(true);
    try {
      const res = await advancedApi.advancedApi.testPromptGuard({ prompt: promptTest });
      setPromptResult(res);
    } catch {
      setPromptResult({ passed: false, risk_score: 0.9, detected_issues: [{ type: 'error', message: '测试失败' }], recommendations: ['请重试'] });
    }
    setLoading(false);
  };

  const handleStartScan = () => {
    setShowScanModal(false);
  };

  const actionLabel: Record<string, string> = { allow: '允许', deny: '拒绝', log: '记录', alert: '告警' };
  const actionVariant: Record<string, any> = { allow: 'success', deny: 'danger', log: 'info', alert: 'warning' };
  const levelVariant: Record<string, any> = { low: 'info', medium: 'warning', high: 'danger', critical: 'danger' };
  const levelLabel: Record<string, string> = { low: '低', medium: '中', high: '高', critical: '严重' };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">安全中心</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">系统安全防护与监控</p>
        </div>
        <Badge variant="success" dot>安全</Badge>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatsCard title={'"防火墙规则"'} value={firewallRules.length} icon={<Shield size={24} />} color="blue" />
        <StatsCard title={'"活跃威胁"'} value={threats.filter(t => t.status === 'active').length} icon={<ShieldAlert size={24} />} color="red" />
        <StatsCard title={'"审计日志"'} value={auditLogs.length} icon={<FileText size={24} />} color="purple" />
        <StatsCard title={'"安全评分"'} value="92" icon={<ShieldCheck size={24} />} color="green" />
      </div>

      <Card>
        <CardContent className="pt-5">
          <Tabs
            tabs={[
              { key: 'firewall', label: '防火墙规则', icon: <Shield size={16} /> },
              { key: 'audit', label: '审计日志', icon: <FileText size={16} /> },
              { key: 'threats', label: '威胁检测', icon: <Bug size={16} /> },
              { key: 'prompt_guard', label: '提示词守卫', icon: <Lock size={16} /> },
            ]}
            activeKey={activeTab}
            onChange={setActiveTab}
          />

          {activeTab === 'firewall' && (
            <TabPanel>
              <div className="flex justify-end mb-4">
                <Button leftIcon={<Plus size={16} />} onClick={() => setShowRuleModal(true)}>添加规则</Button>
              </div>
              {firewallRules.length === 0 ? (
                <EmptyState icon={<Shield size={32} />} title={'"暂无防火墙规则"'} description={'"添加规则以保护系统安全"'} />
              ) : (
                <Table
                  columns={[
                    { key: 'name', title: '规则名称', render: (val: any) => <span className="font-medium text-gray-900 dark:text-white">{val}</span> },
                    { key: 'source_ip', title: '来源 IP' },
                    { key: 'port', title: '端口' },
                    { key: 'protocol', title: '协议' },
                    { key: 'action', title: '动作', render: (val: any) => <Badge variant={actionVariant[val] || 'default'}>{actionLabel[val] || val}</Badge> },
                    { key: 'priority', title: '优先级' },
                    { key: 'actions', title: '操作', render: () => (
                      <div className="flex gap-1">
                        <Button variant="ghost" size="sm">编辑</Button>
                        <Button variant="ghost" size="sm" className="text-red-500" leftIcon={<Trash2 size={14} />}>删除</Button>
                      </div>
                    )},
                  ]}
                  data={firewallRules}
                />
              )}
            </TabPanel>
          )}

          {activeTab === 'audit' && (
            <TabPanel>
              {auditLogs.length === 0 ? (
                <EmptyState icon={<FileText size={32} />} title={'"暂无审计日志"'} />
              ) : (
                <Table
                  columns={[
                    { key: 'timestamp', title: '时间' },
                    { key: 'action', title: '操作', render: (val: any) => <Badge variant="outline">{val}</Badge> },
                    { key: 'user', title: '用户' },
                    { key: 'ip', title: 'IP 地址' },
                    { key: 'details', title: '详情' },
                  ]}
                  data={auditLogs}
                />
              )}
            </TabPanel>
          )}

          {activeTab === 'threats' && (
            <TabPanel>
              <div className="flex justify-end mb-4">
                <Button leftIcon={<Search size={16} />} onClick={() => setShowScanModal(true)}>安全扫描</Button>
              </div>
              {threats.length === 0 ? (
                <EmptyState icon={<ShieldCheck size={32} />} title={'"未检测到威胁"'} description={'"系统安全运行中"'} />
              ) : (
                <Table
                  columns={[
                    { key: 'level', title: '级别', render: (val: any) => <Badge variant={levelVariant[val]}>{levelLabel[val]}</Badge> },
                    { key: 'type', title: '类型', render: (val: any) => <Badge variant="outline">{val}</Badge> },
                    { key: 'description', title: '描述' },
                    { key: 'source', title: '来源' },
                    { key: 'detected_at', title: '检测时间' },
                    { key: 'status', title: '状态', render: (val: any) => <Badge variant={val === 'active' ? 'danger' : val === 'monitoring' ? 'warning' : 'success'}>{val === 'active' ? '活跃' : val === 'monitoring' ? '监控中' : '已解决'}</Badge> },
                  ]}
                  data={threats}
                />
              )}
            </TabPanel>
          )}

          {activeTab === 'prompt_guard' && (
            <TabPanel>
              <div className="space-y-4">
                <Textarea
                  label={'"测试提示词"'}
                  placeholder={'"输入需要安全检测的提示词..."'}
                  value={promptTest}
                  onChange={e => setPromptTest(e.target.value)}
                  rows={3}
                />
                <Button leftIcon={<Search size={16} />} onClick={handleTestPrompt} isLoading={loading} disabled={!promptTest.trim()}>安全检测</Button>

                {promptResult && (
                  <div className={`p-4 rounded-lg border ${promptResult.passed ? 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20' : 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20'}`}>
                    <div className="flex items-center gap-2 mb-2">
                      {promptResult.passed ? <CheckCircle size={20} className="text-green-500" /> : <AlertTriangle size={20} className="text-red-500" />}
                      <span className={`font-medium ${promptResult.passed ? 'text-green-700 dark:text-green-300' : 'text-red-700 dark:text-red-300'}`}>
                        {promptResult.passed ? '通过安全检测' : '检测到安全风险'}
                      </span>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">风险评分: {(promptResult.risk_score * 100).toFixed(0)}%</p>
                    {promptResult.detected_issues?.length > 0 && (
                      <div className="mt-2 space-y-1">
                        {promptResult.detected_issues.map((issue: any, i: number) => (
                          <p key={i} className="text-sm text-red-600 dark:text-red-400">- {issue.message || issue.type}</p>
                        ))}
                      </div>
                    )}
                    {promptResult.recommendations?.length > 0 && (
                      <div className="mt-2">
                        <p className="text-sm font-medium text-gray-700 dark:text-gray-300">建议:</p>
                        {promptResult.recommendations.map((rec: string, i: number) => (
                          <p key={i} className="text-sm text-gray-500 dark:text-gray-400">- {rec}</p>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </TabPanel>
          )}
        </CardContent>
      </Card>

      <Modal isOpen={showRuleModal} onClose={() => setShowRuleModal(false)} title={'"添加防火墙规则"'} size="md" footer={
        <div className="flex gap-2">
          <Button variant="secondary" onClick={() => setShowRuleModal(false)}>取消</Button>
          <Button onClick={handleCreateRule}>添加</Button>
        </div>
      }>
        <div className="space-y-4">
          <Input label={'"规则名称"'} placeholder={'"输入规则名称"'} value={newRule.name} onChange={e => setNewRule({ ...newRule, name: e.target.value })} />
          <div className="grid grid-cols-2 gap-4">
            <Input label={'"来源 IP"'} placeholder={'"* 或 CIDR"'} value={newRule.source_ip} onChange={e => setNewRule({ ...newRule, source_ip: e.target.value })} />
            <Input label={'"目标 IP"'} placeholder={'"* 或 IP 地址"'} value={newRule.destination_ip} onChange={e => setNewRule({ ...newRule, destination_ip: e.target.value })} />
            <Input label={'"端口"'} placeholder={'"80,443 或 *"'} value={newRule.port} onChange={e => setNewRule({ ...newRule, port: e.target.value })} />
            <Select label={'"协议"'} value={newRule.protocol} onChange={e => setNewRule({ ...newRule, protocol: e.target.value })} options={[
              { label: 'TCP', value: 'tcp' }, { label: 'UDP', value: 'udp' }, { label: 'HTTP', value: 'http' }, { label: 'HTTPS', value: 'https' },
            ]} />
            <Select label={'"动作"'} value={newRule.action} onChange={e => setNewRule({ ...newRule, action: e.target.value })} options={[
              { label: '拒绝', value: 'deny' }, { label: '允许', value: 'allow' }, { label: '记录', value: 'log' }, { label: '告警', value: 'alert' },
            ]} />
            <Input label={'"优先级"'} type="number" value={newRule.priority} onChange={e => setNewRule({ ...newRule, priority: Number(e.target.value) })} />
          </div>
        </div>
      </Modal>

      <Modal isOpen={showScanModal} onClose={() => setShowScanModal(false)} title={'"安全扫描"'} size="md" footer={
        <div className="flex gap-2">
          <Button variant="secondary" onClick={() => setShowScanModal(false)}>取消</Button>
          <Button onClick={handleStartScan}>开始扫描</Button>
        </div>
      }>
        <div className="space-y-4">
          <Select label={'"扫描类型"'} value={scanType} onChange={e => setScanType(e.target.value)} options={[
            { label: '漏洞扫描', value: 'vulnerability' },
            { label: '恶意软件扫描', value: 'malware' },
            { label: '渗透测试', value: 'penetration' },
            { label: '合规检查', value: 'compliance' },
            { label: '提示注入检测', value: 'prompt_injection' },
          ]} />
          <Input label={'"扫描目标"'} placeholder={'"输入目标地址或范围"'} value={scanTarget} onChange={e => setScanTarget(e.target.value)} />
        </div>
      </Modal>
    </div>
  );
}
