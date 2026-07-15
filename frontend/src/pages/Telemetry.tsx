import { useState } from 'react';
import { Activity, AlertTriangle, TrendingUp, FileText, Search, Download } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Badge, Input, Select, Tabs, TabPanel, EmptyState, Loading, Table, StatsCard } from '../components/common';
import { useSystemMetrics, useAlerts, useHardwareInfo } from '../hooks/useSystem';

export default function Telemetry() {
  const [activeTab, setActiveTab] = useState('metrics');
  const { data: metrics } = useSystemMetrics();
  const { data: alerts } = useAlerts({ page_size: 20 });
  const { data: hardware } = useHardwareInfo();

  return (
    <div className="space-y-6">
      <div><h1 className="text-2xl font-bold text-gray-900 dark:text-white">监控遥测</h1><p className="text-sm text-gray-500 mt-1">实时系统监控和遥测数据</p></div>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatsCard title={'"CPU 使用率"'} value={`${(metrics as any)?.cpu_usage_percent?.toFixed(1) || 0}%`} icon={<Activity size={24} />} color="blue" />
        <StatsCard title={'"内存使用率"'} value={`${(metrics as any)?.memory_usage_percent?.toFixed(1) || 0}%`} icon={<TrendingUp size={24} />} color="green" />
        <StatsCard title={'"活跃告警"'} value={alerts?.data?.filter((a: any) => a.status === 'active').length || 0} icon={<AlertTriangle size={24} />} color="red" />
        <StatsCard title={'"活跃连接"'} value={(metrics as any)?.active_connections || 0} icon={<Search size={24} />} color="purple" />
      </div>
      <Tabs tabs={[{ key: 'metrics', label: '系统指标' }, { key: 'alerts', label: '告警管理' }, { key: 'logs', label: '日志查看' }, { key: 'traces', label: '分布式追踪' }]} activeKey={activeTab} onChange={setActiveTab} />
      {activeTab === 'metrics' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card><CardHeader><CardTitle>资源使用率</CardTitle></CardHeader><CardContent>
            <div className="space-y-4">
              {[
                { label: 'CPU', value: (metrics as any)?.cpu_usage_percent || 0 },
                { label: '内存', value: (metrics as any)?.memory_usage_percent || 0 },
                { label: '磁盘', value: (metrics as any)?.disk_usage_percent || 0 },
                { label: 'GPU', value: (metrics as any)?.gpu_usage_percent || 0 },
              ].map(item => (
                <div key={item.label} className="space-y-1">
                  <div className="flex justify-between text-sm"><span className="text-gray-600 dark:text-gray-400">{item.label}</span><span className="font-medium">{item.value.toFixed(1)}%</span></div>
                  <div className="h-3 bg-gray-100 dark:bg-dark-600 rounded-full overflow-hidden"><div className="h-full bg-gradient-to-r from-primary-500 to-primary-400 rounded-full transition-all" style={{ width: `${item.value}%` }} /></div>
                </div>
              ))}
            </div>
          </CardContent></Card>
          <Card><CardHeader><CardTitle>网络流量</CardTitle></CardHeader><CardContent>
            <div className="space-y-4 text-center py-8">
              <p className="text-3xl font-bold text-gray-900 dark:text-white">{((metrics as any)?.network_in_bytes / 1024 / 1024 || 0).toFixed(1)} MB</p>
              <p className="text-sm text-gray-500">入站流量</p>
              <p className="text-3xl font-bold text-gray-900 dark:text-white mt-4">{((metrics as any)?.network_out_bytes / 1024 / 1024 || 0).toFixed(1)} MB</p>
              <p className="text-sm text-gray-500">出站流量</p>
            </div>
          </CardContent></Card>
        </div>
      )}
      {activeTab === 'alerts' && <Card padding="none"><Table columns={[{ key: 'name', title: '告警名称' }, { key: 'severity', title: '严重程度', render: (v) => <Badge variant={v === 'critical' ? 'danger' : v === 'high' ? 'warning' : 'info'}>{v}</Badge> }, { key: 'status', title: '状态', render: (v) => <Badge variant={v === 'active' ? 'danger' : 'success'}>{v}</Badge> }, { key: 'created_at', title: '时间' }]} data={(alerts?.data || []) as any[]} /></Card>}
      {activeTab === 'logs' && <Card><CardContent><EmptyState icon={<FileText size={48} />} title={'"日志查看"'} description={'"查看系统日志和审计记录"'} action={<Button leftIcon={<Download size={16} />}>导出日志</Button>} /></CardContent></Card>}
      {activeTab === 'traces' && <Card><CardContent><EmptyState icon={<Activity size={48} />} title={'"分布式追踪"'} description={'"查看请求链路和性能分析"'} /></CardContent></Card>}
    </div>
  );
}
