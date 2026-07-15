import { useEffect, useState } from 'react';
import { Activity, Cpu, Users, Clock, TrendingUp, Zap, AlertTriangle, CheckCircle } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, StatsCard, Badge, Loading } from '../components/common';
import { useSystemMetrics, useAlerts } from '../hooks/useSystem';
import { useModels } from '../hooks/useModels';
import type { SystemMetrics } from '../types/system';

export default function Dashboard() {
  const { data: metrics, isLoading: metricsLoading } = useSystemMetrics();
  const { data: alerts } = useAlerts({ page_size: 5 });
  const { data: modelsData } = useModels();
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);

  useEffect(() => {
    if (metrics) setSystemMetrics(metrics as unknown as SystemMetrics);
  }, [metrics]);

  if (metricsLoading) return <Loading size="lg" text={'"加载仪表盘..."'} />;

  const stats = [
    { title: '活跃模型', value: modelsData?.length || 0, icon: <Cpu size={24} />, color: 'blue' as const, change: 12 },
    { title: 'CPU 使用率', value: systemMetrics ? `${(systemMetrics as any).cpu_usage_percent?.toFixed(1) || 0}%` : '0%', icon: <Activity size={24} />, color: 'green' as const },
    { title: '内存使用率', value: systemMetrics ? `${(systemMetrics as any).memory_usage_percent?.toFixed(1) || 0}%` : '0%', icon: <Zap size={24} />, color: 'purple' as const },
    { title: '系统运行时间', value: systemMetrics ? `${Math.floor(((systemMetrics as any).uptime_seconds || 0) / 3600)}h` : '0h', icon: <Clock size={24} />, color: 'indigo' as const },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">仪表盘</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">UFO AGI 统一框架 - 系统概览</p>
        </div>
        <Badge variant="success" dot>系统正常</Badge>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat, i) => <StatsCard key={i} {...stat} />)}
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* System Status */}
        <Card className="lg:col-span-2">
          <CardHeader><CardTitle>系统状态</CardTitle></CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              {[
                { label: 'CPU', value: (systemMetrics as any)?.cpu_usage_percent || 0, max: 100, unit: '%' },
                { label: '内存', value: (systemMetrics as any)?.memory_usage_percent || 0, max: 100, unit: '%' },
                { label: '磁盘', value: (systemMetrics as any)?.disk_usage_percent || 0, max: 100, unit: '%' },
                { label: 'GPU', value: (systemMetrics as any)?.gpu_usage_percent || 0, max: 100, unit: '%' },
              ].map(item => (
                <div key={item.label} className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500 dark:text-gray-400">{item.label}</span>
                    <span className="font-medium text-gray-900 dark:text-white">{item.value.toFixed(1)}{item.unit}</span>
                  </div>
                  <div className="h-2 bg-gray-100 dark:bg-dark-600 rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-primary-500 to-primary-400 rounded-full transition-all duration-500" style={{ width: `${Math.min(item.value, item.max)}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Recent Alerts */}
        <Card>
          <CardHeader><CardTitle>最近告警</CardTitle></CardHeader>
          <CardContent>
            <div className="space-y-3">
              {(alerts?.data || []).length === 0 ? (
                <div className="text-center py-8">
                  <CheckCircle size={32} className="mx-auto text-green-400 mb-2" />
                  <p className="text-sm text-gray-500">暂无告警</p>
                </div>
              ) : (
                (alerts?.data || []).slice(0, 5).map((alert: any) => (
                  <div key={alert.id} className="flex items-start gap-3 p-2 rounded-lg hover:bg-gray-50 dark:hover:bg-dark-700">
                    <AlertTriangle size={16} className={alert.severity === 'critical' ? 'text-red-500' : 'text-yellow-500 mt-0.5 shrink-0'} />
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-gray-900 dark:text-white truncate">{alert.name}</p>
                      <p className="text-xs text-gray-500 truncate">{alert.description}</p>
                    </div>
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions */}
      <Card>
        <CardHeader><CardTitle>快速操作</CardTitle></CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
            {[
              { label: '新建对话', icon: <Users size={20} />, path: '/chat', color: 'bg-blue-500' },
              { label: '管理模型', icon: <Cpu size={20} />, path: '/model-manager', color: 'bg-purple-500' },
              { label: '创建工作流', icon: <TrendingUp size={20} />, path: '/workflow', color: 'bg-green-500' },
              { label: '开始训练', icon: <Zap size={20} />, path: '/training', color: 'bg-yellow-500' },
              { label: '生成图像', icon: <Activity size={20} />, path: '/image-gen', color: 'bg-pink-500' },
              { label: '系统设置', icon: <Clock size={20} />, path: '/settings', color: 'bg-gray-500' },
            ].map(action => (
              <a key={action.label} href={action.path} className="flex flex-col items-center gap-2 p-4 rounded-xl hover:bg-gray-50 dark:hover:bg-dark-700 transition-colors group">
                <div className={`w-10 h-10 rounded-lg ${action.color} flex items-center justify-center text-white group-hover:scale-110 transition-transform`}>{action.icon}</div>
                <span className="text-xs text-gray-600 dark:text-gray-400">{action.label}</span>
              </a>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
