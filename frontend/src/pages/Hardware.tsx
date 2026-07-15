import { HardDrive, Cpu, MemoryStick, Wifi, Monitor, Thermometer } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Badge, StatsCard, Loading } from '../components/common';
import { useHardwareInfo } from '../hooks/useSystem';

export default function Hardware() {
  const { data: hardware, isLoading } = useHardwareInfo();
  if (isLoading) return <Loading size="lg" text={'"加载硬件信息..."'} />;
  const hw = hardware as any;

  return (
    <div className="space-y-6">
      <div><h1 className="text-2xl font-bold text-gray-900 dark:text-white">硬件管理</h1><p className="text-sm text-gray-500 mt-1">查看和管理硬件资源</p></div>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatsCard title="CPU" value={hw?.cpu_model || 'N/A'} icon={<Cpu size={24} />} color="blue" />
        <StatsCard title={'"内存"'} value={hw?.memory_total_gb || 'N/A'} icon={<MemoryStick size={24} />} color="green" />
        <StatsCard title={'"GPU 数量"'} value={hw?.gpu_count || 0} icon={<Monitor size={24} />} color="purple" />
        <StatsCard title={'"平台"'} value={hw?.platform || 'N/A'} icon={<HardDrive size={24} />} color="indigo" />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card><CardHeader><CardTitle>CPU 信息</CardTitle></CardHeader><CardContent>
          <div className="space-y-3">
            <div className="flex justify-between"><span className="text-gray-500">型号</span><span className="font-medium">{hw?.cpu_model}</span></div>
            <div className="flex justify-between"><span className="text-gray-500">核心数</span><span className="font-medium">{hw?.cpu_count}</span></div>
            <div className="flex justify-between"><span className="text-gray-500">Python</span><span className="font-medium">{hw?.python_version}</span></div>
          </div>
        </CardContent></Card>
        <Card><CardHeader><CardTitle>GPU 信息</CardTitle></CardHeader><CardContent>
          {hw?.gpus?.length === 0 ? <p className="text-center text-gray-400 py-4">未检测到 GPU</p> : hw?.gpus?.map((gpu: any, i: number) => (
            <div key={i} className="p-3 rounded-lg border border-gray-200 dark:border-dark-600 mb-3 last:mb-0">
              <div className="flex items-center justify-between mb-2"><span className="font-medium">{gpu.name}</span><Badge variant={gpu.utilization_percent > 80 ? 'danger' : 'success'}>{gpu.utilization_percent?.toFixed(0)}%</Badge></div>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div><span className="text-gray-500">显存:</span> {gpu.memory_used_gb?.toFixed(1)}/{gpu.memory_total_gb?.toFixed(1)} GB</div>
                <div><span className="text-gray-500">温度:</span> {gpu.temperature_celsius}°C</div>
                <div><span className="text-gray-500">功耗:</span> {gpu.power_draw_watts?.toFixed(0)}/{gpu.power_limit_watts}W</div>
                <div><span className="text-gray-500">驱动:</span> {gpu.driver_version}</div>
              </div>
            </div>
          ))}
        </CardContent></Card>
      </div>
    </div>
  );
}
