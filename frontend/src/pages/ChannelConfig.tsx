import { Settings, Zap } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Input, Select, Badge, EmptyState } from '../components/common';

export default function ChannelConfig() {
  return (
    <div className="space-y-6">
      <div><h1 className="text-2xl font-bold text-gray-900 dark:text-white">渠道配置</h1><p className="text-sm text-gray-500 mt-1">配置消息渠道参数</p></div>
      <Card><CardContent><EmptyState icon={<Zap size={48} />} title={'"选择渠道进行配置"'} description={'"在渠道管理页面选择一个渠道来配置其参数"'} /></CardContent></Card>
    </div>
  );
}
