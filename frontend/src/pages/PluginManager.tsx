import { Plug, Power, Trash2, Settings } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Badge, EmptyState, Table } from '../components/common';

export default function PluginManager() {
  return (
    <div className="space-y-6">
      <div><h1 className="text-2xl font-bold text-gray-900 dark:text-white">插件管理</h1><p className="text-sm text-gray-500 mt-1">管理已安装的插件</p></div>
      <Card><CardContent><EmptyState icon={<Plug size={48} />} title={'"暂无已安装插件"'} description={'"前往插件市场浏览和安装插件"'} action={<Button>前往插件市场</Button>} /></CardContent></Card>
    </div>
  );
}
