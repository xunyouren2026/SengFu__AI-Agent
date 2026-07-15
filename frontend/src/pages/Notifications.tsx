import { Bell, CheckCheck, Trash2 } from 'lucide-react';
import { Card, CardContent, Button, EmptyState } from '../components/common';

export default function Notifications() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div><h1 className="text-2xl font-bold text-gray-900 dark:text-white">通知中心</h1><p className="text-sm text-gray-500 mt-1">查看系统通知</p></div>
        <Button variant="ghost" leftIcon={<CheckCheck size={16} />}>全部已读</Button>
      </div>
      <Card><CardContent><EmptyState icon={<Bell size={48} />} title={'"暂无通知"'} /></CardContent></Card>
    </div>
  );
}
