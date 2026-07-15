import { User, MessageSquare, Hash, Calendar } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Input, StatsCard } from '../components/common';

export default function Profile() {
  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <div><h1 className="text-2xl font-bold text-gray-900 dark:text-white">个人资料</h1><p className="text-sm text-gray-500 mt-1">管理你的个人信息和账户设置</p></div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatsCard title={'"对话次数"'} value={0} icon={<MessageSquare size={24} />} color="blue" />
        <StatsCard title={'"消息数量"'} value={0} icon={<Hash size={24} />} color="green" />
        <StatsCard title={'"使用天数"'} value={1} icon={<Calendar size={24} />} color="purple" />
      </div>
      <Card>
        <CardHeader><CardTitle>基本信息</CardTitle></CardHeader>
        <CardContent>
          <div className="flex items-center gap-6 mb-6">
            <div className="w-20 h-20 rounded-full bg-gradient-to-br from-primary-400 to-primary-600 flex items-center justify-center text-white text-3xl font-bold">U</div>
            <div><h2 className="text-xl font-bold text-gray-900 dark:text-white">User</h2><p className="text-sm text-gray-500">user@ufo.ai</p><Button variant="outline" size="sm" className="mt-2">更换头像</Button></div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <Input label={'"用户名"'} defaultValue="user" />
            <Input label={'"邮箱"'} defaultValue="user@ufo.ai" />
            <Input label={'"全名"'} placeholder={'"输入全名"'} />
            <Input label={'"手机号"'} placeholder={'"输入手机号"'} />
          </div>
          <div className="flex justify-end mt-4"><Button>保存修改</Button></div>
        </CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle>修改密码</CardTitle></CardHeader>
        <CardContent>
          <div className="space-y-4 max-w-md">
            <Input label={'"当前密码"'} type="password" />
            <Input label={'"新密码"'} type="password" />
            <Input label={'"确认新密码"'} type="password" />
            <Button>修改密码</Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
