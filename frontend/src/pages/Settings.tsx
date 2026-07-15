import { useState } from 'react';
import { Settings as SettingsIcon, Users, Key, Shield, Database, Palette, Bell, Globe, Save, Download, Upload, RefreshCw } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Input, Select, Badge, Modal, Tabs, TabPanel, Loading, Table } from '../components/common';
import { useSettings, useUpdateSettings, useUsers, useCreateUser, useDeleteUser, useBackups, useCreateBackup } from '../hooks/useSystem';

export default function Settings() {
  const [activeTab, setActiveTab] = useState('general');
  const { data: settings, isLoading } = useSettings();
  const updateSettings = useUpdateSettings();
  const { data: users } = useUsers();
  const { data: backups } = useBackups();
  const createUser = useCreateUser();
  const deleteUser = useDeleteUser();
  const createBackup = useCreateBackup();

  const settingGroups = [
    { key: 'general', label: '通用设置', icon: <SettingsIcon size={16} /> },
    { key: 'appearance', label: '外观', icon: <Palette size={16} /> },
    { key: 'notifications', label: '通知', icon: <Bell size={16} /> },
    { key: 'security', label: '安全', icon: <Shield size={16} /> },
    { key: 'api-keys', label: 'API 密钥', icon: <Key size={16} /> },
    { key: 'data', label: '数据管理', icon: <Database size={16} /> },
    { key: 'backup', label: '备份恢复', icon: <Download size={16} /> },
    { key: 'users', label: '用户管理', icon: <Users size={16} /> },
  ];

  return (
    <div className="space-y-6">
      <div><h1 className="text-2xl font-bold text-gray-900 dark:text-white">系统设置</h1><p className="text-sm text-gray-500 mt-1">管理系统配置、用户和备份</p></div>
      <div className="flex gap-6">
        <Card className="w-56 shrink-0 self-start" padding="sm">
          <nav className="space-y-1">{settingGroups.map(group => (
            <button key={group.key} onClick={() => setActiveTab(group.key)} className={`flex items-center gap-2 w-full px-3 py-2 rounded-lg text-sm transition-colors ${activeTab === group.key ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-600 font-medium' : 'text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-dark-700'}`}>
              {group.icon}{group.label}
            </button>
          ))}</nav>
        </Card>
        <div className="flex-1">
          {activeTab === 'general' && <Card><CardHeader><CardTitle>通用设置</CardTitle></CardHeader><CardContent>
            <div className="space-y-4">
              <Input label={'"系统名称"'} defaultValue={'"UFO AGI 统一框架"'} />
              <Select label={'"语言"'} options={[{ label: '中文', value: 'zh' }, { label: 'English', value: 'en' }]} defaultValue="zh" />
              <Select label={'"时区"'} options={[{ label: 'Asia/Shanghai (UTC+8)', value: 'Asia/Shanghai' }, { label: 'America/New_York (UTC-5)', value: 'America/New_York' }]} defaultValue="Asia/Shanghai" />
              <div className="flex justify-end pt-4"><Button leftIcon={<Save size={16} />}>保存设置</Button></div>
            </div>
          </CardContent></Card>}
          {activeTab === 'appearance' && <Card><CardHeader><CardTitle>外观设置</CardTitle></CardHeader><CardContent>
            <div className="space-y-4">
              <Select label={'"主题"'} options={[{ label: '浅色', value: 'light' }, { label: '深色', value: 'dark' }, { label: '跟随系统', value: 'system' }]} />
              <Select label={'"字体大小"'} options={[{ label: '小', value: 'sm' }, { label: '中', value: 'md' }, { label: '大', value: 'lg' }]} defaultValue="md" />
              <Select label={'"侧边栏位置"'} options={[{ label: '左侧', value: 'left' }, { label: '右侧', value: 'right' }]} defaultValue="left" />
            </div>
          </CardContent></Card>}
          {activeTab === 'users' && <Card><CardHeader><CardTitle>用户管理</CardTitle></CardHeader><CardContent>
            <Table columns={[
              { key: 'username', title: '用户名', render: (_, row) => <span className="font-medium">{row.username}</span> },
              { key: 'email', title: '邮箱' },
              { key: 'status', title: '状态', render: (v) => <Badge variant={v === 'active' ? 'success' : 'warning'}>{v}</Badge> },
              { key: 'roles', title: '角色', render: (v) => (v as string[]).map(r => <Badge key={r} variant="info" size="sm">{r}</Badge>) },
              { key: 'actions', title: '操作', width: '100px', render: (_, row) => <Button variant="ghost" size="sm" onClick={() => deleteUser.mutate(row.id)} className="text-red-500">删除</Button> },
            ]} data={(users?.data || []) as any[]} />
          </CardContent></Card>}
          {activeTab === 'backup' && <Card><CardHeader><CardTitle>备份恢复</CardTitle></CardHeader><CardContent>
            <div className="space-y-4">
              <Button leftIcon={<Download size={16} />} onClick={() => createBackup.mutate({})}>创建备份</Button>
              {(backups?.data || []).length === 0 ? <p className="text-center text-gray-400 py-8">暂无备份</p> : (backups?.data || []).map((b: any) => (
                <div key={b.id} className="flex items-center justify-between p-3 rounded-lg border border-gray-200 dark:border-dark-600">
                  <div><p className="font-medium">{b.name}</p><p className="text-xs text-gray-500">{b.size_formatted} · {b.created_at}</p></div>
                  <div className="flex gap-2"><Button variant="ghost" size="sm">恢复</Button><Button variant="ghost" size="sm">下载</Button></div>
                </div>
              ))}
            </div>
          </CardContent></Card>}
          {activeTab !== 'general' && activeTab !== 'users' && activeTab !== 'backup' && activeTab !== 'appearance' && (
            <Card><CardContent><p className="text-center text-gray-400 py-8">设置项开发中...</p></CardContent></Card>
          )}
        </div>
      </div>
    </div>
  );
}
