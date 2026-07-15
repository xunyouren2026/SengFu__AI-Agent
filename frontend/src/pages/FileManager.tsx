import { Upload, Folder, File, Trash2, Download, Search, Grid3X3, List } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Input, Badge, EmptyState } from '../components/common';

export default function FileManager() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div><h1 className="text-2xl font-bold text-gray-900 dark:text-white">文件管理</h1><p className="text-sm text-gray-500 mt-1">上传和管理文件</p></div>
        <Button leftIcon={<Upload size={16} />}>上传文件</Button>
      </div>
      <Card>
        <div className="flex items-center gap-3 p-4 border-b border-gray-100 dark:border-dark-600">
          <Input placeholder={'"搜索文件..."'} leftIcon={<Search size={16} />} className="w-64" />
          <Button variant="ghost" size="sm"><Grid3X3 size={16} /></Button>
          <Button variant="ghost" size="sm"><List size={16} /></Button>
        </div>
        <CardContent>
          <EmptyState icon={<Folder size={48} />} title={'"暂无文件"'} description={'"上传文件到文件管理器"'} action={<Button leftIcon={<Upload size={16} />}>上传文件</Button>} />
        </CardContent>
      </Card>
    </div>
  );
}
