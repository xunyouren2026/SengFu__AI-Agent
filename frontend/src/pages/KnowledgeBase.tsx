import { BookOpen, Upload, Search, Trash2, Plus } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Input, Badge, EmptyState, Table } from '../components/common';

export default function KnowledgeBase() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div><h1 className="text-2xl font-bold text-gray-900 dark:text-white">知识库管理</h1><p className="text-sm text-gray-500 mt-1">管理知识库和文档</p></div>
        <Button leftIcon={<Plus size={16} />}>创建知识库</Button>
      </div>
      <Card><CardContent><EmptyState icon={<BookOpen size={48} />} title={'"暂无知识库"'} description={'"创建知识库并上传文档以启用 RAG 检索"'} action={<Button leftIcon={<Plus size={16} />}>创建知识库</Button>} /></CardContent></Card>
    </div>
  );
}
