import { Star, Download, Search } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Input, Badge, EmptyState } from '../components/common';

export default function SkillMarket() {
  return (
    <div className="space-y-6">
      <div><h1 className="text-2xl font-bold text-gray-900 dark:text-white">技能市场</h1><p className="text-sm text-gray-500 mt-1">浏览和安装 AI 技能</p></div>
      <Input placeholder={'"搜索技能..."'} leftIcon={<Search size={16} />} className="max-w-md" />
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <Card hoverable variant="bordered"><div className="flex items-center gap-2 mb-2"><Star size={16} className="text-yellow-500" /><Badge variant="info">推荐</Badge></div><h3 className="font-medium">代码生成技能</h3><p className="text-sm text-gray-500 mt-1">高级代码生成和理解能力</p><div className="flex items-center gap-2 mt-3"><Badge variant="outline">⭐ 4.8</Badge><Badge variant="outline">📥 1.2k</Badge></div></Card>
        <Card hoverable variant="bordered"><div className="flex items-center gap-2 mb-2"><Star size={16} className="text-yellow-500" /><Badge variant="info">热门</Badge></div><h3 className="font-medium">数据分析技能</h3><p className="text-sm text-gray-500 mt-1">数据分析和可视化能力</p><div className="flex items-center gap-2 mt-3"><Badge variant="outline">⭐ 4.6</Badge><Badge variant="outline">📥 890</Badge></div></Card>
        <Card hoverable variant="bordered"><div className="flex items-center gap-2 mb-2"><Star size={16} className="text-yellow-500" /><Badge variant="info">新</Badge></div><h3 className="font-medium">多语言翻译技能</h3><p className="text-sm text-gray-500 mt-1">100+ 语言翻译能力</p><div className="flex items-center gap-2 mt-3"><Badge variant="outline">⭐ 4.9</Badge><Badge variant="outline">📥 2.1k</Badge></div></Card>
      </div>
    </div>
  );
}
