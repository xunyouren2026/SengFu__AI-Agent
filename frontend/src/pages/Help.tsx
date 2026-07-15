import { useState } from 'react';
import { HelpCircle, Search, Book, MessageSquare, FileText, ChevronRight, ExternalLink, ThumbsUp, ThumbsDown } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Badge, Input, Tabs, TabPanel, EmptyState, Loading } from '../components/common';
import { useHelpDocs, useFAQs } from '../hooks/useSystem';

export default function Help() {
  const [activeTab, setActiveTab] = useState('docs');
  const [searchQuery, setSearchQuery] = useState('');
  const { data: docs } = useHelpDocs();
  const { data: faqs } = useFAQs();

  return (
    <div className="space-y-6">
      <div><h1 className="text-2xl font-bold text-gray-900 dark:text-white">帮助文档</h1><p className="text-sm text-gray-500 mt-1">查看文档、FAQ 和提交反馈</p></div>
      <div className="max-w-2xl mx-auto"><Input placeholder={'"搜索文档和 FAQ..."'} value={searchQuery} onChange={e => setSearchQuery(e.target.value)} leftIcon={<Search size={18} />} size="lg" /></div>
      <Tabs tabs={[{ key: 'docs', label: '文档' }, { key: 'faq', label: '常见问题' }, { key: 'feedback', label: '反馈' }, { key: 'shortcuts', label: '快捷键' }, { key: 'changelog', label: '更新日志' }]} activeKey={activeTab} onChange={setActiveTab} />
      {activeTab === 'docs' && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {(docs?.data || []).map((doc: any) => (
            <Card key={doc.id} hoverable variant="bordered">
              <div className="flex items-center gap-2 mb-2"><Book size={16} className="text-primary-500" /><Badge variant="info">{doc.category}</Badge></div>
              <h3 className="font-medium text-gray-900 dark:text-white mb-1">{doc.title}</h3>
              <p className="text-sm text-gray-500 line-clamp-2">{doc.summary || doc.description}</p>
              <div className="flex items-center gap-2 mt-3 text-xs text-gray-400"><span>v{doc.version}</span><span>·</span><span>{doc.last_updated}</span></div>
            </Card>
          ))}
        </div>
      )}
      {activeTab === 'faq' && (
        <div className="max-w-3xl mx-auto space-y-3">
          {(faqs?.data || []).map((faq: any) => (
            <Card key={faq.id} variant="bordered" padding="sm">
              <div className="flex items-start gap-3">
                <HelpCircle size={18} className="text-primary-500 mt-0.5 shrink-0" />
                <div className="flex-1">
                  <h3 className="font-medium text-gray-900 dark:text-white">{faq.question}</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{faq.answer}</p>
                  <div className="flex items-center gap-3 mt-2 text-xs text-gray-400">
                    <span className="flex items-center gap-1"><ThumbsUp size={12} /> {faq.helpful_count}</span>
                    <span className="flex items-center gap-1"><ThumbsDown size={12} /> {faq.not_helpful_count}</span>
                    <Badge variant="outline" size="sm">{faq.category}</Badge>
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}
      {activeTab === 'feedback' && <Card><CardContent><EmptyState icon={<MessageSquare size={48} />} title={'"提交反馈"'} description={'"帮助我们改进 UFO AGI 框架"'} /></CardContent></Card>}
      {activeTab === 'shortcuts' && <Card><CardContent><EmptyState icon={<FileText size={48} />} title={'"快捷键"'} description={'"查看所有可用的键盘快捷键"'} /></CardContent></Card>}
      {activeTab === 'changelog' && <Card><CardContent><EmptyState icon={<FileText size={48} />} title={'"更新日志"'} description={'"查看版本更新历史"'} /></CardContent></Card>}
    </div>
  );
}
