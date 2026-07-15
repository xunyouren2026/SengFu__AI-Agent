import { useState, useRef } from 'react';
import { Search, Upload, FileText, Database, Plus, Trash2, Download, RefreshCw, BookOpen, Tag, Clock, ChevronRight } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Badge, Input, Select, Textarea, Tabs, TabPanel, EmptyState, Loading, StatsCard, Modal, Table } from '../components/common';
import { useDocuments, useRAGSearch, useKnowledgeBases } from '../hooks/useAdvanced';

export default function RAG() {
  const [activeTab, setActiveTab] = useState('search');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [selectedKB, setSelectedKB] = useState('');
  const [searchStrategy, setSearchStrategy] = useState('hybrid');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const { data: docsData, isLoading: docsLoading } = useDocuments();
  const { data: knowledgeBases } = useKnowledgeBases();
  const ragSearch = useRAGSearch();

  const documents = (docsData?.data || []) as any[];
  const kbList = (knowledgeBases || []) as any[];

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    setSearchLoading(true);
    try {
      const res = await ragSearch.mutateAsync({
        query: searchQuery,
        strategy: searchStrategy as any,
        top_k: 10,
      });
      setSearchResults((res as any)?.results || []);
    } catch {
      setSearchResults([]);
    }
    setSearchLoading(false);
  };

  const handleUpload = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // Would call uploadDocument API
      console.log('Uploading file:', file.name);
    }
  };

  const statusLabel: Record<string, string> = { processing: '处理中', completed: '已完成', failed: '失败', pending: '等待中' };
  const statusVariant: Record<string, any> = { processing: 'warning', completed: 'success', failed: 'danger', pending: 'info' };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">知识检索</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">RAG 知识库管理与智能检索</p>
        </div>
        <Button leftIcon={<Upload size={16} />} onClick={handleUpload}>上传文档</Button>
        <input ref={fileInputRef} type="file" className="hidden" onChange={handleFileChange} accept=".pdf,.doc,.docx,.txt,.md,.html,.json" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatsCard title={'"文档总数"'} value={documents.length} icon={<FileText size={24} />} color="blue" />
        <StatsCard title={'"知识库"'} value={kbList.length} icon={<Database size={24} />} color="green" />
        <StatsCard title={'"总块数"'} value={kbList.reduce((sum: number, kb: any) => sum + (kb.total_chunks || 0), 0)} icon={<BookOpen size={24} />} color="purple" />
        <StatsCard title={'"搜索次数"'} value="1,234" icon={<Search size={24} />} color="indigo" />
      </div>

      <Card>
        <CardContent className="pt-5">
          <Tabs
            tabs={[
              { key: 'search', label: '智能检索', icon: <Search size={16} /> },
              { key: 'documents', label: '文档管理', icon: <FileText size={16} /> },
              { key: 'knowledge_bases', label: '知识库', icon: <Database size={16} /> },
            ]}
            activeKey={activeTab}
            onChange={setActiveTab}
          />

          {activeTab === 'search' && (
            <TabPanel>
              <div className="space-y-4">
                <div className="flex gap-2">
                  <Input
                    placeholder={'"输入搜索查询..."'}
                    value={searchQuery}
                    onChange={e => setSearchQuery(e.target.value)}
                    leftIcon={<Search size={16} />}
                    className="flex-1"
                    onKeyDown={e => e.key === 'Enter' && handleSearch()}
                  />
                  <Select value={searchStrategy} onChange={e => setSearchStrategy(e.target.value)} options={[
                    { label: '混合搜索', value: 'hybrid' },
                    { label: '语义搜索', value: 'semantic' },
                    { label: '关键词搜索', value: 'keyword' },
                    { label: '向量搜索', value: 'vector' },
                  ]} />
                  <Button leftIcon={<Search size={16} />} onClick={handleSearch} isLoading={searchLoading}>搜索</Button>
                </div>

                {searchLoading && <Loading />}
                {searchResults.length > 0 && (
                  <div className="space-y-3">
                    <p className="text-sm text-gray-500 dark:text-gray-400">找到 {searchResults.length} 条结果</p>
                    {searchResults.map((result: any, i: number) => (
                      <div key={i} className="p-4 border border-gray-200 dark:border-dark-600 rounded-lg hover:bg-gray-50 dark:hover:bg-dark-700 transition-colors">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-gray-900 dark:text-white">{result.document_name}</span>
                          <Badge variant="info">相似度: {(result.score * 100).toFixed(1)}%</Badge>
                        </div>
                        <p className="text-sm text-gray-600 dark:text-gray-300 line-clamp-3">{result.content}</p>
                        <div className="flex items-center gap-2 mt-2">
                          <Badge variant="outline" size="sm">块 #{result.chunk_index}</Badge>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
                {!searchLoading && searchResults.length === 0 && searchQuery && (
                  <EmptyState icon={<Search size={32} />} title={'"未找到相关结果"'} description={'"尝试修改搜索关键词或切换搜索策略"'} />
                )}
              </div>
            </TabPanel>
          )}

          {activeTab === 'documents' && (
            <TabPanel>
              <div className="flex justify-end mb-4">
                <Button leftIcon={<Upload size={16} />} onClick={handleUpload}>上传文档</Button>
              </div>
              {docsLoading ? (
                <Loading />
              ) : documents.length === 0 ? (
                <EmptyState icon={<FileText size={32} />} title={'"暂无文档"'} description={'"上传文档以构建知识库"'} />
              ) : (
                <Table
                  columns={[
                    { key: 'filename', title: '文件名', render: (val: any) => (
                      <div className="flex items-center gap-2">
                        <FileText size={16} className="text-gray-400" />
                        <span className="font-medium text-gray-900 dark:text-white">{val}</span>
                      </div>
                    )},
                    { key: 'type', title: '类型', render: (val: any) => <Badge variant="outline">{val?.toUpperCase()}</Badge> },
                    { key: 'size', title: '大小', render: (val: any) => val ? `${(val / 1024).toFixed(1)} KB` : '-' },
                    { key: 'chunks_count', title: '分块数' },
                    { key: 'status', title: '状态', render: (val: any) => <Badge variant={statusVariant[val]}>{statusLabel[val] || val}</Badge> },
                    { key: 'uploaded_at', title: '上传时间' },
                    { key: 'actions', title: '操作', render: () => (
                      <div className="flex gap-1">
                        <Button variant="ghost" size="sm">详情</Button>
                        <Button variant="ghost" size="sm" className="text-red-500" leftIcon={<Trash2 size={14} />}>删除</Button>
                      </div>
                    )},
                  ]}
                  data={documents}
                />
              )}
            </TabPanel>
          )}

          {activeTab === 'knowledge_bases' && (
            <TabPanel>
              <div className="flex justify-end mb-4">
                <Button leftIcon={<Plus size={16} />}>创建知识库</Button>
              </div>
              {kbList.length === 0 ? (
                <EmptyState icon={<Database size={32} />} title={'"暂无知识库"'} description={'"创建知识库以组织和管理文档"'} />
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {kbList.map((kb: any) => (
                    <Card key={kb.id} hoverable>
                      <CardContent>
                        <div className="flex items-start justify-between mb-3">
                          <div className="flex items-center gap-2">
                            <div className="w-10 h-10 rounded-lg bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center">
                              <Database size={20} className="text-primary-600 dark:text-primary-400" />
                            </div>
                            <div>
                              <p className="font-medium text-gray-900 dark:text-white">{kb.name}</p>
                              <p className="text-xs text-gray-500">{kb.embedding_model}</p>
                            </div>
                          </div>
                        </div>
                        {kb.description && <p className="text-sm text-gray-500 dark:text-gray-400 mb-3">{kb.description}</p>}
                        <div className="flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
                          <span>{kb.document_count} 文档</span>
                          <span>{kb.total_chunks} 块</span>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </TabPanel>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
