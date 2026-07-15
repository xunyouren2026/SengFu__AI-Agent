import { useState } from 'react';
import { Puzzle, Download, Upload, Settings, Star, Search, Power, PowerOff, Trash2, ExternalLink, Tag, CheckCircle, Package } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Badge, Input, Select, Tabs, TabPanel, EmptyState, Loading, StatsCard, Modal } from '../components/common';
import { usePlugins, useInstallPlugin, usePluginMarketplace } from '../hooks/useAdvanced';

export default function Plugins() {
  const [activeTab, setActiveTab] = useState('installed');
  const [searchQuery, setSearchQuery] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('');
  const [showInstallModal, setShowInstallModal] = useState(false);
  const [installSource, setInstallSource] = useState('');
  const [installVersion, setInstallVersion] = useState('');

  const { data: pluginsData, isLoading } = usePlugins();
  const { data: marketplaceData, isLoading: marketplaceLoading } = usePluginMarketplace();
  const installPlugin = useInstallPlugin();

  const plugins = (pluginsData?.data || []) as any[];
  const marketplace = (marketplaceData || []) as any[];

  const handleInstall = (source?: string, version?: string) => {
    const src = source || installSource;
    if (!src.trim()) return;
    installPlugin.mutate({ source: src, version: version || installVersion || undefined });
    setShowInstallModal(false);
    setInstallSource('');
    setInstallVersion('');
  };

  const statusLabel: Record<string, string> = { installed: '已安装', enabled: '已启用', disabled: '已禁用', error: '错误', updating: '更新中' };
  const statusVariant: Record<string, any> = { installed: 'info', enabled: 'success', disabled: 'default', error: 'danger', updating: 'warning' };
  const categoryLabel: Record<string, string> = { generation: '生成', integration: '集成', automation: '自动化', analytics: '分析', security: '安全', custom: '自定义' };

  const filteredMarketplace = marketplace.filter((p: any) => {
    const matchSearch = !searchQuery || p.name.toLowerCase().includes(searchQuery.toLowerCase()) || p.description.toLowerCase().includes(searchQuery.toLowerCase());
    const matchCategory = !categoryFilter || p.category === categoryFilter;
    return matchSearch && matchCategory;
  });

  const isInstalled = (name: string) => plugins.some((p: any) => p.name === name);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">插件管理</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">插件市场与已安装插件管理</p>
        </div>
        <Button leftIcon={<Upload size={16} />} onClick={() => setShowInstallModal(true)}>安装插件</Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatsCard title={'"已安装"'} value={plugins.length} icon={<Package size={24} />} color="blue" />
        <StatsCard title={'"已启用"'} value={plugins.filter((p: any) => p.status === 'enabled').length} icon={<CheckCircle size={24} />} color="green" />
        <StatsCard title={'"已禁用"'} value={plugins.filter((p: any) => p.status === 'disabled').length} icon={<PowerOff size={24} />} color="yellow" />
        <StatsCard title={'"市场可用"'} value={marketplace.length} icon={<Puzzle size={24} />} color="purple" />
      </div>

      <Card>
        <CardContent className="pt-5">
          <Tabs
            tabs={[
              { key: 'installed', label: '已安装插件', badge: plugins.length },
              { key: 'marketplace', label: '插件市场', badge: marketplace.length },
            ]}
            activeKey={activeTab}
            onChange={setActiveTab}
          />

          {activeTab === 'installed' && (
            <TabPanel>
              {isLoading ? (
                <Loading />
              ) : plugins.length === 0 ? (
                <EmptyState icon={<Puzzle size={32} />} title={'"暂无已安装插件"'} description={'"从插件市场安装或手动安装插件"'} action={<Button leftIcon={<Upload size={16} />} onClick={() => setShowInstallModal(true)}>安装插件</Button>} />
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {plugins.map((plugin: any) => (
                    <Card key={plugin.id} hoverable>
                      <CardContent>
                        <div className="flex items-start justify-between mb-3">
                          <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-lg bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center">
                              <Puzzle size={20} className="text-primary-600 dark:text-primary-400" />
                            </div>
                            <div>
                              <p className="font-medium text-gray-900 dark:text-white">{plugin.name}</p>
                              <p className="text-xs text-gray-500">v{plugin.version} by {plugin.author}</p>
                            </div>
                          </div>
                          <Badge variant={statusVariant[plugin.status]}>{statusLabel[plugin.status]}</Badge>
                        </div>
                        <p className="text-sm text-gray-500 dark:text-gray-400 mb-3 line-clamp-2">{plugin.description}</p>
                        <div className="flex items-center gap-2 mb-3">
                          <Badge variant="outline" size="sm">{categoryLabel[plugin.category] || plugin.category}</Badge>
                          {plugin.permissions?.length > 0 && (
                            <Badge variant="outline" size="sm">{plugin.permissions.length} 权限</Badge>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          {plugin.status === 'enabled' ? (
                            <Button variant="secondary" size="sm" leftIcon={<PowerOff size={14} />}>禁用</Button>
                          ) : (
                            <Button variant="primary" size="sm" leftIcon={<Power size={14} />}>启用</Button>
                          )}
                          <Button variant="ghost" size="sm" leftIcon={<Settings size={14} />}>配置</Button>
                          <Button variant="ghost" size="sm" className="text-red-500" leftIcon={<Trash2 size={14} />}>卸载</Button>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </TabPanel>
          )}

          {activeTab === 'marketplace' && (
            <TabPanel>
              <div className="flex gap-2 mb-4">
                <Input
                  placeholder={'"搜索插件..."'}
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                  leftIcon={<Search size={16} />}
                  className="flex-1"
                />
                <Select value={categoryFilter} onChange={e => setCategoryFilter(e.target.value)} options={[
                  { label: '全部分类', value: '' },
                  { label: '生成', value: 'generation' },
                  { label: '集成', value: 'integration' },
                  { label: '自动化', value: 'automation' },
                  { label: '分析', value: 'analytics' },
                  { label: '安全', value: 'security' },
                  { label: '自定义', value: 'custom' },
                ]} />
              </div>

              {marketplaceLoading ? (
                <Loading />
              ) : filteredMarketplace.length === 0 ? (
                <EmptyState icon={<Puzzle size={32} />} title={'"未找到匹配的插件"'} description={'"尝试修改搜索条件"'} />
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {filteredMarketplace.map((item: any) => (
                    <Card key={item.id} hoverable>
                      <CardContent>
                        <div className="flex items-start justify-between mb-3">
                          <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-primary-500 to-purple-500 flex items-center justify-center text-white">
                              <Puzzle size={20} />
                            </div>
                            <div>
                              <div className="flex items-center gap-1">
                                <p className="font-medium text-gray-900 dark:text-white">{item.name}</p>
                                {item.is_official && <Badge variant="primary" size="sm">官方</Badge>}
                              </div>
                              <p className="text-xs text-gray-500">v{item.version} by {item.author}</p>
                            </div>
                          </div>
                        </div>
                        <p className="text-sm text-gray-500 dark:text-gray-400 mb-3 line-clamp-2">{item.description}</p>
                        <div className="flex items-center gap-3 mb-3 text-sm text-gray-500 dark:text-gray-400">
                          <span className="flex items-center gap-1"><Star size={14} className="text-yellow-400" /> {item.rating}</span>
                          <span className="flex items-center gap-1"><Download size={14} /> {item.download_count}</span>
                          <Badge variant="outline" size="sm">{categoryLabel[item.category] || item.category}</Badge>
                        </div>
                        {item.tags?.length > 0 && (
                          <div className="flex flex-wrap gap-1 mb-3">
                            {item.tags.slice(0, 3).map((tag: string, i: number) => (
                              <span key={i} className="px-2 py-0.5 text-xs bg-gray-100 dark:bg-dark-700 text-gray-500 dark:text-gray-400 rounded">{tag}</span>
                            ))}
                          </div>
                        )}
                        <div className="flex items-center gap-2">
                          {isInstalled(item.name) ? (
                            <Badge variant="success">已安装</Badge>
                          ) : (
                            <Button size="sm" leftIcon={<Download size={14} />} onClick={() => handleInstall(item.id, item.version)}>安装</Button>
                          )}
                          <Button variant="ghost" size="sm" leftIcon={<ExternalLink size={14} />}>详情</Button>
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

      <Modal isOpen={showInstallModal} onClose={() => setShowInstallModal(false)} title={'"安装插件"'} size="md" footer={
        <div className="flex gap-2">
          <Button variant="secondary" onClick={() => setShowInstallModal(false)}>取消</Button>
          <Button onClick={() => handleInstall()} isLoading={installPlugin.isPending}>安装</Button>
        </div>
      }>
        <div className="space-y-4">
          <Input label={'"插件来源"'} placeholder={'"输入插件名称、Git URL 或本地路径"'} value={installSource} onChange={e => setInstallSource(e.target.value)} />
          <Input label={'"版本"'} placeholder={'"留空使用最新版本"'} value={installVersion} onChange={e => setInstallVersion(e.target.value)} hint={'"例如: 1.0.0, latest"'} />
          <p className="text-sm text-gray-500 dark:text-gray-400">支持从 Git 仓库、本地路径或插件名称安装</p>
        </div>
      </Modal>
    </div>
  );
}
