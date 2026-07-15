import { useState, useCallback } from 'react';
import { Monitor, MousePointer, Keyboard, ArrowDown, ArrowUp, Globe, Camera, Circle, Square, Download, RefreshCw, History, Settings } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Badge, Input, Select, Tabs, TabPanel, EmptyState, Loading, StatsCard } from '../components/common';
import * as advancedApi from '../api/advanced';

export default function ComputerUse() {
  const [activeTab, setActiveTab] = useState('control');
  const [screenshot, setScreenshot] = useState<string | null>(null);
  const [url, setUrl] = useState('');
  const [clickX, setClickX] = useState('');
  const [clickY, setClickY] = useState('');
  const [typeText, setTypeText] = useState('');
  const [scrollAmount, setScrollAmount] = useState('300');
  const [scrollDirection, setScrollDirection] = useState('down');
  const [isRecording, setIsRecording] = useState(false);
  const [loading, setLoading] = useState(false);
  const [actionLog, setActionLog] = useState<{ action: string; time: string; status: string }[]>([]);

  const addLog = useCallback((action: string, status: string) => {
    setActionLog(prev => [{ action, time: new Date().toLocaleTimeString(), status }, ...prev].slice(0, 50));
  }, []);

  const handleScreenshot = async () => {
    setLoading(true);
    try {
      const res = await advancedApi.advancedApi.screenshot();
      setScreenshot((res as any)?.image_url || (res as any)?.image_data || null);
      addLog('截图', '成功');
    } catch { addLog('截图', '失败'); }
    setLoading(false);
  };

  const handleNavigate = async () => {
    if (!url.trim()) return;
    setLoading(true);
    try {
      await advancedApi.advancedApi.navigate({ url });
      addLog(`导航到 ${url}`, '成功');
      await handleScreenshot();
    } catch { addLog(`导航到 ${url}`, '失败'); }
    setLoading(false);
  };

  const handleClick = async () => {
    if (!clickX || !clickY) return;
    setLoading(true);
    try {
      await advancedApi.advancedApi.click({ x: Number(clickX), y: Number(clickY) });
      addLog(`点击 (${clickX}, ${clickY})`, '成功');
      await handleScreenshot();
    } catch { addLog(`点击 (${clickX}, ${clickY})`, '失败'); }
    setLoading(false);
  };

  const handleType = async () => {
    if (!typeText.trim()) return;
    setLoading(true);
    try {
      await advancedApi.advancedApi.type({ text: typeText });
      addLog(`输入文本`, '成功');
      await handleScreenshot();
    } catch { addLog(`输入文本`, '失败'); }
    setLoading(false);
  };

  const handleScroll = async () => {
    setLoading(true);
    try {
      await advancedApi.advancedApi.scroll({ direction: scrollDirection, amount: Number(scrollAmount) });
      addLog(`滚动 ${scrollDirection} ${scrollAmount}px`, '成功');
      await handleScreenshot();
    } catch { addLog(`滚动`, '失败'); }
    setLoading(false);
  };

  const handleRecording = async () => {
    if (!isRecording) {
      try {
        await advancedApi.advancedApi.startRecording({ fps: 30, resolution: '1920x1080', format: 'mp4' });
        setIsRecording(true);
        addLog('开始录制', '成功');
      } catch { addLog('开始录制', '失败'); }
    } else {
      try {
        await advancedApi.advancedApi.stopRecording('current');
        setIsRecording(false);
        addLog('停止录制', '成功');
      } catch { addLog('停止录制', '失败'); }
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">计算机控制</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">AI 驱动的计算机自动化操作</p>
        </div>
        <div className="flex items-center gap-2">
          {isRecording && <Badge variant="danger" dot>录制中</Badge>}
          <Button leftIcon={<Camera size={16} />} onClick={handleScreenshot} isLoading={loading}>截图</Button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatsCard title={'"操作次数"'} value={actionLog.length} icon={<MousePointer size={24} />} color="blue" />
        <StatsCard title={'"成功操作"'} value={actionLog.filter(l => l.status === '成功').length} icon={<Circle size={24} />} color="green" />
        <StatsCard title={'"失败操作"'} value={actionLog.filter(l => l.status === '失败').length} icon={<Square size={24} />} color="red" />
        <StatsCard title={'"录制状态"'} value={isRecording ? '录制中' : '未录制'} icon={<Camera size={24} />} color={isRecording ? 'red' : 'purple'} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2">
          <CardHeader><CardTitle>控制面板</CardTitle></CardHeader>
          <CardContent>
            <Tabs
              tabs={[
                { key: 'control', label: '操作控制' },
                { key: 'navigate', label: '页面导航' },
                { key: 'recording', label: '录制' },
              ]}
              activeKey={activeTab}
              onChange={setActiveTab}
            />

            {activeTab === 'control' && (
              <TabPanel>
                <div className="space-y-4">
                  {screenshot ? (
                    <div className="relative border rounded-lg overflow-hidden">
                      <img src={screenshot} alt={'"屏幕截图"'} className="w-full" />
                    </div>
                  ) : (
                    <div className="border-2 border-dashed border-gray-300 dark:border-dark-600 rounded-xl p-12 text-center">
                      <Monitor size={48} className="mx-auto text-gray-300 dark:text-gray-600 mb-3" />
                      <p className="text-sm text-gray-500 dark:text-gray-400">点击「截图」按钮获取当前屏幕</p>
                    </div>
                  )}

                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <p className="text-sm font-medium text-gray-700 dark:text-gray-300">点击操作</p>
                      <div className="flex gap-2">
                        <Input placeholder="X" type="number" value={clickX} onChange={e => setClickX(e.target.value)} />
                        <Input placeholder="Y" type="number" value={clickY} onChange={e => setClickY(e.target.value)} />
                        <Button leftIcon={<MousePointer size={14} />} onClick={handleClick} isLoading={loading}>点击</Button>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <p className="text-sm font-medium text-gray-700 dark:text-gray-300">滚动操作</p>
                      <div className="flex gap-2">
                        <Select value={scrollDirection} onChange={e => setScrollDirection(e.target.value)} options={[
                          { label: '向下', value: 'down' },
                          { label: '向上', value: 'up' },
                        ]} />
                        <Input placeholder={'"像素"'} type="number" value={scrollAmount} onChange={e => setScrollAmount(e.target.value)} />
                        <Button leftIcon={<ArrowDown size={14} />} onClick={handleScroll} isLoading={loading}>滚动</Button>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <p className="text-sm font-medium text-gray-700 dark:text-gray-300">文本输入</p>
                    <div className="flex gap-2">
                      <Input placeholder={'"输入文本内容..."'} value={typeText} onChange={e => setTypeText(e.target.value)} className="flex-1" />
                      <Button leftIcon={<Keyboard size={14} />} onClick={handleType} isLoading={loading}>输入</Button>
                    </div>
                  </div>
                </div>
              </TabPanel>
            )}

            {activeTab === 'navigate' && (
              <TabPanel>
                <div className="space-y-4">
                  <div className="flex gap-2">
                    <Input placeholder={'"输入 URL 地址..."'} value={url} onChange={e => setUrl(e.target.value)} leftIcon={<Globe size={16} />} className="flex-1" />
                    <Button leftIcon={<Globe size={16} />} onClick={handleNavigate} isLoading={loading}>导航</Button>
                  </div>
                  <div className="flex gap-2">
                    <Input placeholder={'"CSS 选择器..."'} hint={'"可选：指定元素选择器"'} />
                    <Button variant="secondary">等待加载</Button>
                  </div>
                </div>
              </TabPanel>
            )}

            {activeTab === 'recording' && (
              <TabPanel>
                <div className="space-y-4">
                  <div className="flex items-center justify-center p-8">
                    <Button
                      variant={isRecording ? 'danger' : 'primary'}
                      size="lg"
                      leftIcon={isRecording ? <Square size={20} /> : <Circle size={20} />}
                      onClick={handleRecording}
                    >
                      {isRecording ? '停止录制' : '开始录制'}
                    </Button>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <Select label={'"帧率"'} options={[
                      { label: '30 FPS', value: '30' },
                      { label: '60 FPS', value: '60' },
                    ]} />
                    <Select label={'"分辨率"'} options={[
                      { label: '1920x1080', value: '1920x1080' },
                      { label: '1280x720', value: '1280x720' },
                      { label: '3840x2160', value: '3840x2160' },
                    ]} />
                  </div>
                </div>
              </TabPanel>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>操作日志</CardTitle></CardHeader>
          <CardContent>
            {actionLog.length === 0 ? (
              <EmptyState icon={<History size={32} />} title={'"暂无操作记录"'} description={'"开始控制计算机以查看操作日志"'} />
            ) : (
              <div className="space-y-2 max-h-[500px] overflow-y-auto">
                {actionLog.map((log, i) => (
                  <div key={i} className="flex items-center gap-2 p-2 rounded-lg hover:bg-gray-50 dark:hover:bg-dark-700 text-sm">
                    <Badge variant={log.status === '成功' ? 'success' : 'danger'} size="sm">{log.status}</Badge>
                    <span className="text-gray-700 dark:text-gray-300 flex-1 truncate">{log.action}</span>
                    <span className="text-xs text-gray-400">{log.time}</span>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
