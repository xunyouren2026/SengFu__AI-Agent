import { useState } from 'react';
import { Bot, Play, Square, RotateCcw, MapPin, Battery, Wifi, WifiOff, Activity, Settings, Eye, AlertTriangle, ChevronUp, ChevronDown, ChevronLeft, ChevronRight, Zap } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Badge, Input, Select, Tabs, TabPanel, EmptyState, Loading, StatsCard, Modal } from '../components/common';
import { useRobots } from '../hooks/useAdvanced';

export default function Robot() {
  const [activeTab, setActiveTab] = useState('control');
  const [selectedRobot, setSelectedRobot] = useState<string | null>(null);
  const [speed, setSpeed] = useState(50);
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const [commandLog, setCommandLog] = useState<{ command: string; time: string; status: string }[]>([]);

  const { data: robotsData, isLoading } = useRobots();
  const robots = (robotsData || []) as any[];

  const selectedRobotData = robots.find((r: any) => r.id === selectedRobot);

  const addLog = (command: string, status: string) => {
    setCommandLog(prev => [{ command, time: new Date().toLocaleTimeString(), status }, ...prev].slice(0, 50));
  };

  const handleMove = async (direction: string) => {
    if (!selectedRobot) return;
    addLog(`移动: ${direction}`, '执行中');
    try {
      const movements: Record<string, any> = {
        forward: { movement_type: 'linear', relative_movement: { x: 0, y: 0.1, z: 0 } },
        backward: { movement_type: 'linear', relative_movement: { x: 0, y: -0.1, z: 0 } },
        left: { movement_type: 'linear', relative_movement: { x: -0.1, y: 0, z: 0 } },
        right: { movement_type: 'linear', relative_movement: { x: 0.1, y: 0, z: 0 } },
        up: { movement_type: 'linear', relative_movement: { x: 0, y: 0, z: 0.1 } },
        down: { movement_type: 'linear', relative_movement: { x: 0, y: 0, z: -0.1 } },
      };
      await import('../api/advanced').then(api => api.advancedApi.moveRobot(selectedRobot, movements[direction]));
      addLog(`移动: ${direction}`, '成功');
    } catch {
      addLog(`移动: ${direction}`, '失败');
    }
  };

  const handleStop = async () => {
    if (!selectedRobot) return;
    addLog('紧急停止', '执行中');
    try {
      await import('../api/advanced').then(api => api.advancedApi.stopRobot(selectedRobot));
      addLog('紧急停止', '成功');
    } catch {
      addLog('紧急停止', '失败');
    }
  };

  const statusLabel: Record<string, string> = { idle: '空闲', busy: '忙碌', error: '错误', offline: '离线', emergency: '紧急停止' };
  const statusVariant: Record<string, any> = { idle: 'success', busy: 'primary', error: 'danger', offline: 'default', emergency: 'danger' };
  const typeLabel: Record<string, string> = { arm: '机械臂', mobile: '移动机器人', humanoid: '人形机器人', drone: '无人机', wheelchair: '轮椅' };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">机器人控制</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">实时机器人控制与监控</p>
        </div>
        <Badge variant={robots.some((r: any) => r.status === 'error') ? 'danger' : 'success'} dot>
          {robots.some((r: any) => r.status === 'error') ? '存在异常' : '系统正常'}
        </Badge>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatsCard title={'"机器人总数"'} value={robots.length} icon={<Bot size={24} />} color="blue" />
        <StatsCard title={'"在线"'} value={robots.filter((r: any) => r.status !== 'offline').length} icon={<Wifi size={24} />} color="green" />
        <StatsCard title={'"忙碌"'} value={robots.filter((r: any) => r.status === 'busy').length} icon={<Activity size={24} />} color="purple" />
        <StatsCard title={'"异常"'} value={robots.filter((r: any) => r.status === 'error').length} icon={<AlertTriangle size={24} />} color="red" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2">
          <CardHeader><CardTitle>控制面板</CardTitle></CardHeader>
          <CardContent>
            <Tabs
              tabs={[
                { key: 'control', label: '运动控制', icon: <Play size={16} /> },
                { key: 'status', label: '状态监控', icon: <Activity size={16} /> },
              ]}
              activeKey={activeTab}
              onChange={setActiveTab}
            />

            {activeTab === 'control' && (
              <TabPanel>
                {!selectedRobot ? (
                  <EmptyState icon={<Bot size={32} />} title={'"请选择机器人"'} description={'"从右侧列表选择一个机器人进行控制"'} />
                ) : (
                  <div className="space-y-6">
                    <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-dark-700 rounded-lg">
                      <div className="flex items-center gap-3">
                        <Bot size={24} className="text-primary-500" />
                        <div>
                          <p className="font-medium text-gray-900 dark:text-white">{selectedRobotData?.name}</p>
                          <p className="text-xs text-gray-500">{typeLabel[selectedRobotData?.type] || selectedRobotData?.type}</p>
                        </div>
                      </div>
                      <Badge variant={statusVariant[selectedRobotData?.status]} dot>{statusLabel[selectedRobotData?.status]}</Badge>
                    </div>

                    <div className="flex items-center justify-center">
                      <div className="grid grid-cols-3 gap-2 w-48">
                        <div />
                        <Button variant="secondary" size="lg" leftIcon={<ChevronUp size={20} />} onClick={() => handleMove('forward')} disabled={selectedRobotData?.status === 'offline'}>前进</Button>
                        <div />
                        <Button variant="secondary" size="lg" leftIcon={<ChevronLeft size={20} />} onClick={() => handleMove('left')} disabled={selectedRobotData?.status === 'offline'}>左移</Button>
                        <Button variant="danger" size="lg" leftIcon={<Square size={20} />} onClick={handleStop}>停止</Button>
                        <Button variant="secondary" size="lg" leftIcon={<ChevronRight size={20} />} onClick={() => handleMove('right')} disabled={selectedRobotData?.status === 'offline'}>右移</Button>
                        <div />
                        <Button variant="secondary" size="lg" leftIcon={<ChevronDown size={20} />} onClick={() => handleMove('backward')} disabled={selectedRobotData?.status === 'offline'}>后退</Button>
                        <div />
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">速度控制</p>
                        <input type="range" min="0" max="100" value={speed} onChange={e => setSpeed(Number(e.target.value))} className="w-full" />
                        <p className="text-xs text-gray-500 mt-1">{speed}%</p>
                      </div>
                      <div>
                        <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">快速操作</p>
                        <div className="flex gap-2">
                          <Button variant="secondary" size="sm" leftIcon={<RotateCcw size={14} />}>复位</Button>
                          <Button variant="secondary" size="sm" leftIcon={<Settings size={14} />} onClick={() => setShowSettingsModal(true)}>设置</Button>
                        </div>
                      </div>
                    </div>

                    {selectedRobotData?.battery_level !== undefined && (
                      <div className="p-3 bg-gray-50 dark:bg-dark-700 rounded-lg">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm text-gray-700 dark:text-gray-300">电池电量</span>
                          <span className="text-sm font-medium text-gray-900 dark:text-white">{selectedRobotData.battery_level}%</span>
                        </div>
                        <div className="h-2 bg-gray-200 dark:bg-dark-600 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${selectedRobotData.battery_level > 50 ? 'bg-green-500' : selectedRobotData.battery_level > 20 ? 'bg-yellow-500' : 'bg-red-500'}`}
                            style={{ width: `${selectedRobotData.battery_level}%` }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </TabPanel>
            )}

            {activeTab === 'status' && (
              <TabPanel>
                {!selectedRobot ? (
                  <EmptyState icon={<Activity size={32} />} title={'"请选择机器人"'} description={'"选择机器人以查看详细状态"'} />
                ) : (
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="p-4 bg-gray-50 dark:bg-dark-700 rounded-lg">
                        <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">状态</p>
                        <Badge variant={statusVariant[selectedRobotData?.status]} size="md">{statusLabel[selectedRobotData?.status]}</Badge>
                      </div>
                      <div className="p-4 bg-gray-50 dark:bg-dark-700 rounded-lg">
                        <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">电池</p>
                        <p className="text-lg font-bold text-gray-900 dark:text-white">{selectedRobotData?.battery_level !== undefined ? `${selectedRobotData.battery_level}%` : 'N/A'}</p>
                      </div>
                      <div className="p-4 bg-gray-50 dark:bg-dark-700 rounded-lg">
                        <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">当前位置</p>
                        <p className="text-sm font-mono text-gray-900 dark:text-white">
                          {selectedRobotData?.current_position
                            ? `X: ${selectedRobotData.current_position.x?.toFixed(2)} Y: ${selectedRobotData.current_position.y?.toFixed(2)} Z: ${selectedRobotData.current_position.z?.toFixed(2)}`
                            : '未知'}
                        </p>
                      </div>
                      <div className="p-4 bg-gray-50 dark:bg-dark-700 rounded-lg">
                        <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">能力</p>
                        <div className="flex flex-wrap gap-1">
                          {(selectedRobotData?.capabilities || []).map((cap: string, i: number) => (
                            <Badge key={i} variant="outline" size="sm">{cap}</Badge>
                          ))}
                        </div>
                      </div>
                    </div>
                    <div className="p-4 bg-gray-50 dark:bg-dark-700 rounded-lg">
                      <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">设备信息</p>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <span className="text-gray-500">型号:</span><span className="text-gray-900 dark:text-white">{selectedRobotData?.model || '-'}</span>
                        <span className="text-gray-500">制造商:</span><span className="text-gray-900 dark:text-white">{selectedRobotData?.manufacturer || '-'}</span>
                        <span className="text-gray-500">连接时间:</span><span className="text-gray-900 dark:text-white">{selectedRobotData?.connected_at || '-'}</span>
                        <span className="text-gray-500">最后心跳:</span><span className="text-gray-900 dark:text-white">{selectedRobotData?.last_heartbeat || '-'}</span>
                      </div>
                    </div>
                  </div>
                )}
              </TabPanel>
            )}
          </CardContent>
        </Card>

        <div className="space-y-6">
          <Card>
            <CardHeader><CardTitle>机器人列表</CardTitle></CardHeader>
            <CardContent>
              {isLoading ? (
                <Loading />
              ) : robots.length === 0 ? (
                <EmptyState icon={<Bot size={32} />} title={'"暂无机器人"'} description={'"连接机器人以开始控制"'} />
              ) : (
                <div className="space-y-2">
                  {robots.map((robot: any) => (
                    <div
                      key={robot.id}
                      className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                        selectedRobot === robot.id
                          ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                          : 'border-gray-200 dark:border-dark-600 hover:bg-gray-50 dark:hover:bg-dark-700'
                      }`}
                      onClick={() => setSelectedRobot(robot.id)}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Bot size={18} className={selectedRobot === robot.id ? 'text-primary-500' : 'text-gray-400'} />
                          <div>
                            <p className="text-sm font-medium text-gray-900 dark:text-white">{robot.name}</p>
                            <p className="text-xs text-gray-500">{typeLabel[robot.type] || robot.type}</p>
                          </div>
                        </div>
                        <Badge variant={statusVariant[robot.status]} dot size="sm">{statusLabel[robot.status]}</Badge>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle>命令日志</CardTitle></CardHeader>
            <CardContent>
              {commandLog.length === 0 ? (
                <EmptyState icon={<Settings size={32} />} title={'"暂无命令记录"'} description={'"控制机器人以查看命令日志"'} />
              ) : (
                <div className="space-y-2 max-h-[300px] overflow-y-auto">
                  {commandLog.map((log, i) => (
                    <div key={i} className="flex items-center gap-2 p-2 rounded-lg hover:bg-gray-50 dark:hover:bg-dark-700 text-sm">
                      <Badge variant={log.status === '成功' ? 'success' : log.status === '失败' ? 'danger' : 'info'} size="sm">{log.status}</Badge>
                      <span className="text-gray-700 dark:text-gray-300 flex-1 truncate">{log.command}</span>
                      <span className="text-xs text-gray-400">{log.time}</span>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      <Modal isOpen={showSettingsModal} onClose={() => setShowSettingsModal(false)} title={'"机器人设置"'} size="md" footer={
        <div className="flex gap-2">
          <Button variant="secondary" onClick={() => setShowSettingsModal(false)}>关闭</Button>
          <Button>保存</Button>
        </div>
      }>
        <div className="space-y-4">
          <Input label={'"最大速度"'} type="number" defaultValue={100} hint={'"单位: mm/s"'} />
          <Input label={'"最大加速度"'} type="number" defaultValue={500} hint={'"单位: mm/s^2"'} />
          <Select label={'"坐标系"'} options={[
            { label: '世界坐标', value: 'world' },
            { label: '工具坐标', value: 'tool' },
            { label: '关节坐标', value: 'joint' },
          ]} />
          <div className="flex items-center gap-2">
            <input type="checkbox" id="collision_detection" defaultChecked className="rounded" />
            <label htmlFor="collision_detection" className="text-sm text-gray-700 dark:text-gray-300">碰撞检测</label>
          </div>
          <div className="flex items-center gap-2">
            <input type="checkbox" id="smooth_motion" defaultChecked className="rounded" />
            <label htmlFor="smooth_motion" className="text-sm text-gray-700 dark:text-gray-300">平滑运动</label>
          </div>
        </div>
      </Modal>
    </div>
  );
}
