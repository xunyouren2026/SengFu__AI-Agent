import { useState } from 'react';
import { Box, Upload, Play, Download, RefreshCw, Sparkles, History, Clock, RotateCcw, Image, Settings, Hexagon } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Badge, Input, Select, Textarea, Tabs, TabPanel, EmptyState, Loading, StatsCard } from '../components/common';
import { useGenerationStats, useGenerate3D, useGenerationTasks } from '../hooks/useGeneration';

export default function ThreeDGen() {
  const [prompt, setPrompt] = useState('');
  const [activeTab, setActiveTab] = useState('text2three_d');
  const [engine, setEngine] = useState('shap-e');
  const [steps, setSteps] = useState(50);
  const [guidanceScale, setGuidanceScale] = useState(7.5);

  const { data: stats } = useGenerationStats();
  const { data: tasks, isLoading: tasksLoading } = useGenerationTasks();
  const generate3D = useGenerate3D();

  const handleGenerate = () => {
    if (!prompt.trim()) return;
    generate3D.mutate({
      prompt,
      engine,
      num_inference_steps: steps,
      guidance_scale: guidanceScale,
      type: activeTab,
    } as any);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">3D 模型生成</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">AI 驱动的三维模型生成与处理</p>
        </div>
        <Badge variant="success" dot>服务就绪</Badge>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatsCard title={'"总任务"'} value={stats?.total_tasks || 0} icon={<Box size={24} />} color="blue" />
        <StatsCard title={'"已完成"'} value={stats?.completed_tasks || 0} icon={<Sparkles size={24} />} color="green" />
        <StatsCard title={'"失败"'} value={stats?.failed_tasks || 0} icon={<RefreshCw size={24} />} color="red" />
        <StatsCard title={'"平均推理时间"'} value={`${stats?.avg_inference_time?.toFixed(1) || 0}s`} icon={<Clock size={24} />} color="purple" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2">
          <CardHeader><CardTitle>生成设置</CardTitle></CardHeader>
          <CardContent>
            <Tabs
              tabs={[
                { key: 'text2three_d', label: '文本生成 3D' },
                { key: 'img2three_d', label: '图像生成 3D' },
              ]}
              activeKey={activeTab}
              onChange={setActiveTab}
            />

            {activeTab === 'text2three_d' && (
              <TabPanel>
                <div className="space-y-4">
                  <Textarea
                    label={'"提示词"'}
                    placeholder={'"描述你想要生成的 3D 模型..."'}
                    value={prompt}
                    onChange={e => setPrompt(e.target.value)}
                    rows={4}
                  />
                  <Textarea
                    label={'"负面提示词"'}
                    placeholder={'"不希望出现的内容..."'}
                    rows={2}
                  />
                  <div className="grid grid-cols-2 gap-4">
                    <Select
                      label={'"引擎"'}
                      value={engine}
                      onChange={e => setEngine(e.target.value)}
                      options={[
                        { label: 'Shap-E', value: 'shap-e' },
                        { label: 'Point-E', value: 'point-e' },
                        { label: 'TripoSR', value: 'triposr' },
                        { label: 'LRM', value: 'lrm' },
                      ]}
                    />
                    <Select
                      label={'"输出格式"'}
                      options={[
                        { label: 'GLB', value: 'glb' },
                        { label: 'OBJ', value: 'obj' },
                        { label: 'STL', value: 'stl' },
                        { label: 'PLY', value: 'ply' },
                      ]}
                    />
                    <Input label={'"推理步数"'} type="number" value={steps} onChange={e => setSteps(Number(e.target.value))} />
                    <Input label={'"引导系数"'} type="number" step="0.5" value={guidanceScale} onChange={e => setGuidanceScale(Number(e.target.value))} />
                  </div>
                  <Button
                    className="w-full"
                    leftIcon={<Hexagon size={16} />}
                    onClick={handleGenerate}
                    isLoading={generate3D.isPending}
                    disabled={!prompt.trim()}
                  >
                    生成 3D 模型
                  </Button>
                </div>
              </TabPanel>
            )}

            {activeTab === 'img2three_d' && (
              <TabPanel>
                <div className="space-y-4">
                  <div className="border-2 border-dashed border-gray-300 dark:border-dark-600 rounded-xl p-8 text-center hover:border-primary-400 transition-colors cursor-pointer">
                    <Upload size={32} className="mx-auto text-gray-400 mb-2" />
                    <p className="text-sm text-gray-500 dark:text-gray-400">拖拽或点击上传参考图片</p>
                    <p className="text-xs text-gray-400 mt-1">建议使用正面视角的清晰图片</p>
                  </div>
                  <Textarea label={'"补充描述"'} placeholder={'"可选：补充描述模型的细节..."'} rows={2} />
                  <div className="grid grid-cols-2 gap-4">
                    <Select
                      label={'"引擎"'}
                      value={engine}
                      onChange={e => setEngine(e.target.value)}
                      options={[
                        { label: 'TripoSR', value: 'triposr' },
                        { label: 'LRM', value: 'lrm' },
                        { label: 'Shap-E', value: 'shap-e' },
                      ]}
                    />
                    <Select
                      label={'"输出格式"'}
                      options={[
                        { label: 'GLB', value: 'glb' },
                        { label: 'OBJ', value: 'obj' },
                        { label: 'STL', value: 'stl' },
                      ]}
                    />
                  </div>
                  <Button className="w-full" leftIcon={<Image size={16} />}>生成 3D 模型</Button>
                </div>
              </TabPanel>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>生成历史</CardTitle></CardHeader>
          <CardContent>
            {tasksLoading ? (
              <Loading />
            ) : (tasks?.tasks || []).length === 0 ? (
              <EmptyState icon={<History size={32} />} title={'"暂无生成记录"'} description={'"创建你的第一个 3D 模型生成任务"'} />
            ) : (
              <div className="space-y-3">
                {(tasks?.tasks || []).slice(0, 10).map((task: any) => (
                  <div key={task.id} className="flex items-center gap-3 p-2 rounded-lg hover:bg-gray-50 dark:hover:bg-dark-700 transition-colors">
                    <Badge
                      variant={
                        task.status === 'completed' ? 'success' :
                        task.status === 'failed' ? 'danger' :
                        task.status === 'processing' ? 'warning' : 'info'
                      }
                    >
                      {task.status === 'completed' ? '完成' : task.status === 'failed' ? '失败' : task.status === 'processing' ? '处理中' : '等待中'}
                    </Badge>
                    <div className="min-w-0 flex-1">
                      <p className="text-sm truncate text-gray-900 dark:text-white">{task.type || task.prompt}</p>
                      <p className="text-xs text-gray-400">{task.progress?.toFixed(0) || 0}%</p>
                    </div>
                    {task.status === 'completed' && (
                      <Button variant="ghost" size="sm" leftIcon={<Download size={14} />}>下载</Button>
                    )}
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
