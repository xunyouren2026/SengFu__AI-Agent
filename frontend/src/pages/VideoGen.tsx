import { useState } from 'react';
import { Video, Upload, Play, Download, RefreshCw, Settings, Sparkles, History, Film, Clock } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Badge, Input, Select, Textarea, Tabs, TabPanel, EmptyState, Loading, StatsCard } from '../components/common';
import { useGenerationStats, useGenerateVideo, useGenerationTasks } from '../hooks/useGeneration';

export default function VideoGen() {
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [activeTab, setActiveTab] = useState('text2video');
  const [engine, setEngine] = useState('cogvideox');
  const [numFrames, setNumFrames] = useState(16);
  const [fps, setFps] = useState(8);
  const [steps, setSteps] = useState(50);
  const [guidanceScale, setGuidanceScale] = useState(7.5);

  const { data: stats } = useGenerationStats();
  const { data: tasks, isLoading: tasksLoading } = useGenerationTasks();
  const generateVideo = useGenerateVideo();

  const handleGenerate = () => {
    if (!prompt.trim()) return;
    generateVideo.mutate({
      prompt,
      negative_prompt: negativePrompt,
      num_frames: numFrames,
      fps,
      engine,
      guidance_scale: guidanceScale,
      num_inference_steps: steps,
    } as any);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">视频生成</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">AI 驱动的视频内容生成</p>
        </div>
        <Badge variant="success" dot>服务就绪</Badge>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatsCard title={'"总任务"'} value={stats?.total_tasks || 0} icon={<Video size={24} />} color="blue" />
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
                { key: 'text2video', label: '文本生成视频' },
                { key: 'img2vid', label: '图生视频' },
              ]}
              activeKey={activeTab}
              onChange={setActiveTab}
            />

            {activeTab === 'text2video' && (
              <TabPanel>
                <div className="space-y-4">
                  <Textarea
                    label={'"提示词"'}
                    placeholder={'"描述你想要生成的视频内容..."'}
                    value={prompt}
                    onChange={e => setPrompt(e.target.value)}
                    rows={4}
                  />
                  <Textarea
                    label={'"负面提示词"'}
                    placeholder={'"不希望出现的内容..."'}
                    value={negativePrompt}
                    onChange={e => setNegativePrompt(e.target.value)}
                    rows={2}
                  />
                  <div className="grid grid-cols-2 gap-4">
                    <Select
                      label={'"引擎"'}
                      value={engine}
                      onChange={e => setEngine(e.target.value)}
                      options={[
                        { label: 'CogVideoX', value: 'cogvideox' },
                        { label: 'AnimateDiff', value: 'animatediff' },
                        { label: 'SVD', value: 'svd' },
                      ]}
                    />
                    <Input label={'"帧数"'} type="number" value={numFrames} onChange={e => setNumFrames(Number(e.target.value))} />
                    <Input label="FPS" type="number" value={fps} onChange={e => setFps(Number(e.target.value))} />
                    <Input label={'"推理步数"'} type="number" value={steps} onChange={e => setSteps(Number(e.target.value))} />
                    <Input label={'"引导系数"'} type="number" step="0.5" value={guidanceScale} onChange={e => setGuidanceScale(Number(e.target.value))} />
                    <Select
                      label={'"分辨率"'}
                      options={[
                        { label: '512x512', value: '512x512' },
                        { label: '768x768', value: '768x768' },
                        { label: '1024x576', value: '1024x576' },
                      ]}
                    />
                  </div>
                  <Button
                    className="w-full"
                    leftIcon={<Play size={16} />}
                    onClick={handleGenerate}
                    isLoading={generateVideo.isPending}
                    disabled={!prompt.trim()}
                  >
                    生成视频
                  </Button>
                </div>
              </TabPanel>
            )}

            {activeTab === 'img2vid' && (
              <TabPanel>
                <div className="space-y-4">
                  <div className="border-2 border-dashed border-gray-300 dark:border-dark-600 rounded-xl p-8 text-center hover:border-primary-400 transition-colors cursor-pointer">
                    <Upload size={32} className="mx-auto text-gray-400 mb-2" />
                    <p className="text-sm text-gray-500 dark:text-gray-400">拖拽或点击上传图片</p>
                    <p className="text-xs text-gray-400 mt-1">支持 PNG, JPG, WEBP 格式</p>
                  </div>
                  <Textarea
                    label={'"运动描述"'}
                    placeholder={'"描述图片中的运动方式..."'}
                    rows={3}
                  />
                  <div className="grid grid-cols-2 gap-4">
                    <Input label={'"帧数"'} type="number" defaultValue={16} />
                    <Input label="FPS" type="number" defaultValue={8} />
                    <Input label={'"推理步数"'} type="number" defaultValue={50} />
                    <Input label={'"运动幅度"'} type="number" defaultValue={1.0} step={0.1} />
                  </div>
                  <Button className="w-full" leftIcon={<Film size={16} />}>
                    生成视频
                  </Button>
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
              <EmptyState icon={<History size={32} />} title={'"暂无生成记录"'} description={'"创建你的第一个视频生成任务"'} />
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
