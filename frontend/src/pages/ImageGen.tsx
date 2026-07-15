import { useState } from 'react';
import { Image, Upload, Play, Download, RefreshCw, Sparkles, History, Paintbrush, Expand, Wand2, Clock, Layers } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Badge, Input, Select, Textarea, Tabs, TabPanel, EmptyState, Loading, StatsCard } from '../components/common';
import { useGenerationStats, useGenerateImage, useGenerationTasks } from '../hooks/useGeneration';

export default function ImageGen() {
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [activeTab, setActiveTab] = useState('text2img');
  const [engine, setEngine] = useState('sd3');
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [steps, setSteps] = useState(30);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [batchSize, setBatchSize] = useState(1);

  const { data: stats } = useGenerationStats();
  const { data: tasks, isLoading: tasksLoading } = useGenerationTasks();
  const generateImage = useGenerateImage();

  const handleGenerate = () => {
    if (!prompt.trim()) return;
    generateImage.mutate({
      prompt,
      negative_prompt: negativePrompt,
      width,
      height,
      num_inference_steps: steps,
      guidance_scale: guidanceScale,
      batch_size: batchSize,
      engine,
    } as any);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">图像生成</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">AI 驱动的图像内容生成与编辑</p>
        </div>
        <Badge variant="success" dot>服务就绪</Badge>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatsCard title={'"总任务"'} value={stats?.total_tasks || 0} icon={<Image size={24} />} color="blue" />
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
                { key: 'text2img', label: '文本生成图像' },
                { key: 'img2img', label: '图生图像' },
                { key: 'inpaint', label: '局部重绘' },
                { key: 'outpaint', label: '图像扩展' },
              ]}
              activeKey={activeTab}
              onChange={setActiveTab}
            />

            {(activeTab === 'text2img' || activeTab === 'img2img') && (
              <TabPanel>
                <div className="space-y-4">
                  {activeTab === 'img2img' && (
                    <div className="border-2 border-dashed border-gray-300 dark:border-dark-600 rounded-xl p-8 text-center hover:border-primary-400 transition-colors cursor-pointer">
                      <Upload size={32} className="mx-auto text-gray-400 mb-2" />
                      <p className="text-sm text-gray-500 dark:text-gray-400">拖拽或点击上传参考图片</p>
                    </div>
                  )}
                  <Textarea
                    label={'"提示词"'}
                    placeholder={'"描述你想要生成的图像内容..."'}
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
                        { label: 'Stable Diffusion 3', value: 'sd3' },
                        { label: 'DALL-E 3', value: 'dall-e-3' },
                        { label: 'Midjourney', value: 'midjourney' },
                        { label: 'Flux', value: 'flux' },
                      ]}
                    />
                    <Select
                      label={'"批量大小"'}
                      value={String(batchSize)}
                      onChange={e => setBatchSize(Number(e.target.value))}
                      options={[
                        { label: '1 张', value: '1' },
                        { label: '2 张', value: '2' },
                        { label: '4 张', value: '4' },
                      ]}
                    />
                    <Input label={'"宽度"'} type="number" value={width} onChange={e => setWidth(Number(e.target.value))} />
                    <Input label={'"高度"'} type="number" value={height} onChange={e => setHeight(Number(e.target.value))} />
                    <Input label={'"推理步数"'} type="number" value={steps} onChange={e => setSteps(Number(e.target.value))} />
                    <Input label={'"引导系数"'} type="number" step="0.5" value={guidanceScale} onChange={e => setGuidanceScale(Number(e.target.value))} />
                  </div>
                  {activeTab === 'img2img' && (
                    <Input label={'"重绘强度"'} type="number" defaultValue={0.7} step={0.05} hint={'"值越高变化越大"'} />
                  )}
                  <Button
                    className="w-full"
                    leftIcon={<Wand2 size={16} />}
                    onClick={handleGenerate}
                    isLoading={generateImage.isPending}
                    disabled={!prompt.trim()}
                  >
                    生成图像
                  </Button>
                </div>
              </TabPanel>
            )}

            {activeTab === 'inpaint' && (
              <TabPanel>
                <div className="space-y-4">
                  <div className="border-2 border-dashed border-gray-300 dark:border-dark-600 rounded-xl p-8 text-center hover:border-primary-400 transition-colors cursor-pointer">
                    <Paintbrush size={32} className="mx-auto text-gray-400 mb-2" />
                    <p className="text-sm text-gray-500 dark:text-gray-400">上传图片并用画笔标记重绘区域</p>
                    <p className="text-xs text-gray-400 mt-1">在图片上涂抹需要重绘的部分</p>
                  </div>
                  <Textarea label={'"重绘提示词"'} placeholder={'"描述重绘区域的内容..."'} rows={3} />
                  <div className="grid grid-cols-2 gap-4">
                    <Input label={'"重绘强度"'} type="number" defaultValue={0.8} step={0.05} />
                    <Input label={'"推理步数"'} type="number" defaultValue={30} />
                  </div>
                  <Button className="w-full" leftIcon={<Paintbrush size={16} />}>开始重绘</Button>
                </div>
              </TabPanel>
            )}

            {activeTab === 'outpaint' && (
              <TabPanel>
                <div className="space-y-4">
                  <div className="border-2 border-dashed border-gray-300 dark:border-dark-600 rounded-xl p-8 text-center hover:border-primary-400 transition-colors cursor-pointer">
                    <Expand size={32} className="mx-auto text-gray-400 mb-2" />
                    <p className="text-sm text-gray-500 dark:text-gray-400">上传图片并设置扩展方向</p>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <Select
                      label={'"扩展方向"'}
                      options={[
                        { label: '四周扩展', value: 'all' },
                        { label: '向右扩展', value: 'right' },
                        { label: '向下扩展', value: 'bottom' },
                        { label: '向左扩展', value: 'left' },
                      ]}
                    />
                    <Input label={'"扩展像素"'} type="number" defaultValue={256} />
                    <Input label={'"推理步数"'} type="number" defaultValue={30} />
                    <Input label={'"引导系数"'} type="number" defaultValue={7.5} step={0.5} />
                  </div>
                  <Button className="w-full" leftIcon={<Expand size={16} />}>开始扩展</Button>
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
              <EmptyState icon={<History size={32} />} title={'"暂无生成记录"'} description={'"创建你的第一个图像生成任务"'} />
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
