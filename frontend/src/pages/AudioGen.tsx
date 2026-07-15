import { useState } from 'react';
import { Music, Mic, Volume2, Upload, Play, Download, RefreshCw, Sparkles, History, Clock, Guitar, Radio, FileAudio } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Badge, Input, Select, Textarea, Tabs, TabPanel, EmptyState, Loading, StatsCard } from '../components/common';
import { useGenerationStats, useGenerateAudio, useGenerationTasks, useVoices } from '../hooks/useGeneration';

export default function AudioGen() {
  const [prompt, setPrompt] = useState('');
  const [activeTab, setActiveTab] = useState('tts');
  const [voiceId, setVoiceId] = useState('');
  const [engine, setEngine] = useState('bark');
  const [duration, setDuration] = useState(10);
  const [speed, setSpeed] = useState(1.0);

  const { data: stats } = useGenerationStats();
  const { data: tasks, isLoading: tasksLoading } = useGenerationTasks();
  const { data: voices } = useVoices();
  const generateAudio = useGenerateAudio();

  const handleGenerate = () => {
    if (!prompt.trim()) return;
    generateAudio.mutate({
      prompt,
      voice_id: voiceId || undefined,
      engine,
      duration,
      speed,
      type: activeTab,
    } as any);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">音频生成</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">AI 驱动的音频内容生成与处理</p>
        </div>
        <Badge variant="success" dot>服务就绪</Badge>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatsCard title={'"总任务"'} value={stats?.total_tasks || 0} icon={<Music size={24} />} color="blue" />
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
                { key: 'tts', label: '文本转语音', icon: <Mic size={16} /> },
                { key: 'voice_clone', label: '声音克隆', icon: <Volume2 size={16} /> },
                { key: 'music', label: '音乐生成', icon: <Guitar size={16} /> },
                { key: 'sfx', label: '音效生成', icon: <Radio size={16} /> },
              ]}
              activeKey={activeTab}
              onChange={setActiveTab}
            />

            {activeTab === 'tts' && (
              <TabPanel>
                <div className="space-y-4">
                  <Textarea
                    label={'"文本内容"'}
                    placeholder={'"输入需要转换为语音的文本..."'}
                    value={prompt}
                    onChange={e => setPrompt(e.target.value)}
                    rows={4}
                  />
                  <div className="grid grid-cols-2 gap-4">
                    <Select
                      label={'"语音"'}
                      value={voiceId}
                      onChange={e => setVoiceId(e.target.value)}
                      options={[
                        { label: '默认女声', value: 'female_default' },
                        { label: '默认男声', value: 'male_default' },
                        ...((voices || []) as any[]).map((v: any) => ({ label: v.name || v.id, value: v.id })),
                      ]}
                    />
                    <Select
                      label={'"引擎"'}
                      value={engine}
                      onChange={e => setEngine(e.target.value)}
                      options={[
                        { label: 'Bark', value: 'bark' },
                        { label: 'VITS', value: 'vits' },
                        { label: 'ChatTTS', value: 'chattts' },
                        { label: 'CosyVoice', value: 'cosyvoice' },
                      ]}
                    />
                    <Input label={'"语速"'} type="number" step={0.1} value={speed} onChange={e => setSpeed(Number(e.target.value))} />
                    <Select
                      label={'"输出格式"'}
                      options={[
                        { label: 'WAV', value: 'wav' },
                        { label: 'MP3', value: 'mp3' },
                        { label: 'OGG', value: 'ogg' },
                      ]}
                    />
                  </div>
                  <Button
                    className="w-full"
                    leftIcon={<Play size={16} />}
                    onClick={handleGenerate}
                    isLoading={generateAudio.isPending}
                    disabled={!prompt.trim()}
                  >
                    生成语音
                  </Button>
                </div>
              </TabPanel>
            )}

            {activeTab === 'voice_clone' && (
              <TabPanel>
                <div className="space-y-4">
                  <div className="border-2 border-dashed border-gray-300 dark:border-dark-600 rounded-xl p-8 text-center hover:border-primary-400 transition-colors cursor-pointer">
                    <Upload size={32} className="mx-auto text-gray-400 mb-2" />
                    <p className="text-sm text-gray-500 dark:text-gray-400">上传参考音频文件</p>
                    <p className="text-xs text-gray-400 mt-1">建议 10-30 秒的清晰语音样本</p>
                  </div>
                  <Textarea label={'"合成文本"'} placeholder={'"输入需要用克隆声音朗读的文本..."'} rows={3} />
                  <div className="grid grid-cols-2 gap-4">
                    <Input label={'"语速"'} type="number" step={0.1} defaultValue={1.0} />
                    <Select
                      label={'"输出格式"'}
                      options={[
                        { label: 'WAV', value: 'wav' },
                        { label: 'MP3', value: 'mp3' },
                      ]}
                    />
                  </div>
                  <Button className="w-full" leftIcon={<Volume2 size={16} />}>克隆并生成</Button>
                </div>
              </TabPanel>
            )}

            {activeTab === 'music' && (
              <TabPanel>
                <div className="space-y-4">
                  <Textarea
                    label={'"音乐描述"'}
                    placeholder={'"描述你想要生成的音乐风格、情绪、乐器..."'}
                    value={prompt}
                    onChange={e => setPrompt(e.target.value)}
                    rows={3}
                  />
                  <div className="grid grid-cols-2 gap-4">
                    <Select
                      label={'"音乐风格"'}
                      options={[
                        { label: '流行', value: 'pop' },
                        { label: '古典', value: 'classical' },
                        { label: '电子', value: 'electronic' },
                        { label: '爵士', value: 'jazz' },
                        { label: '摇滚', value: 'rock' },
                        { label: '民谣', value: 'folk' },
                      ]}
                    />
                    <Select
                      label={'引擎'}
                      value={engine}
                      onChange={e => setEngine(e.target.value)}
                      options={[
                        { label: 'MusicGen', value: 'musicgen' },
                        { label: 'Suno', value: 'suno' },
                        { label: 'Udio', value: 'udio' },
                      ]}
                    />
                    <Input label={'"时长（秒）"'} type="number" value={duration} onChange={e => setDuration(Number(e.target.value))} />
                    <Input label={'"温度"'} type="number" step={0.1} defaultValue={1.0} />
                  </div>
                  <Button
                    className="w-full"
                    leftIcon={<Guitar size={16} />}
                    onClick={handleGenerate}
                    isLoading={generateAudio.isPending}
                    disabled={!prompt.trim()}
                  >
                    生成音乐
                  </Button>
                </div>
              </TabPanel>
            )}

            {activeTab === 'sfx' && (
              <TabPanel>
                <div className="space-y-4">
                  <Textarea
                    label={'"音效描述"'}
                    placeholder={'"描述你想要生成的音效，如：雨声、键盘敲击、爆炸声..."'}
                    value={prompt}
                    onChange={e => setPrompt(e.target.value)}
                    rows={3}
                  />
                  <div className="grid grid-cols-2 gap-4">
                    <Input label={'"时长（秒）"'} type="number" defaultValue={5} />
                    <Input label={'"温度"'} type="number" step={0.1} defaultValue={1.0} />
                  </div>
                  <Button
                    className="w-full"
                    leftIcon={<Radio size={16} />}
                    onClick={handleGenerate}
                    isLoading={generateAudio.isPending}
                    disabled={!prompt.trim()}
                  >
                    生成音效
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
              <EmptyState icon={<FileAudio size={32} />} title={'"暂无生成记录"'} description={'"创建你的第一个音频生成任务"'} />
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
