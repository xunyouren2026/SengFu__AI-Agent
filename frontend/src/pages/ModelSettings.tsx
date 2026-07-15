import { useState } from 'react';
import { Settings, Eye, EyeOff, TestTube, Save } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Input, Select, Badge, Tabs, TabPanel } from '../components/common';

const providers = [
  { id: 'openai', name: 'OpenAI', models: ['gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo'] },
  { id: 'anthropic', name: 'Anthropic', models: ['claude-3.5-sonnet', 'claude-3-opus'] },
  { id: 'deepseek', name: 'DeepSeek', models: ['deepseek-chat', 'deepseek-coder'] },
  { id: 'zhipu', name: '智谱AI', models: ['glm-4', 'glm-3-turbo'] },
  { id: 'moonshot', name: '月之暗面', models: ['moonshot-v1-8k', 'moonshot-v1-32k'] },
  { id: 'qwen', name: '通义千问', models: ['qwen-max', 'qwen-plus'] },
  { id: 'doubao', name: '字节豆包', models: ['doubao-pro'] },
  { id: 'baichuan', name: '百川', models: ['baichuan-4'] },
  { id: 'minimax', name: 'MiniMax', models: ['abab-6.5s'] },
  { id: 'local', name: '本地模型', models: ['custom'] },
];

export default function ModelSettings() {
  const [selectedProvider, setSelectedProvider] = useState(providers[0].id);
  const [showKey, setShowKey] = useState(false);
  const provider = providers.find(p => p.id === selectedProvider)!;

  return (
    <div className="space-y-6">
      <div><h1 className="text-2xl font-bold text-gray-900 dark:text-white">模型设置</h1><p className="text-sm text-gray-500 mt-1">配置 LLM 提供商和模型参数</p></div>
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <Card className="lg:col-span-1">
          <CardHeader><CardTitle>提供商</CardTitle></CardHeader>
          <CardContent>
            <div className="space-y-1">
              {providers.map(p => (
                <button key={p.id} onClick={() => setSelectedProvider(p.id)} className={`flex items-center justify-between w-full px-3 py-2 rounded-lg text-sm transition-colors ${selectedProvider === p.id ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-600 font-medium' : 'text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-dark-700'}`}>
                  <span>{p.name}</span>
                  <Badge variant="outline" size="sm">{p.models.length}</Badge>
                </button>
              ))}
            </div>
          </CardContent>
        </Card>
        <Card className="lg:col-span-3">
          <CardHeader><CardTitle>{provider.name} 配置</CardTitle></CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div><label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">API 密钥</label>
                <div className="flex gap-2"><Input type={showKey ? 'text' : 'password'} placeholder="sk-..." className="flex-1" /><Button variant="ghost" onClick={() => setShowKey(!showKey)}>{showKey ? <EyeOff size={16} /> : <Eye size={16} />}</Button><Button variant="outline" size="sm" leftIcon={<TestTube size={14} />}>测试</Button></div>
              </div>
              <Input label={'"API 端点"'} placeholder="https://api.openai.com/v1" />
              <Select label={'"默认模型"'} options={provider.models.map(m => ({ label: m, value: m }))} />
              <div className="grid grid-cols-2 gap-4">
                <Input label="Temperature" type="number" defaultValue="0.7" step="0.1" min="0" max="2" />
                <Input label="Top P" type="number" defaultValue="0.9" step="0.1" min="0" max="1" />
                <Input label="Max Tokens" type="number" defaultValue="4096" />
                <Input label={'"频率惩罚"'} type="number" defaultValue="0" step="0.1" min="-2" max="2" />
              </div>
              <div className="flex items-center gap-4">
                <label className="flex items-center gap-2 text-sm"><input type="checkbox" defaultChecked />视觉能力</label>
                <label className="flex items-center gap-2 text-sm"><input type="checkbox" />函数调用</label>
                <label className="flex items-center gap-2 text-sm"><input type="checkbox" />JSON 模式</label>
              </div>
              <div className="flex justify-end"><Button leftIcon={<Save size={16} />}>保存配置</Button></div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
