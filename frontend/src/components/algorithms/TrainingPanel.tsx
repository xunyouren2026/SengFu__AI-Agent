/**
 * 训练算法面板组件
 */

import React, { useState } from 'react';
import { useRLHF, useDPO, usePPO, useBurstAction, useTrainingJobs } from '../../hooks/useTraining';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { Input } from '../common/Input';
import { Select } from '../common/Select';

export const TrainingPanel: React.FC = () => {
  const { currentJob: rlhfJob, start: startRLHF, getStatus: getRLHFStatus } = useRLHF();
  const { currentJob: dpoJob, start: startDPO } = useDPO();
  const { currentJob: ppoJob, start: startPPO } = usePPO();
  const { actionResult, availableActions, execute: executeBurst, fetchActions } = useBurstAction();
  const { jobs, fetchJobs } = useTrainingJobs();

  const [modelName, setModelName] = useState('gpt-2');
  const [datasetPath, setDatasetPath] = useState('/data/preferences.json');
  const [depressionScore, setDepressionScore] = useState(0.7);

  const handleStartRLHF = async () => {
    await startRLHF({
      model_name: modelName,
      dataset_path: datasetPath,
      beta: 0.1,
      lr: 1e-5,
      epochs: 3
    });
    fetchJobs();
  };

  const handleStartDPO = async () => {
    await startDPO({
      model_name: modelName,
      dataset_path: datasetPath,
      beta: 0.1,
      variant: 'dpo'
    });
    fetchJobs();
  };

  const handleStartPPO = async () => {
    await startPPO({
      env_name: 'CartPole-v1',
      model_name: modelName,
      lr: 3e-4,
      epochs: 10
    });
    fetchJobs();
  };

  const handleBurst = async () => {
    await executeBurst(depressionScore);
  };

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">训练算法</h2>

      {/* 训练配置 */}
      <Card title="训练配置">
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium mb-1">模型名称</label>
            <Input
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">数据集路径</label>
            <Input
              value={datasetPath}
              onChange={(e) => setDatasetPath(e.target.value)}
            />
          </div>
        </div>
        <div className="flex gap-2">
          <Button onClick={handleStartRLHF}>启动RLHF</Button>
          <Button onClick={handleStartDPO}>启动DPO</Button>
          <Button onClick={handleStartPPO}>启动PPO</Button>
        </div>
      </Card>

      {/* 训练状态 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {rlhfJob && (
          <Card title="RLHF状态">
            <div className="space-y-2">
              <div>Job ID: {rlhfJob.job_id}</div>
              <div>进度: {(rlhfJob.progress * 100).toFixed(1)}%</div>
              <div>Loss: {rlhfJob.loss?.toFixed(4)}</div>
              <div>KL Div: {rlhfJob.kl_div?.toFixed(4)}</div>
              <div>Reward: {rlhfJob.reward?.toFixed(4)}</div>
            </div>
          </Card>
        )}
        {dpoJob && (
          <Card title="DPO状态">
            <div className="space-y-2">
              <div>Job ID: {dpoJob.job_id}</div>
              <div>进度: {(dpoJob.progress * 100).toFixed(1)}%</div>
              <div>Loss: {dpoJob.loss?.toFixed(4)}</div>
              <div>Accuracy: {(dpoJob.accuracy * 100).toFixed(1)}%</div>
            </div>
          </Card>
        )}
        {ppoJob && (
          <Card title="PPO状态">
            <div className="space-y-2">
              <div>Job ID: {ppoJob.job_id}</div>
              <div>Episode: {ppoJob.episode}</div>
              <div>Reward Mean: {ppoJob.reward_mean?.toFixed(2)}</div>
              <div>Policy Loss: {ppoJob.policy_loss?.toFixed(4)}</div>
            </div>
          </Card>
        )}
      </div>

      {/* 爆发动作 */}
      <Card title="爆发动作">
        <div className="flex gap-4 mb-4">
          <div className="flex-1">
            <label className="block text-sm font-medium mb-1">郁值分数</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={depressionScore}
              onChange={(e) => setDepressionScore(parseFloat(e.target.value))}
              className="w-full"
            />
            <div className="text-center">{depressionScore.toFixed(1)}</div>
          </div>
          <Button onClick={handleBurst} className="self-end">
            执行爆发动作
          </Button>
        </div>
        
        {actionResult && (
          <div className="p-4 bg-blue-50 rounded-lg">
            <div className="font-medium">执行的动作: {actionResult.action}</div>
            <div className="text-sm text-gray-600">{actionResult.effect}</div>
            <div className="text-sm">新郁值: {actionResult.new_depression?.toFixed(2)}</div>
          </div>
        )}

        <Button onClick={fetchActions} variant="secondary" className="mt-4">
          查看可用动作
        </Button>
        
        {availableActions.length > 0 && (
          <div className="mt-4 grid grid-cols-2 gap-2">
            {availableActions.map((action: any) => (
              <div key={action.id} className="p-2 bg-gray-50 rounded text-sm">
                <div className="font-medium">{action.name}</div>
                <div className="text-xs text-gray-500">阈值: {action.threshold}</div>
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* 任务列表 */}
      <Card title="训练任务">
        <Button onClick={() => fetchJobs()} className="mb-4">
          刷新任务列表
        </Button>
        <div className="space-y-2">
          {jobs.map((job: any) => (
            <div key={job.id} className="flex justify-between p-3 bg-gray-50 rounded-lg">
              <div>
                <div className="font-medium">{job.id}</div>
                <div className="text-sm text-gray-500">{job.type} | {job.status}</div>
              </div>
              <div className="text-sm">
                进度: {(job.progress * 100).toFixed(1)}%
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
};

export default TrainingPanel;
