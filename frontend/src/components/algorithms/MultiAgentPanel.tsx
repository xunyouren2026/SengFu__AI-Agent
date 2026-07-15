/**
 * 多智能体协作面板组件
 */

import React, { useState } from 'react';
import { useAlliance, useDebate, useReputation } from '../../hooks/useMultiagent';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { Input } from '../common/Input';

export const MultiAgentPanel: React.FC = () => {
  const { alliances, createAlliance, joinAlliance, fetchAlliances } = useAlliance();
  const { debates, startDebate, fetchDebates } = useDebate();
  const { leaderboard, fetchLeaderboard } = useReputation();

  const [newAllianceName, setNewAllianceName] = useState('');
  const [debateTopic, setDebateTopic] = useState('');

  const handleCreateAlliance = async () => {
    if (newAllianceName) {
      await createAlliance({
        name: newAllianceName,
        description: '新创建的联盟',
        goals: ['协作', '学习']
      });
      setNewAllianceName('');
      fetchAlliances();
    }
  };

  const handleStartDebate = async () => {
    if (debateTopic) {
      await startDebate(debateTopic, ['agent1', 'agent2', 'agent3']);
      setDebateTopic('');
      fetchDebates();
    }
  };

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">多智能体协作</h2>

      {/* 联盟管理 */}
      <Card title="联盟管理">
        <div className="flex gap-2 mb-4">
          <Input
            value={newAllianceName}
            onChange={(e) => setNewAllianceName(e.target.value)}
            placeholder="输入联盟名称"
          />
          <Button onClick={handleCreateAlliance}>创建联盟</Button>
        </div>
        <div className="space-y-2">
          {alliances.map((alliance: any) => (
            <div key={alliance.id} className="p-3 bg-gray-50 rounded-lg">
              <div className="font-medium">{alliance.name}</div>
              <div className="text-sm text-gray-500">
                成员: {alliance.members?.length || 0}/{alliance.max_members}
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* 辩论系统 */}
      <Card title="辩论系统">
        <div className="flex gap-2 mb-4">
          <Input
            value={debateTopic}
            onChange={(e) => setDebateTopic(e.target.value)}
            placeholder="输入辩题"
          />
          <Button onClick={handleStartDebate}>开始辩论</Button>
        </div>
        <div className="space-y-2">
          {debates.map((debate: any) => (
            <div key={debate.id} className="p-3 bg-gray-50 rounded-lg">
              <div className="font-medium">{debate.topic}</div>
              <div className="text-sm text-gray-500">
                状态: {debate.status} | 论点: {debate.argument_count || 0}
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* 信誉排行榜 */}
      <Card title="信誉排行榜">
        <Button onClick={() => fetchLeaderboard(10)} className="mb-4">
          刷新排行榜
        </Button>
        <div className="space-y-2">
          {leaderboard.map((entry: any) => (
            <div key={entry.agent_id} className="flex justify-between p-3 bg-gray-50 rounded-lg">
              <div>
                <span className="font-bold mr-2">#{entry.rank}</span>
                <span>{entry.agent_id}</span>
              </div>
              <div className="text-blue-600 font-medium">
                {entry.score.toFixed(2)}
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
};

export default MultiAgentPanel;
