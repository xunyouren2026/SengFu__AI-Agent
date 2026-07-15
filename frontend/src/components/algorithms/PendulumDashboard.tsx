/**
 * 胜复学算法仪表盘组件
 */

import React, { useEffect, useState } from 'react';
import { useDepressionMonitor, useAlgorithmStats, usePendulumCycle } from '../../hooks/useAlgorithms';
import { Card } from '../common/Card';
import { StatsCard } from '../common/StatsCard';
import { Button } from '../common/Button';

export const PendulumDashboard: React.FC = () => {
  const { depressionScore, depressionLevel, fetchStatus } = useDepressionMonitor();
  const { stats, fetchStats } = useAlgorithmStats();
  const { cycleResult, loading, executeCycle } = usePendulumCycle();
  
  const [testObservation, setTestObservation] = useState({ text: '测试输入' });

  useEffect(() => {
    fetchStatus();
    fetchStats();
    const interval = setInterval(() => {
      fetchStatus();
      fetchStats();
    }, 10000);
    return () => clearInterval(interval);
  }, [fetchStatus, fetchStats]);

  const handleTestCycle = async () => {
    await executeCycle(testObservation, '完成测试任务');
  };

  const getDepressionColor = (level: string) => {
    switch (level) {
      case 'low': return 'text-green-500';
      case 'medium': return 'text-yellow-500';
      case 'high': return 'text-orange-500';
      case 'critical': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">胜复学算法仪表盘</h2>
      
      {/* 郁值监控 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatsCard
          title="当前郁值"
          value={depressionScore.toFixed(2)}
          subtitle={depressionLevel}
          trend={depressionScore > 0.5 ? 'up' : 'down'}
          className={getDepressionColor(depressionLevel)}
        />
        <StatsCard
          title="算法调用次数"
          value={stats?.swing?.total_calls || 0}
          subtitle="总调用"
        />
        <StatsCard
          title="成功率"
          value={`${((stats?.swing?.success_rate || 0) * 100).toFixed(1)}%`}
          subtitle="平均"
        />
      </div>

      {/* 测试区域 */}
      <Card title="胜复学闭环测试">
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">测试输入</label>
            <input
              type="text"
              value={testObservation.text}
              onChange={(e) => setTestObservation({ text: e.target.value })}
              className="w-full px-3 py-2 border rounded-lg"
            />
          </div>
          <Button onClick={handleTestCycle} loading={loading}>
            执行胜复学闭环
          </Button>
          
          {cycleResult && (
            <div className="mt-4 p-4 bg-gray-50 rounded-lg">
              <h4 className="font-medium mb-2">执行结果</h4>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>动作: {cycleResult.action}</div>
                <div>置信度: {cycleResult.confidence?.toFixed(2)}</div>
                <div>郁值: {cycleResult.depression_score?.toFixed(2)}</div>
                <div>内在奖励: {cycleResult.intrinsic_reward?.toFixed(4)}</div>
                <div>是否停止: {cycleResult.should_halt ? '是' : '否'}</div>
              </div>
            </div>
          )}
        </div>
      </Card>

      {/* 算法统计 */}
      <Card title="算法统计">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Object.entries(stats).map(([algo, data]: [string, any]) => (
            <div key={algo} className="p-3 bg-gray-50 rounded-lg">
              <div className="text-sm font-medium capitalize">{algo}</div>
              <div className="text-xs text-gray-500">
                调用: {data?.total_calls || 0}
              </div>
              <div className="text-xs text-gray-500">
                成功率: {((data?.success_rate || 0) * 100).toFixed(1)}%
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
};

export default PendulumDashboard;
