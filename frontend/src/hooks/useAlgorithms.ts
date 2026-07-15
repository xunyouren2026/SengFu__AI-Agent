/**
 * 胜复学算法Hooks
 * 
 * 提供胜复学算法的React Hooks封装
 */

import { useState, useCallback, useEffect } from 'react';
import { algorithmsAPI } from '../api/algorithms';

// =============================================================================
// Swing Engine Hook
// =============================================================================

export function useSwingEngine() {
  const [action, setAction] = useState<string>('');
  const [confidence, setConfidence] = useState<number>(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const executeSwing = useCallback(async (observation: any, training: boolean = true) => {
    setLoading(true);
    setError(null);
    try {
      const result = await algorithmsAPI.swingExecute({
        observation,
        training,
        force_exploration: false
      });
      setAction(result.data?.action || '');
      setConfidence(result.data?.confidence || 0);
      return result;
    } catch (err: any) {
      setError(err.message || 'Swing execution failed');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return { action, confidence, loading, error, executeSwing };
}

// =============================================================================
// Reflexion Hook
// =============================================================================

export function useReflexion() {
  const [reflections, setReflections] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const think = useCallback(async (params: {
    task_id: string;
    task_description: string;
    outcome: string;
    evaluation: number;
  }) => {
    setLoading(true);
    setError(null);
    try {
      const result = await algorithmsAPI.reflexionThink({
        ...params,
        use_llm: false,
        tags: []
      });
      return result;
    } catch (err: any) {
      setError(err.message || 'Reflexion failed');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchHistory = useCallback(async (taskId?: string) => {
    setLoading(true);
    try {
      const result = await algorithmsAPI.reflexionHistory(taskId);
      setReflections(result.data || []);
      return result;
    } catch (err: any) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return { reflections, loading, error, think, fetchHistory };
}

// =============================================================================
// Depression Monitor Hook (郁值监控)
// =============================================================================

export function useDepressionMonitor() {
  const [depressionScore, setDepressionScore] = useState<number>(0.3);
  const [depressionLevel, setDepressionLevel] = useState<string>('low');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const checkBalance = useCallback(async (metric: string, value: number, target: number) => {
    setLoading(true);
    setError(null);
    try {
      const result = await algorithmsAPI.balanceAdjust({
        metric,
        value,
        target
      });
      const data = result.data;
      setDepressionScore(data?.depression_score || 0);
      setDepressionLevel(data?.depression_level || 'low');
      return result;
    } catch (err: any) {
      setError(err.message || 'Balance check failed');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchStatus = useCallback(async () => {
    try {
      const result = await algorithmsAPI.balanceStatus();
      const data = result.data;
      setDepressionScore(data?.current_depression || 0);
      setDepressionLevel(data?.level || 'low');
      return result;
    } catch (err: any) {
      setError(err.message);
      throw err;
    }
  }, []);

  // 自动监控
  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000); // 每30秒检查一次
    return () => clearInterval(interval);
  }, [fetchStatus]);

  return {
    depressionScore,
    depressionLevel,
    loading,
    error,
    checkBalance,
    fetchStatus
  };
}

// =============================================================================
// Confidence & Halt Hook
// =============================================================================

export function useConfidenceCalculator() {
  const [confidence, setConfidence] = useState<number>(0);
  const [isReliable, setIsReliable] = useState<boolean>(false);
  const [loading, setLoading] = useState(false);

  const calculate = useCallback(async (observation: any, prediction: any) => {
    setLoading(true);
    try {
      const result = await algorithmsAPI.calculateConfidence({
        observation,
        prediction
      });
      const data = result.data;
      setConfidence(data?.confidence || 0);
      setIsReliable(data?.is_reliable || false);
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  return { confidence, isReliable, loading, calculate };
}

export function useHaltDetector() {
  const [shouldHalt, setShouldHalt] = useState<boolean>(false);
  const [haltReason, setHaltReason] = useState<string>('');
  const [loading, setLoading] = useState(false);

  const checkHalt = useCallback(async (observations: any[], threshold: number = 0.5) => {
    setLoading(true);
    try {
      const result = await algorithmsAPI.checkHalt({
        observations,
        threshold
      });
      const data = result.data;
      setShouldHalt(data?.should_halt || false);
      setHaltReason(data?.halt_reason || '');
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  return { shouldHalt, haltReason, loading, checkHalt };
}

// =============================================================================
// Intrinsic Motivation Hook
// =============================================================================

export function useIntrinsicMotivation() {
  const [reward, setReward] = useState<number>(0);
  const [noveltyScore, setNoveltyScore] = useState<number>(0);
  const [loading, setLoading] = useState(false);

  const calculateReward = useCallback(async (
    currentState: any,
    nextState: any,
    motivationType: string = 'novelty'
  ) => {
    setLoading(true);
    try {
      const result = await algorithmsAPI.calculateMotivation({
        current_state: currentState,
        next_state: nextState,
        motivation_type: motivationType,
        extra_info: {}
      });
      const data = result.data;
      setReward(data?.intrinsic_reward || 0);
      setNoveltyScore(data?.novelty_score || 0);
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  return { reward, noveltyScore, loading, calculateReward };
}

// =============================================================================
// MoE Hook
// =============================================================================

export function useMixtureOfExperts() {
  const [selectedExpert, setSelectedExpert] = useState<string>('');
  const [routingWeights, setRoutingWeights] = useState<Record<string, number>>({});
  const [experts, setExperts] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const route = useCallback(async (inputData: any, availableExperts?: string[]) => {
    setLoading(true);
    try {
      const result = await algorithmsAPI.moeRoute({
        input_data: inputData,
        available_experts: availableExperts,
        temperature: 0.7
      });
      const data = result.data;
      setSelectedExpert(data?.selected_expert || '');
      setRoutingWeights(data?.routing_weights || {});
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchExperts = useCallback(async () => {
    try {
      const result = await algorithmsAPI.expertsList();
      setExperts(result.data || []);
      return result;
    } catch (err) {
      console.error('Failed to fetch experts:', err);
      throw err;
    }
  }, []);

  return {
    selectedExpert,
    routingWeights,
    experts,
    loading,
    route,
    fetchExperts
  };
}

// =============================================================================
// Pendulum Cycle Hook (胜复学闭环)
// =============================================================================

export function usePendulumCycle() {
  const [cycleResult, setCycleResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const executeCycle = useCallback(async (observation: any, goal: string) => {
    setLoading(true);
    setError(null);
    try {
      const result = await algorithmsAPI.pendulumCycle({
        observation,
        goal
      });
      setCycleResult(result);
      return result;
    } catch (err: any) {
      setError(err.message || 'Pendulum cycle failed');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return { cycleResult, loading, error, executeCycle };
}

// =============================================================================
// Algorithm Stats Hook
// =============================================================================

export function useAlgorithmStats() {
  const [stats, setStats] = useState<Record<string, any>>({});
  const [loading, setLoading] = useState(false);

  const fetchStats = useCallback(async () => {
    setLoading(true);
    try {
      const result = await algorithmsAPI.algorithmsStats();
      setStats(result.data || {});
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 60000); // 每分钟更新
    return () => clearInterval(interval);
  }, [fetchStats]);

  return { stats, loading, fetchStats };
}

// =============================================================================
// Goal Validation Hook
// =============================================================================

export function useGoalValidation() {
  const [validationResult, setValidationResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const validate = useCallback(async (goal: string, currentState?: Record<string, any>) => {
    setLoading(true);
    try {
      const result = await algorithmsAPI.goalValidate({
        goal,
        current_state: currentState || {}
      });
      setValidationResult(result.data);
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  return { validationResult, loading, validate };
}

// =============================================================================
// Burst Action Hook
// =============================================================================

export function useBurstAction() {
  const [actionResult, setActionResult] = useState<any>(null);
  const [availableActions, setAvailableActions] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const executeBurst = useCallback(async (depressionScore: number, actionType?: string) => {
    setLoading(true);
    try {
      const result = await algorithmsAPI.triggerBurst({
        depression_score: depressionScore,
        action_type: actionType,
        context: {}
      });
      setActionResult(result.data);
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchActions = useCallback(async () => {
    try {
      const result = await algorithmsAPI.burstActions();
      setAvailableActions(result.data || []);
      return result;
    } catch (err) {
      console.error('Failed to fetch burst actions:', err);
      throw err;
    }
  }, []);

  return {
    actionResult,
    availableActions,
    loading,
    executeBurst,
    fetchActions
  };
}

// =============================================================================
// Memory Hook
// =============================================================================

export function useAlgorithmMemory() {
  const [memoryId, setMemoryId] = useState<string>('');
  const [retrievalResults, setRetrievalResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const storeHot = useCallback(async (key: string, value: any) => {
    setLoading(true);
    try {
      const result = await algorithmsAPI.memoryHot({ key, value, memory_type: 'hot' });
      setMemoryId(result.data?.memory_id || '');
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  const storeWarm = useCallback(async (key: string, value: any) => {
    setLoading(true);
    try {
      const result = await algorithmsAPI.memoryWarm({ key, value, memory_type: 'warm' });
      setMemoryId(result.data?.memory_id || '');
      setRetrievalResults(result.data?.retrieval_results || []);
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  const storeCold = useCallback(async (key: string, value: any) => {
    setLoading(true);
    try {
      const result = await algorithmsAPI.memoryCold({ key, value, memory_type: 'cold' });
      setMemoryId(result.data?.memory_id || '');
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    memoryId,
    retrievalResults,
    loading,
    storeHot,
    storeWarm,
    storeCold
  };
}

// =============================================================================
// 导出所有Hooks
// =============================================================================

export const algorithmHooks = {
  useSwingEngine,
  useReflexion,
  useDepressionMonitor,
  useConfidenceCalculator,
  useHaltDetector,
  useIntrinsicMotivation,
  useMixtureOfExperts,
  usePendulumCycle,
  useAlgorithmStats,
  useGoalValidation,
  useBurstAction,
  useAlgorithmMemory
};

export default algorithmHooks;
