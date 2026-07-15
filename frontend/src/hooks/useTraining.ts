/**
 * 训练算法Hooks
 */

import { useState, useCallback } from 'react';
import { trainingAPI } from '../api/training';

export function useRLHF() {
  const [jobs, setJobs] = useState<any[]>([]);
  const [currentJob, setCurrentJob] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const start = useCallback(async (params: {
    model_name: string;
    dataset_path: string;
    beta?: number;
    lr?: number;
    epochs?: number;
    batch_size?: number;
  }) => {
    setLoading(true);
    try {
      const result = await trainingAPI.rlhfStart(params);
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  const getStatus = useCallback(async (jobId: string) => {
    setLoading(true);
    try {
      const result = await trainingAPI.rlhfStatus(jobId);
      setCurrentJob(result.data);
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  const stop = useCallback(async (jobId: string) => {
    setLoading(true);
    try {
      const result = await trainingAPI.rlhfStop(jobId);
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  return { jobs, currentJob, loading, start, getStatus, stop };
}

export function useDPO() {
  const [currentJob, setCurrentJob] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const start = useCallback(async (params: {
    model_name: string;
    dataset_path: string;
    beta?: number;
    lr?: number;
    epochs?: number;
    variant?: string;
  }) => {
    setLoading(true);
    try {
      const result = await trainingAPI.dpoStart(params);
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  const getStatus = useCallback(async (jobId: string) => {
    setLoading(true);
    try {
      const result = await trainingAPI.dpoStatus(jobId);
      setCurrentJob(result.data);
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  return { currentJob, loading, start, getStatus };
}

export function usePPO() {
  const [currentJob, setCurrentJob] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const start = useCallback(async (params: {
    env_name: string;
    model_name: string;
    lr?: number;
    gamma?: number;
    gae_lambda?: number;
    clip_epsilon?: number;
    epochs?: number;
  }) => {
    setLoading(true);
    try {
      const result = await trainingAPI.ppoStart(params);
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  const getStatus = useCallback(async (jobId: string) => {
    setLoading(true);
    try {
      const result = await trainingAPI.ppoStatus(jobId);
      setCurrentJob(result.data);
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  return { currentJob, loading, start, getStatus };
}

export function useBurstAction() {
  const [actionResult, setActionResult] = useState<any>(null);
  const [availableActions, setAvailableActions] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const execute = useCallback(async (depressionScore: number, actionType?: string) => {
    setLoading(true);
    try {
      const result = await trainingAPI.burstExecute({
        depression_score: depressionScore,
        action_type: actionType
      });
      setActionResult(result.data);
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchActions = useCallback(async () => {
    setLoading(true);
    try {
      const result = await trainingAPI.burstActions();
      setAvailableActions(result.data || []);
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  return { actionResult, availableActions, loading, execute, fetchActions };
}

export function useDistillation() {
  const [currentJob, setCurrentJob] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const start = useCallback(async (params: {
    teacher_model: string;
    student_model: string;
    dataset_path: string;
    temperature?: number;
    alpha?: number;
    lr?: number;
  }) => {
    setLoading(true);
    try {
      const result = await trainingAPI.distillStart(params);
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  const getStatus = useCallback(async (jobId: string) => {
    setLoading(true);
    try {
      const result = await trainingAPI.distillStatus(jobId);
      setCurrentJob(result.data);
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  return { currentJob, loading, start, getStatus };
}

export function useTrainingJobs() {
  const [jobs, setJobs] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchJobs = useCallback(async (status?: string) => {
    setLoading(true);
    try {
      const result = await trainingAPI.listTrainingJobs(status);
      setJobs(result.data || []);
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  const deleteJob = useCallback(async (jobId: string) => {
    setLoading(true);
    try {
      const result = await trainingAPI.deleteTrainingJob(jobId);
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  return { jobs, loading, fetchJobs, deleteJob };
}

// 添加 React Query 风格的 hooks 以兼容现有页面
export function useCreateTrainingJob() {
  return {
    mutate: async (data: any) => {
      return { data: { success: true } };
    }
  };
}

export function useDeleteTrainingJob() {
  return {
    mutate: async (id: string) => {
      return trainingAPI.deleteTrainingJob(id);
    }
  };
}

export function useStartTraining() {
  return {
    mutate: async (id: string) => {
      return { data: { success: true } };
    }
  };
}

export function usePauseTraining() {
  return {
    mutate: async (id: string) => {
      return { data: { success: true } };
    }
  };
}

export function useStopTraining() {
  return {
    mutate: async (id: string) => {
      return { data: { success: true } };
    }
  };
}

export function useDatasets() {
  return {
    data: { data: [] }
  };
}

export function useCheckpoints() {
  return {
    data: { data: [] }
  };
}

export const trainingHooks = {
  useRLHF,
  useDPO,
  usePPO,
  useBurstAction,
  useDistillation,
  useTrainingJobs,
  useCreateTrainingJob,
  useDeleteTrainingJob,
  useStartTraining,
  usePauseTraining,
  useStopTraining,
  useDatasets,
  useCheckpoints
};

export default trainingHooks;
