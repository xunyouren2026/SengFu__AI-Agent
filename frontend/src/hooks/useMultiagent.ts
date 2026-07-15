/**
 * 多智能体算法Hooks
 */

import { useState, useCallback } from 'react';
import { multiagentAPI } from '../api/multiagent';

export function useAlliance() {
  const [alliances, setAlliances] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const createAlliance = useCallback(async (params: {
    name: string;
    description: string;
    goals?: string[];
    max_members?: number;
  }) => {
    setLoading(true);
    try {
      const result = await multiagentAPI.createAlliance(params);
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  const joinAlliance = useCallback(async (allianceId: string, agentId: string, contribution: string) => {
    setLoading(true);
    try {
      const result = await multiagentAPI.joinAlliance(allianceId, {
        agent_id: agentId,
        contribution
      });
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  const vote = useCallback(async (allianceId: string, voterId: string, vote: string, reason?: string) => {
    setLoading(true);
    try {
      const result = await multiagentAPI.allianceVote(allianceId, {
        voter_id: voterId,
        vote: vote as any,
        reason
      });
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchAlliances = useCallback(async () => {
    setLoading(true);
    try {
      const result = await multiagentAPI.listAlliances();
      setAlliances(result.data || []);
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  return { alliances, loading, createAlliance, joinAlliance, vote, fetchAlliances };
}

export function useDebate() {
  const [debates, setDebates] = useState<any[]>([]);
  const [currentDebate, setCurrentDebate] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const startDebate = useCallback(async (topic: string, participants: string[]) => {
    setLoading(true);
    try {
      const result = await multiagentAPI.startDebate({
        topic,
        participants,
        debate_type: 'formal'
      });
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  const submitArgument = useCallback(async (debateId: string, participantId: string, content: string, evidence?: string[]) => {
    setLoading(true);
    try {
      const result = await multiagentAPI.submitArgument(debateId, {
        participant_id: participantId,
        content,
        evidence
      });
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  const getResult = useCallback(async (debateId: string) => {
    setLoading(true);
    try {
      const result = await multiagentAPI.getDebateResult(debateId);
      setCurrentDebate(result.data);
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchDebates = useCallback(async () => {
    setLoading(true);
    try {
      const result = await multiagentAPI.listDebates();
      setDebates(result.data || []);
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  return { debates, currentDebate, loading, startDebate, submitArgument, getResult, fetchDebates };
}

export function useReputation() {
  const [reputation, setReputation] = useState<any>(null);
  const [leaderboard, setLeaderboard] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const getReputation = useCallback(async (agentId: string) => {
    setLoading(true);
    try {
      const result = await multiagentAPI.getReputation(agentId);
      setReputation(result.data);
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  const updateReputation = useCallback(async (agentId: string, delta: number, reason: string) => {
    setLoading(true);
    try {
      const result = await multiagentAPI.updateReputation({
        agent_id: agentId,
        delta,
        reason
      });
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchLeaderboard = useCallback(async (limit: number = 10) => {
    setLoading(true);
    try {
      const result = await multiagentAPI.reputationLeaderboard(limit);
      setLeaderboard(result.data || []);
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  return { reputation, leaderboard, loading, getReputation, updateReputation, fetchLeaderboard };
}

export const multiagentHooks = {
  useAlliance,
  useDebate,
  useReputation
};

export default multiagentHooks;
