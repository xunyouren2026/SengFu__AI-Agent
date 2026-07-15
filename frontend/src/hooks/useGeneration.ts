import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { generationApi } from '../api/generation';
import type { TTSRequest, ImageGenerationRequest, VideoGenerationRequest, AudioGenerationRequest, ThreeDGenerationRequest } from '../types/generation';

export const generationKeys = {
  all: ['generation'] as const,
  tasks: () => [...generationKeys.all, 'tasks'] as const,
  task: (id: string) => [...generationKeys.all, 'task', id] as const,
  stats: () => [...generationKeys.all, 'stats'] as const,
};

export const useVoices = (params?: Record<string, unknown>) =>
  useQuery({ queryKey: [...generationKeys.all, 'voices', params], queryFn: () => generationApi.getVoices(params) });

export const useSynthesize = () =>
  useMutation({ mutationFn: (data: TTSRequest) => generationApi.synthesize(data) });

export const useGenerateImage = () =>
  useMutation({ mutationFn: (data: ImageGenerationRequest) => generationApi.generateImage(data) });

export const useGenerateVideo = () =>
  useMutation({ mutationFn: (data: VideoGenerationRequest) => generationApi.generateVideo(data) });

export const useGenerateAudio = () =>
  useMutation({ mutationFn: (data: AudioGenerationRequest) => generationApi.generateAudio(data) });

export const useGenerate3D = () =>
  useMutation({ mutationFn: (data: ThreeDGenerationRequest) => generationApi.generate3D(data) });

export const useGenerationTasks = (params?: Record<string, unknown>) =>
  useQuery({ queryKey: [...generationKeys.tasks(), params], queryFn: () => generationApi.getTasks(params), refetchInterval: 5000 });

export const useGenerationTask = (id: string) =>
  useQuery({ queryKey: generationKeys.task(id), queryFn: () => generationApi.getTask(id), enabled: !!id, refetchInterval: 3000 });

export const useGenerationStats = () =>
  useQuery({ queryKey: generationKeys.stats(), queryFn: () => generationApi.getStats(), staleTime: 30000 });
