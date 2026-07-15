import apiClient from './client';
import { wrapListResponse } from './utils';
import type { TTSRequest, TTSResponse, ImageGenerationRequest, ImageGenerationResponse, VideoGenerationRequest, VideoGenerationResponse, AudioGenerationRequest, AudioGenerationResponse, ThreeDGenerationRequest, ThreeDGenerationResponse, GenerationTask, GenerationStats } from '../types/generation';

export const generationApi = {
  // TTS
  synthesize: (data: TTSRequest) =>
    apiClient.post<TTSResponse>('/generation/tts', data),
  getVoices: (params?: { language?: string; engine?: string }) =>
    apiClient.get('/generation/tts/voices', { params }),
  // Image
  generateImage: (data: ImageGenerationRequest) =>
    apiClient.post<ImageGenerationResponse>('/generation/image', data),
  img2img: (file: File, params?: Record<string, unknown>) => {
    const formData = new FormData();
    formData.append('file', file);
    if (params) Object.entries(params).forEach(([k, v]) => formData.append(k, String(v)));
    return apiClient.post<ImageGenerationResponse>('/generation/image/img2img', formData, { headers: { 'Content-Type': 'multipart/form-data' } });
  },
  getImageModels: (params?: Record<string, unknown>) =>
    apiClient.get('/generation/image/models', { params }),
  // Video
  generateVideo: (data: VideoGenerationRequest) =>
    apiClient.post<VideoGenerationResponse>('/generation/video', data),
  img2vid: (file: File, params?: Record<string, unknown>) => {
    const formData = new FormData();
    formData.append('file', file);
    if (params) Object.entries(params).forEach(([k, v]) => formData.append(k, String(v)));
    return apiClient.post<VideoGenerationResponse>('/generation/video/img2vid', formData, { headers: { 'Content-Type': 'multipart/form-data' } });
  },
  // Audio
  generateAudio: (data: AudioGenerationRequest) =>
    apiClient.post<AudioGenerationResponse>('/generation/audio', data),
  // 3D
  generate3D: (data: ThreeDGenerationRequest) =>
    apiClient.post<ThreeDGenerationResponse>('/generation/3d', data),
  img2obj: (file: File, params?: Record<string, unknown>) => {
    const formData = new FormData();
    formData.append('file', file);
    if (params) Object.entries(params).forEach(([k, v]) => formData.append(k, String(v)));
    return apiClient.post('/generation/3d/img2obj', formData, { headers: { 'Content-Type': 'multipart/form-data' } });
  },
  // Tasks
  submitTask: (type: string, prompt: string, config?: Record<string, unknown>, priority?: number) =>
    apiClient.post('/generation/task/submit', null, { params: { type, prompt, config: JSON.stringify(config), priority } }),
  getTask: (taskId: string) =>
    apiClient.get<GenerationTask>(`/generation/task/${taskId}`),
  getTaskProgress: (taskId: string) =>
    apiClient.get(`/generation/task/${taskId}/progress`),
  deleteTask: (taskId: string) =>
    apiClient.delete(`/generation/task/${taskId}`),
  getTasks: async (params?: Record<string, unknown>) => {
    const response = await apiClient.get('/generation/tasks', { params });
    // 页面使用 tasks?.tasks 访问
    if (Array.isArray(response)) {
      return { tasks: response };
    }
    if (response && typeof response === 'object' && 'tasks' in response) {
      return response;
    }
    return { tasks: Array.isArray(response) ? response : [response] };
  },
  getStats: () =>
    apiClient.get<GenerationStats>('/generation/stats'),
};
