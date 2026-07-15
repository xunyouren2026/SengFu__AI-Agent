import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';
import WebSocket from 'ws';
import { EventEmitter } from 'events';

export interface ApiConfig {
    apiUrl: string;
    apiKey: string;
    wsUrl: string;
    timeout: number;
}

export interface Model {
    id: string;
    name: string;
    version: string;
    status: 'active' | 'inactive' | 'training' | 'deploying';
    type: string;
    size: number;
    createdAt: string;
    updatedAt: string;
}

export interface TrainingJob {
    id: string;
    modelId: string;
    modelName: string;
    status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped';
    progress: number;
    epoch: number;
    totalEpochs: number;
    loss: number;
    accuracy: number;
    startedAt: string;
    estimatedEndTime?: string;
}

export interface Agent {
    id: string;
    name: string;
    status: 'idle' | 'busy' | 'error';
    capabilities: string[];
    lastActive: string;
}

export class AgiApiClient extends EventEmitter {
    private httpClient: AxiosInstance;
    private wsClient: WebSocket | null = null;
    private config: ApiConfig;

    constructor(config: ApiConfig) {
        super();
        this.config = config;
        this.httpClient = this.createHttpClient();
        this.connectWebSocket();
    }

    private createHttpClient(): AxiosInstance {
        const client = axios.create({
            baseURL: this.config.apiUrl,
            timeout: this.config.timeout,
            headers: {
                'Content-Type': 'application/json'
            }
        });

        client.interceptors.request.use((config) => {
            if (this.config.apiKey) {
                config.headers.Authorization = `Bearer ${this.config.apiKey}`;
            }
            return config;
        });

        return client;
    }

    private connectWebSocket(): void {
        try {
            this.wsClient = new WebSocket(this.config.wsUrl, {
                headers: this.config.apiKey ? { Authorization: `Bearer ${this.config.apiKey}` } : {}
            });

            this.wsClient.on('open', () => {
                this.emit('connected');
            });

            this.wsClient.on('message', (data: WebSocket.Data) => {
                try {
                    const message = JSON.parse(data.toString());
                    this.emit('message', message);
                    
                    if (message.type === 'training_update') {
                        this.emit('trainingUpdate', message.data);
                    } else if (message.type === 'model_update') {
                        this.emit('modelUpdate', message.data);
                    }
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            });

            this.wsClient.on('error', (error) => {
                this.emit('error', error);
            });

            this.wsClient.on('close', () => {
                this.emit('disconnected');
                // Reconnect after 5 seconds
                setTimeout(() => this.connectWebSocket(), 5000);
            });
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
        }
    }

    updateConfig(config: Partial<ApiConfig>): void {
        this.config = { ...this.config, ...config };
        this.httpClient = this.createHttpClient();
        
        if (this.wsClient) {
            this.wsClient.close();
        }
        this.connectWebSocket();
    }

    // Model API
    async getModels(): Promise<Model[]> {
        const response = await this.httpClient.get('/api/v1/models');
        return response.data;
    }

    async getModel(id: string): Promise<Model> {
        const response = await this.httpClient.get(`/api/v1/models/${id}`);
        return response.data;
    }

    async deployModel(id: string): Promise<void> {
        await this.httpClient.post(`/api/v1/models/${id}/deploy`);
    }

    async deleteModel(id: string): Promise<void> {
        await this.httpClient.delete(`/api/v1/models/${id}`);
    }

    // Training API
    async getTrainingJobs(): Promise<TrainingJob[]> {
        const response = await this.httpClient.get('/api/v1/training/jobs');
        return response.data;
    }

    async startTraining(modelId: string, config?: Record<string, any>): Promise<TrainingJob> {
        const response = await this.httpClient.post('/api/v1/training/jobs', {
            modelId,
            config
        });
        return response.data;
    }

    async stopTraining(jobId: string): Promise<void> {
        await this.httpClient.post(`/api/v1/training/jobs/${jobId}/stop`);
    }

    async getTrainingLogs(jobId: string): Promise<string[]> {
        const response = await this.httpClient.get(`/api/v1/training/jobs/${jobId}/logs`);
        return response.data;
    }

    // Agent API
    async getAgents(): Promise<Agent[]> {
        const response = await this.httpClient.get('/api/v1/agents');
        return response.data;
    }

    async createAgent(name: string, config?: Record<string, any>): Promise<Agent> {
        const response = await this.httpClient.post('/api/v1/agents', {
            name,
            config
        });
        return response.data;
    }

    async deleteAgent(id: string): Promise<void> {
        await this.httpClient.delete(`/api/v1/agents/${id}`);
    }

    // Inference API
    async runInference(prompt: string, options?: Record<string, any>): Promise<string> {
        const response = await this.httpClient.post('/api/v1/inference', {
            prompt,
            ...options
        });
        return response.data.result;
    }

    async getCompletions(code: string, language: string): Promise<string[]> {
        const response = await this.httpClient.post('/api/v1/completions', {
            code,
            language
        });
        return response.data.completions;
    }

    // Health check
    async healthCheck(): Promise<boolean> {
        try {
            const response = await this.httpClient.get('/health');
            return response.status === 200;
        } catch {
            return false;
        }
    }

    dispose(): void {
        if (this.wsClient) {
            this.wsClient.close();
        }
    }
}
