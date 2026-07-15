import * as vscode from 'vscode';
import { AgiApiClient, TrainingJob } from '../api/client';

export class TrainingJobItem extends vscode.TreeItem {
    constructor(
        public readonly job: TrainingJob,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState
    ) {
        super(job.modelName, collapsibleState);
        
        this.tooltip = this.getTooltip();
        this.description = this.getDescription();
        
        // Set icon based on status
        switch (job.status) {
            case 'running':
                this.iconPath = new vscode.ThemeIcon('sync~spin', new vscode.ThemeColor('testing.iconQueued'));
                break;
            case 'completed':
                this.iconPath = new vscode.ThemeIcon('check', new vscode.ThemeColor('testing.iconPassed'));
                break;
            case 'failed':
                this.iconPath = new vscode.ThemeIcon('error', new vscode.ThemeColor('testing.iconFailed'));
                break;
            case 'stopped':
                this.iconPath = new vscode.ThemeIcon('stop-circle', new vscode.ThemeColor('disabledForeground'));
                break;
            case 'pending':
                this.iconPath = new vscode.ThemeIcon('clock', new vscode.ThemeColor('testing.iconUnset'));
                break;
            default:
                this.iconPath = new vscode.ThemeIcon('question');
        }
        
        this.contextValue = 'trainingJob';
    }

    private getTooltip(): string {
        const lines = [
            `Job ID: ${this.job.id}`,
            `Model: ${this.job.modelName}`,
            `Status: ${this.job.status}`,
            `Progress: ${(this.job.progress * 100).toFixed(1)}%`,
            `Epoch: ${this.job.epoch}/${this.job.totalEpochs}`,
            `Loss: ${this.job.loss?.toFixed(4) || 'N/A'}`,
            `Accuracy: ${this.job.accuracy ? (this.job.accuracy * 100).toFixed(2) + '%' : 'N/A'}`
        ];
        return lines.join('\n');
    }

    private getDescription(): string {
        if (this.job.status === 'running') {
            return `${(this.job.progress * 100).toFixed(0)}% | Epoch ${this.job.epoch}/${this.job.totalEpochs}`;
        }
        return this.job.status;
    }
}

export class TrainingMonitorProvider implements vscode.TreeDataProvider<TrainingJobItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<TrainingJobItem | undefined | null | void> = new vscode.EventEmitter<TrainingJobItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<TrainingJobItem | undefined | null | void> = this._onDidChangeTreeData.event;

    private jobs: TrainingJob[] = [];
    private refreshInterval: NodeJS.Timeout | null = null;

    constructor(private apiClient: AgiApiClient) {
        this.refresh();
        this.startAutoRefresh();
        
        // Listen for training updates from WebSocket
        this.apiClient.on('trainingUpdate', (data: TrainingJob) => {
            const index = this.jobs.findIndex(j => j.id === data.id);
            if (index !== -1) {
                this.jobs[index] = data;
            } else {
                this.jobs.push(data);
            }
            this._onDidChangeTreeData.fire();
        });
    }

    refresh(): void {
        this._onDidChangeTreeData.fire();
        this.loadJobs();
    }

    private startAutoRefresh(): void {
        // Refresh every 5 seconds for more responsive updates
        this.refreshInterval = setInterval(() => {
            this.loadJobs();
        }, 5000);
    }

    private async loadJobs(): Promise<void> {
        try {
            this.jobs = await this.apiClient.getTrainingJobs();
            this._onDidChangeTreeData.fire();
        } catch (error) {
            console.error('Failed to load training jobs:', error);
        }
    }

    getTreeItem(element: TrainingJobItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: TrainingJobItem): Thenable<vscode.TreeItem[]> {
        if (element) {
            // Return job details as children
            const items: vscode.TreeItem[] = [
                new TrainingDetailItem('Job ID', element.job.id),
                new TrainingDetailItem('Model ID', element.job.modelId),
                new TrainingDetailItem('Status', element.job.status),
                new TrainingDetailItem('Progress', `${(element.job.progress * 100).toFixed(2)}%`),
                new TrainingDetailItem('Epoch', `${element.job.epoch} / ${element.job.totalEpochs}`),
                new TrainingDetailItem('Loss', element.job.loss?.toFixed(6) || 'N/A'),
                new TrainingDetailItem('Accuracy', element.job.accuracy ? (element.job.accuracy * 100).toFixed(2) + '%' : 'N/A'),
                new TrainingDetailItem('Started', new Date(element.job.startedAt).toLocaleString())
            ];
            
            if (element.job.estimatedEndTime) {
                items.push(new TrainingDetailItem('Est. End', new Date(element.job.estimatedEndTime).toLocaleString()));
            }
            
            return Promise.resolve(items);
        }

        // Group jobs by status
        const runningJobs = this.jobs.filter(j => j.status === 'running');
        const pendingJobs = this.jobs.filter(j => j.status === 'pending');
        const completedJobs = this.jobs.filter(j => j.status === 'completed');
        const failedJobs = this.jobs.filter(j => j.status === 'failed');
        const stoppedJobs = this.jobs.filter(j => j.status === 'stopped');

        const items: vscode.TreeItem[] = [];

        if (runningJobs.length > 0) {
            items.push(new TrainingCategoryItem('Running', runningJobs.length, 'testing.iconQueued'));
            items.push(...runningJobs.map(job => 
                new TrainingJobItem(job, vscode.TreeItemCollapsibleState.Collapsed)
            ));
        }

        if (pendingJobs.length > 0) {
            items.push(new TrainingCategoryItem('Pending', pendingJobs.length, 'testing.iconUnset'));
            items.push(...pendingJobs.map(job => 
                new TrainingJobItem(job, vscode.TreeItemCollapsibleState.Collapsed)
            ));
        }

        if (completedJobs.length > 0) {
            items.push(new TrainingCategoryItem('Completed', completedJobs.length, 'testing.iconPassed'));
            items.push(...completedJobs.map(job => 
                new TrainingJobItem(job, vscode.TreeItemCollapsibleState.Collapsed)
            ));
        }

        if (failedJobs.length > 0) {
            items.push(new TrainingCategoryItem('Failed', failedJobs.length, 'testing.iconFailed'));
            items.push(...failedJobs.map(job => 
                new TrainingJobItem(job, vscode.TreeItemCollapsibleState.Collapsed)
            ));
        }

        if (stoppedJobs.length > 0) {
            items.push(new TrainingCategoryItem('Stopped', stoppedJobs.length, 'disabledForeground'));
            items.push(...stoppedJobs.map(job => 
                new TrainingJobItem(job, vscode.TreeItemCollapsibleState.Collapsed)
            ));
        }

        if (items.length === 0) {
            const emptyItem = new vscode.TreeItem('No training jobs');
            emptyItem.iconPath = new vscode.ThemeIcon('info');
            items.push(emptyItem);
        }

        return Promise.resolve(items);
    }

    dispose(): void {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
    }
}

class TrainingCategoryItem extends vscode.TreeItem {
    constructor(
        label: string,
        count: number,
        colorId: string
    ) {
        super(`${label} (${count})`, vscode.TreeItemCollapsibleState.None);
        this.iconPath = new vscode.ThemeIcon('folder', new vscode.ThemeColor(colorId));
        this.contextValue = 'category';
    }
}

class TrainingDetailItem extends vscode.TreeItem {
    constructor(
        label: string,
        value: string
    ) {
        super(`${label}: ${value}`, vscode.TreeItemCollapsibleState.None);
        this.iconPath = new vscode.ThemeIcon('info');
        this.contextValue = 'detail';
    }
}
