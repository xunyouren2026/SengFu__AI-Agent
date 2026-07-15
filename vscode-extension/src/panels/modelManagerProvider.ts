import * as vscode from 'vscode';
import { AgiApiClient, Model } from '../api/client';

export class ModelItem extends vscode.TreeItem {
    constructor(
        public readonly model: Model,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState
    ) {
        super(model.name, collapsibleState);
        
        this.tooltip = `${model.name} (${model.version}) - ${model.status}`;
        this.description = `${model.version} | ${model.type}`;
        
        // Set icon based on status
        switch (model.status) {
            case 'active':
                this.iconPath = new vscode.ThemeIcon('check', new vscode.ThemeColor('testing.iconPassed'));
                break;
            case 'training':
                this.iconPath = new vscode.ThemeIcon('sync~spin', new vscode.ThemeColor('testing.iconQueued'));
                break;
            case 'deploying':
                this.iconPath = new vscode.ThemeIcon('cloud-upload', new vscode.ThemeColor('testing.iconQueued'));
                break;
            case 'inactive':
                this.iconPath = new vscode.ThemeIcon('circle-outline', new vscode.ThemeColor('disabledForeground'));
                break;
            default:
                this.iconPath = new vscode.ThemeIcon('question');
        }
        
        this.contextValue = 'model';
        
        // Format size
        const sizeStr = model.size < 1024 * 1024 
            ? `${(model.size / 1024).toFixed(2)} KB`
            : `${(model.size / (1024 * 1024)).toFixed(2)} MB`;
        
        this.detail = `Size: ${sizeStr} | Created: ${new Date(model.createdAt).toLocaleDateString()}`;
    }
}

export class ModelManagerProvider implements vscode.TreeDataProvider<ModelItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<ModelItem | undefined | null | void> = new vscode.EventEmitter<ModelItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<ModelItem | undefined | null | void> = this._onDidChangeTreeData.event;

    private models: Model[] = [];
    private refreshInterval: NodeJS.Timeout | null = null;

    constructor(private apiClient: AgiApiClient) {
        this.refresh();
        this.startAutoRefresh();
        
        // Listen for model updates from WebSocket
        this.apiClient.on('modelUpdate', () => {
            this.refresh();
        });
    }

    refresh(): void {
        this._onDidChangeTreeData.fire();
        this.loadModels();
    }

    private startAutoRefresh(): void {
        // Refresh every 30 seconds
        this.refreshInterval = setInterval(() => {
            this.loadModels();
        }, 30000);
    }

    private async loadModels(): Promise<void> {
        try {
            this.models = await this.apiClient.getModels();
            this._onDidChangeTreeData.fire();
        } catch (error) {
            console.error('Failed to load models:', error);
        }
    }

    getTreeItem(element: ModelItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: ModelItem): Thenable<ModelItem[]> {
        if (element) {
            // Return details as children
            return Promise.resolve([
                new ModelDetailItem('ID', element.model.id),
                new ModelDetailItem('Version', element.model.version),
                new ModelDetailItem('Type', element.model.type),
                new ModelDetailItem('Status', element.model.status),
                new ModelDetailItem('Size', `${(element.model.size / (1024 * 1024)).toFixed(2)} MB`),
                new ModelDetailItem('Created', new Date(element.model.createdAt).toLocaleString()),
                new ModelDetailItem('Updated', new Date(element.model.updatedAt).toLocaleString())
            ]);
        }

        // Group models by status
        const activeModels = this.models.filter(m => m.status === 'active');
        const trainingModels = this.models.filter(m => m.status === 'training');
        const deployingModels = this.models.filter(m => m.status === 'deploying');
        const inactiveModels = this.models.filter(m => m.status === 'inactive');

        const items: ModelItem[] = [];

        if (trainingModels.length > 0) {
            items.push(new ModelCategoryItem('Training', trainingModels.length, 'testing.iconQueued'));
        }
        if (deployingModels.length > 0) {
            items.push(new ModelCategoryItem('Deploying', deployingModels.length, 'testing.iconQueued'));
        }
        if (activeModels.length > 0) {
            items.push(new ModelCategoryItem('Active', activeModels.length, 'testing.iconPassed'));
        }
        if (inactiveModels.length > 0) {
            items.push(new ModelCategoryItem('Inactive', inactiveModels.length, 'disabledForeground'));
        }

        // Add all models
        items.push(...this.models.map(model => 
            new ModelItem(model, vscode.TreeItemCollapsibleState.Collapsed)
        ));

        return Promise.resolve(items);
    }

    dispose(): void {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
    }
}

class ModelCategoryItem extends vscode.TreeItem {
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

class ModelDetailItem extends vscode.TreeItem {
    constructor(
        label: string,
        value: string
    ) {
        super(`${label}: ${value}`, vscode.TreeItemCollapsibleState.None);
        this.iconPath = new vscode.ThemeIcon('info');
        this.contextValue = 'detail';
    }
}
