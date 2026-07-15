import * as vscode from 'vscode';
import { AgiApiClient, Agent } from '../api/client';

export class AgentItem extends vscode.TreeItem {
    constructor(
        public readonly agent: Agent,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState
    ) {
        super(agent.name, collapsibleState);
        
        this.tooltip = this.getTooltip();
        this.description = `${agent.status} | ${agent.capabilities.length} capabilities`;
        
        // Set icon based on status
        switch (agent.status) {
            case 'idle':
                this.iconPath = new vscode.ThemeIcon('circle-filled', new vscode.ThemeColor('testing.iconPassed'));
                break;
            case 'busy':
                this.iconPath = new vscode.ThemeIcon('sync~spin', new vscode.ThemeColor('testing.iconQueued'));
                break;
            case 'error':
                this.iconPath = new vscode.ThemeIcon('error', new vscode.ThemeColor('testing.iconFailed'));
                break;
            default:
                this.iconPath = new vscode.ThemeIcon('question');
        }
        
        this.contextValue = 'agent';
    }

    private getTooltip(): string {
        const lines = [
            `Agent: ${this.agent.name}`,
            `ID: ${this.agent.id}`,
            `Status: ${this.agent.status}`,
            `Capabilities: ${this.agent.capabilities.join(', ')}`,
            `Last Active: ${new Date(this.agent.lastActive).toLocaleString()}`
        ];
        return lines.join('\n');
    }
}

export class AgentExplorerProvider implements vscode.TreeDataProvider<AgentItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<AgentItem | undefined | null | void> = new vscode.EventEmitter<AgentItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<AgentItem | undefined | null | void> = this._onDidChangeTreeData.event;

    private agents: Agent[] = [];
    private refreshInterval: NodeJS.Timeout | null = null;

    constructor(private apiClient: AgiApiClient) {
        this.refresh();
        this.startAutoRefresh();
    }

    refresh(): void {
        this._onDidChangeTreeData.fire();
        this.loadAgents();
    }

    private startAutoRefresh(): void {
        // Refresh every 10 seconds
        this.refreshInterval = setInterval(() => {
            this.loadAgents();
        }, 10000);
    }

    private async loadAgents(): Promise<void> {
        try {
            this.agents = await this.apiClient.getAgents();
            this._onDidChangeTreeData.fire();
        } catch (error) {
            console.error('Failed to load agents:', error);
        }
    }

    getTreeItem(element: AgentItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: AgentItem): Thenable<vscode.TreeItem[]> {
        if (element) {
            // Return agent details as children
            const items: vscode.TreeItem[] = [
                new AgentDetailItem('ID', element.agent.id),
                new AgentDetailItem('Status', element.agent.status),
                new AgentDetailItem('Last Active', new Date(element.agent.lastActive).toLocaleString())
            ];
            
            if (element.agent.capabilities.length > 0) {
                items.push(new AgentDetailItem('Capabilities', element.agent.capabilities.join(', ')));
            }
            
            return Promise.resolve(items);
        }

        // Group agents by status
        const idleAgents = this.agents.filter(a => a.status === 'idle');
        const busyAgents = this.agents.filter(a => a.status === 'busy');
        const errorAgents = this.agents.filter(a => a.status === 'error');

        const items: vscode.TreeItem[] = [];

        if (idleAgents.length > 0) {
            items.push(new AgentCategoryItem('Idle', idleAgents.length, 'testing.iconPassed'));
            items.push(...idleAgents.map(agent => 
                new AgentItem(agent, vscode.TreeItemCollapsibleState.Collapsed)
            ));
        }

        if (busyAgents.length > 0) {
            items.push(new AgentCategoryItem('Busy', busyAgents.length, 'testing.iconQueued'));
            items.push(...busyAgents.map(agent => 
                new AgentItem(agent, vscode.TreeItemCollapsibleState.Collapsed)
            ));
        }

        if (errorAgents.length > 0) {
            items.push(new AgentCategoryItem('Error', errorAgents.length, 'testing.iconFailed'));
            items.push(...errorAgents.map(agent => 
                new AgentItem(agent, vscode.TreeItemCollapsibleState.Collapsed)
            ));
        }

        if (items.length === 0) {
            const emptyItem = new vscode.TreeItem('No agents found');
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

class AgentCategoryItem extends vscode.TreeItem {
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

class AgentDetailItem extends vscode.TreeItem {
    constructor(
        label: string,
        value: string
    ) {
        super(`${label}: ${value}`, vscode.TreeItemCollapsibleState.None);
        this.iconPath = new vscode.ThemeIcon('info');
        this.contextValue = 'detail';
    }
}
