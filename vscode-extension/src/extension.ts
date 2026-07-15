import * as vscode from 'vscode';
import { ModelManagerProvider } from './panels/modelManagerProvider';
import { TrainingMonitorProvider } from './panels/trainingMonitorProvider';
import { AgentExplorerProvider } from './panels/agentExplorerProvider';
import { AgiCompletionProvider } from './providers/completionProvider';
import { AgiApiClient } from './api/client';
import { Logger } from './utils/logger';

let apiClient: AgiApiClient;
let logger: Logger;

export function activate(context: vscode.ExtensionContext) {
    logger = new Logger();
    logger.info('AGI Framework extension is now active!');

    // Initialize API client
    const config = vscode.workspace.getConfiguration('agiFramework');
    apiClient = new AgiApiClient({
        apiUrl: config.get<string>('apiUrl') || 'http://localhost:8000',
        apiKey: config.get<string>('apiKey') || '',
        wsUrl: config.get<string>('wsUrl') || 'ws://localhost:8000/ws',
        timeout: config.get<number>('timeout') || 30000
    });

    // Set context as enabled
    vscode.commands.executeCommand('setContext', 'agiFramework.enabled', true);

    // Register tree data providers
    const modelManagerProvider = new ModelManagerProvider(apiClient);
    const trainingMonitorProvider = new TrainingMonitorProvider(apiClient);
    const agentExplorerProvider = new AgentExplorerProvider(apiClient);

    vscode.window.registerTreeDataProvider('agiFramework.modelManager', modelManagerProvider);
    vscode.window.registerTreeDataProvider('agiFramework.trainingMonitor', trainingMonitorProvider);
    vscode.window.registerTreeDataProvider('agiFramework.agentExplorer', agentExplorerProvider);

    // Register completion provider
    const completionProvider = vscode.languages.registerCompletionItemProvider(
        ['python', 'javascript', 'typescript'],
        new AgiCompletionProvider(apiClient),
        '.'
    );

    // Register commands
    const commands = [
        vscode.commands.registerCommand('agiFramework.showModelManager', () => {
            vscode.commands.executeCommand('agiFramework.modelManager.focus');
        }),

        vscode.commands.registerCommand('agiFramework.showTrainingMonitor', () => {
            vscode.commands.executeCommand('agiFramework.trainingMonitor.focus');
        }),

        vscode.commands.registerCommand('agiFramework.showAgentPanel', () => {
            vscode.commands.executeCommand('agiFramework.agentExplorer.focus');
        }),

        vscode.commands.registerCommand('agiFramework.refreshModels', () => {
            modelManagerProvider.refresh();
            vscode.window.showInformationMessage('Models refreshed');
        }),

        vscode.commands.registerCommand('agiFramework.deployModel', async (model) => {
            try {
                await apiClient.deployModel(model.id);
                vscode.window.showInformationMessage(`Model ${model.name} deployed successfully`);
                modelManagerProvider.refresh();
            } catch (error) {
                vscode.window.showErrorMessage(`Failed to deploy model: ${error}`);
            }
        }),

        vscode.commands.registerCommand('agiFramework.trainModel', async (model) => {
            try {
                await apiClient.startTraining(model.id);
                vscode.window.showInformationMessage(`Training started for model ${model.name}`);
                trainingMonitorProvider.refresh();
            } catch (error) {
                vscode.window.showErrorMessage(`Failed to start training: ${error}`);
            }
        }),

        vscode.commands.registerCommand('agiFramework.stopTraining', async (job) => {
            try {
                await apiClient.stopTraining(job.id);
                vscode.window.showInformationMessage('Training stopped');
                trainingMonitorProvider.refresh();
            } catch (error) {
                vscode.window.showErrorMessage(`Failed to stop training: ${error}`);
            }
        }),

        vscode.commands.registerCommand('agiFramework.openSettings', () => {
            vscode.commands.executeCommand('workbench.action.openSettings', 'agiFramework');
        }),

        vscode.commands.registerCommand('agiFramework.createAgent', async () => {
            const name = await vscode.window.showInputBox({
                prompt: 'Enter agent name',
                placeHolder: 'my-agent'
            });
            if (name) {
                try {
                    await apiClient.createAgent(name);
                    vscode.window.showInformationMessage(`Agent ${name} created`);
                    agentExplorerProvider.refresh();
                } catch (error) {
                    vscode.window.showErrorMessage(`Failed to create agent: ${error}`);
                }
            }
        }),

        vscode.commands.registerCommand('agiFramework.runInference', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showErrorMessage('No active editor');
                return;
            }

            const selectedText = editor.document.getText(editor.selection);
            if (!selectedText) {
                vscode.window.showErrorMessage('No text selected');
                return;
            }

            try {
                const result = await apiClient.runInference(selectedText);
                const outputChannel = vscode.window.createOutputChannel('AGI Inference');
                outputChannel.appendLine(result);
                outputChannel.show();
            } catch (error) {
                vscode.window.showErrorMessage(`Inference failed: ${error}`);
            }
        })
    ];

    // Add all disposables to context
    context.subscriptions.push(
        completionProvider,
        ...commands
    );

    // Watch for configuration changes
    vscode.workspace.onDidChangeConfiguration(e => {
        if (e.affectsConfiguration('agiFramework')) {
            const newConfig = vscode.workspace.getConfiguration('agiFramework');
            apiClient.updateConfig({
                apiUrl: newConfig.get<string>('apiUrl') || 'http://localhost:8000',
                apiKey: newConfig.get<string>('apiKey') || '',
                wsUrl: newConfig.get<string>('wsUrl') || 'ws://localhost:8000/ws',
                timeout: newConfig.get<number>('timeout') || 30000
            });
            logger.info('Configuration updated');
        }
    });
}

export function deactivate() {
    logger?.info('AGI Framework extension is now deactivated');
}
