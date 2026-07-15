import * as vscode from 'vscode';

export class Logger {
    private outputChannel: vscode.OutputChannel;

    constructor() {
        this.outputChannel = vscode.window.createOutputChannel('AGI Framework');
    }

    private getTimestamp(): string {
        return new Date().toISOString();
    }

    private log(level: string, message: string): void {
        const timestamp = this.getTimestamp();
        this.outputChannel.appendLine(`[${timestamp}] [${level}] ${message}`);
    }

    debug(message: string): void {
        const config = vscode.workspace.getConfiguration('agiFramework');
        if (config.get<string>('logLevel') === 'debug') {
            this.log('DEBUG', message);
        }
    }

    info(message: string): void {
        this.log('INFO', message);
    }

    warn(message: string): void {
        this.log('WARN', message);
    }

    error(message: string): void {
        this.log('ERROR', message);
    }

    show(): void {
        this.outputChannel.show();
    }

    dispose(): void {
        this.outputChannel.dispose();
    }
}
