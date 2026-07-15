import * as vscode from 'vscode';
import { AgiApiClient } from '../api/client';

export class AgiCompletionProvider implements vscode.CompletionItemProvider {
    constructor(private apiClient: AgiApiClient) {}

    async provideCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
        token: vscode.CancellationToken,
        context: vscode.CompletionContext
    ): Promise<vscode.CompletionItem[]> {
        const config = vscode.workspace.getConfiguration('agiFramework');
        if (!config.get<boolean>('enableAutoComplete')) {
            return [];
        }

        // Get the current line text up to the cursor
        const lineText = document.lineAt(position).text;
        const textBeforeCursor = lineText.substring(0, position.character);

        // Only trigger on specific patterns
        if (!this.shouldTriggerCompletion(textBeforeCursor, context)) {
            return [];
        }

        try {
            // Get code context (previous lines for context)
            const startLine = Math.max(0, position.line - 10);
            const contextRange = new vscode.Range(startLine, 0, position.line, position.character);
            const codeContext = document.getText(contextRange);

            // Call API for completions
            const completions = await this.apiClient.getCompletions(
                codeContext,
                document.languageId
            );

            return completions.map((completion, index) => {
                const item = new vscode.CompletionItem(
                    completion,
                    vscode.CompletionItemKind.Snippet
                );
                item.detail = 'AGI Framework Suggestion';
                item.documentation = new vscode.MarkdownString(
                    `AI-powered code completion suggestion\n\n\`\`\`${document.languageId}\n${completion}\n\`\`\``
                );
                item.insertText = completion;
                item.sortText = String(index).padStart(3, '0');
                
                // Add command to track completion usage
                item.command = {
                    command: 'agiFramework.logCompletion',
                    title: 'Log Completion',
                    arguments: [completion]
                };

                return item;
            });
        } catch (error) {
            console.error('Failed to get completions:', error);
            return [];
        }
    }

    private shouldTriggerCompletion(textBeforeCursor: string, context: vscode.CompletionContext): boolean {
        // Trigger on dot access
        if (textBeforeCursor.endsWith('.')) {
            return true;
        }

        // Trigger on specific keywords
        const triggerKeywords = [
            'def ',
            'class ',
            'import ',
            'from ',
            'if ',
            'for ',
            'while ',
            'try:',
            'with ',
            'async ',
            'await '
        ];

        for (const keyword of triggerKeywords) {
            if (textBeforeCursor.endsWith(keyword)) {
                return true;
            }
        }

        // Trigger on manual invocation (Ctrl+Space)
        if (context.triggerKind === vscode.CompletionTriggerKind.Invoke) {
            return true;
        }

        return false;
    }
}

// Inline completion provider for ghost text
export class AgiInlineCompletionProvider implements vscode.InlineCompletionItemProvider {
    constructor(private apiClient: AgiApiClient) {}

    async provideInlineCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
        context: vscode.InlineCompletionContext,
        token: vscode.CancellationToken
    ): Promise<vscode.InlineCompletionList | null> {
        const config = vscode.workspace.getConfiguration('agiFramework');
        if (!config.get<boolean>('enableInlineHints')) {
            return null;
        }

        // Only provide inline completions when typing
        if (context.triggerKind !== vscode.InlineCompletionTriggerKind.Automatic) {
            return null;
        }

        try {
            // Get code context
            const startLine = Math.max(0, position.line - 5);
            const contextRange = new vscode.Range(startLine, 0, position.line, position.character);
            const codeContext = document.getText(contextRange);

            // Get completions
            const completions = await this.apiClient.getCompletions(
                codeContext,
                document.languageId
            );

            if (completions.length === 0) {
                return null;
            }

            const items = completions.map(completion => {
                return new vscode.InlineCompletionItem(
                    completion,
                    new vscode.Range(position, position)
                );
            });

            return new vscode.InlineCompletionList(items);
        } catch (error) {
            console.error('Failed to get inline completions:', error);
            return null;
        }
    }
}
