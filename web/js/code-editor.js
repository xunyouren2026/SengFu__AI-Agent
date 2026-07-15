/**
 * AGI Unified Framework - Code Editor Library
 * Monaco Editor wrapper with advanced features
 * @version 1.0.0
 * @author AGI Framework Team
 */

(function(global) {
    'use strict';

    // Code Editor Manager
    class CodeEditor {
        constructor(container, options = {}) {
            this.container = typeof container === 'string'
                ? document.querySelector(container)
                : container;

            if (!this.container) {
                throw new Error('Code editor container not found');
            }

            this.options = {
                language: 'javascript',
                theme: 'vs-dark',
                value: '',
                readOnly: false,
                fontSize: 14,
                lineNumbers: true,
                minimap: { enabled: true },
                automaticLayout: true,
                scrollBeyondLastLine: false,
                wordWrap: 'on',
                folding: true,
                renderWhitespace: 'selection',
                bracketPairColorization: { enabled: true },
                formatOnPaste: true,
                formatOnType: true,
                suggestOnTriggerCharacters: true,
                acceptSuggestionOnEnter: 'on',
                quickSuggestions: true,
                ...options
            };

            this.editor = null;
            this.decorations = [];
            this.markers = [];

            this.init();
        }

        async init() {
            // Check if Monaco is available
            if (typeof monaco === 'undefined') {
                console.warn('Monaco Editor not loaded. Loading from CDN...');
                await this.loadMonaco();
            }

            this.createEditor();
            this.setupEvents();
        }

        loadMonaco() {
            return new Promise((resolve, reject) => {
                const script = document.createElement('script');
                script.src = 'https://cdn.jsdelivr.net/npm/monaco-editor@0.44.0/min/vs/loader.js';
                script.onload = () => {
                    require.config({
                        paths: { vs: 'https://cdn.jsdelivr.net/npm/monaco-editor@0.44.0/min/vs' }
                    });
                    require(['vs/editor/editor.main'], () => {
                        resolve();
                    });
                };
                script.onerror = reject;
                document.head.appendChild(script);
            });
        }

        createEditor() {
            this.editor = monaco.editor.create(this.container, {
                value: this.options.value,
                language: this.options.language,
                theme: this.options.theme,
                readOnly: this.options.readOnly,
                fontSize: this.options.fontSize,
                lineNumbers: this.options.lineNumbers,
                minimap: this.options.minimap,
                automaticLayout: this.options.automaticLayout,
                scrollBeyondLastLine: this.options.scrollBeyondLastLine,
                wordWrap: this.options.wordWrap,
                folding: this.options.folding,
                renderWhitespace: this.options.renderWhitespace,
                bracketPairColorization: this.options.bracketPairColorization,
                formatOnPaste: this.options.formatOnPaste,
                formatOnType: this.options.formatOnType,
                suggestOnTriggerCharacters: this.options.suggestOnTriggerCharacters,
                acceptSuggestionOnEnter: this.options.acceptSuggestionOnEnter,
                quickSuggestions: this.options.quickSuggestions
            });

            // Set container size
            this.container.style.height = this.options.height || '400px';
        }

        setupEvents() {
            // Content change
            this.editor.onDidChangeModelContent((e) => {
                if (this.options.onChange) {
                    this.options.onChange(this.getValue(), e);
                }
            });

            // Cursor position change
            this.editor.onDidChangeCursorPosition((e) => {
                if (this.options.onCursorChange) {
                    this.options.onCursorChange(e.position);
                }
            });

            // Selection change
            this.editor.onDidChangeCursorSelection((e) => {
                if (this.options.onSelectionChange) {
                    this.options.onSelectionChange(e.selection);
                }
            });

            // Focus
            this.editor.onDidFocusEditorText(() => {
                if (this.options.onFocus) {
                    this.options.onFocus();
                }
            });

            // Blur
            this.editor.onDidBlurEditorText(() => {
                if (this.options.onBlur) {
                    this.options.onBlur();
                }
            });
        }

        // Value operations
        getValue() {
            return this.editor ? this.editor.getValue() : '';
        }

        setValue(value) {
            if (this.editor) {
                this.editor.setValue(value);
            }
        }

        insertValue(value, position = null) {
            if (!this.editor) return;

            if (position) {
                this.editor.executeEdits('', [{
                    range: new monaco.Range(position.lineNumber, position.column, position.lineNumber, position.column),
                    text: value
                }]);
            } else {
                const selection = this.editor.getSelection();
                this.editor.executeEdits('', [{
                    range: selection,
                    text: value
                }]);
            }
        }

        // Selection operations
        getSelection() {
            return this.editor ? this.editor.getSelection() : null;
        }

        setSelection(range) {
            if (this.editor) {
                this.editor.setSelection(range);
            }
        }

        getSelectedText() {
            if (!this.editor) return '';
            const selection = this.editor.getSelection();
            return this.editor.getModel().getValueInRange(selection);
        }

        replaceSelection(text) {
            if (this.editor) {
                this.editor.executeEdits('', [{
                    range: this.editor.getSelection(),
                    text: text
                }]);
            }
        }

        // Position operations
        getPosition() {
            return this.editor ? this.editor.getPosition() : null;
        }

        setPosition(position) {
            if (this.editor) {
                this.editor.setPosition(position);
            }
        }

        revealLine(lineNumber) {
            if (this.editor) {
                this.editor.revealLine(lineNumber);
            }
        }

        revealLineInCenter(lineNumber) {
            if (this.editor) {
                this.editor.revealLineInCenter(lineNumber);
            }
        }

        // Language and theme
        setLanguage(language) {
            if (this.editor) {
                monaco.editor.setModelLanguage(this.editor.getModel(), language);
            }
        }

        setTheme(theme) {
            if (typeof monaco !== 'undefined') {
                monaco.editor.setTheme(theme);
            }
        }

        // Formatting
        formatDocument() {
            if (this.editor) {
                this.editor.getAction('editor.action.formatDocument').run();
            }
        }

        formatSelection() {
            if (this.editor) {
                this.editor.getAction('editor.action.formatSelection').run();
            }
        }

        // Undo/Redo
        undo() {
            if (this.editor) {
                this.editor.trigger('keyboard', 'undo', null);
            }
        }

        redo() {
            if (this.editor) {
                this.editor.trigger('keyboard', 'redo', null);
            }
        }

        // Find and replace
        find(query, options = {}) {
            if (!this.editor) return null;

            const model = this.editor.getModel();
            const matches = model.findMatches(
                query,
                false,
                options.regex || false,
                options.matchCase || false,
                options.wholeWord ? ' \t\n\r' : null,
                true
            );

            return matches;
        }

        replace(findText, replaceText, options = {}) {
            if (!this.editor) return;

            const matches = this.find(findText, options);
            if (matches && matches.length > 0) {
                const edits = matches.map(match => ({
                    range: match.range,
                    text: replaceText
                }));
                this.editor.executeEdits('replace', edits);
            }
        }

        replaceAll(findText, replaceText, options = {}) {
            this.replace(findText, replaceText, options);
        }

        // Decorations
        addDecoration(range, options = {}) {
            if (!this.editor) return;

            const decoration = {
                range: range,
                options: {
                    isWholeLine: options.isWholeLine || false,
                    className: options.className,
                    glyphMarginClassName: options.glyphMarginClassName,
                    overviewRuler: options.overviewRuler,
                    minimap: options.minimap,
                    linesDecorationsClassName: options.linesDecorationsClassName,
                    inlineClassName: options.inlineClassName,
                    hoverMessage: options.hoverMessage ? { value: options.hoverMessage } : undefined
                }
            };

            const decorations = this.editor.deltaDecorations([], [decoration]);
            this.decorations.push(...decorations);
            return decorations;
        }

        removeDecorations(decorationIds) {
            if (this.editor) {
                this.editor.deltaDecorations(decorationIds, []);
                this.decorations = this.decorations.filter(id => !decorationIds.includes(id));
            }
        }

        clearDecorations() {
            if (this.editor) {
                this.editor.deltaDecorations(this.decorations, []);
                this.decorations = [];
            }
        }

        // Markers (errors, warnings, info)
        addMarker(marker) {
            if (!this.editor) return;

            const model = this.editor.getModel();
            const markers = [{
                ...marker,
                resource: model.uri
            }];

            monaco.editor.setModelMarkers(model, 'owner', markers);
            this.markers.push(marker);
        }

        clearMarkers() {
            if (this.editor) {
                monaco.editor.setModelMarkers(this.editor.getModel(), 'owner', []);
                this.markers = [];
            }
        }

        // Highlight line
        highlightLine(lineNumber, className = 'line-highlight') {
            const range = new monaco.Range(lineNumber, 1, lineNumber, 1);
            return this.addDecoration(range, {
                isWholeLine: true,
                className: className
            });
        }

        // Diff editor
        static createDiffEditor(container, originalValue, modifiedValue, options = {}) {
            const diffEditor = monaco.editor.createDiffEditor(container, {
                enableSplitViewResizing: true,
                renderSideBySide: true,
                ...options
            });

            diffEditor.setModel({
                original: monaco.editor.createModel(originalValue, options.language || 'javascript'),
                modified: monaco.editor.createModel(modifiedValue, options.language || 'javascript')
            });

            return diffEditor;
        }

        // Minimap
        toggleMinimap() {
            if (this.editor) {
                const current = this.editor.getOption(monaco.editor.EditorOption.minimap);
                this.editor.updateOptions({
                    minimap: { enabled: !current.enabled }
                });
            }
        }

        // Word wrap
        toggleWordWrap() {
            if (this.editor) {
                const current = this.editor.getOption(monaco.editor.EditorOption.wordWrap);
                this.editor.updateOptions({
                    wordWrap: current === 'on' ? 'off' : 'on'
                });
            }
        }

        // Read only
        setReadOnly(readOnly) {
            if (this.editor) {
                this.editor.updateOptions({ readOnly });
            }
        }

        // Focus
        focus() {
            if (this.editor) {
                this.editor.focus();
            }
        }

        // Layout
        layout() {
            if (this.editor) {
                this.editor.layout();
            }
        }

        // Dispose
        dispose() {
            if (this.editor) {
                this.editor.dispose();
                this.editor = null;
            }
        }
    }

    // Code Diff Viewer
    class CodeDiffViewer {
        constructor(container, options = {}) {
            this.container = typeof container === 'string'
                ? document.querySelector(container)
                : container;

            this.options = {
                language: 'javascript',
                theme: 'vs-dark',
                original: '',
                modified: '',
                renderSideBySide: true,
                ...options
            };

            this.diffEditor = null;
            this.init();
        }

        async init() {
            if (typeof monaco === 'undefined') {
                await this.loadMonaco();
            }

            this.createDiffEditor();
        }

        loadMonaco() {
            return new Promise((resolve, reject) => {
                const script = document.createElement('script');
                script.src = 'https://cdn.jsdelivr.net/npm/monaco-editor@0.44.0/min/vs/loader.js';
                script.onload = () => {
                    require.config({
                        paths: { vs: 'https://cdn.jsdelivr.net/npm/monaco-editor@0.44.0/min/vs' }
                    });
                    require(['vs/editor/editor.main'], () => {
                        resolve();
                    });
                };
                script.onerror = reject;
                document.head.appendChild(script);
            });
        }

        createDiffEditor() {
            this.diffEditor = monaco.editor.createDiffEditor(this.container, {
                enableSplitViewResizing: true,
                renderSideBySide: this.options.renderSideBySide,
                theme: this.options.theme,
                readOnly: true
            });

            this.setContent(this.options.original, this.options.modified);
        }

        setContent(original, modified) {
            if (!this.diffEditor) return;

            const originalModel = monaco.editor.createModel(original, this.options.language);
            const modifiedModel = monaco.editor.createModel(modified, this.options.language);

            this.diffEditor.setModel({
                original: originalModel,
                modified: modifiedModel
            });
        }

        getOriginalValue() {
            return this.diffEditor ? this.diffEditor.getOriginalEditor().getValue() : '';
        }

        getModifiedValue() {
            return this.diffEditor ? this.diffEditor.getModifiedEditor().getValue() : '';
        }

        setTheme(theme) {
            if (typeof monaco !== 'undefined') {
                monaco.editor.setTheme(theme);
            }
        }

        layout() {
            if (this.diffEditor) {
                this.diffEditor.layout();
            }
        }

        dispose() {
            if (this.diffEditor) {
                this.diffEditor.dispose();
                this.diffEditor = null;
            }
        }
    }

    // Code Snippet Manager
    class CodeSnippetManager {
        constructor() {
            this.snippets = new Map();
            this.loadDefaultSnippets();
        }

        loadDefaultSnippets() {
            // JavaScript snippets
            this.snippets.set('javascript', {
                'function': {
                    prefix: 'func',
                    body: [
                        'function ${1:name}(${2:params}) {',
                        '\t${3:// body}',
                        '}'
                    ],
                    description: 'Function declaration'
                },
                'arrow': {
                    prefix: 'arrow',
                    body: [
                        'const ${1:name} = (${2:params}) => {',
                        '\t${3:// body}',
                        '};'
                    ],
                    description: 'Arrow function'
                },
                'class': {
                    prefix: 'class',
                    body: [
                        'class ${1:Name} {',
                        '\tconstructor(${2:params}) {',
                        '\t\t${3:// initialization}',
                        '\t}',
                        '}'
                    ],
                    description: 'Class declaration'
                },
                'for': {
                    prefix: 'for',
                    body: [
                        'for (let ${1:i} = 0; ${1:i} < ${2:length}; ${1:i}++) {',
                        '\t${3:// body}',
                        '}'
                    ],
                    description: 'For loop'
                },
                'foreach': {
                    prefix: 'foreach',
                    body: [
                        '${1:array}.forEach((${2:item}) => {',
                        '\t${3:// body}',
                        '});'
                    ],
                    description: 'forEach loop'
                },
                'if': {
                    prefix: 'if',
                    body: [
                        'if (${1:condition}) {',
                        '\t${2:// body}',
                        '}'
                    ],
                    description: 'If statement'
                },
                'try': {
                    prefix: 'try',
                    body: [
                        'try {',
                        '\t${1:// code}',
                        '} catch (${2:error}) {',
                        '\t${3:// handle error}',
                        '}'
                    ],
                    description: 'Try-catch block'
                },
                'console': {
                    prefix: 'log',
                    body: 'console.log(${1:message});',
                    description: 'Console log'
                }
            });

            // Python snippets
            this.snippets.set('python', {
                'function': {
                    prefix: 'def',
                    body: [
                        'def ${1:name}(${2:params}):',
                        '    ${3:# body}'
                    ],
                    description: 'Function definition'
                },
                'class': {
                    prefix: 'class',
                    body: [
                        'class ${1:Name}:',
                        '    def __init__(self${2:, params}):',
                        '        ${3:# initialization}'
                    ],
                    description: 'Class definition'
                },
                'for': {
                    prefix: 'for',
                    body: [
                        'for ${1:item} in ${2:iterable}:',
                        '    ${3:# body}'
                    ],
                    description: 'For loop'
                },
                'if': {
                    prefix: 'if',
                    body: [
                        'if ${1:condition}:',
                        '    ${2:# body}'
                    ],
                    description: 'If statement'
                },
                'try': {
                    prefix: 'try',
                    body: [
                        'try:',
                        '    ${1:# code}',
                        'except ${2:Exception} as ${3:e}:',
                        '    ${4:# handle error}'
                    ],
                    description: 'Try-except block'
                }
            });

            // HTML snippets
            this.snippets.set('html', {
                'div': {
                    prefix: 'div',
                    body: '<div class="${1:class}">${2:content}</div>',
                    description: 'Div element'
                },
                'span': {
                    prefix: 'span',
                    body: '<span class="${1:class}">${2:content}</span>',
                    description: 'Span element'
                },
                'link': {
                    prefix: 'link',
                    body: '<link rel="stylesheet" href="${1:style.css}">',
                    description: 'Stylesheet link'
                },
                'script': {
                    prefix: 'script',
                    body: '<script src="${1:script.js}"></script>',
                    description: 'Script tag'
                },
                'img': {
                    prefix: 'img',
                    body: '<img src="${1:source}" alt="${2:description}">',
                    description: 'Image tag'
                }
            });
        }

        getSnippets(language) {
            return this.snippets.get(language) || {};
        }

        addSnippet(language, name, snippet) {
            if (!this.snippets.has(language)) {
                this.snippets.set(language, {});
            }
            this.snippets.get(language)[name] = snippet;
        }

        removeSnippet(language, name) {
            const snippets = this.snippets.get(language);
            if (snippets) {
                delete snippets[name];
            }
        }
    }

    // Code Formatter
    class CodeFormatter {
        constructor() {
            this.formatters = new Map();
        }

        registerFormatter(language, formatter) {
            this.formatters.set(language, formatter);
        }

        async format(code, language, options = {}) {
            const formatter = this.formatters.get(language);
            if (formatter) {
                return await formatter(code, options);
            }

            // Default formatting
            return this.defaultFormat(code, language);
        }

        defaultFormat(code, language) {
            // Simple indentation formatting
            const lines = code.split('\n');
            let indent = 0;
            const indentSize = 4;
            const formatted = [];

            const increaseIndent = ['{', '[', '('];
            const decreaseIndent = ['}', ']', ')'];

            lines.forEach(line => {
                const trimmed = line.trim();

                if (decreaseIndent.some(c => trimmed.startsWith(c))) {
                    indent = Math.max(0, indent - 1);
                }

                formatted.push(' '.repeat(indent * indentSize) + trimmed);

                if (increaseIndent.some(c => trimmed.endsWith(c))) {
                    indent++;
                }
            });

            return formatted.join('\n');
        }
    }

    // Code Linter
    class CodeLinter {
        constructor() {
            this.linters = new Map();
            this.rules = new Map();
        }

        registerLinter(language, linter) {
            this.linters.set(language, linter);
        }

        addRule(language, ruleName, rule) {
            if (!this.rules.has(language)) {
                this.rules.set(language, new Map());
            }
            this.rules.get(language).set(ruleName, rule);
        }

        async lint(code, language) {
            const linter = this.linters.get(language);
            if (linter) {
                return await linter(code);
            }

            // Default linting
            return this.defaultLint(code, language);
        }

        defaultLint(code, language) {
            const diagnostics = [];
            const lines = code.split('\n');

            lines.forEach((line, index) => {
                // Check for trailing whitespace
                if (/\s+$/.test(line)) {
                    diagnostics.push({
                        line: index + 1,
                        column: line.length,
                        message: 'Trailing whitespace',
                        severity: 'warning'
                    });
                }

                // Check for tabs
                if (/\t/.test(line)) {
                    diagnostics.push({
                        line: index + 1,
                        column: line.indexOf('\t') + 1,
                        message: 'Use spaces instead of tabs',
                        severity: 'warning'
                    });
                }

                // Check line length
                if (line.length > 120) {
                    diagnostics.push({
                        line: index + 1,
                        column: 121,
                        message: 'Line exceeds 120 characters',
                        severity: 'warning'
                    });
                }
            });

            return diagnostics;
        }
    }

    // Code Completion Engine
    class CodeCompletionEngine {
        constructor() {
            this.providers = new Map();
            this.suggestions = new Map();
        }

        registerProvider(language, provider) {
            this.providers.set(language, provider);
        }

        async getCompletions(model, position) {
            const language = model.getLanguageId();
            const provider = this.providers.get(language);

            if (provider) {
                return await provider(model, position);
            }

            return this.getDefaultCompletions(model, position);
        }

        getDefaultCompletions(model, position) {
            const word = model.getWordUntilPosition(position);
            const range = {
                startLineNumber: position.lineNumber,
                endLineNumber: position.lineNumber,
                startColumn: word.startColumn,
                endColumn: word.endColumn
            };

            // Default keyword completions
            const keywords = [
                'function', 'const', 'let', 'var', 'if', 'else', 'for', 'while',
                'return', 'class', 'import', 'export', 'async', 'await'
            ];

            return keywords.map(keyword => ({
                label: keyword,
                kind: monaco.languages.CompletionItemKind.Keyword,
                insertText: keyword,
                range: range
            }));
        }
    }

    // Code Folding Provider
    class CodeFoldingProvider {
        constructor() {
            this.foldingRanges = [];
        }

        getFoldingRanges(model) {
            const ranges = [];
            const lineCount = model.getLineCount();
            let startLine = null;

            for (let i = 1; i <= lineCount; i++) {
                const line = model.getLineContent(i);

                // Check for fold start (functions, classes, blocks)
                if (/\{\s*$/.test(line) || /\(\s*$/.test(line)) {
                    startLine = i;
                }

                // Check for fold end
                if (startLine && /^\s*\}/.test(line)) {
                    ranges.push({
                        start: startLine,
                        end: i,
                        kind: monaco.languages.FoldingRangeKind.Region
                    });
                    startLine = null;
                }
            }

            return ranges;
        }
    }

    // Code Outline Provider
    class CodeOutlineProvider {
        constructor() {
            this.symbols = [];
        }

        getDocumentSymbols(model) {
            const symbols = [];
            const lineCount = model.getLineCount();

            for (let i = 1; i <= lineCount; i++) {
                const line = model.getLineContent(i);

                // Function detection
                const functionMatch = line.match(/(?:function|const|let|var)\s+(\w+)\s*\(/);
                if (functionMatch) {
                    symbols.push({
                        name: functionMatch[1],
                        kind: monaco.languages.SymbolKind.Function,
                        location: {
                            uri: model.uri,
                            range: {
                                startLineNumber: i,
                                startColumn: line.indexOf(functionMatch[1]) + 1,
                                endLineNumber: i,
                                endColumn: line.indexOf(functionMatch[1]) + functionMatch[1].length + 1
                            }
                        }
                    });
                }

                // Class detection
                const classMatch = line.match(/class\s+(\w+)/);
                if (classMatch) {
                    symbols.push({
                        name: classMatch[1],
                        kind: monaco.languages.SymbolKind.Class,
                        location: {
                            uri: model.uri,
                            range: {
                                startLineNumber: i,
                                startColumn: line.indexOf(classMatch[1]) + 1,
                                endLineNumber: i,
                                endColumn: line.indexOf(classMatch[1]) + classMatch[1].length + 1
                            }
                        }
                    });
                }
            }

            return symbols;
        }
    }

    // Code Search Engine
    class CodeSearchEngine {
        constructor() {
            this.index = new Map();
        }

        indexDocument(uri, model) {
            const content = model.getValue();
            const lines = content.split('\n');

            const documentIndex = {
                uri: uri,
                lines: lines,
                tokens: this.tokenize(content)
            };

            this.index.set(uri.toString(), documentIndex);
        }

        tokenize(text) {
            return text.toLowerCase()
                .replace(/[^a-z0-9_]/g, ' ')
                .split(/\s+/)
                .filter(token => token.length > 2);
        }

        search(query, options = {}) {
            const results = [];
            const queryTokens = this.tokenize(query);

            this.index.forEach((doc, uri) => {
                let score = 0;

                queryTokens.forEach(token => {
                    doc.tokens.forEach((docToken, index) => {
                        if (docToken.includes(token)) {
                            score += 1;
                        }
                    });
                });

                if (score > 0) {
                    results.push({
                        uri: uri,
                        score: score,
                        preview: this.getPreview(doc, queryTokens[0])
                    });
                }
            });

            return results.sort((a, b) => b.score - a.score);
        }

        getPreview(doc, token) {
            for (let i = 0; i < doc.lines.length; i++) {
                if (doc.lines[i].toLowerCase().includes(token)) {
                    return {
                        line: i + 1,
                        text: doc.lines[i].trim()
                    };
                }
            }
            return null;
        }
    }

    // Code Execution Engine
    class CodeExecutionEngine {
        constructor() {
            this.sandboxes = new Map();
            this.timeout = 5000; // 5 seconds
        }

        createSandbox(id) {
            const iframe = document.createElement('iframe');
            iframe.style.display = 'none';
            iframe.sandbox = 'allow-scripts';
            document.body.appendChild(iframe);

            this.sandboxes.set(id, iframe);
            return iframe;
        }

        async execute(code, language, options = {}) {
            const sandboxId = options.sandboxId || 'default';

            if (!this.sandboxes.has(sandboxId)) {
                this.createSandbox(sandboxId);
            }

            const iframe = this.sandboxes.get(sandboxId);

            return new Promise((resolve, reject) => {
                const timeoutId = setTimeout(() => {
                    reject(new Error('Execution timeout'));
                }, this.timeout);

                // Set up message listener
                const handleMessage = (event) => {
                    if (event.source === iframe.contentWindow) {
                        clearTimeout(timeoutId);
                        window.removeEventListener('message', handleMessage);
                        resolve(event.data);
                    }
                };

                window.addEventListener('message', handleMessage);

                // Execute code in sandbox
                const script = `
                    try {
                        const result = (function() {
                            ${code}
                        })();
                        parent.postMessage({ success: true, result: String(result) }, '*');
                    } catch (error) {
                        parent.postMessage({ success: false, error: error.message }, '*');
                    }
                `;

                iframe.srcdoc = `<script>${script}</script>`;
            });
        }

        destroySandbox(id) {
            const iframe = this.sandboxes.get(id);
            if (iframe) {
                iframe.remove();
                this.sandboxes.delete(id);
            }
        }
    }

    // Code Collaboration Manager
    class CodeCollaborationManager {
        constructor() {
            this.cursors = new Map();
            this.selections = new Map();
            this.websocket = null;
        }

        connect(url, roomId) {
            this.websocket = new WebSocket(url);

            this.websocket.onopen = () => {
                this.websocket.send(JSON.stringify({
                    type: 'join',
                    room: roomId
                }));
            };

            this.websocket.onmessage = (event) => {
                const message = JSON.parse(event.data);
                this.handleMessage(message);
            };
        }

        handleMessage(message) {
            switch (message.type) {
                case 'cursor':
                    this.updateRemoteCursor(message.userId, message.position);
                    break;
                case 'selection':
                    this.updateRemoteSelection(message.userId, message.selection);
                    break;
                case 'edit':
                    this.applyRemoteEdit(message.edit);
                    break;
            }
        }

        updateRemoteCursor(userId, position) {
            this.cursors.set(userId, position);
            // Update cursor decoration in editor
        }

        updateRemoteSelection(userId, selection) {
            this.selections.set(userId, selection);
            // Update selection decoration in editor
        }

        applyRemoteEdit(edit) {
            // Apply edit from remote user
        }

        sendCursor(position) {
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify({
                    type: 'cursor',
                    position: position
                }));
            }
        }

        sendSelection(selection) {
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify({
                    type: 'selection',
                    selection: selection
                }));
            }
        }

        disconnect() {
            if (this.websocket) {
                this.websocket.close();
                this.websocket = null;
            }
        }
    }

    // Advanced Language Support Manager
    class LanguageSupportManager {
        constructor() {
            this.languages = new Map();
            this.monacoLanguages = new Map();
            this.initDefaultLanguages();
        }
        
        initDefaultLanguages() {
            // JavaScript/TypeScript
            this.registerLanguage('javascript', {
                extensions: ['.js', '.mjs', '.cjs'],
                aliases: ['JavaScript', 'JS'],
                mimetypes: ['text/javascript'],
                configuration: {
                    comments: { lineComment: '//', blockComment: ['/*', '*/'] },
                    brackets: [['{', '}'], ['[', ']'], ['(', ')']],
                    autoClosingPairs: [
                        { open: '{', close: '}' },
                        { open: '[', close: ']' },
                        { open: '(', close: ')' },
                        { open: '"', close: '"', notIn: ['string'] },
                        { open: "'", close: "'", notIn: ['string', 'comment'] },
                        { open: '`', close: '`', notIn: ['string', 'comment'] }
                    ],
                    surroundingPairs: [
                        { open: '{', close: '}' },
                        { open: '[', close: ']' },
                        { open: '(', close: ')' },
                        { open: '"', close: '"' },
                        { open: "'", close: "'" },
                        { open: '`', close: '`' }
                    ],
                    folding: { markers: { start: /^\/\/\s*#?region\b/, end: /^\/\/\s*#?endregion\b/ } }
                },
                tokens: this.getJSTokens()
            });
            
            // TypeScript
            this.registerLanguage('typescript', {
                extensions: ['.ts', '.tsx'],
                aliases: ['TypeScript', 'TS'],
                mimetypes: ['text/typescript'],
                configuration: this.languages.get('javascript')?.configuration,
                tokens: this.getTSTokens()
            });
            
            // Python
            this.registerLanguage('python', {
                extensions: ['.py', '.pyw', '.py3'],
                aliases: ['Python', 'py'],
                mimetypes: ['text/x-python'],
                configuration: {
                    comments: { lineComment: '#', blockComment: ['"""', '"""'] },
                    brackets: [['{', '}'], ['[', ']'], ['(', ')']],
                    autoClosingPairs: [
                        { open: '{', close: '}' },
                        { open: '[', close: ']' },
                        { open: '(', close: ')' },
                        { open: '"', close: '"', notIn: ['string'] },
                        { open: "'", close: "'", notIn: ['string', 'comment'] }
                    ],
                    indentationRules: { increaseIndentPattern: /^\s*.*:\s*$/, decreaseIndentPattern: /^\s*(return|break|continue|raise|pass)\b/ }
                },
                tokens: this.getPythonTokens()
            });
            
            // Java
            this.registerLanguage('java', {
                extensions: ['.java'],
                aliases: ['Java'],
                mimetypes: ['text/x-java-source'],
                configuration: {
                    comments: { lineComment: '//', blockComment: ['/*', '*/'] },
                    brackets: [['{', '}'], ['[', ']'], ['(', ')']],
                    autoClosingPairs: [
                        { open: '{', close: '}' },
                        { open: '[', close: ']' },
                        { open: '(', close: ')' },
                        { open: '"', close: '"' },
                        { open: "'", close: "'" }
                    ]
                },
                tokens: this.getJavaTokens()
            });
            
            // C/C++
            this.registerLanguage('cpp', {
                extensions: ['.c', '.cpp', '.cc', '.cxx', '.h', '.hpp'],
                aliases: ['C++', 'CPP', 'C'],
                mimetypes: ['text/x-c++src'],
                configuration: {
                    comments: { lineComment: '//', blockComment: ['/*', '*/'] },
                    brackets: [['{', '}'], ['[', ']'], ['(', ')']],
                    autoClosingPairs: [
                        { open: '{', close: '}' },
                        { open: '[', close: ']' },
                        { open: '(', close: ')' },
                        { open: '"', close: '"' },
                        { open: "'", close: "'" }
                    ]
                },
                tokens: this.getCPPTokens()
            });
            
            // Rust
            this.registerLanguage('rust', {
                extensions: ['.rs'],
                aliases: ['Rust'],
                mimetypes: ['text/x-rustsrc'],
                configuration: {
                    comments: { lineComment: '//', blockComment: ['/*', '*/'] },
                    brackets: [['{', '}'], ['[', ']'], ['(', ')']],
                    autoClosingPairs: [
                        { open: '{', close: '}' },
                        { open: '[', close: ']' },
                        { open: '(', close: ')' },
                        { open: '"', close: '"' },
                        { open: "'", close: "'" }
                    ]
                },
                tokens: this.getRustTokens()
            });
            
            // Go
            this.registerLanguage('go', {
                extensions: ['.go'],
                aliases: ['Go'],
                mimetypes: ['text/x-go'],
                configuration: {
                    comments: { lineComment: '//', blockComment: ['/*', '*/'] },
                    brackets: [['{', '}'], ['[', ']'], ['(', ')']],
                    autoClosingPairs: [
                        { open: '{', close: '}' },
                        { open: '[', close: ']' },
                        { open: '(', close: ')' },
                        { open: '"', close: '"' },
                        { open: "'", close: "'" },
                        { open: '`', close: '`' }
                    ]
                },
                tokens: this.getGoTokens()
            });
            
            // SQL
            this.registerLanguage('sql', {
                extensions: ['.sql'],
                aliases: ['SQL'],
                mimetypes: ['text/x-sql'],
                configuration: {
                    comments: { lineComment: '--', blockComment: ['/*', '*/'] },
                    brackets: [['(', ')']],
                    autoClosingPairs: [
                        { open: '(', close: ')' },
                        { open: '"', close: '"' },
                        { open: "'", close: "'" }
                    ]
                },
                tokens: this.getSQLTokens()
            });
            
            // JSON
            this.registerLanguage('json', {
                extensions: ['.json', '.jsonc'],
                aliases: ['JSON'],
                mimetypes: ['application/json'],
                configuration: {
                    brackets: [['{', '}'], ['[', ']']],
                    autoClosingPairs: [
                        { open: '{', close: '}' },
                        { open: '[', close: ']' },
                        { open: '"', close: '"' }
                    ]
                },
                tokens: this.getJSONTokens()
            });
            
            // YAML
            this.registerLanguage('yaml', {
                extensions: ['.yaml', '.yml'],
                aliases: ['YAML'],
                mimetypes: ['text/x-yaml'],
                configuration: {
                    comments: { lineComment: '#' },
                    brackets: [['{', '}'], ['[', ']']],
                    autoClosingPairs: [
                        { open: '{', close: '}' },
                        { open: '[', close: ']' },
                        { open: '"', close: '"' },
                        { open: "'", close: "'" }
                    ]
                },
                tokens: this.getYAMLTokens()
            });
            
            // Markdown
            this.registerLanguage('markdown', {
                extensions: ['.md', '.markdown'],
                aliases: ['Markdown'],
                mimetypes: ['text/markdown'],
                configuration: {
                    brackets: [['[', ']'], ['(', ')']],
                    autoClosingPairs: [
                        { open: '[', close: ']' },
                        { open: '(', close: ')' },
                        { open: '"', close: '"' },
                        { open: "'", close: "'" },
                        { open: '`', close: '`' }
                    ]
                },
                tokens: this.getMarkdownTokens()
            });
            
            // Shell/Bash
            this.registerLanguage('shell', {
                extensions: ['.sh', '.bash', '.zsh'],
                aliases: ['Shell', 'Bash', 'sh'],
                mimetypes: ['text/x-sh'],
                configuration: {
                    comments: { lineComment: '#' },
                    brackets: [['{', '}'], ['[', ']'], ['(', ')']],
                    autoClosingPairs: [
                        { open: '{', close: '}' },
                        { open: '[', close: ']' },
                        { open: '(', close: ')' },
                        { open: '"', close: '"' },
                        { open: "'", close: "'" }
                    ]
                },
                tokens: this.getShellTokens()
            });
        }
        
        getJSTokens() {
            return {
                keywords: ['async', 'await', 'break', 'case', 'catch', 'class', 'const', 'continue', 'debugger', 'default', 'delete', 'do', 'else', 'export', 'extends', 'finally', 'for', 'function', 'if', 'import', 'in', 'instanceof', 'let', 'new', 'of', 'return', 'static', 'super', 'switch', 'this', 'throw', 'try', 'typeof', 'var', 'void', 'while', 'with', 'yield'],
                builtins: ['Array', 'ArrayBuffer', 'BigInt', 'Boolean', 'DataView', 'Date', 'Error', 'EvalError', 'Float32Array', 'Float64Array', 'Function', 'Int16Array', 'Int32Array', 'Int8Array', 'Map', 'Number', 'Object', 'Promise', 'Proxy', 'RangeError', 'ReferenceError', 'RegExp', 'Set', 'String', 'Symbol', 'SyntaxError', 'TypeError', 'Uint16Array', 'Uint32Array', 'Uint8Array', 'Uint8ClampedArray', 'URIError', 'WeakMap', 'WeakSet', 'console', 'JSON', 'Math', 'Reflect', 'Atomics', 'Intl', 'WebAssembly'],
                constants: ['null', 'undefined', 'true', 'false', 'NaN', 'Infinity']
            };
        }
        
        getTSTokens() {
            const js = this.getJSTokens();
            return {
                keywords: [...js.keywords, 'abstract', 'as', 'declare', 'enum', 'implements', 'interface', 'is', 'namespace', 'never', 'private', 'protected', 'public', 'readonly', 'require', 'type', 'typeof', 'unique', 'unknown'],
                builtins: js.builtins,
                constants: js.constants
            };
        }
        
        getPythonTokens() {
            return {
                keywords: ['False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield'],
                builtins: ['abs', 'all', 'any', 'bin', 'bool', 'bytes', 'callable', 'chr', 'classmethod', 'compile', 'complex', 'delattr', 'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec', 'filter', 'float', 'format', 'frozenset', 'getattr', 'globals', 'hasattr', 'hash', 'help', 'hex', 'id', 'input', 'int', 'isinstance', 'issubclass', 'iter', 'len', 'list', 'locals', 'map', 'max', 'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord', 'pow', 'print', 'property', 'range', 'repr', 'reversed', 'round', 'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super', 'tuple', 'type', 'vars', 'zip'],
                constants: ['True', 'False', 'None', 'Ellipsis', 'NotImplemented']
            };
        }
        
        getJavaTokens() {
            return {
                keywords: ['abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', 'const', 'continue', 'default', 'do', 'double', 'else', 'enum', 'extends', 'final', 'finally', 'float', 'for', 'goto', 'if', 'implements', 'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new', 'package', 'private', 'protected', 'public', 'return', 'short', 'static', 'strictfp', 'super', 'switch', 'synchronized', 'this', 'throw', 'throws', 'transient', 'try', 'void', 'volatile', 'while'],
                builtins: ['String', 'Integer', 'Long', 'Double', 'Float', 'Boolean', 'Character', 'Byte', 'Short', 'Object', 'Class', 'System', 'Math', 'Runtime', 'Thread', 'Runnable', 'Exception', 'Error', 'ArrayList', 'HashMap', 'HashSet', 'LinkedList', 'Arrays', 'Collections', 'Optional'],
                constants: ['true', 'false', 'null']
            };
        }
        
        getCPPTokens() {
            return {
                keywords: ['alignas', 'alignof', 'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor', 'bool', 'break', 'case', 'catch', 'char', 'char8_t', 'char16_t', 'char32_t', 'class', 'compl', 'concept', 'const', 'consteval', 'constexpr', 'constinit', 'const_cast', 'continue', 'co_await', 'co_return', 'co_yield', 'decltype', 'default', 'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum', 'explicit', 'export', 'extern', 'false', 'float', 'for', 'friend', 'goto', 'if', 'inline', 'int', 'long', 'mutable', 'namespace', 'new', 'noexcept', 'not', 'not_eq', 'nullptr', 'operator', 'or', 'or_eq', 'private', 'protected', 'public', 'register', 'reinterpret_cast', 'requires', 'return', 'short', 'signed', 'sizeof', 'static', 'static_assert', 'static_cast', 'struct', 'switch', 'template', 'this', 'thread_local', 'throw', 'true', 'try', 'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using', 'virtual', 'void', 'volatile', 'wchar_t', 'while', 'xor', 'xor_eq'],
                builtins: ['std', 'vector', 'string', 'map', 'set', 'list', 'array', 'deque', 'queue', 'stack', 'pair', 'tuple', 'shared_ptr', 'unique_ptr', 'weak_ptr', 'make_shared', 'make_unique', 'cout', 'cin', 'endl', 'cerr', 'clog'],
                constants: ['true', 'false', 'nullptr', 'NULL']
            };
        }
        
        getRustTokens() {
            return {
                keywords: ['as', 'async', 'await', 'break', 'const', 'continue', 'crate', 'dyn', 'else', 'enum', 'extern', 'false', 'fn', 'for', 'if', 'impl', 'in', 'let', 'loop', 'match', 'mod', 'move', 'mut', 'pub', 'ref', 'return', 'self', 'Self', 'static', 'struct', 'super', 'trait', 'true', 'type', 'unsafe', 'use', 'where', 'while'],
                builtins: ['Vec', 'String', 'Option', 'Result', 'Box', 'Rc', 'Arc', 'Cell', 'RefCell', 'Mutex', 'RwLock', 'HashMap', 'HashSet', 'BTreeMap', 'BTreeSet', 'VecDeque', 'LinkedList', 'Cow', 'fmt', 'io', 'fs', 'net', 'thread', 'sync', 'time', 'path', 'process', 'env', 'error', 'result', 'option', 'iter', 'slice', 'str', 'char', 'bool', 'i8', 'i16', 'i32', 'i64', 'i128', 'isize', 'u8', 'u16', 'u32', 'u64', 'u128', 'usize', 'f32', 'f64'],
                constants: ['true', 'false', 'Some', 'None', 'Ok', 'Err']
            };
        }
        
        getGoTokens() {
            return {
                keywords: ['break', 'case', 'chan', 'const', 'continue', 'default', 'defer', 'else', 'fallthrough', 'for', 'func', 'go', 'goto', 'if', 'import', 'interface', 'map', 'package', 'range', 'return', 'select', 'struct', 'switch', 'type', 'var'],
                builtins: ['append', 'cap', 'close', 'complex', 'copy', 'delete', 'imag', 'len', 'make', 'new', 'panic', 'print', 'println', 'real', 'recover', 'error', 'string', 'int', 'int8', 'int16', 'int32', 'int64', 'uint', 'uint8', 'uint16', 'uint32', 'uint64', 'uintptr', 'float32', 'float64', 'complex64', 'complex128', 'bool', 'byte', 'rune', 'any', 'comparable'],
                constants: ['true', 'false', 'nil', 'iota']
            };
        }
        
        getSQLTokens() {
            return {
                keywords: ['ADD', 'ALL', 'ALTER', 'AND', 'ANY', 'AS', 'ASC', 'BACKUP', 'BETWEEN', 'CASE', 'CHECK', 'COLUMN', 'CONSTRAINT', 'CREATE', 'DATABASE', 'DEFAULT', 'DELETE', 'DESC', 'DISTINCT', 'DROP', 'EXEC', 'EXISTS', 'FOREIGN', 'FROM', 'FULL', 'GROUP', 'HAVING', 'IN', 'INDEX', 'INNER', 'INSERT', 'INTO', 'IS', 'JOIN', 'KEY', 'LEFT', 'LIKE', 'LIMIT', 'NOT', 'NULL', 'OR', 'ORDER', 'OUTER', 'PRIMARY', 'PROCEDURE', 'RIGHT', 'ROWNUM', 'SELECT', 'SET', 'TABLE', 'TOP', 'TRUNCATE', 'UNION', 'UNIQUE', 'UPDATE', 'VALUES', 'VIEW', 'WHERE', 'WITH'],
                builtins: ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'COALESCE', 'NULLIF', 'CAST', 'CONVERT', 'CONCAT', 'SUBSTRING', 'LENGTH', 'UPPER', 'LOWER', 'TRIM', 'DATE', 'NOW', 'YEAR', 'MONTH', 'DAY'],
                constants: ['TRUE', 'FALSE', 'NULL']
            };
        }
        
        getJSONTokens() {
            return {
                keywords: [],
                builtins: [],
                constants: ['true', 'false', 'null']
            };
        }
        
        getYAMLTokens() {
            return {
                keywords: [],
                builtins: [],
                constants: ['true', 'false', 'null', 'yes', 'no', 'on', 'off']
            };
        }
        
        getMarkdownTokens() {
            return {
                keywords: [],
                builtins: [],
                constants: []
            };
        }
        
        getShellTokens() {
            return {
                keywords: ['if', 'then', 'else', 'elif', 'fi', 'case', 'esac', 'for', 'select', 'while', 'until', 'do', 'done', 'in', 'function', 'time', 'break', 'continue', 'return', 'exit', 'local', 'declare', 'readonly', 'export', 'unset', 'shift', 'set', 'source'],
                builtins: ['echo', 'printf', 'read', 'cd', 'pwd', 'ls', 'mkdir', 'rmdir', 'rm', 'cp', 'mv', 'touch', 'cat', 'head', 'tail', 'grep', 'sed', 'awk', 'find', 'sort', 'uniq', 'wc', 'cut', 'tr', 'chmod', 'chown', 'ln', 'tar', 'gzip', 'gunzip', 'ssh', 'scp', 'rsync', 'curl', 'wget', 'ps', 'kill', 'killall', 'top', 'htop', 'df', 'du', 'free', 'uname', 'date', 'whoami', 'which', 'whereis', 'man', 'help'],
                constants: ['true', 'false']
            };
        }
        
        registerLanguage(id, config) {
            this.languages.set(id, config);
        }
        
        getLanguage(id) {
            return this.languages.get(id);
        }
        
        getLanguageByExtension(ext) {
            for (const [id, config] of this.languages) {
                if (config.extensions && config.extensions.includes(ext)) {
                    return id;
                }
            }
            return null;
        }
        
        setupMonacoLanguage(id) {
            if (typeof monaco === 'undefined') return;
            
            const config = this.languages.get(id);
            if (!config) return;
            
            monaco.languages.register({ id });
            
            if (config.configuration) {
                monaco.languages.setLanguageConfiguration(id, config.configuration);
            }
            
            if (config.tokens) {
                monaco.languages.setMonarchTokensProvider(id, this.createMonarchTokensProvider(config.tokens));
            }
        }
        
        createMonarchTokensProvider(tokens) {
            return {
                keywords: tokens.keywords,
                builtins: tokens.builtins,
                constants: tokens.constants,
                tokenizer: {
                    root: [
                        [/[a-zA-Z_$][\w$]*/, {
                            cases: {
                                '@keywords': 'keyword',
                                '@builtins': 'type.identifier',
                                '@constants': 'constant',
                                '@default': 'identifier'
                            }
                        }],
                        [/[{}()\[\]]/, '@brackets'],
                        [/[<>](?!@symbols)/, '@brackets'],
                        [/["']/, 'string'],
                        [/\d*\.\d+([eE][\-+]?\d+)?/, 'number.float'],
                        [/\d+/, 'number'],
                        [/#.*$/, 'comment'],
                        [/\/\/.*$/, 'comment'],
                        [/\/\*/, 'comment', '@comment']
                    ],
                    comment: [
                        [/\*\//, 'comment', '@pop'],
                        [/./, 'comment']
                    ],
                    string: [
                        [/[^\\'"]+/, 'string'],
                        [/\\./, 'string.escape'],
                        [/['"]/, 'string', '@pop']
                    ]
                }
            };
        }
    }

    // Static Code Analyzer
    class StaticCodeAnalyzer {
        constructor() {
            this.analyzers = new Map();
            this.registerDefaultAnalyzers();
        }
        
        registerDefaultAnalyzers() {
            // Complexity Analyzer
            this.registerAnalyzer('complexity', (ast, language) => {
                const results = [];
                // Cyclomatic complexity calculation
                this.traverseAST(ast, (node) => {
                    if (['IfStatement', 'SwitchStatement', 'ForStatement', 'WhileStatement', 'DoWhileStatement', 'ConditionalExpression', 'LogicalExpression', 'CatchClause'].includes(node.type)) {
                        results.push({
                            type: 'complexity',
                            node: node,
                            value: 1
                        });
                    }
                });
                return results;
            });
            
            // Unused Variable Analyzer
            this.registerAnalyzer('unused', (ast, language) => {
                const results = [];
                const declared = new Map();
                const used = new Set();
                
                this.traverseAST(ast, (node) => {
                    if (node.type === 'VariableDeclarator' && node.id.name) {
                        declared.set(node.id.name, node);
                    }
                    if (node.type === 'Identifier' && node.name) {
                        used.add(node.name);
                    }
                });
                
                declared.forEach((node, name) => {
                    if (!used.has(name)) {
                        results.push({
                            type: 'unused',
                            severity: 'warning',
                            message: `Variable '${name}' is declared but never used`,
                            node: node
                        });
                    }
                });
                
                return results;
            });
            
            // Duplicate Code Analyzer
            this.registerAnalyzer('duplicate', (ast, language) => {
                const results = [];
                const hashes = new Map();
                
                this.traverseAST(ast, (node) => {
                    const hash = this.hashNode(node);
                    if (hashes.has(hash)) {
                        results.push({
                            type: 'duplicate',
                            severity: 'info',
                            message: 'Similar code block found',
                            node: node,
                            original: hashes.get(hash)
                        });
                    } else {
                        hashes.set(hash, node);
                    }
                });
                
                return results;
            });
            
            // Security Analyzer
            this.registerAnalyzer('security', (ast, language) => {
                const results = [];
                
                this.traverseAST(ast, (node) => {
                    // Check for eval usage
                    if (node.type === 'CallExpression' && node.callee.name === 'eval') {
                        results.push({
                            type: 'security',
                            severity: 'error',
                            message: 'Use of eval() is potentially dangerous',
                            node: node
                        });
                    }
                    
                    // Check for innerHTML
                    if (node.type === 'AssignmentExpression' && 
                        node.left.property && node.left.property.name === 'innerHTML') {
                        results.push({
                            type: 'security',
                            severity: 'warning',
                            message: 'Setting innerHTML can lead to XSS vulnerabilities',
                            node: node
                        });
                    }
                });
                
                return results;
            });
            
            // Performance Analyzer
            this.registerAnalyzer('performance', (ast, language) => {
                const results = [];
                
                this.traverseAST(ast, (node) => {
                    // Check for synchronous operations in async context
                    if (node.type === 'FunctionDeclaration' && node.async) {
                        this.traverseAST(node.body, (child) => {
                            if (child.type === 'CallExpression' && 
                                child.callee.property && 
                                ['readFileSync', 'writeFileSync', 'existsSync'].includes(child.callee.property.name)) {
                                results.push({
                                    type: 'performance',
                                    severity: 'warning',
                                    message: 'Synchronous operation in async function',
                                    node: child
                                });
                            }
                        });
                    }
                });
                
                return results;
            });
        }
        
        registerAnalyzer(name, analyzer) {
            this.analyzers.set(name, analyzer);
        }
        
        analyze(code, language, options = {}) {
            const results = {};
            const ast = this.parseToAST(code, language);
            
            this.analyzers.forEach((analyzer, name) => {
                if (!options.excludes || !options.excludes.includes(name)) {
                    results[name] = analyzer(ast, language);
                }
            });
            
            return results;
        }
        
        parseToAST(code, language) {
            // Simplified AST representation
            const ast = {
                type: 'Program',
                body: [],
                source: code
            };
            
            // Basic parsing logic
            const lines = code.split('\n');
            let currentScope = ast.body;
            
            lines.forEach((line, index) => {
                const trimmed = line.trim();
                
                // Function declaration
                const funcMatch = trimmed.match(/(?:async\s+)?function\s+(\w+)\s*\(/);
                if (funcMatch) {
                    currentScope.push({
                        type: 'FunctionDeclaration',
                        name: funcMatch[1],
                        async: trimmed.startsWith('async'),
                        body: [],
                        loc: { start: { line: index + 1 } }
                    });
                }
                
                // Variable declaration
                const varMatch = trimmed.match(/(?:const|let|var)\s+(\w+)/);
                if (varMatch) {
                    currentScope.push({
                        type: 'VariableDeclarator',
                        id: { name: varMatch[1] },
                        loc: { start: { line: index + 1 } }
                    });
                }
                
                // If statement
                if (trimmed.startsWith('if')) {
                    currentScope.push({
                        type: 'IfStatement',
                        loc: { start: { line: index + 1 } }
                    });
                }
                
                // For loop
                if (trimmed.startsWith('for')) {
                    currentScope.push({
                        type: 'ForStatement',
                        loc: { start: { line: index + 1 } }
                    });
                }
                
                // While loop
                if (trimmed.startsWith('while')) {
                    currentScope.push({
                        type: 'WhileStatement',
                        loc: { start: { line: index + 1 } }
                    });
                }
            });
            
            return ast;
        }
        
        traverseAST(node, callback) {
            callback(node);
            for (const key in node) {
                if (node[key] && typeof node[key] === 'object') {
                    if (Array.isArray(node[key])) {
                        node[key].forEach(child => this.traverseAST(child, callback));
                    } else {
                        this.traverseAST(node[key], callback);
                    }
                }
            }
        }
        
        hashNode(node) {
            // Simple hash for duplicate detection
            const str = JSON.stringify(node, (key, value) => {
                if (key === 'loc' || key === 'range') return undefined;
                return value;
            });
            let hash = 0;
            for (let i = 0; i < str.length; i++) {
                hash = ((hash << 5) - hash) + str.charCodeAt(i);
                hash |= 0;
            }
            return hash;
        }
    }

    // Code Metrics Calculator
    class CodeMetricsCalculator {
        constructor() {
            this.metrics = new Map();
        }
        
        calculate(code, language) {
            const lines = code.split('\n');
            const metrics = {
                linesOfCode: lines.length,
                linesOfCodeNonEmpty: lines.filter(l => l.trim().length > 0).length,
                linesOfCodeComment: lines.filter(l => l.trim().startsWith('//') || l.trim().startsWith('#') || l.trim().startsWith('/*')).length,
                characters: code.length,
                charactersNonWhitespace: code.replace(/\s/g, '').length,
                averageLineLength: 0,
                maxLineLength: 0,
                indentStyle: this.detectIndentStyle(code),
                cyclomaticComplexity: 0,
                cognitiveComplexity: 0,
                maintainabilityIndex: 0,
                halsteadVolume: 0,
                functionCount: 0,
                classCount: 0,
                importCount: 0
            };
            
            // Calculate average and max line length
            let totalLength = 0;
            lines.forEach(line => {
                totalLength += line.length;
                if (line.length > metrics.maxLineLength) {
                    metrics.maxLineLength = line.length;
                }
            });
            metrics.averageLineLength = totalLength / lines.length;
            
            // Count functions
            const functionMatches = code.match(/(?:function\s+\w+|(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)/g);
            metrics.functionCount = functionMatches ? functionMatches.length : 0;
            
            // Count classes
            const classMatches = code.match(/class\s+\w+/g);
            metrics.classCount = classMatches ? classMatches.length : 0;
            
            // Count imports
            const importMatches = code.match(/(?:import\s+|require\s*\(|from\s+['"])/g);
            metrics.importCount = importMatches ? importMatches.length : 0;
            
            // Calculate cyclomatic complexity (simplified)
            const decisionPoints = (code.match(/\b(if|else|for|while|case|catch|\?|&&|\|\|)\b/g) || []).length;
            metrics.cyclomaticComplexity = decisionPoints + 1;
            
            // Calculate Halstead volume (simplified)
            const operators = (code.match(/[+\-*/%=<>!&|^~?:]/g) || []).length;
            const operands = (code.match(/[a-zA-Z_][a-zA-Z0-9_]*/g) || []).length;
            const vocabulary = operators + operands;
            const length = operators + operands;
            metrics.halsteadVolume = length * Math.log2(vocabulary || 1);
            
            // Calculate maintainability index
            const V = metrics.halsteadVolume;
            const G = metrics.cyclomaticComplexity;
            const LOC = metrics.linesOfCodeNonEmpty;
            metrics.maintainabilityIndex = Math.max(0, (171 - 5.2 * Math.log(V || 1) - 0.23 * G - 16.2 * Math.log(LOC)) * 100 / 171);
            
            return metrics;
        }
        
        detectIndentStyle(code) {
            const lines = code.split('\n');
            let spaces = 0;
            let tabs = 0;
            let spaceSizes = [];
            
            lines.forEach(line => {
                const leadingSpace = line.match(/^( {2,})/);
                if (leadingSpace) {
                    spaces++;
                    spaceSizes.push(leadingSpace[1].length);
                }
                if (line.match(/^\t+/)) {
                    tabs++;
                }
            });
            
            if (tabs > spaces) return { type: 'tab', size: 1 };
            if (spaces > 0) {
                const gcd = this.gcdArray(spaceSizes);
                return { type: 'space', size: gcd || 2 };
            }
            return { type: 'unknown', size: 0 };
        }
        
        gcdArray(arr) {
            if (arr.length === 0) return 0;
            return arr.reduce((a, b) => this.gcd(a, b));
        }
        
        gcd(a, b) {
            a = Math.abs(a);
            b = Math.abs(b);
            while (b) {
                const t = b;
                b = a % b;
                a = t;
            }
            return a;
        }
        
        compareMetrics(metrics1, metrics2) {
            return {
                linesOfCode: metrics2.linesOfCode - metrics1.linesOfCode,
                cyclomaticComplexity: metrics2.cyclomaticComplexity - metrics1.cyclomaticComplexity,
                maintainabilityIndex: metrics2.maintainabilityIndex - metrics1.maintainabilityIndex,
                functionCount: metrics2.functionCount - metrics1.functionCount,
                classCount: metrics2.classCount - metrics1.classCount
            };
        }
    }

    // Code Refactoring Engine
    class CodeRefactoringEngine {
        constructor() {
            this.refactorings = new Map();
            this.registerDefaultRefactorings();
        }
        
        registerDefaultRefactorings() {
            // Extract Method
            this.registerRefactoring('extractMethod', {
                applicable: (code, selection) => selection.length > 0,
                apply: (code, selection, options) => {
                    const selectedCode = code.substring(selection.start, selection.end);
                    const methodName = options.methodName || 'extractedMethod';
                    const indent = '    ';
                    
                    const newMethod = `${indent}${methodName}() {\n${indent}${indent}${selectedCode.split('\n').join('\n' + indent + indent)}\n${indent}}\n`;
                    
                    return {
                        code: code.substring(0, selection.start) + `${methodName}()` + code.substring(selection.end),
                        insertPosition: this.findInsertPosition(code),
                        insertCode: newMethod
                    };
                }
            });
            
            // Rename Variable
            this.registerRefactoring('renameVariable', {
                applicable: (code, selection) => true,
                apply: (code, selection, options) => {
                    const oldName = options.oldName;
                    const newName = options.newName;
                    const regex = new RegExp(`\\b${oldName}\\b`, 'g');
                    return { code: code.replace(regex, newName) };
                }
            });
            
            // Inline Variable
            this.registerRefactoring('inlineVariable', {
                applicable: (code, selection) => true,
                apply: (code, selection, options) => {
                    const varName = options.varName;
                    const varValue = options.varValue;
                    const varDecl = new RegExp(`(?:const|let|var)\\s+${varName}\\s*=\\s*[^;]+;?\\n?`, 'g');
                    const varUsage = new RegExp(`\\b${varName}\\b`, 'g');
                    return { code: code.replace(varDecl, '').replace(varUsage, varValue) };
                }
            });
            
            // Convert to Arrow Function
            this.registerRefactoring('toArrowFunction', {
                applicable: (code, selection) => {
                    const selected = code.substring(selection.start, selection.end);
                    return /function\s*\([^)]*\)\s*\{/.test(selected);
                },
                apply: (code, selection, options) => {
                    const selected = code.substring(selection.start, selection.end);
                    const converted = selected
                        .replace(/function\s*\(([^)]*)\)\s*\{/, '($1) => {')
                        .replace(/function\s+(\w+)\s*\(([^)]*)\)\s*\{/, 'const $1 = ($2) => {');
                    return { code: code.substring(0, selection.start) + converted + code.substring(selection.end) };
                }
            });
            
            // Extract Interface
            this.registerRefactoring('extractInterface', {
                applicable: (code, selection) => /class\s+\w+/.test(code),
                apply: (code, selection, options) => {
                    const className = options.className || 'ExtractedInterface';
                    const methods = options.methods || [];
                    const interfaceDef = `interface ${className} {\n${methods.map(m => `    ${m}(): void;`).join('\n')}\n}\n\n`;
                    return { code: interfaceDef + code };
                }
            });
        }
        
        registerRefactoring(name, refactoring) {
            this.refactorings.set(name, refactoring);
        }
        
        getRefactoring(name) {
            return this.refactorings.get(name);
        }
        
        getApplicableRefactorings(code, selection) {
            const applicable = [];
            this.refactorings.forEach((refactoring, name) => {
                if (refactoring.applicable(code, selection)) {
                    applicable.push(name);
                }
            });
            return applicable;
        }
        
        applyRefactoring(name, code, selection, options = {}) {
            const refactoring = this.refactorings.get(name);
            if (!refactoring) {
                throw new Error(`Unknown refactoring: ${name}`);
            }
            return refactoring.apply(code, selection, options);
        }
        
        findInsertPosition(code) {
            const lines = code.split('\n');
            let lastClassEnd = 0;
            let braceCount = 0;
            let inClass = false;
            
            for (let i = 0; i < lines.length; i++) {
                if (lines[i].match(/class\s+\w+/)) {
                    inClass = true;
                }
                braceCount += (lines[i].match(/{/g) || []).length;
                braceCount -= (lines[i].match(/}/g) || []).length;
                if (inClass && braceCount === 0) {
                    lastClassEnd = i;
                    break;
                }
            }
            
            return lastClassEnd;
        }
    }

    // Export
    global.AGIEditor = {
        CodeEditor,
        CodeDiffViewer,
        CodeSnippetManager,
        CodeFormatter,
        CodeLinter,
        CodeCompletionEngine,
        CodeFoldingProvider,
        CodeOutlineProvider,
        CodeSearchEngine,
        CodeExecutionEngine,
        CodeCollaborationManager,
        LanguageSupportManager,
        StaticCodeAnalyzer,
        CodeMetricsCalculator,
        CodeRefactoringEngine
    };

})(window);
