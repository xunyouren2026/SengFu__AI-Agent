/**
 * AGI Framework - Embeddable Chat Widget SDK
 * Similar to Intercom, supports iframe embedding and JS SDK
 */

(function(global, factory) {
  if (typeof module === 'object' && typeof module.exports === 'object') {
    module.exports = factory();
  } else if (typeof define === 'function' && define.amd) {
    define(factory);
  } else {
    global.AgiChat = factory();
  }
})(typeof window !== 'undefined' ? window : this, function() {
  'use strict';

  const DEFAULT_CONFIG = {
    apiUrl: 'http://localhost:8000',
    wsUrl: 'ws://localhost:8000/ws',
    apiKey: '',
    position: 'bottom-right',
    theme: 'light',
    primaryColor: '#007bff',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    title: 'AGI Assistant',
    subtitle: 'How can we help you?',
    placeholder: 'Type a message...',
    greeting: 'Hello! How can I help you today?',
    showAvatar: true,
    avatarUrl: '',
    botName: 'AGI Assistant',
    userName: 'You',
    allowAttachments: true,
    maxFileSize: 10 * 1024 * 1024, // 10MB
    supportedFileTypes: ['image/*', 'application/pdf', '.doc', '.docx'],
    autoOpen: false,
    autoOpenDelay: 5000,
    persistConversation: true,
    soundEnabled: true,
    typingIndicator: true,
    headerBackground: '',
    headerTextColor: '#ffffff'
  };

  class AgiChatWidget {
    constructor(config = {}) {
      this.config = { ...DEFAULT_CONFIG, ...config };
      this.isOpen = false;
      this.isTyping = false;
      this.messages = [];
      this.ws = null;
      this.sessionId = this.getOrCreateSessionId();
      this.container = null;
      this.iframe = null;
      this.initialized = false;
      this.unreadCount = 0;
      
      this.init();
    }

    init() {
      if (this.initialized) return;
      
      this.injectStyles();
      this.createContainer();
      this.createLauncher();
      this.createIframe();
      this.connectWebSocket();
      
      if (this.config.autoOpen) {
        setTimeout(() => this.open(), this.config.autoOpenDelay);
      }
      
      this.initialized = true;
      this.emit('ready');
    }

    injectStyles() {
      if (document.getElementById('agi-chat-styles')) return;
      
      const styles = `
        .agi-chat-widget {
          --agi-primary: ${this.config.primaryColor};
          --agi-bg: ${this.config.theme === 'dark' ? '#1a1a1a' : '#ffffff'};
          --agi-text: ${this.config.theme === 'dark' ? '#ffffff' : '#333333'};
          --agi-text-secondary: ${this.config.theme === 'dark' ? '#aaaaaa' : '#666666'};
          --agi-border: ${this.config.theme === 'dark' ? '#333333' : '#e0e0e0'};
          --agi-shadow: 0 4px 20px rgba(0,0,0,0.15);
          font-family: ${this.config.fontFamily};
        }
        
        .agi-chat-launcher {
          position: fixed;
          ${this.getPositionStyles()}
          width: 60px;
          height: 60px;
          border-radius: 50%;
          background: var(--agi-primary);
          color: white;
          border: none;
          cursor: pointer;
          box-shadow: var(--agi-shadow);
          display: flex;
          align-items: center;
          justify-content: center;
          transition: transform 0.2s, box-shadow 0.2s;
          z-index: 999999;
        }
        
        .agi-chat-launcher:hover {
          transform: scale(1.05);
          box-shadow: 0 6px 25px rgba(0,0,0,0.2);
        }
        
        .agi-chat-launcher svg {
          width: 28px;
          height: 28px;
        }
        
        .agi-chat-badge {
          position: absolute;
          top: -2px;
          right: -2px;
          background: #ff4757;
          color: white;
          font-size: 12px;
          font-weight: bold;
          min-width: 20px;
          height: 20px;
          border-radius: 10px;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 0 6px;
        }
        
        .agi-chat-container {
          position: fixed;
          ${this.getContainerPositionStyles()}
          width: 380px;
          height: 600px;
          max-height: calc(100vh - 100px);
          background: var(--agi-bg);
          border-radius: 16px;
          box-shadow: var(--agi-shadow);
          overflow: hidden;
          display: flex;
          flex-direction: column;
          opacity: 0;
          visibility: hidden;
          transform: translateY(20px) scale(0.95);
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          z-index: 999998;
        }
        
        .agi-chat-container.open {
          opacity: 1;
          visibility: visible;
          transform: translateY(0) scale(1);
        }
        
        .agi-chat-header {
          background: ${this.config.headerBackground || 'var(--agi-primary)'};
          color: ${this.config.headerTextColor};
          padding: 16px 20px;
          display: flex;
          align-items: center;
          gap: 12px;
        }
        
        .agi-chat-avatar {
          width: 40px;
          height: 40px;
          border-radius: 50%;
          background: rgba(255,255,255,0.2);
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 18px;
        }
        
        .agi-chat-avatar img {
          width: 100%;
          height: 100%;
          border-radius: 50%;
          object-fit: cover;
        }
        
        .agi-chat-header-info {
          flex: 1;
        }
        
        .agi-chat-header-title {
          font-weight: 600;
          font-size: 16px;
          margin: 0;
        }
        
        .agi-chat-header-subtitle {
          font-size: 13px;
          opacity: 0.9;
          margin: 2px 0 0;
        }
        
        .agi-chat-close {
          background: none;
          border: none;
          color: inherit;
          cursor: pointer;
          padding: 4px;
          opacity: 0.8;
          transition: opacity 0.2s;
        }
        
        .agi-chat-close:hover {
          opacity: 1;
        }
        
        .agi-chat-messages {
          flex: 1;
          overflow-y: auto;
          padding: 20px;
          display: flex;
          flex-direction: column;
          gap: 12px;
        }
        
        .agi-chat-message {
          max-width: 85%;
          padding: 12px 16px;
          border-radius: 16px;
          font-size: 14px;
          line-height: 1.5;
          word-wrap: break-word;
          animation: agi-message-appear 0.3s ease;
        }
        
        @keyframes agi-message-appear {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        .agi-chat-message.user {
          align-self: flex-end;
          background: var(--agi-primary);
          color: white;
          border-bottom-right-radius: 4px;
        }
        
        .agi-chat-message.bot {
          align-self: flex-start;
          background: ${this.config.theme === 'dark' ? '#333333' : '#f0f0f0'};
          color: var(--agi-text);
          border-bottom-left-radius: 4px;
        }
        
        .agi-chat-message-time {
          font-size: 11px;
          opacity: 0.6;
          margin-top: 4px;
        }
        
        .agi-chat-typing {
          display: flex;
          align-items: center;
          gap: 4px;
          padding: 16px;
        }
        
        .agi-chat-typing-dot {
          width: 8px;
          height: 8px;
          background: var(--agi-text-secondary);
          border-radius: 50%;
          animation: agi-typing 1.4s infinite ease-in-out both;
        }
        
        .agi-chat-typing-dot:nth-child(1) { animation-delay: -0.32s; }
        .agi-chat-typing-dot:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes agi-typing {
          0%, 80%, 100% { transform: scale(0.6); opacity: 0.5; }
          40% { transform: scale(1); opacity: 1; }
        }
        
        .agi-chat-input-container {
          padding: 12px 16px;
          border-top: 1px solid var(--agi-border);
          display: flex;
          align-items: flex-end;
          gap: 8px;
        }
        
        .agi-chat-input {
          flex: 1;
          border: 1px solid var(--agi-border);
          border-radius: 20px;
          padding: 10px 16px;
          font-size: 14px;
          background: var(--agi-bg);
          color: var(--agi-text);
          resize: none;
          min-height: 20px;
          max-height: 120px;
          outline: none;
          transition: border-color 0.2s;
        }
        
        .agi-chat-input:focus {
          border-color: var(--agi-primary);
        }
        
        .agi-chat-input::placeholder {
          color: var(--agi-text-secondary);
        }
        
        .agi-chat-send {
          width: 40px;
          height: 40px;
          border-radius: 50%;
          background: var(--agi-primary);
          color: white;
          border: none;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: opacity 0.2s;
          flex-shrink: 0;
        }
        
        .agi-chat-send:hover:not(:disabled) {
          opacity: 0.9;
        }
        
        .agi-chat-send:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        
        .agi-chat-attach {
          width: 40px;
          height: 40px;
          border-radius: 50%;
          background: transparent;
          color: var(--agi-text-secondary);
          border: none;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.2s;
          flex-shrink: 0;
        }
        
        .agi-chat-attach:hover {
          background: ${this.config.theme === 'dark' ? '#333333' : '#f0f0f0'};
          color: var(--agi-text);
        }
        
        @media (max-width: 480px) {
          .agi-chat-container {
            width: 100%;
            height: 100%;
            max-height: 100vh;
            border-radius: 0;
            ${this.config.position.includes('right') ? 'right: 0;' : 'left: 0;'}
            bottom: 0;
          }
        }
      `;
      
      const styleEl = document.createElement('style');
      styleEl.id = 'agi-chat-styles';
      styleEl.textContent = styles;
      document.head.appendChild(styleEl);
    }

    getPositionStyles() {
      const positions = {
        'bottom-right': 'bottom: 20px; right: 20px;',
        'bottom-left': 'bottom: 20px; left: 20px;',
        'top-right': 'top: 20px; right: 20px;',
        'top-left': 'top: 20px; left: 20px;'
      };
      return positions[this.config.position] || positions['bottom-right'];
    }

    getContainerPositionStyles() {
      const positions = {
        'bottom-right': 'bottom: 90px; right: 20px;',
        'bottom-left': 'bottom: 90px; left: 20px;',
        'top-right': 'top: 90px; right: 20px;',
        'top-left': 'top: 90px; left: 20px;'
      };
      return positions[this.config.position] || positions['bottom-right'];
    }

    createContainer() {
      this.container = document.createElement('div');
      this.container.className = 'agi-chat-widget';
      document.body.appendChild(this.container);
    }

    createLauncher() {
      const launcher = document.createElement('button');
      launcher.className = 'agi-chat-launcher';
      launcher.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/>
        </svg>
        <span class="agi-chat-badge" style="display: none;">0</span>
      `;
      launcher.onclick = () => this.toggle();
      this.container.appendChild(launcher);
      this.launcher = launcher;
    }

    createIframe() {
      const chatContainer = document.createElement('div');
      chatContainer.className = 'agi-chat-container';
      
      chatContainer.innerHTML = `
        <div class="agi-chat-header">
          ${this.config.showAvatar ? `
            <div class="agi-chat-avatar">
              ${this.config.avatarUrl ? `<img src="${this.config.avatarUrl}" alt="">` : '🤖'}
            </div>
          ` : ''}
          <div class="agi-chat-header-info">
            <h3 class="agi-chat-header-title">${this.config.title}</h3>
            <p class="agi-chat-header-subtitle">${this.config.subtitle}</p>
          </div>
          <button class="agi-chat-close" onclick="this.closest('.agi-chat-container').classList.remove('open')">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M18 6L6 18M6 6l12 12"/>
            </svg>
          </button>
        </div>
        <div class="agi-chat-messages"></div>
        <div class="agi-chat-typing" style="display: none;">
          <div class="agi-chat-typing-dot"></div>
          <div class="agi-chat-typing-dot"></div>
          <div class="agi-chat-typing-dot"></div>
        </div>
        <div class="agi-chat-input-container">
          ${this.config.allowAttachments ? `
            <button class="agi-chat-attach" title="Attach file">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
              </svg>
            </button>
          ` : ''}
          <textarea class="agi-chat-input" placeholder="${this.config.placeholder}" rows="1"></textarea>
          <button class="agi-chat-send" disabled>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
            </svg>
          </button>
        </div>
      `;
      
      this.container.appendChild(chatContainer);
      this.chatContainer = chatContainer;
      this.messagesContainer = chatContainer.querySelector('.agi-chat-messages');
      this.inputEl = chatContainer.querySelector('.agi-chat-input');
      this.sendBtn = chatContainer.querySelector('.agi-chat-send');
      this.typingIndicator = chatContainer.querySelector('.agi-chat-typing');
      
      // Setup event listeners
      this.setupEventListeners();
      
      // Load persisted messages
      this.loadMessages();
      
      // Show greeting if no messages
      if (this.messages.length === 0 && this.config.greeting) {
        this.addMessage('bot', this.config.greeting);
      }
    }

    setupEventListeners() {
      // Send button
      this.sendBtn.onclick = () => this.sendMessage();
      
      // Input textarea
      this.inputEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          this.sendMessage();
        }
      });
      
      this.inputEl.addEventListener('input', () => {
        this.sendBtn.disabled = !this.inputEl.value.trim();
        this.inputEl.style.height = 'auto';
        this.inputEl.style.height = Math.min(this.inputEl.scrollHeight, 120) + 'px';
      });
      
      // Close button
      const closeBtn = this.chatContainer.querySelector('.agi-chat-close');
      closeBtn.onclick = () => this.close();
      
      // Attachment button
      const attachBtn = this.chatContainer.querySelector('.agi-chat-attach');
      if (attachBtn) {
        attachBtn.onclick = () => this.handleAttachment();
      }
    }

    connectWebSocket() {
      try {
        this.ws = new WebSocket(`${this.config.wsUrl}?session=${this.sessionId}`);
        
        this.ws.onopen = () => {
          console.log('AGI Chat: WebSocket connected');
          this.emit('connected');
        };
        
        this.ws.onmessage = (event) => {
          const data = JSON.parse(event.data);
          this.handleWebSocketMessage(data);
        };
        
        this.ws.onclose = () => {
          console.log('AGI Chat: WebSocket disconnected');
          this.emit('disconnected');
          // Reconnect after 5 seconds
          setTimeout(() => this.connectWebSocket(), 5000);
        };
        
        this.ws.onerror = (error) => {
          console.error('AGI Chat: WebSocket error:', error);
          this.emit('error', error);
        };
      } catch (error) {
        console.error('AGI Chat: Failed to connect WebSocket:', error);
      }
    }

    handleWebSocketMessage(data) {
      switch (data.type) {
        case 'message':
          this.hideTyping();
          this.addMessage('bot', data.content);
          break;
        case 'typing':
          this.showTyping();
          break;
        case 'error':
          this.hideTyping();
          this.addMessage('bot', 'Sorry, an error occurred. Please try again.');
          break;
      }
    }

    sendMessage() {
      const text = this.inputEl.value.trim();
      if (!text) return;
      
      this.addMessage('user', text);
      this.inputEl.value = '';
      this.inputEl.style.height = 'auto';
      this.sendBtn.disabled = true;
      
      // Send via WebSocket if connected
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({
          type: 'message',
          content: text,
          sessionId: this.sessionId
        }));
        this.showTyping();
      } else {
        // Fallback to HTTP
        this.sendViaHttp(text);
      }
      
      this.emit('messageSent', { text });
    }

    async sendViaHttp(text) {
      this.showTyping();
      
      try {
        const response = await fetch(`${this.config.apiUrl}/api/v1/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${this.config.apiKey}`
          },
          body: JSON.stringify({
            message: text,
            sessionId: this.sessionId
          })
        });
        
        const data = await response.json();
        this.hideTyping();
        this.addMessage('bot', data.response);
      } catch (error) {
        this.hideTyping();
        this.addMessage('bot', 'Sorry, I could not process your message. Please try again.');
        console.error('AGI Chat: HTTP request failed:', error);
      }
    }

    addMessage(sender, text, attachments = []) {
      const message = {
        id: Date.now().toString(),
        sender,
        text,
        attachments,
        timestamp: new Date().toISOString()
      };
      
      this.messages.push(message);
      this.renderMessage(message);
      this.saveMessages();
      
      if (!this.isOpen && sender === 'bot') {
        this.incrementUnread();
      }
      
      this.emit('messageReceived', message);
    }

    renderMessage(message) {
      const messageEl = document.createElement('div');
      messageEl.className = `agi-chat-message ${message.sender}`;
      
      let content = message.text;
      
      // Convert URLs to links
      content = content.replace(
        /(https?:\/\/[^\s]+)/g,
        '<a href="$1" target="_blank" rel="noopener" style="color: inherit; text-decoration: underline;">$1</a>'
      );
      
      // Convert markdown-style formatting
      content = content
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/`(.+?)`/g, '<code style="background: rgba(0,0,0,0.1); padding: 2px 4px; border-radius: 3px;">$1</code>');
      
      messageEl.innerHTML = `
        ${content}
        <div class="agi-chat-message-time">${this.formatTime(message.timestamp)}</div>
      `;
      
      this.messagesContainer.appendChild(messageEl);
      this.scrollToBottom();
    }

    showTyping() {
      this.isTyping = true;
      this.typingIndicator.style.display = 'flex';
      this.scrollToBottom();
    }

    hideTyping() {
      this.isTyping = false;
      this.typingIndicator.style.display = 'none';
    }

    scrollToBottom() {
      this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }

    formatTime(timestamp) {
      const date = new Date(timestamp);
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    getOrCreateSessionId() {
      let sessionId = localStorage.getItem('agi-chat-session');
      if (!sessionId) {
        sessionId = 'sess_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('agi-chat-session', sessionId);
      }
      return sessionId;
    }

    loadMessages() {
      if (!this.config.persistConversation) return;
      
      try {
        const stored = localStorage.getItem(`agi-chat-messages-${this.sessionId}`);
        if (stored) {
          this.messages = JSON.parse(stored);
          this.messages.forEach(msg => this.renderMessage(msg));
        }
      } catch (error) {
        console.error('AGI Chat: Failed to load messages:', error);
      }
    }

    saveMessages() {
      if (!this.config.persistConversation) return;
      
      try {
        localStorage.setItem(
          `agi-chat-messages-${this.sessionId}`,
          JSON.stringify(this.messages.slice(-100)) // Keep last 100 messages
        );
      } catch (error) {
        console.error('AGI Chat: Failed to save messages:', error);
      }
    }

    incrementUnread() {
      this.unreadCount++;
      const badge = this.launcher.querySelector('.agi-chat-badge');
      badge.textContent = this.unreadCount;
      badge.style.display = 'flex';
    }

    clearUnread() {
      this.unreadCount = 0;
      const badge = this.launcher.querySelector('.agi-chat-badge');
      badge.style.display = 'none';
    }

    toggle() {
      if (this.isOpen) {
        this.close();
      } else {
        this.open();
      }
    }

    open() {
      this.isOpen = true;
      this.chatContainer.classList.add('open');
      this.clearUnread();
      this.inputEl.focus();
      this.emit('opened');
    }

    close() {
      this.isOpen = false;
      this.chatContainer.classList.remove('open');
      this.emit('closed');
    }

    handleAttachment() {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = this.config.supportedFileTypes.join(',');
      input.onchange = (e) => {
        const file = e.target.files[0];
        if (file) {
          this.uploadFile(file);
        }
      };
      input.click();
    }

    async uploadFile(file) {
      if (file.size > this.config.maxFileSize) {
        alert(`File too large. Maximum size is ${this.config.maxFileSize / 1024 / 1024}MB`);
        return;
      }
      
      const formData = new FormData();
      formData.append('file', file);
      formData.append('sessionId', this.sessionId);
      
      try {
        const response = await fetch(`${this.config.apiUrl}/api/v1/upload`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${this.config.apiKey}`
          },
          body: formData
        });
        
        const data = await response.json();
        this.addMessage('user', `📎 ${file.name}`, [data.fileUrl]);
      } catch (error) {
        console.error('AGI Chat: File upload failed:', error);
        alert('Failed to upload file. Please try again.');
      }
    }

    // Event emitter
    emit(event, data) {
      const listeners = this._listeners || {};
      if (listeners[event]) {
        listeners[event].forEach(cb => cb(data));
      }
    }

    on(event, callback) {
      if (!this._listeners) this._listeners = {};
      if (!this._listeners[event]) this._listeners[event] = [];
      this._listeners[event].push(callback);
    }

    off(event, callback) {
      if (!this._listeners || !this._listeners[event]) return;
      this._listeners[event] = this._listeners[event].filter(cb => cb !== callback);
    }

    // Public API
    destroy() {
      if (this.ws) {
        this.ws.close();
      }
      if (this.container) {
        this.container.remove();
      }
      const styles = document.getElementById('agi-chat-styles');
      if (styles) {
        styles.remove();
      }
    }

    setConfig(newConfig) {
      this.config = { ...this.config, ...newConfig };
      // Re-render with new config
      this.destroy();
      this.init();
    }

    sendCustomMessage(text, sender = 'user') {
      this.addMessage(sender, text);
    }
  }

  // Factory function
  function createAgiChat(config) {
    return new AgiChatWidget(config);
  }

  // Auto-initialize if config is present
  if (typeof window !== 'undefined' && window.agiChatConfig) {
    window.agiChat = createAgiChat(window.agiChatConfig);
  }

  return {
    create: createAgiChat,
    Widget: AgiChatWidget
  };
});
