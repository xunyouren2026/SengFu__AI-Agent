# AGI Framework - Embeddable Chat Widget

可嵌入的聊天组件，类似 Intercom 的体验，支持 iframe 嵌入和 JS SDK 两种方式。

## 特性

- **双模式嵌入**: 支持 JS SDK 和 iframe 两种方式
- **实时通信**: WebSocket 支持，消息即时送达
- **主题自定义**: 浅色/深色主题，可自定义颜色和样式
- **文件上传**: 支持图片、文档等文件上传
- **响应式设计**: 完美适配桌面和移动设备
- **消息通知**: 未读消息角标提醒
- **对话持久化**: 本地存储对话历史

## 快速开始

### 方式一: JavaScript SDK

```html
<!-- 引入 SDK -->
<script src="https://your-domain.com/embed/embed.js"></script>

<script>
// 初始化聊天组件
const chat = AgiChat.create({
  apiUrl: 'https://api.example.com',
  apiKey: 'your-api-key',
  theme: 'light',
  position: 'bottom-right',
  title: 'AI 助手',
  greeting: '你好！有什么可以帮助您的吗？'
});
</script>
```

### 方式二: Iframe 嵌入

```html
<iframe
  src="https://your-domain.com/embed/iframe.html?apiUrl=https://api.example.com&theme=dark"
  width="380"
  height="600"
  frameborder="0"
  style="position: fixed; bottom: 20px; right: 20px; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.15);"
></iframe>
```

## 配置选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `apiUrl` | string | 'http://localhost:8000' | API 服务器地址 |
| `wsUrl` | string | 'ws://localhost:8000/ws' | WebSocket 地址 |
| `apiKey` | string | '' | API 密钥 |
| `theme` | string | 'light' | 主题: 'light' 或 'dark' |
| `position` | string | 'bottom-right' | 位置: 'bottom-right', 'bottom-left', 'top-right', 'top-left' |
| `primaryColor` | string | '#007bff' | 主题色 |
| `title` | string | 'AGI Assistant' | 聊天窗口标题 |
| `subtitle` | string | 'How can we help you?' | 副标题 |
| `greeting` | string | 'Hello! How can I help you today?' | 欢迎消息 |
| `placeholder` | string | 'Type a message...' | 输入框占位符 |
| `autoOpen` | boolean | false | 是否自动打开 |
| `autoOpenDelay` | number | 5000 | 自动打开延迟（毫秒） |
| `persistConversation` | boolean | true | 是否持久化对话 |
| `allowAttachments` | boolean | true | 是否允许附件上传 |
| `maxFileSize` | number | 10485760 | 最大文件大小（字节） |

## SDK API

### 方法

```javascript
// 打开聊天窗口
chat.open();

// 关闭聊天窗口
chat.close();

// 切换聊天窗口
chat.toggle();

// 发送自定义消息
chat.sendCustomMessage('Hello', 'user');

// 销毁组件
chat.destroy();

// 更新配置
chat.setConfig({ theme: 'dark' });
```

### 事件

```javascript
// 监听事件
chat.on('ready', () => console.log('组件已就绪'));
chat.on('opened', () => console.log('聊天窗口已打开'));
chat.on('closed', () => console.log('聊天窗口已关闭'));
chat.on('messageSent', (data) => console.log('消息已发送:', data));
chat.on('messageReceived', (data) => console.log('收到消息:', data));
chat.on('connected', () => console.log('WebSocket 已连接'));
chat.on('disconnected', () => console.log('WebSocket 已断开'));

// 取消监听
chat.off('ready', callback);
```

## 文件结构

```
web/embed/
├── embed.js          # JS SDK
├── embed.css         # 样式文件
├── iframe.html       # Iframe 版本
├── example.html      # 示例页面
└── README.md         # 说明文档
```

## 浏览器兼容性

- Chrome 60+
- Firefox 60+
- Safari 12+
- Edge 79+

## 许可证

MIT
