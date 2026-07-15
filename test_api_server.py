#!/usr/bin/env python3
"""
简单的API测试服务器
仅用于测试chat_integrated API端点
"""

import json
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

class APIHandler(SimpleHTTPRequestHandler):
    """处理API请求"""
    
    def log_message(self, format, *args):
        """自定义日志格式"""
        print(f"[{time.strftime('%H:%M:%S')}] {args[0]}")
    
    def do_GET(self):
        """处理GET请求"""
        parsed = urlparse(self.path)
        
        if parsed.path == '/api/v1/chat-integrated/conversations/integrated':
            self.send_json({
                "success": True,
                "data": {
                    "items": [],
                    "total": 0,
                    "page": 1,
                    "limit": 50
                }
            })
        elif parsed.path.startswith('/api/v1/models'):
            self.send_json({
                "success": True,
                "data": {
                    "items": [
                        {"id": "gpt-4", "name": "GPT-4", "status": "active"},
                        {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "status": "active"},
                        {"id": "claude-3", "name": "Claude-3", "status": "active"}
                    ]
                }
            })
        elif parsed.path == '/api/v1/health/integrated':
            self.send_json({
                "success": True,
                "data": {
                    "status": "healthy",
                    "advanced_modules_available": True,
                    "memory_system": True,
                    "hierarchical_memory": True,
                    "compressor": True,
                    "model_gateway": True,
                    "rag_pipeline": True
                }
            })
        else:
            self.send_error(404, "Not Found")
    
    def do_POST(self):
        """处理POST请求"""
        parsed = urlparse(self.path)
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        
        try:
            data = json.loads(body) if body else {}
        except:
            data = {}
        
        if parsed.path == '/api/v1/chat-integrated/conversations/integrated':
            self.send_json({
                "success": True,
                "data": {
                    "id": f"conv_{int(time.time())}",
                    "title": data.get('title', '新对话'),
                    "model": data.get('model', 'gpt-4'),
                    "created_at": time.strftime('%Y-%m-%dT%H:%M:%SZ')
                }
            })
        elif parsed.path == '/api/v1/chat-integrated/messages/integrated':
            self.send_json({
                "success": True,
                "data": {
                    "id": f"msg_{int(time.time())}",
                    "conversation_id": data.get('conversation_id'),
                    "role": data.get('role', 'user'),
                    "content": data.get('content', ''),
                    "created_at": time.strftime('%Y-%m-%dT%H:%M:%SZ')
                }
            })
        elif parsed.path == '/api/v1/chat-integrated/stream/integrated':
            # SSE流式响应
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()
            
            # 模拟流式响应
            response_text = f"这是模拟的AI回复。关于您的问题：{data.get('message', '你好')}，我来为您详细解答。"
            
            for i, char in enumerate(response_text):
                if i % 5 == 0:  # 每5个字符发送一个事件
                    event = json.dumps({"content": char})
                    self.wfile.write(f"data: {event}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.05)
            
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            self.send_error(404, "Not Found")
    
    def do_DELETE(self):
        """处理DELETE请求"""
        self.send_json({"success": True, "message": "Deleted"})
    
    def send_json(self, data):
        """发送JSON响应"""
        response = json.dumps(data)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(response))
        self.end_headers()
        self.wfile.write(response.encode())

def run_server(port=8000):
    """运行服务器"""
    server = HTTPServer(('0.0.0.0', port), APIHandler)
    print(f"🚀 API测试服务器运行在 http://localhost:{port}")
    print(f"📖 API文档: http://localhost:{port}/docs")
    print(f"💬 聊天页面: http://localhost:{port}/pages/chat.html")
    print("")
    print("可用端点:")
    print("  GET  /api/v1/chat-integrated/conversations/integrated")
    print("  POST /api/v1/chat-integrated/conversations/integrated")
    print("  GET  /api/v1/models")
    print("  POST /api/v1/chat-integrated/messages/integrated")
    print("  POST /api/v1/chat-integrated/stream/integrated (SSE)")
    print("  GET  /api/v1/health/integrated")
    print("")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")
        server.shutdown()

if __name__ == '__main__':
    run_server(8000)
