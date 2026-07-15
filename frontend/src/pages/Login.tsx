import { useState } from 'react';
import { Zap, Eye, EyeOff } from 'lucide-react';
import { Card, Button, Input } from '../components/common';

export default function Login() {
  const [isLogin, setIsLogin] = useState(true);
  const [showPassword, setShowPassword] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Login logic would go here
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-primary-50 to-primary-100 dark:from-dark-900 dark:to-dark-800 p-4">
      <Card className="w-full max-w-md" variant="elevated">
        <div className="text-center mb-6">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center text-white text-2xl font-bold mx-auto mb-4">U</div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">UFO AGI 统一框架</h1>
          <p className="text-sm text-gray-500 mt-1">{isLogin ? '登录到你的账户' : '创建新账户'}</p>
        </div>
        <form onSubmit={handleSubmit} className="space-y-4">
          {!isLogin && <Input label={'"邮箱"'} type="email" placeholder="your@email.com" value={username} onChange={e => setUsername(e.target.value)} />}
          <Input label={'"用户名"'} placeholder={'"输入用户名"'} value={username} onChange={e => setUsername(e.target.value)} />
          <div className="relative">
            <Input label={'"密码"'} type={showPassword ? 'text' : 'password'} placeholder={'"输入密码"'} value={password} onChange={e => setPassword(e.target.value)} rightIcon={<button type="button" onClick={() => setShowPassword(!showPassword)}>{showPassword ? <EyeOff size={16} /> : <Eye size={16} />}</button>} />
          </div>
          {isLogin && <div className="flex items-center justify-between text-sm"><label className="flex items-center gap-2"><input type="checkbox" className="rounded" />记住我</label><a href="#" className="text-primary-500 hover:underline">忘记密码?</a></div>}
          <Button type="submit" className="w-full">{isLogin ? '登录' : '注册'}</Button>
        </form>
        <p className="text-center text-sm text-gray-500 mt-4">
          {isLogin ? '还没有账户？' : '已有账户？'}
          <button onClick={() => setIsLogin(!isLogin)} className="text-primary-500 hover:underline ml-1">{isLogin ? '注册' : '登录'}</button>
        </p>
      </Card>
    </div>
  );
}
