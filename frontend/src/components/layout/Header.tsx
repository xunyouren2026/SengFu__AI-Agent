import { useState, useRef, useEffect } from 'react';
import { Search, Bell, Moon, Sun, Globe, User, LogOut, Settings, ChevronDown } from 'lucide-react';
import { cn } from '../common/Button';
import { useSystemStore } from '../../stores/useSystemStore';

export default function Header() {
  const [searchOpen, setSearchOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const [notifOpen, setNotifOpen] = useState(false);
  const profileRef = useRef<HTMLDivElement>(null);
  const notifRef = useRef<HTMLDivElement>(null);
  const { theme, toggleTheme, language, toggleLanguage } = useSystemStore();

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (profileRef.current && !profileRef.current.contains(e.target as Node)) setProfileOpen(false);
      if (notifRef.current && !notifRef.current.contains(e.target as Node)) setNotifOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  return (
    <header className="h-16 bg-white dark:bg-dark-800 border-b border-gray-200 dark:border-dark-600 flex items-center justify-between px-6 shrink-0">
      {/* Search */}
      <div className="flex-1 max-w-md">
        <div className="relative">
          <Search size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
          <input type="text" placeholder={'搜索功能、模型、设置...'} className="w-full pl-10 pr-4 py-2 rounded-lg border border-gray-200 dark:border-dark-600 bg-gray-50 dark:bg-dark-700 text-sm text-gray-900 dark:text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent" onFocus={() => setSearchOpen(true)} onBlur={() => setSearchOpen(false)} />
        </div>
      </div>
      {/* Actions */}
      <div className="flex items-center gap-2">
        <button onClick={toggleLanguage} className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-dark-700 text-gray-500 dark:text-gray-400" title={'切换语言'}>
          <Globe size={18} />
          <span className="text-xs ml-1">{language === 'zh' ? '中' : 'EN'}</span>
        </button>
        <button onClick={toggleTheme} className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-dark-700 text-gray-500 dark:text-gray-400" title={'切换主题'}>
          {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
        </button>
        {/* Notifications */}
        <div ref={notifRef} className="relative">
          <button onClick={() => setNotifOpen(!notifOpen)} className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-dark-700 text-gray-500 dark:text-gray-400 relative">
            <Bell size={18} />
            <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full" />
          </button>
          {notifOpen && (
            <div className="absolute right-0 top-full mt-2 w-80 bg-white dark:bg-dark-800 rounded-xl shadow-xl border border-gray-200 dark:border-dark-600 z-50">
              <div className="p-4 border-b border-gray-100 dark:border-dark-600 flex items-center justify-between">
                <h3 className="font-medium text-gray-900 dark:text-white">通知</h3>
                <button className="text-xs text-primary-500 hover:underline">全部已读</button>
              </div>
              <div className="p-4 text-center text-sm text-gray-400">暂无新通知</div>
            </div>
          )}
        </div>
        {/* Profile */}
        <div ref={profileRef} className="relative">
          <button onClick={() => setProfileOpen(!profileOpen)} className="flex items-center gap-2 p-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-dark-700">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary-400 to-primary-600 flex items-center justify-center text-white text-sm font-medium">U</div>
            {!profileOpen && <ChevronDown size={14} className="text-gray-400" />}
          </button>
          {profileOpen && (
            <div className="absolute right-0 top-full mt-2 w-48 bg-white dark:bg-dark-800 rounded-xl shadow-xl border border-gray-200 dark:border-dark-600 z-50 py-1">
              <button className="flex items-center gap-2 w-full px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-dark-700"><User size={16} />个人资料</button>
              <button className="flex items-center gap-2 w-full px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-dark-700"><Settings size={16} />用户设置</button>
              <hr className="my-1 border-gray-100 dark:border-dark-600" />
              <button className="flex items-center gap-2 w-full px-4 py-2 text-sm text-red-500 hover:bg-gray-50 dark:hover:bg-dark-700"><LogOut size={16} />退出登录</button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
