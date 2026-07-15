import { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { cn } from '../common/Button';
import {
  LayoutDashboard, MessageSquare, Cpu, Workflow, Users, Brain, GraduationCap,
  Video, Image, Music, Box, Atom, Monitor, Shield, Globe, Search, Plug,
  UserCircle, Database, GitBranch, Scale, Bot, Activity, HardDrive, Settings,
  HelpCircle, Bell, ChevronDown, ChevronRight, Zap
} from 'lucide-react';

interface MenuItem { key: string; label: string; icon: React.ReactNode; path?: string; children?: MenuItem[]; }

const iconMap: Record<string, React.ReactNode> = {
  dashboard: <LayoutDashboard size={18} />, chat: <MessageSquare size={18} />, 'model-manager': <Cpu size={18} />,
  orchestration: <Workflow size={18} />, workflow: <GitBranch size={18} />, multiagent: <Users size={18} />,
  cognitive: <Brain size={18} />, training: <GraduationCap size={18} />, 'video-gen': <Video size={18} />,
  'image-gen': <Image size={18} />, 'audio-gen': <Music size={18} />, '3d-gen': <Box size={18} />,
  'physics-engine': <Atom size={18} />, 'computer-use': <Monitor size={18} />, security: <Shield size={18} />,
  federated: <Globe size={18} />, rag: <Search size={18} />, channels: <Zap size={18} />,
  plugins: <Plug size={18} />, personality: <UserCircle size={18} />, 'data-pipeline': <Database size={18} />,
  alignment: <Scale size={18} />, robot: <Bot size={18} />, telemetry: <Activity size={18} />,
  hardware: <HardDrive size={18} />, settings: <Settings size={18} />, help: <HelpCircle size={18} />,
  notifications: <Bell size={18} />,
};

const menuGroups: { label: string; items: MenuItem[] }[] = [
  { label: '主要功能', items: [{ key: 'dashboard', label: '仪表盘', icon: iconMap.dashboard, path: '/' }, { key: 'chat', label: '智能对话', icon: iconMap.chat, path: '/chat' }] },
  { label: '模型管理', items: [{ key: 'model-manager', label: '模型管理', icon: iconMap['model-manager'], path: '/model-manager' }, { key: 'orchestration', label: '模型编排', icon: iconMap.orchestration, path: '/orchestration' }, { key: 'workflow', label: '工作流', icon: iconMap.workflow, path: '/workflow' }] },
  { label: 'AI 系统', items: [{ key: 'multiagent', label: '多智能体', icon: iconMap.multiagent, path: '/multiagent' }, { key: 'cognitive', label: '认知系统', icon: iconMap.cognitive, path: '/cognitive' }, { key: 'training', label: '训练中心', icon: iconMap.training, path: '/training' }] },
  { label: '内容生成', items: [{ key: 'video-gen', label: '视频生成', icon: iconMap['video-gen'], path: '/video-gen' }, { key: 'image-gen', label: '图像生成', icon: iconMap['image-gen'], path: '/image-gen' }, { key: 'audio-gen', label: '音频生成', icon: iconMap['audio-gen'], path: '/audio-gen' }, { key: '3d-gen', label: '3D 生成', icon: iconMap['3d-gen'], path: '/3d-gen' }] },
  { label: '高级功能', items: [{ key: 'physics-engine', label: '物理引擎', icon: iconMap['physics-engine'], path: '/physics-engine' }, { key: 'computer-use', label: '计算机使用', icon: iconMap['computer-use'], path: '/computer-use' }, { key: 'security', label: '安全中心', icon: iconMap.security, path: '/security' }, { key: 'federated', label: '联邦学习', icon: iconMap.federated, path: '/federated' }, { key: 'rag', label: '知识检索', icon: iconMap.rag, path: '/rag' }, { key: 'channels', label: '渠道管理', icon: iconMap.channels, path: '/channels' }, { key: 'plugins', label: '插件市场', icon: iconMap.plugins, path: '/plugins' }, { key: 'personality', label: '人格引擎', icon: iconMap.personality, path: '/personality' }, { key: 'data-pipeline', label: '数据管道', icon: iconMap['data-pipeline'], path: '/data-pipeline' }, { key: 'alignment', label: '价值对齐', icon: iconMap.alignment, path: '/alignment' }, { key: 'robot', label: '机器人控制', icon: iconMap.robot, path: '/robot' }] },
  { label: '系统管理', items: [{ key: 'telemetry', label: '监控遥测', icon: iconMap.telemetry, path: '/telemetry' }, { key: 'hardware', label: '硬件管理', icon: iconMap.hardware, path: '/hardware' }, { key: 'settings', label: '系统设置', icon: iconMap.settings, path: '/settings' }, { key: 'help', label: '帮助文档', icon: iconMap.help, path: '/help' }] },
];

interface SidebarProps { collapsed?: boolean; onToggle?: () => void; }

export default function Sidebar({ collapsed = false }: SidebarProps) {
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set(['主要功能', '模型管理', 'AI 系统']));
  const location = useLocation();

  const toggleGroup = (label: string) => {
    setExpandedGroups(prev => { const next = new Set(prev); if (next.has(label)) next.delete(label); else next.add(label); return next; });
  };

  return (
    <aside className={cn('h-screen bg-white dark:bg-dark-800 border-r border-gray-200 dark:border-dark-600 flex flex-col transition-all duration-300 overflow-hidden', collapsed ? 'w-16' : 'w-64')}>
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 h-16 border-b border-gray-200 dark:border-dark-600 shrink-0">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center text-white font-bold text-sm shrink-0">U</div>
        {!collapsed && <span className="font-bold text-lg text-gray-900 dark:text-white">UFO AGI</span>}
      </div>
      {/* Nav */}
      <nav className="flex-1 overflow-y-auto py-2">
        {menuGroups.map(group => (
          <div key={group.label} className="mb-1">
            {!collapsed && <button onClick={() => toggleGroup(group.label)} className="flex items-center justify-between w-full px-4 py-2 text-xs font-semibold text-gray-400 uppercase tracking-wider hover:text-gray-600 dark:hover:text-gray-300">
              <span>{group.label}</span>
              {expandedGroups.has(group.label) ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            </button>}
            {(collapsed || expandedGroups.has(group.label)) && group.items.map(item => (
              <NavLink key={item.key} to={item.path || '/'} className={({ isActive }) => cn('flex items-center gap-3 mx-2 px-3 py-2 rounded-lg text-sm transition-colors', isActive ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-600 dark:text-primary-400 font-medium' : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-dark-700', collapsed && 'justify-center px-2')}>
                {item.icon}
                {!collapsed && <span className="truncate">{item.label}</span>}
              </NavLink>
            ))}
          </div>
        ))}
      </nav>
    </aside>
  );
}
