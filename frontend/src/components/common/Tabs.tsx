import { useState } from 'react';
import { cn } from './Button';

interface Tab {
  key: string;
  label: string;
  icon?: React.ReactNode;
  badge?: number | string;
  disabled?: boolean;
}

interface TabsProps {
  tabs: Tab[];
  activeKey?: string;
  onChange?: (key: string) => void;
  variant?: 'line' | 'card' | 'pill';
  className?: string;
}

export default function Tabs({ tabs, activeKey, onChange, variant = 'line', className }: TabsProps) {
  const [internalKey, setInternalKey] = useState(tabs[0]?.key || '');
  const currentKey = activeKey ?? internalKey;

  const handleClick = (key: string) => {
    if (!tabs.find(t => t.key === key)?.disabled) {
      setInternalKey(key);
      onChange?.(key);
    }
  };

  const variants = {
    line: { container: 'border-b border-gray-200 dark:border-dark-600', tab: 'border-b-2 border-transparent', active: 'border-primary-500 text-primary-600 dark:text-primary-400' },
    card: { container: 'bg-gray-100 dark:bg-dark-700 rounded-lg p-1', tab: 'rounded-md', active: 'bg-white dark:bg-dark-800 shadow-sm text-gray-900 dark:text-white' },
    pill: { container: 'gap-1', tab: 'rounded-full px-4 py-1.5', active: 'bg-primary-500 text-white' },
  };

  const v = variants[variant];

  return (
    <div className={cn('flex', v.container, className)}>
      {tabs.map(tab => (
        <button key={tab.key} onClick={() => handleClick(tab.key)} disabled={tab.disabled}
          className={cn('flex items-center gap-2 px-4 py-2 text-sm font-medium transition-colors whitespace-nowrap', v.tab, currentKey === tab.key ? v.active : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300', tab.disabled && 'opacity-50 cursor-not-allowed')}>
          {tab.icon}
          {tab.label}
          {tab.badge !== undefined && <span className={cn('px-1.5 py-0.5 text-xs rounded-full', currentKey === tab.key ? 'bg-primary-100 text-primary-600' : 'bg-gray-100 text-gray-500')}>{tab.badge}</span>}
        </button>
      ))}
    </div>
  );
}

interface TabPanelProps { children: React.ReactNode; className?: string }
export function TabPanel({ children, className }: TabPanelProps) {
  return <div className={cn('mt-4', className)}>{children}</div>;
}
