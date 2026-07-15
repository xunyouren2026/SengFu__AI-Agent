import { cn } from './Button';

interface StatsCardProps {
  title: string;
  value: string | number;
  icon?: React.ReactNode;
  change?: number;
  changeLabel?: string;
  color?: 'blue' | 'green' | 'yellow' | 'red' | 'purple' | 'indigo';
  className?: string;
}

export default function StatsCard({ title, value, icon, change, changeLabel, color = 'blue', className }: StatsCardProps) {
  const colors = { blue: 'from-blue-500 to-blue-600', green: 'from-green-500 to-green-600', yellow: 'from-yellow-500 to-yellow-600', red: 'from-red-500 to-red-600', purple: 'from-purple-500 to-purple-600', indigo: 'from-indigo-500 to-indigo-600' };
  return (
    <div className={cn('bg-white dark:bg-dark-800 rounded-xl p-5 shadow-sm border border-gray-100 dark:border-dark-600', className)}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-500 dark:text-gray-400">{title}</p>
          <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">{value}</p>
          {change !== undefined && (
            <div className="flex items-center gap-1 mt-1">
              <span className={cn('text-xs font-medium', change >= 0 ? 'text-green-500' : 'text-red-500')}>
                {change >= 0 ? '↑' : '↓'} {Math.abs(change)}%
              </span>
              {changeLabel && <span className="text-xs text-gray-400">{changeLabel}</span>}
            </div>
          )}
        </div>
        {icon && <div className={cn('w-12 h-12 rounded-xl bg-gradient-to-br flex items-center justify-center text-white', colors[color])}>{icon}</div>}
      </div>
    </div>
  );
}
