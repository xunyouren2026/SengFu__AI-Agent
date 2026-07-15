import { cn } from './Button';

interface LoadingProps {
  size?: 'sm' | 'md' | 'lg';
  text?: string;
  fullScreen?: boolean;
  className?: string;
}

export default function Loading({ size = 'md', text, fullScreen = false, className }: LoadingProps) {
  const sizes = { sm: 'w-5 h-5', md: 'w-8 h-8', lg: 'w-12 h-12' };
  const content = (
    <div className={cn('flex flex-col items-center justify-center gap-3', className)}>
      <div className={cn('animate-spin rounded-full border-2 border-gray-300 border-t-primary-500', sizes[size])} />
      {text && <p className="text-sm text-gray-500 dark:text-gray-400">{text}</p>}
    </div>
  );
  if (fullScreen) return <div className="fixed inset-0 z-50 flex items-center justify-center bg-white/80 dark:bg-dark-900/80 backdrop-blur-sm">{content}</div>;
  return content;
}

export function Skeleton({ className, rows = 1 }: { className?: string; rows?: number }) {
  return (
    <div className="space-y-2">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className={cn('h-4 bg-gray-200 dark:bg-dark-600 rounded animate-pulse', className)} />
      ))}
    </div>
  );
}

export function PageLoading() {
  return (
    <div className="flex items-center justify-center min-h-[400px]">
      <Loading size="lg" text={'加载中...'} />
    </div>
  );
}
