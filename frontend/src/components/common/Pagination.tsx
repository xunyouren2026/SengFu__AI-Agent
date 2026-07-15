import { ChevronLeft, ChevronRight } from 'lucide-react';
import { cn } from './Button';

interface PaginationProps {
  page: number;
  pageSize: number;
  total: number;
  onChange: (page: number) => void;
  className?: string;
}

export default function Pagination({ page, pageSize, total, onChange, className }: PaginationProps) {
  const totalPages = Math.ceil(total / pageSize);
  if (totalPages <= 1) return null;
  return (
    <div className={cn('flex items-center justify-between px-1 py-3', className)}>
      <p className="text-sm text-gray-500">共 {total} 条</p>
      <div className="flex items-center gap-1">
        <button onClick={() => onChange(page - 1)} disabled={page <= 1} className="p-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-dark-700 disabled:opacity-50 disabled:cursor-not-allowed"><ChevronLeft size={18} /></button>
        {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
          let pageNum: number;
          if (totalPages <= 5) pageNum = i + 1;
          else if (page <= 3) pageNum = i + 1;
          else if (page >= totalPages - 2) pageNum = totalPages - 4 + i;
          else pageNum = page - 2 + i;
          return <button key={pageNum} onClick={() => onChange(pageNum)} className={cn('w-8 h-8 rounded-lg text-sm font-medium', page === pageNum ? 'bg-primary-500 text-white' : 'hover:bg-gray-100 dark:hover:bg-dark-700 text-gray-700 dark:text-gray-300')}>{pageNum}</button>;
        })}
        <button onClick={() => onChange(page + 1)} disabled={page >= totalPages} className="p-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-dark-700 disabled:opacity-50 disabled:cursor-not-allowed"><ChevronRight size={18} /></button>
      </div>
    </div>
  );
}
