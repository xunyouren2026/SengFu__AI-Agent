import { cn } from './Button';

interface Column<T> { key: string; title: string; width?: string; render?: (value: unknown, row: T, index: number) => React.ReactNode; sortable?: boolean; align?: 'left' | 'center' | 'right'; }

interface TableProps<T> { columns: Column<T>[]; data: T[]; keyField?: string; loading?: boolean; emptyText?: string; onRowClick?: (row: T) => void; className?: string; }

export default function Table<T extends Record<string, unknown>>({ columns, data, keyField = 'id', loading, emptyText = '暂无数据', onRowClick, className }: TableProps<T>) {
  if (loading) return <div className="animate-pulse space-y-2 p-4">{Array.from({ length: 5 }).map((_, i) => <div key={i} className="h-10 bg-gray-100 dark:bg-dark-700 rounded" />)}</div>;
  return (
    <div className={cn('overflow-x-auto rounded-lg border border-gray-200 dark:border-dark-600', className)}>
      <table className="w-full text-sm">
        <thead><tr className="bg-gray-50 dark:bg-dark-700">{columns.map(col => (
          <th key={col.key} className={cn('px-4 py-3 font-medium text-gray-500 dark:text-gray-400 text-left', col.align === 'center' && 'text-center', col.align === 'right' && 'text-right')} style={col.width ? { width: col.width } : undefined}>{col.title}</th>
        ))}</tr></thead>
        <tbody className="divide-y divide-gray-100 dark:divide-dark-600">
          {data.length === 0 ? <tr><td colSpan={columns.length} className="px-4 py-8 text-center text-gray-400">{emptyText}</td></tr> :
            data.map((row, idx) => (
              <tr key={String(row[keyField] ?? idx)} onClick={() => onRowClick?.(row)} className={cn('hover:bg-gray-50 dark:hover:bg-dark-700/50 transition-colors', onRowClick && 'cursor-pointer')}>
                {columns.map(col => (
                  <td key={col.key} className={cn('px-4 py-3 text-gray-700 dark:text-gray-300', col.align === 'center' && 'text-center', col.align === 'right' && 'text-right')}>
                    {col.render ? col.render(row[col.key], row, idx) : String(row[col.key] ?? '-')}
                  </td>
                ))}
              </tr>
            ))}
        </tbody>
      </table>
    </div>
  );
}
