import { create } from 'zustand';
import { CheckCircle, XCircle, AlertTriangle, Info, X } from 'lucide-react';
import { cn } from './Button';

interface Toast { id: string; type: 'success' | 'error' | 'warning' | 'info'; title: string; message?: string; duration?: number; }

interface ToastStore { toasts: Toast[]; addToast: (toast: Omit<Toast, 'id'>) => void; removeToast: (id: string) => void; }

export const useToastStore = create<ToastStore>((set) => ({
  toasts: [],
  addToast: (toast) => {
    const id = Date.now().toString() + Math.random().toString(36).slice(2);
    set(s => ({ toasts: [...s.toasts, { ...toast, id }] }));
    setTimeout(() => set(s => ({ toasts: s.toasts.filter(t => t.id !== id) })), toast.duration || 5000);
  },
  removeToast: (id) => set(s => ({ toasts: s.toasts.filter(t => t.id !== id) })),
}));

const icons = { success: CheckCircle, error: XCircle, warning: AlertTriangle, info: Info };
const colors = { success: 'border-green-500 bg-green-50 dark:bg-green-900/20', error: 'border-red-500 bg-red-50 dark:bg-red-900/20', warning: 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20', info: 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' };
const iconColors = { success: 'text-green-500', error: 'text-red-500', warning: 'text-yellow-500', info: 'text-blue-500' };

export function ToastContainer() {
  const { toasts, removeToast } = useToastStore();
  return (
    <div className="fixed top-4 right-4 z-[100] flex flex-col gap-2 max-w-sm">
      {toasts.map(toast => {
        const Icon = icons[toast.type];
        return (
          <div key={toast.id} className={cn('flex items-start gap-3 p-4 rounded-lg border-l-4 shadow-lg animate-slideInRight', colors[toast.type])}>
            <Icon size={20} className={cn('mt-0.5 shrink-0', iconColors[toast.type])} />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-900 dark:text-white">{toast.title}</p>
              {toast.message && <p className="text-sm text-gray-600 dark:text-gray-400 mt-0.5">{toast.message}</p>}
            </div>
            <button onClick={() => removeToast(toast.id)} className="shrink-0 p-0.5 rounded hover:bg-black/10 dark:hover:bg-white/10"><X size={16} className="text-gray-400" /></button>
          </div>
        );
      })}
    </div>
  );
}
