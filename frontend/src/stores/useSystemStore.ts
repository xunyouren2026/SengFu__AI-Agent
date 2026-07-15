import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface SystemState {
  theme: 'light' | 'dark';
  language: 'zh' | 'en';
  sidebarCollapsed: boolean;
  toggleTheme: () => void;
  toggleLanguage: () => void;
  toggleSidebar: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
}

export const useSystemStore = create<SystemState>()(
  persist(
    (set) => ({
      theme: 'light',
      language: 'zh',
      sidebarCollapsed: false,
      toggleTheme: () => set((s) => {
        const next = s.theme === 'light' ? 'dark' : 'light';
        document.documentElement.classList.toggle('dark', next === 'dark');
        return { theme: next };
      }),
      toggleLanguage: () => set((s) => ({ language: s.language === 'zh' ? 'en' : 'zh' })),
      toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
      setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
    }),
    { name: 'ufo-system-settings', partialize: (state) => ({ theme: state.theme, language: state.language, sidebarCollapsed: state.sidebarCollapsed }) }
  )
);
