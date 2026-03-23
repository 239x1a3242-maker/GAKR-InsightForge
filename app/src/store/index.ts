import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { User } from '@/types';

interface AppState {
  user: User | null;
  isAuthenticated: boolean;
  setUser: (user: User | null) => void;
  setAuthenticated: (value: boolean) => void;
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;
  selectedDatasetId: string | null;
  setSelectedDatasetId: (id: string | null) => void;
  activeModelId: string | null;
  setActiveModelId: (id: string | null) => void;
  notifications: Array<{
    id: string;
    type: 'success' | 'error' | 'warning' | 'info';
    message: string;
  }>;
  addNotification: (notification: Omit<AppState['notifications'][0], 'id'>) => void;
  removeNotification: (id: string) => void;
}

export const useStore = create<AppState>()(
  persist(
    (set) => ({
      user: null,
      isAuthenticated: false,
      setUser: (user) => set({ user }),
      setAuthenticated: (value) => set({ isAuthenticated: value }),
      sidebarOpen: true,
      setSidebarOpen: (open) => set({ sidebarOpen: open }),
      selectedDatasetId: null,
      setSelectedDatasetId: (id) => set({ selectedDatasetId: id }),
      activeModelId: null,
      setActiveModelId: (id) => set({ activeModelId: id }),
      notifications: [],
      addNotification: (notification) =>
        set((state) => ({
          notifications: [
            ...state.notifications,
            { ...notification, id: Math.random().toString(36).substr(2, 9) },
          ],
        })),
      removeNotification: (id) =>
        set((state) => ({
          notifications: state.notifications.filter((n) => n.id !== id),
        })),
    }),
    {
      name: 'bi-platform-storage',
      partialize: (state) => ({ user: state.user, isAuthenticated: state.isAuthenticated }),
    }
  )
);
