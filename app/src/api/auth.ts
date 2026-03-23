import apiClient from './client';
import type { User, LoginCredentials, RegisterData, TokenResponse } from '@/types';

export const authApi = {
  login: async (credentials: LoginCredentials): Promise<TokenResponse> => {
    const response = await apiClient.post('/api/auth/login', credentials);
    return response.data;
  },

  register: async (data: RegisterData): Promise<TokenResponse> => {
    const response = await apiClient.post('/api/auth/register', data);
    return response.data;
  },

  logout: async (refreshToken: string): Promise<void> => {
    await apiClient.post('/api/auth/logout', { refresh_token: refreshToken });
  },

  refreshToken: async (refreshToken: string): Promise<TokenResponse> => {
    const response = await apiClient.post('/api/auth/refresh', { refresh_token: refreshToken });
    return response.data;
  },

  getMe: async (): Promise<User> => {
    const response = await apiClient.get('/api/auth/me');
    return response.data;
  },

  updateMe: async (data: Partial<User>): Promise<User> => {
    const response = await apiClient.put('/api/auth/me', data);
    return response.data;
  },

  changePassword: async (currentPassword: string, newPassword: string): Promise<void> => {
    await apiClient.post('/api/auth/change-password', {
      current_password: currentPassword,
      new_password: newPassword,
    });
  },

  // Workspaces
  getWorkspaces: async () => {
    const response = await apiClient.get('/api/auth/workspaces');
    return response.data;
  },

  createWorkspace: async (data: { name: string; description?: string }) => {
    const response = await apiClient.post('/api/auth/workspaces', data);
    return response.data;
  },

  getWorkspace: async (id: string) => {
    const response = await apiClient.get(`/api/auth/workspaces/${id}`);
    return response.data;
  },
};
