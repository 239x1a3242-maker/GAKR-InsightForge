import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import type { User, LoginCredentials, RegisterData } from '@/types';
import { authApi } from '@/api/auth';

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (credentials: LoginCredentials) => Promise<void>;
  register: (data: RegisterData) => Promise<void>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const initAuth = async () => {
      const token = localStorage.getItem('bi_token');
      if (token) {
        try {
          const userData = await authApi.getMe();
          setUser(userData);
        } catch (error) {
          localStorage.removeItem('bi_token');
          localStorage.removeItem('bi_refresh_token');
        }
      }
      setIsLoading(false);
    };

    initAuth();
  }, []);

  const login = useCallback(async (credentials: LoginCredentials) => {
    const response = await authApi.login(credentials);
    localStorage.setItem('bi_token', response.access_token);
    localStorage.setItem('bi_refresh_token', response.refresh_token);
    setUser(response.user);
  }, []);

  const register = useCallback(async (data: RegisterData) => {
    const response = await authApi.register(data);
    localStorage.setItem('bi_token', response.access_token);
    localStorage.setItem('bi_refresh_token', response.refresh_token);
    setUser(response.user);
  }, []);

  const logout = useCallback(async () => {
    const refreshToken = localStorage.getItem('bi_refresh_token');
    if (refreshToken) {
      try {
        await authApi.logout(refreshToken);
      } catch (error) {
        // Ignore error
      }
    }
    localStorage.removeItem('bi_token');
    localStorage.removeItem('bi_refresh_token');
    setUser(null);
  }, []);

  const refreshUser = useCallback(async () => {
    const userData = await authApi.getMe();
    setUser(userData);
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated: !!user,
        login,
        register,
        logout,
        refreshUser,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
