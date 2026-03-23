import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { Layout } from './Layout';

interface PrivateRouteProps {
  children: React.ReactNode;
}

export function PrivateRoute({ children }: PrivateRouteProps) {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen" style={{ background: 'var(--bg-primary)' }}>
        <div className="animate-spin rounded-full h-8 w-8 border-b-2" style={{ borderColor: 'var(--accent)' }} />
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <Layout>{children}</Layout>;
}
