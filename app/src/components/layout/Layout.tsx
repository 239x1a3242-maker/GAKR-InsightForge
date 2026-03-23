import React, { useState } from 'react';
import { Sidebar } from './Sidebar';

interface LayoutProps {
  children: React.ReactNode;
}

export function Layout({ children }: LayoutProps) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: 'var(--bg-primary)' }}>
      <Sidebar collapsed={collapsed} onToggle={() => setCollapsed(!collapsed)} />
      <main
        className="flex-1 overflow-auto transition-all duration-300"
        style={{
          marginLeft: collapsed ? '64px' : '240px',
          background: 'var(--bg-primary)',
        }}
      >
        <div className="p-6 animate-fade-in">
          {children}
        </div>
      </main>
    </div>
  );
}
