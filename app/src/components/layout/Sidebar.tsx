import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { useTheme } from '@/contexts/ThemeContext';
import {
  LayoutDashboard, Database, FileBarChart,
  Sun, Moon, LogOut, ChevronLeft, ChevronRight, Hexagon,
  Brain, BarChart3
} from 'lucide-react';

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

const navItems = [
  { path: '/', icon: LayoutDashboard, label: 'Home' },
  { path: '/datasets', icon: Database, label: 'Datasets' },
  { path: '/analysis', icon: BarChart3, label: 'Analysis' },
  { path: '/reports', icon: FileBarChart, label: 'Reports & Dashboards' },
  { path: '/ml', icon: Brain, label: 'Machine Learning' },
];

export function Sidebar({ collapsed, onToggle }: SidebarProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useAuth();
  const { theme, toggleTheme } = useTheme();

  const isActive = (path: string) => {
    if (path === '/') return location.pathname === '/';
    return location.pathname.startsWith(path);
  };

  return (
    <aside
      className="fixed top-0 left-0 h-full z-30 flex flex-col transition-all duration-300"
      style={{
        width: collapsed ? '64px' : '240px',
        background: 'var(--bg-secondary)',
        borderRight: '1px solid var(--border-color)',
      }}
    >
      {/* Logo */}
      <div 
        className="flex items-center h-14 px-4 gap-3"
        style={{ borderBottom: '1px solid var(--border-color)' }}
      >
        <Hexagon size={24} style={{ color: 'var(--accent)', flexShrink: 0 }} strokeWidth={1.5} />
        {!collapsed && (
          <span className="font-heading font-semibold text-sm tracking-tighter whitespace-nowrap" style={{ color: 'var(--text-primary)' }}>
            BI Platform
          </span>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-3 px-2 space-y-1 overflow-y-auto">
        {navItems.map(({ path, icon: Icon, label }) => (
          <button
            key={path}
            onClick={() => navigate(path)}
            className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-all duration-150"
            style={{
              background: isActive(path) ? 'var(--accent)' : 'transparent',
              color: isActive(path) ? '#fff' : 'var(--text-secondary)',
            }}
            onMouseEnter={(e) => {
              if (!isActive(path)) (e.currentTarget as HTMLButtonElement).style.background = 'var(--surface-hover)';
            }}
            onMouseLeave={(e) => {
              if (!isActive(path)) (e.currentTarget as HTMLButtonElement).style.background = 'transparent';
            }}
          >
            <Icon size={18} strokeWidth={1.5} style={{ flexShrink: 0 }} />
            {!collapsed && <span>{label}</span>}
          </button>
        ))}
      </nav>

      {/* Bottom Actions */}
      <div className="px-2 py-3 space-y-1" style={{ borderTop: '1px solid var(--border-color)' }}>
        <button
          onClick={toggleTheme}
          className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-all duration-150"
          style={{ color: 'var(--text-secondary)' }}
          onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.background = 'var(--surface-hover)'; }}
          onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.background = 'transparent'; }}
        >
          {theme === 'dark' ? <Sun size={18} strokeWidth={1.5} /> : <Moon size={18} strokeWidth={1.5} />}
          {!collapsed && <span>{theme === 'dark' ? 'Light Mode' : 'Dark Mode'}</span>}
        </button>

        <button
          onClick={onToggle}
          className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-all duration-150"
          style={{ color: 'var(--text-secondary)' }}
          onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.background = 'var(--surface-hover)'; }}
          onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.background = 'transparent'; }}
        >
          {collapsed ? <ChevronRight size={18} strokeWidth={1.5} /> : <ChevronLeft size={18} strokeWidth={1.5} />}
          {!collapsed && <span>Collapse</span>}
        </button>

        {user && (
          <button
            onClick={logout}
            className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-all duration-150"
            style={{ color: 'var(--danger)' }}
            onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.background = 'var(--surface-hover)'; }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.background = 'transparent'; }}
          >
            <LogOut size={18} strokeWidth={1.5} />
            {!collapsed && <span>Logout</span>}
          </button>
        )}
      </div>

      {/* User Info */}
      {!collapsed && user && (
        <div className="px-4 py-3" style={{ borderTop: '1px solid var(--border-color)' }}>
          <p className="text-xs font-medium truncate" style={{ color: 'var(--text-primary)' }}>
            {user.name}
          </p>
          <p className="text-xs truncate" style={{ color: 'var(--text-muted)' }}>
            {user.email}
          </p>
        </div>
      )}
    </aside>
  );
}
