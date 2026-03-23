import { useState } from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { LoginForm } from '@/components/auth/LoginForm';
import { RegisterForm } from '@/components/auth/RegisterForm';
import { Hexagon } from 'lucide-react';

export function AuthPage() {
  const [isLogin, setIsLogin] = useState(true);
  const { isAuthenticated } = useAuth();

  if (isAuthenticated) {
    return <Navigate to="/" replace />;
  }

  return (
    <div className="min-h-screen flex" style={{ background: 'var(--bg-primary)' }}>
      {/* Left Panel - Branding */}
      <div 
        className="hidden lg:flex lg:w-1/2 flex-col justify-center items-center relative overflow-hidden"
        style={{ 
          background: 'linear-gradient(135deg, #1e3a5f 0%, #0f172a 50%, #09090B 100%)' 
        }}
      >
        <div className="relative z-10 px-16">
          <div className="flex items-center gap-3 mb-8">
            <Hexagon size={40} className="text-blue-400" strokeWidth={1.5} />
            <span className="font-heading text-3xl font-bold text-white tracking-tighter">
              BI Platform
            </span>
          </div>
          <h1 className="font-heading text-4xl font-bold text-white mb-4 leading-tight">
            Enterprise Analytics.<br />Reimagined.
          </h1>
          <p className="text-blue-200/70 text-lg max-w-md leading-relaxed">
            Build interactive reports, explore data with 20+ chart types, 
            and share insights across your organization.
          </p>
          <div className="mt-12 grid grid-cols-3 gap-6">
            {['20+ Charts', 'Cross-filter', 'Drag & Drop'].map((f) => (
              <div key={f} className="text-center">
                <div className="text-sm font-medium text-blue-300">{f}</div>
              </div>
            ))}
          </div>
        </div>
        
        {/* Background decoration */}
        <div className="absolute inset-0 opacity-10">
          {[...Array(20)].map((_, i) => (
            <div
              key={i}
              className="absolute rounded-full"
              style={{
                width: Math.random() * 300 + 50,
                height: Math.random() * 300 + 50,
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                background: 'radial-gradient(circle, rgba(59,130,246,0.2) 0%, transparent 70%)',
              }}
            />
          ))}
        </div>
      </div>

      {/* Right Panel - Form */}
      <div className="flex-1 flex items-center justify-center p-8">
        {isLogin ? (
          <>
            <LoginForm />
            <p className="mt-6 text-center text-sm absolute bottom-8" style={{ color: 'var(--text-muted)' }}>
              Don't have an account?{' '}
              <button onClick={() => setIsLogin(false)} className="font-medium" style={{ color: 'var(--accent)' }}>
                Sign up
              </button>
            </p>
          </>
        ) : (
          <RegisterForm onToggle={() => setIsLogin(true)} />
        )}
      </div>
    </div>
  );
}
