import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Eye, EyeOff, ArrowRight, Hexagon } from 'lucide-react';

interface RegisterFormProps {
  onToggle: () => void;
}

export function RegisterForm({ onToggle }: RegisterFormProps) {
  const navigate = useNavigate();
  const { register } = useAuth();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      await register({ email, password, name });
      navigate('/');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Registration failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="w-full max-w-sm animate-fade-in">
      <div className="flex items-center gap-3 mb-10 justify-center lg:justify-start">
        <Hexagon size={32} style={{ color: 'var(--accent)' }} strokeWidth={1.5} />
        <span className="font-heading text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>
          BI Platform
        </span>
      </div>

      <h2 className="font-heading text-2xl font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
        Create account
      </h2>
      <p className="text-sm mb-8" style={{ color: 'var(--text-muted)' }}>
        Get started with your analytics
      </p>

      {error && (
        <Alert variant="destructive" className="mb-4">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <Label htmlFor="name" className="text-xs font-medium mb-1.5 block" style={{ color: 'var(--text-secondary)' }}>
            Name
          </Label>
          <Input
            id="name"
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Your name"
            required
            className="w-full"
            style={{
              background: 'var(--bg-tertiary)',
              borderColor: 'var(--border-color)',
              color: 'var(--text-primary)',
            }}
          />
        </div>

        <div>
          <Label htmlFor="email" className="text-xs font-medium mb-1.5 block" style={{ color: 'var(--text-secondary)' }}>
            Email
          </Label>
          <Input
            id="email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@company.com"
            required
            className="w-full"
            style={{
              background: 'var(--bg-tertiary)',
              borderColor: 'var(--border-color)',
              color: 'var(--text-primary)',
            }}
          />
        </div>

        <div>
          <Label htmlFor="password" className="text-xs font-medium mb-1.5 block" style={{ color: 'var(--text-secondary)' }}>
            Password
          </Label>
          <div className="relative">
            <Input
              id="password"
              type={showPassword ? 'text' : 'password'}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Min 6 characters"
              required
              minLength={6}
              className="w-full pr-10"
              style={{
                background: 'var(--bg-tertiary)',
                borderColor: 'var(--border-color)',
                color: 'var(--text-primary)',
              }}
            />
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="absolute right-3 top-1/2 -translate-y-1/2"
              style={{ color: 'var(--text-muted)' }}
            >
              {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
            </button>
          </div>
        </div>

        <Button
          type="submit"
          disabled={isLoading}
          className="w-full flex items-center justify-center gap-2"
          style={{ background: 'var(--accent)' }}
        >
          {isLoading ? 'Creating account...' : 'Create Account'}
          {!isLoading && <ArrowRight size={16} />}
        </Button>
      </form>

      <p className="mt-6 text-center text-sm" style={{ color: 'var(--text-muted)' }}>
        Already have an account?{' '}
        <button onClick={onToggle} className="font-medium" style={{ color: 'var(--accent)' }}>
          Sign in
        </button>
      </p>
    </div>
  );
}
