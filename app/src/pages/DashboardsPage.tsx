import { useEffect, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { reportsApi } from '@/api/reports';
import type { Dashboard } from '@/types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { 
  PieChart, Plus, Trash2, Search, Star, Share2, 
  Layout
} from 'lucide-react';
import { formatDistanceToNow } from '@/utils/format';

export function DashboardsPage() {
  const navigate = useNavigate();
  const [dashboards, setDashboards] = useState<Dashboard[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [newDashboardName, setNewDashboardName] = useState('');
  const [isCreating, setIsCreating] = useState(false);

  const loadDashboards = useCallback(async () => {
    try {
      const data = await reportsApi.getDashboards();
      setDashboards(data);
    } catch (error) {
      console.error('Failed to load dashboards:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadDashboards();
  }, [loadDashboards]);

  const handleCreate = async () => {
    if (!newDashboardName) return;

    setIsCreating(true);
    try {
      const dashboard = await reportsApi.createDashboard({
        name: newDashboardName,
      });
      navigate(`/dashboards/${dashboard.id}`);
    } catch (error) {
      console.error('Create failed:', error);
    } finally {
      setIsCreating(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this dashboard?')) return;

    try {
      await reportsApi.deleteDashboard(id);
      await loadDashboards();
    } catch (error) {
      console.error('Delete failed:', error);
    }
  };

  const handleToggleFavorite = async (dashboard: Dashboard) => {
    try {
      await reportsApi.updateDashboard(dashboard.id, {
        is_favorite: !dashboard.is_favorite,
      });
      await loadDashboards();
    } catch (error) {
      console.error('Update failed:', error);
    }
  };

  const filteredDashboards = dashboards.filter(d =>
    d.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2" style={{ borderColor: 'var(--accent)' }} />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-heading text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>
            Dashboards
          </h1>
          <p style={{ color: 'var(--text-secondary)' }}>
            Monitor your key metrics at a glance
          </p>
        </div>
        <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
          <DialogTrigger asChild>
            <Button style={{ background: 'var(--accent)' }}>
              <Plus size={16} className="mr-2" />
              New Dashboard
            </Button>
          </DialogTrigger>
          <DialogContent style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
            <DialogHeader>
              <DialogTitle style={{ color: 'var(--text-primary)' }}>Create New Dashboard</DialogTitle>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div>
                <Label htmlFor="name" style={{ color: 'var(--text-secondary)' }}>Dashboard Name</Label>
                <Input
                  id="name"
                  value={newDashboardName}
                  onChange={(e) => setNewDashboardName(e.target.value)}
                  placeholder="Enter dashboard name"
                  className="mt-1"
                  style={{
                    background: 'var(--bg-tertiary)',
                    borderColor: 'var(--border-color)',
                    color: 'var(--text-primary)',
                  }}
                />
              </div>
              <Button 
                onClick={handleCreate} 
                disabled={!newDashboardName || isCreating}
                className="w-full"
                style={{ background: 'var(--accent)' }}
              >
                {isCreating ? 'Creating...' : 'Create Dashboard'}
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2" size={18} style={{ color: 'var(--text-muted)' }} />
        <Input
          placeholder="Search dashboards..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="pl-10"
          style={{
            background: 'var(--bg-secondary)',
            borderColor: 'var(--border-color)',
            color: 'var(--text-primary)',
          }}
        />
      </div>

      {/* Dashboards Grid */}
      {filteredDashboards.length === 0 ? (
        <Card style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
          <CardContent className="flex flex-col items-center justify-center py-16">
            <PieChart size={64} style={{ color: 'var(--text-muted)' }} className="mb-4" />
            <h3 className="text-lg font-medium mb-2" style={{ color: 'var(--text-primary)' }}>
              No dashboards yet
            </h3>
            <p className="text-center mb-6 max-w-md" style={{ color: 'var(--text-muted)' }}>
              Create your first dashboard to monitor your key metrics
            </p>
            <Button onClick={() => setCreateDialogOpen(true)} style={{ background: 'var(--accent)' }}>
              <Plus size={16} className="mr-2" />
              Create Dashboard
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredDashboards.map((dashboard) => (
            <Card 
              key={dashboard.id} 
              className="group cursor-pointer transition-all hover:shadow-lg"
              style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}
              onClick={() => navigate(`/dashboards/${dashboard.id}`)}
            >
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <div 
                      className="p-2 rounded-lg"
                      style={{ background: 'var(--warning)20' }}
                    >
                      <Layout size={20} style={{ color: 'var(--warning)' }} />
                    </div>
                    <div>
                      <CardTitle className="text-base font-medium" style={{ color: 'var(--text-primary)' }}>
                        {dashboard.name}
                      </CardTitle>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        {dashboard.widgets?.length || 0} widgets
                      </p>
                    </div>
                  </div>
                  <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleToggleFavorite(dashboard);
                      }}
                    >
                      <Star 
                        size={16} 
                        style={{ 
                          color: dashboard.is_favorite ? 'var(--warning)' : 'var(--text-muted)',
                          fill: dashboard.is_favorite ? 'var(--warning)' : 'none'
                        }} 
                      />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(dashboard.id);
                      }}
                    >
                      <Trash2 size={16} style={{ color: 'var(--danger)' }} />
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between text-sm">
                  <span style={{ color: 'var(--text-muted)' }}>
                    Updated {formatDistanceToNow(dashboard.updated_at)}
                  </span>
                  {dashboard.is_shared && (
                    <Share2 size={14} style={{ color: 'var(--success)' }} />
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
