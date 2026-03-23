import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { datasetsApi } from '@/api/datasets';
import { reportsApi } from '@/api/reports';
import type { Dataset, Report, Dashboard } from '@/types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { 
  Database, FileBarChart, PieChart, 
  Plus, ArrowRight, Activity
} from 'lucide-react';

export function HomePage() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [reports, setReports] = useState<Report[]>([]);
  const [dashboards, setDashboards] = useState<Dashboard[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      try {
        const [dsData, rptData, dbData] = await Promise.all([
          datasetsApi.getDatasets(),
          reportsApi.getReports(),
          reportsApi.getDashboards(),
        ]);
        setDatasets(dsData);
        setReports(rptData);
        setDashboards(dbData);
      } catch (error) {
        console.error('Failed to load data:', error);
      } finally {
        setIsLoading(false);
      }
    };

    loadData();
  }, []);

  const stats = [
    { label: 'Datasets', value: datasets.length, icon: Database, color: 'var(--accent)' },
    { label: 'Reports', value: reports.length, icon: FileBarChart, color: 'var(--success)' },
    { label: 'Dashboards', value: dashboards.length, icon: PieChart, color: 'var(--warning)' },
    { label: 'Total Rows', value: datasets.reduce((acc, ds) => acc + ds.row_count, 0).toLocaleString(), icon: Activity, color: 'var(--danger)' },
  ];

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2" style={{ borderColor: 'var(--accent)' }} />
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Welcome Section */}
      <div>
        <h1 className="font-heading text-3xl font-bold mb-2" style={{ color: 'var(--text-primary)' }}>
          Welcome back, {user?.name?.split(' ')[0] || 'User'}
        </h1>
        <p style={{ color: 'var(--text-secondary)' }}>
          Here's what's happening with your analytics
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat) => (
          <Card key={stat.label} style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm mb-1" style={{ color: 'var(--text-muted)' }}>{stat.label}</p>
                  <p className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>{stat.value}</p>
                </div>
                <div 
                  className="p-3 rounded-lg"
                  style={{ background: `${stat.color}20` }}
                >
                  <stat.icon size={24} style={{ color: stat.color }} />
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent Datasets */}
        <Card style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
              Recent Datasets
            </CardTitle>
            <Button variant="ghost" size="sm" onClick={() => navigate('/data')}>
              View All <ArrowRight size={16} className="ml-1" />
            </Button>
          </CardHeader>
          <CardContent>
            {datasets.length === 0 ? (
              <div className="text-center py-8">
                <Database size={48} className="mx-auto mb-4" style={{ color: 'var(--text-muted)' }} />
                <p style={{ color: 'var(--text-muted)' }}>No datasets yet</p>
                <Button 
                  variant="outline" 
                  className="mt-4"
                  onClick={() => navigate('/data')}
                >
                  <Plus size={16} className="mr-2" />
                  Upload Dataset
                </Button>
              </div>
            ) : (
              <div className="space-y-3">
                {datasets.slice(0, 5).map((ds) => (
                  <div 
                    key={ds.id}
                    className="flex items-center justify-between p-3 rounded-lg cursor-pointer transition-colors"
                    style={{ background: 'var(--bg-tertiary)' }}
                    onClick={() => navigate('/data')}
                  >
                    <div className="flex items-center gap-3">
                      <Database size={18} style={{ color: 'var(--accent)' }} />
                      <div>
                        <p className="font-medium text-sm" style={{ color: 'var(--text-primary)' }}>{ds.name}</p>
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                          {ds.row_count.toLocaleString()} rows
                        </p>
                      </div>
                    </div>
                    <ArrowRight size={16} style={{ color: 'var(--text-muted)' }} />
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Recent Reports */}
        <Card style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
              Recent Reports
            </CardTitle>
            <Button variant="ghost" size="sm" onClick={() => navigate('/reports')}>
              View All <ArrowRight size={16} className="ml-1" />
            </Button>
          </CardHeader>
          <CardContent>
            {reports.length === 0 ? (
              <div className="text-center py-8">
                <FileBarChart size={48} className="mx-auto mb-4" style={{ color: 'var(--text-muted)' }} />
                <p style={{ color: 'var(--text-muted)' }}>No reports yet</p>
                <Button 
                  variant="outline" 
                  className="mt-4"
                  onClick={() => navigate('/reports')}
                >
                  <Plus size={16} className="mr-2" />
                  Create Report
                </Button>
              </div>
            ) : (
              <div className="space-y-3">
                {reports.slice(0, 5).map((report) => (
                  <div 
                    key={report.id}
                    className="flex items-center justify-between p-3 rounded-lg cursor-pointer transition-colors"
                    style={{ background: 'var(--bg-tertiary)' }}
                    onClick={() => navigate(`/reports/${report.id}`)}
                  >
                    <div className="flex items-center gap-3">
                      <FileBarChart size={18} style={{ color: 'var(--success)' }} />
                      <div>
                        <p className="font-medium text-sm" style={{ color: 'var(--text-primary)' }}>{report.name}</p>
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                          {report.page_count || 0} pages
                        </p>
                      </div>
                    </div>
                    <ArrowRight size={16} style={{ color: 'var(--text-muted)' }} />
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Recent Dashboards */}
        <Card style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
              Recent Dashboards
            </CardTitle>
            <Button variant="ghost" size="sm" onClick={() => navigate('/dashboards')}>
              View All <ArrowRight size={16} className="ml-1" />
            </Button>
          </CardHeader>
          <CardContent>
            {dashboards.length === 0 ? (
              <div className="text-center py-8">
                <PieChart size={48} className="mx-auto mb-4" style={{ color: 'var(--text-muted)' }} />
                <p style={{ color: 'var(--text-muted)' }}>No dashboards yet</p>
                <Button 
                  variant="outline" 
                  className="mt-4"
                  onClick={() => navigate('/dashboards')}
                >
                  <Plus size={16} className="mr-2" />
                  Create Dashboard
                </Button>
              </div>
            ) : (
              <div className="space-y-3">
                {dashboards.slice(0, 5).map((db) => (
                  <div 
                    key={db.id}
                    className="flex items-center justify-between p-3 rounded-lg cursor-pointer transition-colors"
                    style={{ background: 'var(--bg-tertiary)' }}
                    onClick={() => navigate(`/dashboards/${db.id}`)}
                  >
                    <div className="flex items-center gap-3">
                      <PieChart size={18} style={{ color: 'var(--warning)' }} />
                      <div>
                        <p className="font-medium text-sm" style={{ color: 'var(--text-primary)' }}>{db.name}</p>
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                          {db.widgets?.length || 0} widgets
                        </p>
                      </div>
                    </div>
                    <ArrowRight size={16} style={{ color: 'var(--text-muted)' }} />
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
