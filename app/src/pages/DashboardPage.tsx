import { useQuery } from '@tanstack/react-query';
import {
  Database,
  Brain,
  BarChart3,
  TrendingUp,
  Activity,
  ArrowRight,
} from 'lucide-react';
import { datasetsApi } from '@/api/datasets';
import { modelsApi } from '@/api/models';
import { Link } from 'react-router-dom';

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle: string;
  icon: React.ElementType;
  trend?: string;
  trendUp?: boolean;
  color: string;
}

function StatCard({ title, value, subtitle, icon: Icon, trend, trendUp, color }: StatCardProps) {
  return (
    <div className="card p-6">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-[var(--text-muted)]">{title}</p>
          <p className="text-2xl font-bold text-[var(--text-primary)] mt-1">{value}</p>
          <p className="text-sm text-[var(--text-secondary)] mt-1">{subtitle}</p>
          {trend && (
            <div className={`flex items-center gap-1 mt-2 text-sm ${trendUp ? 'text-green-400' : 'text-red-400'}`}>
              <TrendingUp className="w-4 h-4" />
              {trend}
            </div>
          )}
        </div>
        <div className={`p-3 rounded-lg ${color}`}>
          <Icon className="w-6 h-6 text-white" />
        </div>
      </div>
    </div>
  );
}

export function DashboardPage() {
  const { data: datasets } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => datasetsApi.list(),
  });

  const { data: models } = useQuery({
    queryKey: ['models'],
    queryFn: () => modelsApi.list(),
  });

  const productionModels = models?.filter(m => m.production_version_id) || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-[var(--text-primary)]">Dashboard</h1>
        <p className="text-[var(--text-muted)]">Overview of your analytics workspace</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Datasets"
          value={datasets?.total || 0}
          subtitle="Active datasets"
          icon={Database}
          color="bg-blue-500"
        />
        <StatCard
          title="ML Models"
          value={models?.length || 0}
          subtitle="Trained models"
          icon={Brain}
          color="bg-purple-500"
        />
        <StatCard
          title="Production Models"
          value={productionModels.length}
          subtitle="In production"
          icon={Activity}
          trend="+2 this week"
          trendUp={true}
          color="bg-green-500"
        />
        <StatCard
          title="Predictions"
          value="12.5K"
          subtitle="Last 30 days"
          icon={BarChart3}
          trend="+18%"
          trendUp={true}
          color="bg-amber-500"
        />
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Datasets */}
        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-[var(--text-primary)]">Recent Datasets</h3>
            <Link to="/datasets" className="text-sm text-blue-400 hover:text-blue-300 flex items-center gap-1">
              View all <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
          
          <div className="space-y-3">
            {datasets?.items.slice(0, 5).map((dataset) => (
              <Link
                key={dataset.id}
                to={`/datasets/${dataset.id}`}
                className="flex items-center justify-between p-3 rounded-lg hover:bg-[var(--bg-tertiary)] transition-colors"
              >
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center">
                    <Database className="w-5 h-5 text-blue-400" />
                  </div>
                  <div>
                    <p className="font-medium text-[var(--text-primary)]">{dataset.name}</p>
                    <p className="text-sm text-[var(--text-muted)]">
                      {dataset.row_count.toLocaleString()} rows • {dataset.column_count} columns
                    </p>
                  </div>
                </div>
                <span className="text-xs text-[var(--text-muted)]">
                  {new Date(dataset.created_at).toLocaleDateString()}
                </span>
              </Link>
            ))}
            
            {(!datasets?.items.length) && (
              <p className="text-center text-[var(--text-muted)] py-8">
                No datasets yet. <Link to="/datasets" className="text-blue-400">Upload one</Link>
              </p>
            )}
          </div>
        </div>

        {/* Production Models */}
        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-[var(--text-primary)]">Production Models</h3>
            <Link to="/ml/models" className="text-sm text-blue-400 hover:text-blue-300 flex items-center gap-1">
              View all <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
          
          <div className="space-y-3">
            {productionModels.slice(0, 5).map((model) => (
              <Link
                key={model.id}
                to="/ml/models"
                className="flex items-center justify-between p-3 rounded-lg hover:bg-[var(--bg-tertiary)] transition-colors"
              >
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-green-500/10 flex items-center justify-center">
                    <Brain className="w-5 h-5 text-green-400" />
                  </div>
                  <div>
                    <p className="font-medium text-[var(--text-primary)]">{model.target_column}</p>
                    <p className="text-sm text-[var(--text-muted)]">
                      {model.task_type} • v{model.production_version?.version_number}
                    </p>
                  </div>
                </div>
                <span className="badge badge-green">Production</span>
              </Link>
            ))}
            
            {(!productionModels.length) && (
              <p className="text-center text-[var(--text-muted)] py-8">
                No production models. <Link to="/ml/training" className="text-blue-400">Train one</Link>
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Quick Links */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Link to="/datasets" className="card p-4 hover:border-blue-500/50 transition-colors">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-blue-500 flex items-center justify-center">
              <Database className="w-5 h-5 text-white" />
            </div>
            <div>
              <p className="font-medium text-[var(--text-primary)]">Upload Dataset</p>
              <p className="text-sm text-[var(--text-muted)]">Import CSV or Excel</p>
            </div>
          </div>
        </Link>

        <Link to="/ml/training" className="card p-4 hover:border-green-500/50 transition-colors">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-green-500 flex items-center justify-center">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <div>
              <p className="font-medium text-[var(--text-primary)]">Train Model</p>
              <p className="text-sm text-[var(--text-muted)]">AutoML with H2O</p>
            </div>
          </div>
        </Link>
      </div>
    </div>
  );
}

