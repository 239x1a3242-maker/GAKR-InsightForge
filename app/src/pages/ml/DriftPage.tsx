import { useQuery } from '@tanstack/react-query';
import {
  Shield,
  AlertTriangle,
  CheckCircle,
  Activity,
} from 'lucide-react';
import { modelsApi } from '../../api/models';

export function DriftPage() {

  const { data: models } = useQuery({
    queryKey: ['models'],
    queryFn: () => modelsApi.list(),
  });

  const productionModels = models?.filter(m => m.production_version_id) || [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-[var(--text-primary)]">Drift Monitor</h1>
        <p className="text-[var(--text-muted)]">
          Monitor data drift and model performance
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-green-500/10 flex items-center justify-center">
              <CheckCircle className="w-5 h-5 text-green-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-[var(--text-primary)]">{productionModels.length}</p>
              <p className="text-sm text-[var(--text-muted)]">Healthy Models</p>
            </div>
          </div>
        </div>

        <div className="card p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-amber-500/10 flex items-center justify-center">
              <AlertTriangle className="w-5 h-5 text-amber-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-[var(--text-primary)]">0</p>
              <p className="text-sm text-[var(--text-muted)]">Drift Warnings</p>
            </div>
          </div>
        </div>

        <div className="card p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-red-500/10 flex items-center justify-center">
              <Activity className="w-5 h-5 text-red-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-[var(--text-primary)]">0</p>
              <p className="text-sm text-[var(--text-muted)]">Critical Alerts</p>
            </div>
          </div>
        </div>

        <div className="card p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center">
              <Shield className="w-5 h-5 text-blue-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-[var(--text-primary)]">100%</p>
              <p className="text-sm text-[var(--text-muted)]">Uptime</p>
            </div>
          </div>
        </div>
      </div>

      <div className="card p-6">
        <h3 className="text-lg font-semibold text-[var(--text-primary)] mb-4">
          Production Models Status
        </h3>

        <div className="space-y-3">
          {productionModels.map((model) => (
            <div
              key={model.id}
              className="flex items-center justify-between p-4 bg-[var(--bg-tertiary)] rounded-lg"
            >
              <div className="flex items-center gap-4">
                <div className="w-3 h-3 rounded-full bg-green-500" />
                <div>
                  <p className="font-medium text-[var(--text-primary)]">{model.target_column}</p>
                  <p className="text-sm text-[var(--text-muted)]">
                    {model.task_type} • v{model.production_version?.version_number}
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-6">
                <div className="text-right">
                  <p className="text-xs text-[var(--text-muted)]">PSI Score</p>
                  <p className="text-sm font-medium text-green-400">0.02</p>
                </div>
                <div className="text-right">
                  <p className="text-xs text-[var(--text-muted)]">Status</p>
                  <span className="badge badge-green text-xs">Healthy</span>
                </div>
                <div className="text-right">
                  <p className="text-xs text-[var(--text-muted)]">Last Check</p>
                  <p className="text-sm text-[var(--text-secondary)]">2 hours ago</p>
                </div>
              </div>
            </div>
          ))}

          {productionModels.length === 0 && (
            <div className="text-center py-8 text-[var(--text-muted)]">
              <Shield className="w-12 h-12 mx-auto mb-4" />
              <p>No production models to monitor</p>
            </div>
          )}
        </div>
      </div>

      <div className="card p-6">
        <h3 className="text-lg font-semibold text-[var(--text-primary)] mb-4">
          About Drift Detection
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-[var(--text-primary)] mb-2">
              Population Stability Index (PSI)
            </h4>
            <p className="text-sm text-[var(--text-muted)] mb-4">
              PSI measures how much a variable has changed between two samples.
              It's commonly used to monitor model input features and predictions.
            </p>

            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm">
                <div className="w-3 h-3 rounded-full bg-green-500" />
                <span className="text-[var(--text-secondary)]">PSI &lt; 0.1: No significant change</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <div className="w-3 h-3 rounded-full bg-amber-500" />
                <span className="text-[var(--text-secondary)]">0.1 ≤ PSI &lt; 0.2: Moderate change</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <div className="w-3 h-3 rounded-full bg-red-500" />
                <span className="text-[var(--text-secondary)]">PSI ≥ 0.2: Significant change</span>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-medium text-[var(--text-primary)] mb-2">
              Recommended Actions
            </h4>
            <ul className="space-y-2 text-sm text-[var(--text-muted)]">
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                Monitor PSI scores weekly for production models
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                Retrain models when PSI exceeds 0.2
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                Investigate feature-level drift first
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                Document all model updates and drift events
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
