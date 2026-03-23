import { useParams, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import {
  ArrowLeft,
  Database,
  FileSpreadsheet,
  BarChart3,
  Brain,
  Loader2,
  Calendar,
  HardDrive,
} from 'lucide-react';
import { datasetsApi } from '@/api/datasets';
import { formatFileSize, formatDate } from '@/utils/format';

export function DatasetDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const { data: dataset, isLoading } = useQuery({
    queryKey: ['dataset', id],
    queryFn: () => datasetsApi.getDataset(id!),
    enabled: !!id,
  });

  const { data: profile } = useQuery({
    queryKey: ['dataset-profile', id],
    queryFn: () => datasetsApi.getProfile(id!),
    enabled: !!id,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="w-8 h-8 animate-spin text-blue-400" />
      </div>
    );
  }

  if (!dataset) {
    return (
      <div className="text-center py-12">
        <p className="text-[var(--text-muted)]">Dataset not found</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <button
          onClick={() => navigate('/data')}
          className="p-2 rounded-lg hover:bg-[var(--bg-tertiary)] text-[var(--text-muted)]"
        >
          <ArrowLeft className="w-5 h-5" />
        </button>
        <div>
          <h1 className="text-2xl font-bold text-[var(--text-primary)]">{dataset.name}</h1>
          <p className="text-[var(--text-muted)]">{dataset.description || 'No description'}</p>
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-3">
        <button
          onClick={() => navigate(`/analysis/${dataset.id}`)}
          className="btn btn-secondary"
        >
          <BarChart3 className="w-4 h-4" />
          Analyze
        </button>
        <button
          onClick={() => navigate('/ml/training', { state: { datasetId: dataset.id } })}
          className="btn btn-primary"
        >
          <Brain className="w-4 h-4" />
          Train Model
        </button>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="card p-4">
          <div className="flex items-center gap-3">
            <Database className="w-5 h-5 text-blue-400" />
            <div>
              <p className="text-lg font-semibold text-[var(--text-primary)]">
                {dataset.row_count.toLocaleString()}
              </p>
              <p className="text-xs text-[var(--text-muted)]">Rows</p>
            </div>
          </div>
        </div>

        <div className="card p-4">
          <div className="flex items-center gap-3">
            <FileSpreadsheet className="w-5 h-5 text-purple-400" />
            <div>
              <p className="text-lg font-semibold text-[var(--text-primary)]">
                {dataset.columns?.length || 0}
              </p>
              <p className="text-xs text-[var(--text-muted)]">Columns</p>
            </div>
          </div>
        </div>

        <div className="card p-4">
          <div className="flex items-center gap-3">
            <HardDrive className="w-5 h-5 text-green-400" />
            <div>
              <p className="text-lg font-semibold text-[var(--text-primary)]">
                {formatFileSize(dataset.size_bytes)}
              </p>
              <p className="text-xs text-[var(--text-muted)]">Size</p>
            </div>
          </div>
        </div>

        <div className="card p-4">
          <div className="flex items-center gap-3">
            <Calendar className="w-5 h-5 text-amber-400" />
            <div>
              <p className="text-lg font-semibold text-[var(--text-primary)]">
                {formatDate(dataset.created_at)}
              </p>
              <p className="text-xs text-[var(--text-muted)]">Created</p>
            </div>
          </div>
        </div>
      </div>

      {/* Schema */}
      <div className="card p-6">
        <h3 className="text-lg font-semibold text-[var(--text-primary)] mb-4">
          Schema
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-[var(--border-color)]">
                <th className="text-left py-2 text-sm text-[var(--text-muted)]">Column</th>
                <th className="text-left py-2 text-sm text-[var(--text-muted)]">Type</th>
                <th className="text-left py-2 text-sm text-[var(--text-muted)]">Data Type</th>
                <th className="text-center py-2 text-sm text-[var(--text-muted)]">Nullable</th>
                <th className="text-right py-2 text-sm text-[var(--text-muted)]">Unique Values</th>
              </tr>
            </thead>
            <tbody>
              {dataset.columns?.map((col) => (
                <tr key={col.name} className="border-b border-[var(--border-color)]">
                  <td className="py-2 text-[var(--text-primary)] font-medium">{col.name}</td>
                  <td className="py-2">
                    <span className={`badge ${
                      col.type === 'numeric' ? 'badge-blue' :
                      col.type === 'string' ? 'badge-green' :
                      col.type === 'datetime' ? 'badge-amber' :
                      'badge-purple'
                    }`}>
                      {col.type}
                    </span>
                  </td>
                  <td className="py-2 text-[var(--text-secondary)]">{col.dtype}</td>
                  <td className="py-2 text-center">
                    {col.nullable ? (
                      <span className="text-amber-400">Yes</span>
                    ) : (
                      <span className="text-green-400">No</span>
                    )}
                  </td>
                  <td className="py-2 text-right text-[var(--text-secondary)]">
                    {col.unique_count?.toLocaleString() || '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Profile Stats */}
      {profile && (
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-[var(--text-primary)] mb-4">
            Data Profile
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-sm text-[var(--text-muted)]">Numeric Columns</p>
              <p className="text-lg font-semibold text-[var(--text-primary)]">
                {profile.numeric_columns}
              </p>
            </div>
            <div>
              <p className="text-sm text-[var(--text-muted)]">Categorical Columns</p>
              <p className="text-lg font-semibold text-[var(--text-primary)]">
                {profile.categorical_columns}
              </p>
            </div>
            <div>
              <p className="text-sm text-[var(--text-muted)]">Memory Usage (MB)</p>
              <p className="text-lg font-semibold text-[var(--text-primary)]">
                {profile.memory_usage_mb.toFixed(2)} MB
              </p>
            </div>
            <div>
              <p className="text-sm text-[var(--text-muted)]">Missing %</p>
              <p className="text-lg font-semibold text-[var(--text-primary)]">
                {profile.missing_percentage.toFixed(1)}%
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
