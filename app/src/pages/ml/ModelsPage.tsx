import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Database, Clock, ChevronDown, ChevronUp, Loader2, Trash2, CheckCircle, XCircle } from 'lucide-react';
import { useStore } from '@/store';
import { formatDate } from '@/utils/format';
import apiClient from '@/api/client';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];

export function ModelsPage() {
  const queryClient = useQueryClient();
  const { addNotification } = useStore();
  const [expandedModel, setExpandedModel] = useState<string | null>(null);

  const { data: modelsData, isLoading } = useQuery({
    queryKey: ['ml-models'],
    queryFn: async () => {
      const res = await apiClient.get('/api/ml/models');
      return res.data;
    },
  });

  const models = modelsData?.items || [];

  const handleDelete = async (modelId: string) => {
    if (!confirm('Delete this model?')) return;
    try {
      await apiClient.delete(`/api/ml/models/${modelId}`);
      queryClient.invalidateQueries({ queryKey: ['ml-models'] });
      addNotification({ type: 'success', message: 'Model deleted' });
    } catch (e) {
      addNotification({ type: 'error', message: 'Failed to delete model' });
    }
  };

  if (isLoading) return (
    <div className="flex items-center justify-center py-12">
      <Loader2 className="w-8 h-8 animate-spin" style={{ color: 'var(--accent)' }} />
    </div>
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>My Models</h1>
        <p style={{ color: 'var(--text-muted)' }}>View and manage trained models</p>
      </div>

      {models.length === 0 ? (
        <div className="card p-12 text-center">
          <Database className="w-12 h-12 mx-auto mb-4" style={{ color: 'var(--text-muted)' }} />
          <h3 className="text-lg font-medium mb-2" style={{ color: 'var(--text-primary)' }}>No models yet</h3>
          <p style={{ color: 'var(--text-muted)' }}>Train your first model from the Training page</p>
        </div>
      ) : (
        <div className="space-y-3">
          {models.map((model: any) => {
            const isExpanded = expandedModel === model.id;
            const targets = Object.keys(model.target_results || {});
            const isCompleted = model.status === 'completed';
            const isFailed = model.status === 'failed';
            const isTraining = model.status === 'training';

            return (
              <div key={model.id} className="card overflow-hidden">
                <div className="p-4 flex items-center justify-between cursor-pointer hover:bg-opacity-80 transition-colors"
                  style={{ background: 'var(--bg-secondary)' }}
                  onClick={() => setExpandedModel(isExpanded ? null : model.id)}>
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-lg flex items-center justify-center"
                      style={{ background: isCompleted ? 'rgba(16,185,129,0.1)' : isFailed ? 'rgba(239,68,68,0.1)' : 'rgba(59,130,246,0.1)' }}>
                      {isCompleted ? <CheckCircle className="w-5 h-5" style={{ color: 'var(--success)' }} />
                        : isFailed ? <XCircle className="w-5 h-5" style={{ color: 'var(--danger)' }} />
                        : <Loader2 className="w-5 h-5 animate-spin" style={{ color: 'var(--accent)' }} />}
                    </div>
                    <div>
                      <h3 className="font-semibold" style={{ color: 'var(--text-primary)' }}>{model.name}</h3>
                      <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                        {isTraining ? 'Training...' : targets.length > 0 ? `Targets: ${targets.join(', ')}` : model.algorithm}
                        {model.total_training_time_seconds && ` · ${model.total_training_time_seconds.toFixed(1)}s`}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-xs px-2 py-1 rounded-full"
                      style={{
                        background: isCompleted ? 'rgba(16,185,129,0.1)' : isFailed ? 'rgba(239,68,68,0.1)' : 'rgba(59,130,246,0.1)',
                        color: isCompleted ? 'var(--success)' : isFailed ? 'var(--danger)' : 'var(--accent)',
                      }}>
                      {model.status}
                    </span>
                    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                      <Clock className="w-3 h-3 inline mr-1" />{formatDate(model.updated_at)}
                    </span>
                    <button onClick={e => { e.stopPropagation(); handleDelete(model.id); }}
                      className="p-1.5 rounded hover:bg-red-500/10 transition-colors">
                      <Trash2 size={14} style={{ color: 'var(--danger)' }} />
                    </button>
                    {isExpanded ? <ChevronUp className="w-4 h-4" style={{ color: 'var(--text-muted)' }} />
                      : <ChevronDown className="w-4 h-4" style={{ color: 'var(--text-muted)' }} />}
                  </div>
                </div>

                {isExpanded && isCompleted && model.target_results && (
                  <div className="p-5 space-y-5" style={{ borderTop: '1px solid var(--border-color)' }}>
                    {Object.entries(model.target_results).map(([target, result]: [string, any]) => (
                      <div key={target}>
                        <div className="flex items-center gap-2 mb-3">
                          <span className="font-medium" style={{ color: 'var(--text-primary)' }}>Target: {target}</span>
                          <span className="text-xs px-2 py-0.5 rounded-full"
                            style={{ background: 'rgba(59,130,246,0.1)', color: 'var(--accent)' }}>
                            {result.task_type}
                          </span>
                          <span className="text-xs px-2 py-0.5 rounded-full"
                            style={{ background: 'rgba(16,185,129,0.1)', color: 'var(--success)' }}>
                            {result.best_algorithm}
                          </span>
                        </div>

                        {/* Metrics */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                          {Object.entries(result.metrics || {}).map(([k, v]: [string, any]) => (
                            <div key={k} className="p-3 rounded-lg" style={{ background: 'var(--bg-tertiary)' }}>
                              <p className="text-xs uppercase" style={{ color: 'var(--text-muted)' }}>{k}</p>
                              <p className="text-base font-semibold" style={{ color: 'var(--text-primary)' }}>
                                {typeof v === 'number' ? (v > 1 ? v.toFixed(4) : (v * 100).toFixed(2) + '%') : v}
                              </p>
                            </div>
                          ))}
                          <div className="p-3 rounded-lg" style={{ background: 'var(--bg-tertiary)' }}>
                            <p className="text-xs uppercase" style={{ color: 'var(--text-muted)' }}>CV Score</p>
                            <p className="text-base font-semibold" style={{ color: 'var(--text-primary)' }}>
                              {((result.cv_mean || 0) * 100).toFixed(2)}%
                            </p>
                          </div>
                        </div>

                        {/* Feature importance */}
                        {result.feature_importance && Object.keys(result.feature_importance).length > 0 && (
                          <div>
                            <p className="text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Feature Importance</p>
                            <ResponsiveContainer width="100%" height={160}>
                              <BarChart layout="vertical"
                                data={Object.entries(result.feature_importance)
                                  .sort(([, a]: any, [, b]: any) => b - a).slice(0, 8)
                                  .map(([name, value]) => ({ name, value: Number((value as number * 100).toFixed(1)) }))}>
                                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                                <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 10 }} stroke="var(--text-secondary)" />
                                <YAxis dataKey="name" type="category" tick={{ fontSize: 10 }} stroke="var(--text-secondary)" width={100} />
                                <Tooltip formatter={(v: any) => `${v}%`} contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-color)', borderRadius: 8 }} />
                                <Bar dataKey="value" radius={[0, 3, 3, 0]}>
                                  {Object.keys(result.feature_importance).slice(0, 8).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                                </Bar>
                              </BarChart>
                            </ResponsiveContainer>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {isExpanded && isFailed && (
                  <div className="p-4" style={{ borderTop: '1px solid var(--border-color)' }}>
                    <p className="text-sm" style={{ color: 'var(--danger)' }}>Error: {model.error || 'Unknown error'}</p>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
