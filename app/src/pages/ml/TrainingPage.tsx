import { useState, useEffect } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Play, Loader2, CheckCircle, Clock, Settings, ChevronDown, ChevronUp,
  Database, Award
} from 'lucide-react';
import { datasetsApi } from '@/api/datasets';
import { useStore } from '@/store';
import apiClient from '@/api/client';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell
} from 'recharts';

const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];

export function TrainingPage() {
  const queryClient = useQueryClient();
  const { addNotification } = useStore();
  const [selectedDataset, setSelectedDataset] = useState('');
  const [targetCols, setTargetCols] = useState<string[]>([]);
  const [featureCols, setFeatureCols] = useState<string[]>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [training, setTraining] = useState(false);
  const [jobResult, setJobResult] = useState<any>(null);
  const [pollInterval, setPollInterval] = useState<ReturnType<typeof setInterval> | null>(null);
  const [config, setConfig] = useState({ test_size: 0.2, cv_folds: 5, problem_type: 'auto' });

  const { data: datasets } = useQuery({ queryKey: ['datasets'], queryFn: () => datasetsApi.list() });
  const selectedDs = datasets?.items.find(d => d.id === selectedDataset);

  const toggleTarget = (col: string) => {
    setTargetCols(prev => prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col]);
    setFeatureCols(prev => prev.filter(c => c !== col));
  };

  const toggleFeature = (col: string) => {
    if (targetCols.includes(col)) return;
    setFeatureCols(prev => prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col]);
  };

  const stopPolling = () => {
    if (pollInterval) { clearInterval(pollInterval); setPollInterval(null); }
  };

  useEffect(() => () => stopPolling(), []);

  const pollJob = (id: string) => {
    const interval = setInterval(async () => {
      try {
        const res = await apiClient.get(`/api/ml/training-jobs/${id}`);
        const job = res.data;
        if (job.status === 'completed' || job.status === 'failed') {
          clearInterval(interval);
          setPollInterval(null);
          setTraining(false);
          setJobResult(job);
          queryClient.invalidateQueries({ queryKey: ['ml-models'] });
          if (job.status === 'completed') {
            addNotification({ type: 'success', message: 'Model training completed!' });
          } else {
            addNotification({ type: 'error', message: `Training failed: ${job.error || 'Unknown error'}` });
          }
        }
      } catch (e) { console.error(e); }
    }, 2000);
    setPollInterval(interval);
  };

  const handleTrain = async () => {
    if (!selectedDataset || targetCols.length === 0) return;
    setTraining(true);
    setJobResult(null);
    try {
      const res = await apiClient.post('/api/ml/train', {
        dataset_id: selectedDataset,
        target_columns: targetCols,
        feature_columns: featureCols.length > 0 ? featureCols : undefined,
        problem_type: config.problem_type,
        test_size: config.test_size,
        cv_folds: config.cv_folds,
      });
      pollJob(res.data.id);
    } catch (e) {
      setTraining(false);
      addNotification({ type: 'error', message: 'Failed to start training' });
    }
  };

  const allCols = selectedDs?.columns || [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>Model Training</h1>
        <p style={{ color: 'var(--text-muted)' }}>Train AutoML models with sklearn</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-4">
          {/* Dataset */}
          <div className="card p-6">
            <h3 className="text-lg font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>1. Select Dataset</h3>
            <select value={selectedDataset} onChange={e => { setSelectedDataset(e.target.value); setTargetCols([]); setFeatureCols([]); }} className="input">
              <option value="">Choose a dataset...</option>
              {datasets?.items.map(d => <option key={d.id} value={d.id}>{d.name} ({d.row_count.toLocaleString()} rows)</option>)}
            </select>
          </div>

          {/* Target columns */}
          {selectedDs && (
            <div className="card p-6">
              <h3 className="text-lg font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>2. Select Target Column(s)</h3>
              <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>The column(s) you want to predict</p>
              <div className="flex flex-wrap gap-2">
                {allCols.map(col => (
                  <button key={col.name} onClick={() => toggleTarget(col.name)}
                    className="px-3 py-1.5 rounded-lg text-sm font-medium transition-colors"
                    style={{
                      background: targetCols.includes(col.name) ? 'var(--accent)' : 'var(--bg-tertiary)',
                      color: targetCols.includes(col.name) ? '#fff' : 'var(--text-secondary)',
                      border: '1px solid var(--border-color)',
                    }}>
                    {col.name} <span className="opacity-60 text-xs">({col.type})</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Feature columns */}
          {selectedDs && targetCols.length > 0 && (
            <div className="card p-6">
              <h3 className="text-lg font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>3. Feature Columns (optional)</h3>
              <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>Leave empty to use all non-target columns</p>
              <div className="flex flex-wrap gap-2">
                {allCols.filter(c => !targetCols.includes(c.name)).map(col => (
                  <button key={col.name} onClick={() => toggleFeature(col.name)}
                    className="px-3 py-1.5 rounded-lg text-sm font-medium transition-colors"
                    style={{
                      background: featureCols.includes(col.name) ? 'rgba(16,185,129,0.2)' : 'var(--bg-tertiary)',
                      color: featureCols.includes(col.name) ? 'var(--success)' : 'var(--text-secondary)',
                      border: `1px solid ${featureCols.includes(col.name) ? 'var(--success)' : 'var(--border-color)'}`,
                    }}>
                    {col.name}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Advanced */}
          <div className="card p-6">
            <button onClick={() => setShowAdvanced(!showAdvanced)} className="flex items-center justify-between w-full">
              <h3 className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>Advanced Configuration</h3>
              {showAdvanced ? <ChevronUp className="w-5 h-5" style={{ color: 'var(--text-muted)' }} /> : <ChevronDown className="w-5 h-5" style={{ color: 'var(--text-muted)' }} />}
            </button>
            {showAdvanced && (
              <div className="mt-4 grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm mb-1" style={{ color: 'var(--text-secondary)' }}>Problem Type</label>
                  <select value={config.problem_type} onChange={e => setConfig(p => ({ ...p, problem_type: e.target.value }))} className="input">
                    <option value="auto">Auto-detect</option>
                    <option value="classification">Classification</option>
                    <option value="regression">Regression</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm mb-1" style={{ color: 'var(--text-secondary)' }}>Test Size</label>
                  <input type="number" value={config.test_size} step={0.05} min={0.1} max={0.5}
                    onChange={e => setConfig(p => ({ ...p, test_size: parseFloat(e.target.value) }))} className="input" />
                </div>
                <div>
                  <label className="block text-sm mb-1" style={{ color: 'var(--text-secondary)' }}>CV Folds</label>
                  <input type="number" value={config.cv_folds} min={2} max={10}
                    onChange={e => setConfig(p => ({ ...p, cv_folds: parseInt(e.target.value) }))} className="input" />
                </div>
              </div>
            )}
          </div>

          <button onClick={handleTrain} disabled={!selectedDataset || targetCols.length === 0 || training}
            className="btn btn-primary w-full py-3">
            {training ? <><Loader2 className="w-5 h-5 animate-spin" /> Training in progress...</> : <><Play className="w-5 h-5" /> Start Training</>}
          </button>
        </div>

        {/* Right panel */}
        <div className="space-y-4">
          <div className="card p-6">
            <h3 className="text-lg font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>Training Info</h3>
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <Database className="w-5 h-5 mt-0.5" style={{ color: 'var(--accent)' }} />
                <div>
                  <p className="font-medium text-sm" style={{ color: 'var(--text-primary)' }}>Dataset</p>
                  <p className="text-sm" style={{ color: 'var(--text-muted)' }}>{selectedDs?.name || 'Not selected'}</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <Settings className="w-5 h-5 mt-0.5" style={{ color: 'var(--accent)' }} />
                <div>
                  <p className="font-medium text-sm" style={{ color: 'var(--text-primary)' }}>Targets</p>
                  <p className="text-sm" style={{ color: 'var(--text-muted)' }}>{targetCols.length > 0 ? targetCols.join(', ') : 'Not selected'}</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <Clock className="w-5 h-5 mt-0.5" style={{ color: 'var(--warning)' }} />
                <div>
                  <p className="font-medium text-sm" style={{ color: 'var(--text-primary)' }}>Status</p>
                  <p className="text-sm" style={{ color: training ? 'var(--warning)' : 'var(--text-muted)' }}>
                    {training ? 'Training...' : jobResult ? jobResult.status : 'Idle'}
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="card p-6">
            <h3 className="text-lg font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>Algorithms</h3>
            <ul className="space-y-2 text-sm" style={{ color: 'var(--text-secondary)' }}>
              {['Random Forest', 'Gradient Boosting', 'Extra Trees', 'AdaBoost', 'Logistic/Linear Regression', 'Ridge / Lasso / ElasticNet', 'Decision Tree', 'Naive Bayes'].map(a => (
                <li key={a} className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4" style={{ color: 'var(--success)' }} /> {a}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* Results */}
      {jobResult && jobResult.status === 'completed' && jobResult.target_results && (
        <div className="space-y-4">
          <h2 className="text-xl font-bold" style={{ color: 'var(--text-primary)' }}>Training Results</h2>
          {Object.entries(jobResult.target_results).map(([target, result]: [string, any]) => (
            <div key={target} className="card p-6 space-y-5">
              <div className="flex items-center gap-3">
                <Award className="w-6 h-6" style={{ color: 'var(--accent)' }} />
                <div>
                  <h3 className="font-semibold text-lg" style={{ color: 'var(--text-primary)' }}>Target: {target}</h3>
                  <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                    {result.task_type === 'classification' ? '🔵 Classification' : '📈 Regression'} · Best: {result.best_algorithm}
                  </p>
                </div>
              </div>

              {/* Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {Object.entries(result.metrics || {}).map(([k, v]: [string, any]) => (
                  <div key={k} className="p-3 rounded-lg" style={{ background: 'var(--bg-tertiary)' }}>
                    <p className="text-xs uppercase" style={{ color: 'var(--text-muted)' }}>{k}</p>
                    <p className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
                      {typeof v === 'number' ? (v > 1 ? v.toFixed(4) : (v * 100).toFixed(2) + '%') : v}
                    </p>
                  </div>
                ))}
                <div className="p-3 rounded-lg" style={{ background: 'var(--bg-tertiary)' }}>
                  <p className="text-xs uppercase" style={{ color: 'var(--text-muted)' }}>CV Score</p>
                  <p className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
                    {((result.cv_mean || 0) * 100).toFixed(2)}%
                  </p>
                </div>
                <div className="p-3 rounded-lg" style={{ background: 'var(--bg-tertiary)' }}>
                  <p className="text-xs uppercase" style={{ color: 'var(--text-muted)' }}>Train Rows</p>
                  <p className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>{result.train_rows}</p>
                </div>
              </div>

              {/* Feature importance */}
              {result.feature_importance && Object.keys(result.feature_importance).length > 0 && (
                <div>
                  <h4 className="font-medium mb-3" style={{ color: 'var(--text-primary)' }}>Feature Importance</h4>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart layout="vertical"
                      data={Object.entries(result.feature_importance)
                        .sort(([, a]: any, [, b]: any) => b - a).slice(0, 10)
                        .map(([name, value]) => ({ name, value: Number((value as number * 100).toFixed(2)) }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                      <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11 }} stroke="var(--text-secondary)" />
                      <YAxis dataKey="name" type="category" tick={{ fontSize: 11 }} stroke="var(--text-secondary)" width={120} />
                      <Tooltip formatter={(v: any) => `${v}%`} contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-color)', borderRadius: 8 }} />
                      <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                        {Object.keys(result.feature_importance).slice(0, 10).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {jobResult && jobResult.status === 'failed' && (
        <div className="card p-6" style={{ border: '1px solid var(--danger)' }}>
          <p className="font-semibold" style={{ color: 'var(--danger)' }}>Training failed</p>
          <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>{jobResult.error}</p>
        </div>
      )}
    </div>
  );
}
