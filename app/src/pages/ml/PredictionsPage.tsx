import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Play, Loader2, TrendingUp, Trash2 } from 'lucide-react';
import { useStore } from '@/store';
import apiClient from '@/api/client';

export function PredictionsPage() {
  const { addNotification } = useStore();
  const queryClient = useQueryClient();
  const [selectedModel, setSelectedModel] = useState('');
  const [features, setFeatures] = useState<Record<string, string>>({});
  const [prediction, setPrediction] = useState<any>(null);
  const [predicting, setPredicting] = useState(false);

  const { data: modelsData, isLoading } = useQuery({
    queryKey: ['ml-models'],
    queryFn: async () => {
      const res = await apiClient.get('/api/ml/models');
      return res.data;
    },
  });

  const models = modelsData?.items || [];
  const completedModels = models.filter((m: any) => m.status === 'completed');
  const selectedModelData = completedModels.find((m: any) => m.id === selectedModel);

  const featureColumns: string[] = selectedModelData?.feature_columns || [];

  const handlePredict = async () => {
    if (!selectedModel || featureColumns.length === 0) return;
    setPredicting(true);
    try {
      // Build model_ids from target_results - use best_model_id which has correct format
      const targetResults = selectedModelData?.target_results || {};
      const modelIds = Object.values(targetResults).map((result: any) => result.best_model_id);
      const res = await apiClient.post('/api/ml/predict', {
        model_ids: modelIds,
        data: [features],
        return_confidence: true,
      });
      setPrediction(res.data);
      addNotification({ type: 'success', message: 'Prediction completed' });
    } catch (e: any) {
      addNotification({ type: 'error', message: e.response?.data?.detail || 'Prediction failed' });
      console.error('Prediction error:', e);
    } finally {
      setPredicting(false);
    }
  };

  const handleDelete = async (modelId: string) => {
    if (!confirm('Delete this model?')) return;
    try {
      await apiClient.delete(`/api/ml/models/${modelId}`);
      queryClient.invalidateQueries({ queryKey: ['ml-models'] });
      if (selectedModel === modelId) { setSelectedModel(''); setPrediction(null); }
      addNotification({ type: 'success', message: 'Model deleted' });
    } catch (e) {
      addNotification({ type: 'error', message: 'Failed to delete model' });
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>Predictions</h1>
        <p style={{ color: 'var(--text-muted)' }}>Make predictions with trained models</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          {/* Model selector */}
          <div className="card p-6">
            <h3 className="text-lg font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>Select Model</h3>
            {isLoading ? (
              <div className="flex items-center gap-2" style={{ color: 'var(--text-muted)' }}>
                <Loader2 size={16} className="animate-spin" /> Loading models...
              </div>
            ) : completedModels.length === 0 ? (
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No trained models yet. Train a model first.</p>
            ) : (
              <div className="space-y-2">
                {completedModels.map((model: any) => {
                  const targets = Object.keys(model.target_results || {});
                  const firstResult = Object.values(model.target_results || {})[0] as any;
                  return (
                    <div key={model.id}
                      className="flex items-center justify-between p-3 rounded-lg cursor-pointer transition-all"
                      style={{
                        background: selectedModel === model.id ? 'rgba(59,130,246,0.1)' : 'var(--bg-tertiary)',
                        border: `1px solid ${selectedModel === model.id ? 'var(--accent)' : 'var(--border-color)'}`,
                      }}
                      onClick={() => { setSelectedModel(model.id); setFeatures({}); setPrediction(null); }}>
                      <div>
                        <p className="font-medium text-sm" style={{ color: 'var(--text-primary)' }}>{model.name}</p>
                        <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
                          Target: {targets.join(', ')} · {firstResult?.task_type} · {firstResult?.best_algorithm}
                        </p>
                      </div>
                      <button onClick={e => { e.stopPropagation(); handleDelete(model.id); }}
                        className="p-1.5 rounded hover:bg-red-500/10 transition-colors"
                        title="Delete model">
                        <Trash2 size={14} style={{ color: 'var(--danger)' }} />
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Feature inputs */}
          {selectedModelData && featureColumns.length > 0 && (
            <div className="card p-6">
              <h3 className="text-lg font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>Input Features</h3>
              <div className="space-y-3 max-h-80 overflow-auto pr-1">
                {featureColumns.map(feat => (
                  <div key={feat}>
                    <label className="block text-sm mb-1" style={{ color: 'var(--text-secondary)' }}>{feat}</label>
                    <input type="text" value={features[feat] || ''}
                      onChange={e => setFeatures(p => ({ ...p, [feat]: e.target.value }))}
                      className="input" placeholder={`Enter ${feat}`} />
                  </div>
                ))}
              </div>
              <button onClick={handlePredict}
                disabled={predicting || featureColumns.some(f => !features[f])}
                className="btn btn-primary w-full mt-4">
                {predicting ? <><Loader2 className="w-4 h-4 animate-spin" /> Predicting...</> : <><Play className="w-4 h-4" /> Predict</>}
              </button>
            </div>
          )}
        </div>

        {/* Results */}
        <div>
          {prediction ? (
            <div className="card p-6 space-y-5">
              <h3 className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>Prediction Results</h3>
              {Object.entries(prediction.predictions || {}).map(([target, result]: [string, any]) => (
                <div key={target}>
                  <p className="text-sm font-medium mb-3" style={{ color: 'var(--text-muted)' }}>Target: {target}</p>
                  {result.error ? (
                    <p className="text-sm" style={{ color: 'var(--danger)' }}>{result.error}</p>
                  ) : (
                    <div className="space-y-3">
                      <div className="text-center p-5 rounded-lg" style={{ background: 'var(--bg-tertiary)' }}>
                        <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Prediction</p>
                        <p className="text-4xl font-bold" style={{ color: 'var(--text-primary)' }}>
                          {Array.isArray(result.predictions) ? String(result.predictions[0]) : '—'}
                        </p>
                        {result.confidence && (
                          <div className="mt-3 flex items-center justify-center gap-2">
                            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Confidence:</span>
                            <div className="w-24 h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--bg-secondary)' }}>
                              <div className="h-full rounded-full" style={{ width: `${(result.confidence[0] || 0) * 100}%`, background: 'var(--success)' }} />
                            </div>
                            <span className="text-xs font-medium" style={{ color: 'var(--success)' }}>
                              {((result.confidence[0] || 0) * 100).toFixed(1)}%
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              ))}
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                Prediction time: {prediction.prediction_time_ms}ms · Input rows: {prediction.input_rows}
              </p>
            </div>
          ) : (
            <div className="card p-12 text-center">
              <TrendingUp className="w-12 h-12 mx-auto mb-4" style={{ color: 'var(--text-muted)' }} />
              <h3 className="text-lg font-medium mb-2" style={{ color: 'var(--text-primary)' }}>No prediction yet</h3>
              <p style={{ color: 'var(--text-muted)' }}>Select a model and fill in the feature values</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
