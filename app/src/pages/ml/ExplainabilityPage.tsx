import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Loader2,
  Info,
} from 'lucide-react';
import { modelsApi } from '../../api/models';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Cell,
} from 'recharts';

export function ExplainabilityPage() {
  const [selectedVersion, setSelectedVersion] = useState('');

  const { data: models } = useQuery({
    queryKey: ['models'],
    queryFn: () => modelsApi.list(),
  });

  const { data: shapData, isLoading: shapLoading } = useQuery({
    queryKey: ['shap-global', selectedVersion],
    queryFn: () => modelsApi.getShapGlobal(selectedVersion),
    enabled: !!selectedVersion,
  });

  const allVersions = models?.flatMap(m => 
    m.versions?.map(v => ({ ...v, target: m.target_column })) || []
  ) || [];

  const chartData = shapData?.feature_importance
    ? Object.entries(shapData.feature_importance)
        .map(([name, value]) => ({ name, value: (value as number) * 100 }))
        .sort((a, b) => b.value - a.value)
        .slice(0, 15)
    : [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-[var(--text-primary)]">Explainability</h1>
        <p className="text-[var(--text-muted)]">
          SHAP explanations and feature importance
        </p>
      </div>

      <div className="card p-6">
        <h3 className="text-lg font-semibold text-[var(--text-primary)] mb-4">
          Select Model Version
        </h3>
        
        <select
          value={selectedVersion}
          onChange={(e) => setSelectedVersion(e.target.value)}
          className="input"
        >
          <option value="">Choose a model version...</option>
          {allVersions.map((version) => (
            <option key={version.id} value={version.id}>
              {version.target} - v{version.version_number} ({version.status})
            </option>
          ))}
        </select>
      </div>

      {selectedVersion && (
        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-[var(--text-primary)]">
              Global Feature Importance
            </h3>
            <div className="flex items-center gap-2 text-sm text-[var(--text-muted)]">
              <Info className="w-4 h-4" />
              Based on SHAP values
            </div>
          </div>

          {shapLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-8 h-8 animate-spin text-blue-400" />
            </div>
          ) : chartData.length > 0 ? (
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={chartData}
                  layout="vertical"
                  margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                  <XAxis 
                    type="number" 
                    stroke="var(--text-muted)"
                    tickFormatter={(v) => `${v.toFixed(0)}%`}
                  />
                  <YAxis 
                    type="category" 
                    dataKey="name" 
                    stroke="var(--text-secondary)"
                    width={90}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'var(--bg-secondary)',
                      border: '1px solid var(--border-color)',
                      borderRadius: '8px',
                    }}
                    formatter={(value: number) => [`${value.toFixed(2)}%`, 'Importance']}
                  />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                    {chartData.map((_, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={`hsl(${210 + index * 5}, 70%, ${60 - index * 2}%)`}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="text-center py-12 text-[var(--text-muted)]">
              <Info className="w-12 h-12 mx-auto mb-4" />
              <p>No feature importance data available</p>
            </div>
          )}
        </div>
      )}

      <div className="card p-6">
        <h3 className="text-lg font-semibold text-[var(--text-primary)] mb-4">
          About SHAP Explanations
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 bg-[var(--bg-tertiary)] rounded-lg">
            <h4 className="font-medium text-[var(--text-primary)] mb-2">
              Global Explanations
            </h4>
            <p className="text-sm text-[var(--text-muted)]">
              Show which features are most important across all predictions.
              Higher values indicate greater impact on model output.
            </p>
          </div>
          
          <div className="p-4 bg-[var(--bg-tertiary)] rounded-lg">
            <h4 className="font-medium text-[var(--text-primary)] mb-2">
              Local Explanations
            </h4>
            <p className="text-sm text-[var(--text-muted)]">
              Explain individual predictions by showing how each feature
              contributed to the specific outcome.
            </p>
          </div>
          
          <div className="p-4 bg-[var(--bg-tertiary)] rounded-lg">
            <h4 className="font-medium text-[var(--text-primary)] mb-2">
              Feature Importance
            </h4>
            <p className="text-sm text-[var(--text-muted)]">
              Aggregated importance scores based on the magnitude of SHAP values.
              Helps identify key drivers in your model.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
