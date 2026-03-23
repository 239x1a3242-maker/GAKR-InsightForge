import { useState } from 'react';
import { useParams } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  BarChart3, TrendingUp, AlertTriangle, Loader2, Database,
  Sparkles, Trash2, CheckCircle
} from 'lucide-react';
import { datasetsApi } from '@/api/datasets';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';
import apiClient from '@/api/client';

const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];

export function AnalysisPage() {
  const { datasetId } = useParams();
  const queryClient = useQueryClient();
  const [selectedDataset, setSelectedDataset] = useState(datasetId || '');
  const [activeTab, setActiveTab] = useState<'overview' | 'columns' | 'correlation' | 'cleaning'>('overview');

  // Cleaning state
  const [cleanOptions, setCleanOptions] = useState({
    remove_missing: false,
    fill_missing_strategy: '',
    remove_duplicates: false,
    remove_outliers: false,
  });
  const [showRenameDialog, setShowRenameDialog] = useState(false);
  const [cleanedName, setCleanedName] = useState('');
  const [cleaning, setCleaning] = useState(false);
  const [cleanResult, setCleanResult] = useState<any>(null);

  const { data: datasets } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => datasetsApi.list(),
  });

  const { data: profile, isLoading: profileLoading } = useQuery({
    queryKey: ['dataset-profile', selectedDataset],
    queryFn: () => datasetsApi.getProfile(selectedDataset),
    enabled: !!selectedDataset,
  });

  const missingData = profile?.column_stats
    ? Object.entries(profile.column_stats)
        .filter(([, s]: [string, any]) => s.missing_percentage > 0)
        .map(([col, s]: [string, any]) => ({ name: col, value: s.missing_percentage }))
        .sort((a, b) => b.value - a.value).slice(0, 10)
    : [];

  const handleClean = async (newName: string) => {
    if (!selectedDataset) return;
    setCleaning(true);
    setShowRenameDialog(false);
    try {
      const res = await apiClient.post(`/api/datasets/${selectedDataset}/clean`, {
        ...cleanOptions,
        new_name: newName,
      });
      setCleanResult(res.data);
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
    } catch (e) { console.error(e); }
    finally { setCleaning(false); }
  };

  const currentDatasetName = datasets?.items.find(d => d.id === selectedDataset)?.name || '';

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>Analysis</h1>
          <p style={{ color: 'var(--text-muted)' }}>Explore and understand your datasets</p>
        </div>
        <div className="w-72">
          <Select value={selectedDataset || '__none__'} onValueChange={v => {
            setSelectedDataset(v === '__none__' ? '' : v);
            setCleanResult(null);
          }}>
            <SelectTrigger style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}>
              <SelectValue placeholder="Select a dataset..." />
            </SelectTrigger>
            <SelectContent style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
              <SelectItem value="__none__">Select a dataset...</SelectItem>
              {datasets?.items.map(d => (
                <SelectItem key={d.id} value={d.id}>{d.name}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {!selectedDataset ? (
        <div className="card p-12 text-center">
          <Database className="w-12 h-12 mx-auto mb-4" style={{ color: 'var(--text-muted)' }} />
          <h3 className="text-lg font-medium mb-2" style={{ color: 'var(--text-primary)' }}>Select a dataset</h3>
          <p style={{ color: 'var(--text-muted)' }}>Choose a dataset above to analyze its structure and quality</p>
        </div>
      ) : profileLoading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 animate-spin" style={{ color: 'var(--accent)' }} />
        </div>
      ) : profile ? (
        <>
          {/* Tabs */}
          <div className="flex gap-1 border-b" style={{ borderColor: 'var(--border-color)' }}>
            {(['overview', 'columns', 'correlation', 'cleaning'] as const).map(tab => (
              <button key={tab} onClick={() => setActiveTab(tab)}
                className="px-4 py-2 text-sm font-medium capitalize transition-colors"
                style={{
                  color: activeTab === tab ? 'var(--accent)' : 'var(--text-muted)',
                  borderBottom: activeTab === tab ? '2px solid var(--accent)' : '2px solid transparent',
                }}>
                {tab === 'cleaning' ? '🧹 Data Cleaning' : tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>

          {/* Overview Tab */}
          {activeTab === 'overview' && (
            <div className="space-y-6">
              {/* Stats */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  { label: 'Total Rows', value: profile.total_rows?.toLocaleString(), icon: Database },
                  { label: 'Total Columns', value: profile.total_columns, icon: BarChart3 },
                  { label: 'Numeric Cols', value: profile.numeric_columns, icon: TrendingUp },
                  { label: 'Categorical Cols', value: profile.categorical_columns, icon: BarChart3 },
                  { label: 'Missing Values', value: `${profile.missing_percentage}%`, icon: AlertTriangle },
                  { label: 'Duplicate Rows', value: profile.duplicate_rows, icon: Trash2 },
                  { label: 'Memory Usage', value: `${profile.memory_usage_mb?.toFixed(2)} MB`, icon: Database },
                  { label: 'Unique Cols', value: profile.total_columns, icon: Sparkles },
                ].map(({ label, value, icon: Icon }) => (
                  <div key={label} className="card p-4">
                    <div className="flex items-center gap-3">
                      <Icon className="w-5 h-5" style={{ color: 'var(--accent)' }} />
                      <div>
                        <p className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>{value}</p>
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{label}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Insights */}
              {profile.insights?.length > 0 && (
                <div className="card p-6">
                  <h3 className="text-lg font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>Insights</h3>
                  <div className="space-y-2">
                    {profile.insights.map((insight: any, i: number) => (
                      <div key={i} className="flex items-start gap-3 p-3 rounded-lg"
                        style={{
                          background: insight.severity === 'high' ? 'rgba(239,68,68,0.08)' : insight.severity === 'medium' ? 'rgba(245,158,11,0.08)' : 'rgba(59,130,246,0.08)',
                          border: `1px solid ${insight.severity === 'high' ? 'rgba(239,68,68,0.2)' : insight.severity === 'medium' ? 'rgba(245,158,11,0.2)' : 'rgba(59,130,246,0.2)'}`,
                        }}>
                        <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5"
                          style={{ color: insight.severity === 'high' ? 'var(--danger)' : insight.severity === 'medium' ? 'var(--warning)' : 'var(--accent)' }} />
                        <div>
                          <p style={{ color: 'var(--text-primary)' }}>{insight.message}</p>
                          {insight.suggestion && <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>Suggestion: {insight.suggestion}</p>}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Missing values chart */}
              {missingData.length > 0 && (
                <div className="card p-6">
                  <h3 className="text-lg font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>Missing Values by Column</h3>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={missingData} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                      <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11 }} stroke="var(--text-secondary)" />
                      <YAxis dataKey="name" type="category" tick={{ fontSize: 11 }} stroke="var(--text-secondary)" width={100} />
                      <Tooltip formatter={(v: any) => `${v}%`} contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-color)', borderRadius: 8 }} />
                      <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                        {missingData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          )}

          {/* Columns Tab */}
          {activeTab === 'columns' && profile.column_stats && (
            <div className="card overflow-hidden">
              <div className="overflow-auto">
                <table className="w-full text-sm">
                  <thead style={{ background: 'var(--bg-tertiary)' }}>
                    <tr>
                      {['Column', 'Type', 'Missing', 'Missing %', 'Unique', 'Mean', 'Std', 'Min', 'Max', 'Median'].map(h => (
                        <th key={h} className="text-left px-4 py-3 font-medium whitespace-nowrap"
                          style={{ color: 'var(--text-secondary)', borderBottom: '1px solid var(--border-color)' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(profile.column_stats).map(([col, stats]: [string, any]) => (
                      <tr key={col} style={{ borderBottom: '1px solid var(--border-color)' }}>
                        <td className="px-4 py-2 font-medium" style={{ color: 'var(--text-primary)' }}>{col}</td>
                        <td className="px-4 py-2">
                          <span className="px-2 py-0.5 rounded text-xs"
                            style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}>
                            {stats.dtype}
                          </span>
                        </td>
                        <td className="px-4 py-2" style={{ color: stats.missing > 0 ? 'var(--warning)' : 'var(--text-muted)' }}>{stats.missing}</td>
                        <td className="px-4 py-2" style={{ color: stats.missing_percentage > 10 ? 'var(--danger)' : 'var(--text-muted)' }}>
                          {stats.missing_percentage}%
                        </td>
                        <td className="px-4 py-2" style={{ color: 'var(--text-muted)' }}>{stats.unique}</td>
                        <td className="px-4 py-2" style={{ color: 'var(--text-muted)' }}>{stats.mean ?? '—'}</td>
                        <td className="px-4 py-2" style={{ color: 'var(--text-muted)' }}>{stats.std ?? '—'}</td>
                        <td className="px-4 py-2" style={{ color: 'var(--text-muted)' }}>{stats.min ?? '—'}</td>
                        <td className="px-4 py-2" style={{ color: 'var(--text-muted)' }}>{stats.max ?? '—'}</td>
                        <td className="px-4 py-2" style={{ color: 'var(--text-muted)' }}>{stats.median ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Correlation Tab */}
          {activeTab === 'correlation' && (
            <div className="card p-6">
              <h3 className="text-lg font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>Correlation Matrix</h3>
              {profile.correlation_matrix ? (
                <div className="overflow-auto">
                  <table className="text-xs">
                    <thead>
                      <tr>
                        <th className="px-2 py-1" />
                        {Object.keys(profile.correlation_matrix).map(col => (
                          <th key={col} className="px-2 py-1 font-medium whitespace-nowrap"
                            style={{ color: 'var(--text-secondary)' }}>{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(profile.correlation_matrix).map(([row, vals]: [string, any]) => (
                        <tr key={row}>
                          <td className="px-2 py-1 font-medium whitespace-nowrap" style={{ color: 'var(--text-secondary)' }}>{row}</td>
                          {Object.values(vals).map((v: any, i) => {
                            const val = Number(v);
                            const bg = val > 0.7 ? 'rgba(59,130,246,0.4)' : val > 0.4 ? 'rgba(59,130,246,0.2)' : val < -0.7 ? 'rgba(239,68,68,0.4)' : val < -0.4 ? 'rgba(239,68,68,0.2)' : 'transparent';
                            return (
                              <td key={i} className="px-2 py-1 text-center" style={{ background: bg, color: 'var(--text-primary)' }}>
                                {isNaN(val) ? '—' : val.toFixed(2)}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p style={{ color: 'var(--text-muted)' }}>No numeric columns available for correlation analysis.</p>
              )}
            </div>
          )}

          {/* Data Cleaning Tab */}
          {activeTab === 'cleaning' && (
            <div className="space-y-4">
              <div className="card p-6">
                <h3 className="text-lg font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>Data Cleaning</h3>
                <p className="text-sm mb-5" style={{ color: 'var(--text-muted)' }}>
                  Apply preprocessing steps and save as a new dataset.
                </p>
                <div className="space-y-4">
                  {/* Remove missing rows */}
                  <label className="flex items-center gap-3 cursor-pointer">
                    <input type="checkbox" checked={cleanOptions.remove_missing}
                      onChange={e => setCleanOptions(p => ({ ...p, remove_missing: e.target.checked, fill_missing_strategy: e.target.checked ? '' : p.fill_missing_strategy }))}
                      className="w-4 h-4 rounded" />
                    <div>
                      <p className="font-medium text-sm" style={{ color: 'var(--text-primary)' }}>Remove rows with missing values</p>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Drops any row that has at least one null value</p>
                    </div>
                  </label>

                  {/* Fill missing */}
                  <div className="space-y-2">
                    <p className="font-medium text-sm" style={{ color: 'var(--text-primary)' }}>Fill missing values with strategy</p>
                    <div className="flex gap-2 flex-wrap">
                      {['mean', 'median', 'mode'].map(s => (
                        <button key={s} onClick={() => setCleanOptions(p => ({
                          ...p,
                          fill_missing_strategy: p.fill_missing_strategy === s ? '' : s,
                          remove_missing: p.fill_missing_strategy === s ? p.remove_missing : false,
                        }))}
                          className="px-3 py-1.5 rounded-lg text-sm font-medium transition-all capitalize"
                          style={{
                            background: cleanOptions.fill_missing_strategy === s ? 'var(--accent)' : 'var(--bg-tertiary)',
                            color: cleanOptions.fill_missing_strategy === s ? '#fff' : 'var(--text-secondary)',
                            border: '1px solid var(--border-color)',
                          }}>
                          {s}
                        </button>
                      ))}
                    </div>
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Fills numeric columns with the selected statistic</p>
                  </div>

                  {/* Remove duplicates */}
                  <label className="flex items-center gap-3 cursor-pointer">
                    <input type="checkbox" checked={cleanOptions.remove_duplicates}
                      onChange={e => setCleanOptions(p => ({ ...p, remove_duplicates: e.target.checked }))}
                      className="w-4 h-4 rounded" />
                    <div>
                      <p className="font-medium text-sm" style={{ color: 'var(--text-primary)' }}>Remove duplicate rows</p>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        {profile.duplicate_rows > 0 ? `${profile.duplicate_rows} duplicates detected` : 'No duplicates detected'}
                      </p>
                    </div>
                  </label>

                  {/* Remove outliers */}
                  <label className="flex items-center gap-3 cursor-pointer">
                    <input type="checkbox" checked={cleanOptions.remove_outliers}
                      onChange={e => setCleanOptions(p => ({ ...p, remove_outliers: e.target.checked }))}
                      className="w-4 h-4 rounded" />
                    <div>
                      <p className="font-medium text-sm" style={{ color: 'var(--text-primary)' }}>Remove outliers (IQR method)</p>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Removes rows where numeric values are beyond 1.5×IQR</p>
                    </div>
                  </label>
                </div>

                <Button
                  className="mt-6 w-full"
                  style={{ background: 'var(--accent)' }}
                  disabled={cleaning || (!cleanOptions.remove_missing && !cleanOptions.fill_missing_strategy && !cleanOptions.remove_duplicates && !cleanOptions.remove_outliers)}
                  onClick={() => {
                    setCleanedName(`${currentDatasetName}_cleaned`);
                    setShowRenameDialog(true);
                  }}>
                  {cleaning ? <><Loader2 size={14} className="mr-2 animate-spin" /> Cleaning...</> : '🧹 Clean & Save Dataset'}
                </Button>
              </div>

              {cleanResult && (
                <div className="card p-5" style={{ border: '1px solid var(--success)' }}>
                  <div className="flex items-center gap-2 mb-3">
                    <CheckCircle size={18} style={{ color: 'var(--success)' }} />
                    <p className="font-semibold" style={{ color: 'var(--text-primary)' }}>Dataset cleaned and saved!</p>
                  </div>
                  <div className="grid grid-cols-3 gap-3 text-sm">
                    <div className="p-3 rounded-lg" style={{ background: 'var(--bg-tertiary)' }}>
                      <p style={{ color: 'var(--text-muted)' }}>Original rows</p>
                      <p className="font-semibold" style={{ color: 'var(--text-primary)' }}>{cleanResult.original_rows?.toLocaleString()}</p>
                    </div>
                    <div className="p-3 rounded-lg" style={{ background: 'var(--bg-tertiary)' }}>
                      <p style={{ color: 'var(--text-muted)' }}>Cleaned rows</p>
                      <p className="font-semibold" style={{ color: 'var(--text-primary)' }}>{cleanResult.cleaned_rows?.toLocaleString()}</p>
                    </div>
                    <div className="p-3 rounded-lg" style={{ background: 'var(--bg-tertiary)' }}>
                      <p style={{ color: 'var(--text-muted)' }}>Rows removed</p>
                      <p className="font-semibold" style={{ color: 'var(--danger)' }}>{cleanResult.rows_removed?.toLocaleString()}</p>
                    </div>
                  </div>
                  <p className="text-sm mt-3" style={{ color: 'var(--text-muted)' }}>
                    Saved as: <span style={{ color: 'var(--accent)' }}>{cleanResult.new_dataset_name}</span>
                  </p>
                </div>
              )}
            </div>
          )}
        </>
      ) : null}

      {/* Rename dialog */}
      <Dialog open={showRenameDialog} onOpenChange={setShowRenameDialog}>
        <DialogContent style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
          <DialogHeader>
            <DialogTitle style={{ color: 'var(--text-primary)' }}>Save Cleaned Dataset</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <div>
              <Label style={{ color: 'var(--text-secondary)' }}>New Dataset Name</Label>
              <Input value={cleanedName} onChange={e => setCleanedName(e.target.value)}
                className="mt-1"
                style={{ background: 'var(--bg-tertiary)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }} />
            </div>
            <Button onClick={() => handleClean(cleanedName)} disabled={!cleanedName}
              className="w-full" style={{ background: 'var(--accent)' }}>
              Save
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
