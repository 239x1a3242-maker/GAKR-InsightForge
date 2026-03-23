import { useState, useRef } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Plus, Search, FileSpreadsheet, FileJson, Trash2, Loader2,
  Upload, GitCompare, Eye, ChevronLeft, ChevronRight, CloudUpload
} from 'lucide-react';
import { datasetsApi } from '@/api/datasets';
import { useStore } from '@/store';
import { formatFileSize } from '@/utils/format';
import type { Dataset } from '@/types';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

export function DatasetsPage() {
  const queryClient = useQueryClient();
  const { addNotification } = useStore();
  const [searchQuery, setSearchQuery] = useState('');
  const [previewDataset, setPreviewDataset] = useState<Dataset | null>(null);
  const [previewData, setPreviewData] = useState<any>(null);
  const [previewPage, setPreviewPage] = useState(1);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  const [showCompare, setShowCompare] = useState(false);
  const [compareA, setCompareA] = useState('');
  const [compareB, setCompareB] = useState('');
  const [compareResult, setCompareResult] = useState<any>(null);
  const [comparing, setComparing] = useState(false);

  const { data, isLoading } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => datasetsApi.list(),
  });

  const deleteMutation = useMutation({
    mutationFn: datasetsApi.deleteDataset,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
      addNotification({ type: 'success', message: 'Dataset deleted' });
    },
  });

  const filteredDatasets = data?.items.filter(d =>
    d.name.toLowerCase().includes(searchQuery.toLowerCase())
  ) || [];

  const getFileIcon = (type: string) => {
    if (type === 'json') return <FileJson className="w-6 h-6" style={{ color: 'var(--warning)' }} />;
    return <FileSpreadsheet className="w-6 h-6" style={{ color: 'var(--success)' }} />;
  };

  const openPreview = async (dataset: Dataset, page = 1) => {
    setPreviewDataset(dataset);
    setPreviewPage(page);
    setPreviewLoading(true);
    try {
      const d = await datasetsApi.getPreview(dataset.id, page, 20);
      setPreviewData(d);
    } catch (e) { console.error(e); }
    finally { setPreviewLoading(false); }
  };

  const handleCompare = async () => {
    if (!compareA || !compareB) return;
    setComparing(true);
    try {
      const res = await fetch('/api/ml/compare-datasets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${localStorage.getItem('token')}` },
        body: JSON.stringify({ dataset_ids: [compareA, compareB] }),
      });
      setCompareResult(await res.json());
    } catch (e) { console.error(e); }
    finally { setComparing(false); }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>Datasets</h1>
          <p style={{ color: 'var(--text-muted)' }}>Manage your data sources</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => setShowCompare(true)}
            style={{ borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}>
            <GitCompare size={16} className="mr-2" /> Compare
          </Button>
          <Button onClick={() => setShowUpload(true)} style={{ background: 'var(--accent)' }}>
            <Plus size={16} className="mr-2" /> Upload Dataset
          </Button>
        </div>
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4" style={{ color: 'var(--text-muted)' }} />
        <input type="text" value={searchQuery} onChange={e => setSearchQuery(e.target.value)}
          placeholder="Search datasets..." className="input pl-10" />
      </div>

      {/* Grid */}
      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 animate-spin" style={{ color: 'var(--accent)' }} />
        </div>
      ) : filteredDatasets.length === 0 ? (
        <div className="card p-12 text-center">
          <Upload className="w-12 h-12 mx-auto mb-4" style={{ color: 'var(--text-muted)' }} />
          <h3 className="text-lg font-medium mb-2" style={{ color: 'var(--text-primary)' }}>No datasets yet</h3>
          <p className="mb-4" style={{ color: 'var(--text-muted)' }}>Upload your first dataset to get started</p>
          <Button onClick={() => setShowUpload(true)} style={{ background: 'var(--accent)' }}>Upload Dataset</Button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredDatasets.map(dataset => (
            <div key={dataset.id} className="card p-4 hover:border-blue-500/50 transition-colors">
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 rounded-lg flex items-center justify-center" style={{ background: 'var(--bg-tertiary)' }}>
                    {getFileIcon(dataset.source_type)}
                  </div>
                  <div>
                    <h3 className="font-medium" style={{ color: 'var(--text-primary)' }}>{dataset.name}</h3>
                    <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                      {dataset.row_count.toLocaleString()} rows · {dataset.columns?.length || 0} cols
                    </p>
                  </div>
                </div>
                <div className="flex gap-1">
                  <button onClick={() => openPreview(dataset)}
                    className="p-2 rounded-lg transition-colors hover:bg-blue-500/10"
                    title="Preview" style={{ color: 'var(--text-muted)' }}>
                    <Eye className="w-4 h-4" />
                  </button>
                  <button onClick={() => { if (confirm('Delete this dataset?')) deleteMutation.mutate(dataset.id); }}
                    className="p-2 rounded-lg transition-colors hover:bg-red-500/10"
                    title="Delete" style={{ color: 'var(--text-muted)' }}>
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
              <div className="mt-4 pt-4 flex items-center justify-between text-sm"
                style={{ borderTop: '1px solid var(--border-color)' }}>
                <span style={{ color: 'var(--text-muted)' }}>{formatFileSize(dataset.size_bytes)}</span>
                <span style={{ color: 'var(--text-muted)' }}>{new Date(dataset.created_at).toLocaleDateString()}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Preview Modal */}
      <Dialog open={!!previewDataset} onOpenChange={() => { setPreviewDataset(null); setPreviewData(null); }}>
        <DialogContent style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)', maxWidth: '90vw', width: 900 }}>
          <DialogHeader>
            <DialogTitle style={{ color: 'var(--text-primary)' }}>
              {previewDataset?.name} — Preview
            </DialogTitle>
          </DialogHeader>
          {previewLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-6 h-6 animate-spin" style={{ color: 'var(--accent)' }} />
            </div>
          ) : previewData ? (
            <div className="space-y-3">
              <div className="overflow-auto max-h-[55vh] rounded-lg" style={{ border: '1px solid var(--border-color)' }}>
                <table className="w-full text-xs">
                  <thead style={{ background: 'var(--bg-tertiary)', position: 'sticky', top: 0 }}>
                    <tr>
                      {previewData.columns.map((col: any) => (
                        <th key={col.name} className="text-left px-3 py-2 font-medium whitespace-nowrap"
                          style={{ color: 'var(--text-secondary)', borderBottom: '1px solid var(--border-color)' }}>
                          {col.name}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {previewData.data.map((row: any, i: number) => (
                      <tr key={i} style={{ borderBottom: '1px solid var(--border-color)' }}>
                        {previewData.columns.map((col: any) => (
                          <td key={col.name} className="px-3 py-1.5 whitespace-nowrap max-w-[200px] truncate"
                            style={{ color: 'var(--text-primary)' }}>
                            {String(row[col.name] ?? '')}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {/* Pagination */}
              <div className="flex items-center justify-between text-sm">
                <span style={{ color: 'var(--text-muted)' }}>
                  Showing rows {(previewPage - 1) * 20 + 1}–{Math.min(previewPage * 20, previewData.total_rows)} of {previewData.total_rows}
                </span>
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" disabled={previewPage <= 1}
                    onClick={() => previewDataset && openPreview(previewDataset, previewPage - 1)}
                    style={{ borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}>
                    <ChevronLeft size={14} />
                  </Button>
                  <Button variant="outline" size="sm" disabled={previewPage * 20 >= previewData.total_rows}
                    onClick={() => previewDataset && openPreview(previewDataset, previewPage + 1)}
                    style={{ borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}>
                    <ChevronRight size={14} />
                  </Button>
                </div>
              </div>
            </div>
          ) : null}
        </DialogContent>
      </Dialog>

      {/* Upload Modal */}
      <Dialog open={showUpload} onOpenChange={setShowUpload}>
        <DialogContent style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)', maxWidth: 560 }}>
          <DialogHeader>
            <DialogTitle style={{ color: 'var(--text-primary)' }}>Upload Dataset</DialogTitle>
          </DialogHeader>
          <UploadForm onClose={() => setShowUpload(false)} onSuccess={() => {
            queryClient.invalidateQueries({ queryKey: ['datasets'] });
            setShowUpload(false);
            addNotification({ type: 'success', message: 'Dataset uploaded successfully' });
          }} />
        </DialogContent>
      </Dialog>

      {/* Compare Modal */}
      <Dialog open={showCompare} onOpenChange={() => { setShowCompare(false); setCompareResult(null); }}>
        <DialogContent style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)', maxWidth: 700 }}>
          <DialogHeader>
            <DialogTitle style={{ color: 'var(--text-primary)' }}>Compare Datasets</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm mb-1 block" style={{ color: 'var(--text-secondary)' }}>Dataset A</label>
                <Select value={compareA} onValueChange={setCompareA}>
                  <SelectTrigger style={{ background: 'var(--bg-tertiary)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}>
                    <SelectValue placeholder="Select dataset" />
                  </SelectTrigger>
                  <SelectContent style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
                    {data?.items.map(ds => <SelectItem key={ds.id} value={ds.id}>{ds.name}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
              <div>
                <label className="text-sm mb-1 block" style={{ color: 'var(--text-secondary)' }}>Dataset B</label>
                <Select value={compareB} onValueChange={setCompareB}>
                  <SelectTrigger style={{ background: 'var(--bg-tertiary)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}>
                    <SelectValue placeholder="Select dataset" />
                  </SelectTrigger>
                  <SelectContent style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
                    {data?.items.filter(ds => ds.id !== compareA).map(ds => <SelectItem key={ds.id} value={ds.id}>{ds.name}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <Button onClick={handleCompare} disabled={!compareA || !compareB || comparing}
              className="w-full" style={{ background: 'var(--accent)' }}>
              {comparing ? <><Loader2 size={14} className="mr-2 animate-spin" /> Comparing...</> : 'Compare'}
            </Button>
            {compareResult && (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  {Object.entries(compareResult.summaries || {}).map(([dsId, summary]: [string, any]) => (
                    <div key={dsId} className="p-3 rounded-lg" style={{ background: 'var(--bg-tertiary)', border: '1px solid var(--border-color)' }}>
                      <p className="font-medium text-sm mb-2" style={{ color: 'var(--text-primary)' }}>{summary.name}</p>
                      <div className="space-y-1 text-xs" style={{ color: 'var(--text-secondary)' }}>
                        <p>Rows: {summary.row_count?.toLocaleString()}</p>
                        <p>Columns: {summary.column_count}</p>
                        <p>Numeric cols: {summary.numeric_columns}</p>
                      </div>
                    </div>
                  ))}
                </div>
                {compareResult.common_columns?.length > 0 && (
                  <div className="p-3 rounded-lg" style={{ background: 'var(--bg-tertiary)', border: '1px solid var(--border-color)' }}>
                    <p className="text-sm font-medium mb-2" style={{ color: 'var(--text-primary)' }}>
                      Common Columns ({compareResult.common_columns.length})
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {compareResult.common_columns.map((c: string) => (
                        <span key={c} className="px-2 py-0.5 rounded text-xs"
                          style={{ background: 'var(--bg-secondary)', color: 'var(--text-secondary)', border: '1px solid var(--border-color)' }}>
                          {c}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function UploadForm({ onClose, onSuccess }: { onClose: () => void; onSuccess: () => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState('');
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = (f: File) => {
    setFile(f);
    if (!name) setName(f.name.replace(/\.[^/.]+$/, ''));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || !name) return;
    setUploading(true);
    try {
      await datasetsApi.uploadFile(file, name);
      onSuccess();
    } catch (err) {
      console.error(err);
    } finally {
      setUploading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Drop zone */}
      <div
        className="rounded-xl border-2 border-dashed p-8 text-center cursor-pointer transition-all"
        style={{
          borderColor: dragOver ? 'var(--accent)' : 'var(--border-color)',
          background: dragOver ? 'rgba(59,130,246,0.05)' : 'var(--bg-tertiary)',
        }}
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={e => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f) handleFile(f); }}
        onClick={() => inputRef.current?.click()}>
        <input ref={inputRef} type="file" accept=".csv,.xlsx,.xls,.json" className="hidden"
          onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />
        <CloudUpload size={36} className="mx-auto mb-3" style={{ color: file ? 'var(--accent)' : 'var(--text-muted)' }} />
        {file ? (
          <div>
            <p className="font-medium" style={{ color: 'var(--text-primary)' }}>{file.name}</p>
            <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>{formatFileSize(file.size)}</p>
          </div>
        ) : (
          <div>
            <p className="font-medium" style={{ color: 'var(--text-primary)' }}>Drop file here or click to browse</p>
            <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>CSV, Excel, JSON supported</p>
          </div>
        )}
      </div>

      <div>
        <label className="block text-sm mb-1" style={{ color: 'var(--text-secondary)' }}>Dataset Name</label>
        <input type="text" value={name} onChange={e => setName(e.target.value)} required
          className="w-full px-3 py-2 rounded-md border text-sm"
          style={{ borderColor: 'var(--border-color)', background: 'var(--bg-primary)', color: 'var(--text-primary)' }}
          placeholder="My Dataset" />
      </div>

      <div className="flex gap-2 justify-end">
        <Button type="button" variant="outline" onClick={onClose} disabled={uploading}
          style={{ borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}>
          Cancel
        </Button>
        <Button type="submit" disabled={uploading || !file || !name} style={{ background: 'var(--accent)' }}>
          {uploading ? <><Loader2 size={14} className="mr-2 animate-spin" /> Uploading...</> : 'Upload'}
        </Button>
      </div>
    </form>
  );
}
