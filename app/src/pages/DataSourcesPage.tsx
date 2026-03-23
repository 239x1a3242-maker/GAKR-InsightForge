import { useEffect, useState, useCallback } from 'react';
import { datasetsApi } from '@/api/datasets';
import type { Dataset } from '@/types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { 
  Database, Upload, FileSpreadsheet, FileJson, Trash2, 
  RefreshCw, Eye, Search, Plus, GitCompare 
} from 'lucide-react';
import { formatDistanceToNow } from '@/utils/format';
import { DatasetComparison } from '@/components/data/DatasetComparison';

export function DataSourcesPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [previewData, setPreviewData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  const loadDatasets = useCallback(async () => {
    try {
      const data = await datasetsApi.getDatasets();
      setDatasets(data);
    } catch (error) {
      console.error('Failed to load datasets:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadDatasets();
  }, [loadDatasets]);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const name = file.name.replace(/\.[^/.]+$/, '');

    setIsUploading(true);
    try {
      await datasetsApi.uploadFile(file, name);
      await loadDatasets();
      setUploadDialogOpen(false);
    } catch (error) {
      console.error('Upload failed:', error);
    } finally {
      setIsUploading(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this dataset?')) return;

    try {
      await datasetsApi.deleteDataset(id);
      await loadDatasets();
    } catch (error) {
      console.error('Delete failed:', error);
    }
  };

  const handlePreview = async (dataset: Dataset) => {
    setSelectedDataset(dataset);
    try {
      const data = await datasetsApi.getPreview(dataset.id);
      setPreviewData(data);
    } catch (error) {
      console.error('Failed to load preview:', error);
    }
  };

  const filteredDatasets = datasets.filter(ds =>
    ds.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getSourceIcon = (sourceType: string) => {
    switch (sourceType) {
      case 'csv':
      case 'excel':
        return <FileSpreadsheet size={20} style={{ color: 'var(--success)' }} />;
      case 'json':
        return <FileJson size={20} style={{ color: 'var(--warning)' }} />;
      default:
        return <Database size={20} style={{ color: 'var(--accent)' }} />;
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2" style={{ borderColor: 'var(--accent)' }} />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-heading text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>
            Data Sources
          </h1>
          <p style={{ color: 'var(--text-secondary)' }}>
            Manage your datasets and connections
          </p>
        </div>
        <Dialog open={uploadDialogOpen} onOpenChange={setUploadDialogOpen}>
          <DialogTrigger asChild>
            <Button style={{ background: 'var(--accent)' }}>
              <Upload size={16} className="mr-2" />
              Upload Data
            </Button>
          </DialogTrigger>
          <DialogContent style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
            <DialogHeader>
              <DialogTitle style={{ color: 'var(--text-primary)' }}>Upload Dataset</DialogTitle>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="border-2 border-dashed rounded-lg p-8 text-center" style={{ borderColor: 'var(--border-color)' }}>
                <Upload size={48} className="mx-auto mb-4" style={{ color: 'var(--text-muted)' }} />
                <p className="mb-2" style={{ color: 'var(--text-primary)' }}>
                  Drag and drop your file here
                </p>
                <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
                  Supports CSV, Excel, and JSON files
                </p>
                <Input
                  type="file"
                  accept=".csv,.xlsx,.xls,.json"
                  onChange={handleFileUpload}
                  disabled={isUploading}
                  className="hidden"
                  id="file-upload"
                />
                <label htmlFor="file-upload">
                  <Button variant="outline" disabled={isUploading} asChild>
                    <span>{isUploading ? 'Uploading...' : 'Select File'}</span>
                  </Button>
                </label>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </div>
      {/* Compare datasets button */}
      <div>
        <Dialog>
          <DialogTrigger asChild>
            <Button variant="outline" style={{ borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}>
              <GitCompare size={16} className="mr-2" />
              Compare Datasets
            </Button>
          </DialogTrigger>
          <DialogContent style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
            <DialogHeader>
              <DialogTitle style={{ color: 'var(--text-primary)' }}>Compare Datasets</DialogTitle>
            </DialogHeader>
            <DatasetComparison datasets={datasets} />
          </DialogContent>
        </Dialog>
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2" size={18} style={{ color: 'var(--text-muted)' }} />
        <Input
          placeholder="Search datasets..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="pl-10"
          style={{
            background: 'var(--bg-secondary)',
            borderColor: 'var(--border-color)',
            color: 'var(--text-primary)',
          }}
        />
      </div>

      {/* Datasets Grid */}
      {filteredDatasets.length === 0 ? (
        <Card style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
          <CardContent className="flex flex-col items-center justify-center py-16">
            <Database size={64} style={{ color: 'var(--text-muted)' }} className="mb-4" />
            <h3 className="text-lg font-medium mb-2" style={{ color: 'var(--text-primary)' }}>
              No datasets yet
            </h3>
            <p className="text-center mb-6 max-w-md" style={{ color: 'var(--text-muted)' }}>
              Upload your first dataset to start building reports and dashboards
            </p>
            <Button onClick={() => setUploadDialogOpen(true)} style={{ background: 'var(--accent)' }}>
              <Plus size={16} className="mr-2" />
              Upload Dataset
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredDatasets.map((dataset) => (
            <Card 
              key={dataset.id} 
              className="group cursor-pointer transition-all hover:shadow-lg"
              style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}
            >
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    {getSourceIcon(dataset.source_type)}
                    <div>
                      <CardTitle className="text-base font-medium" style={{ color: 'var(--text-primary)' }}>
                        {dataset.name}
                      </CardTitle>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        {dataset.row_count.toLocaleString()} rows
                      </p>
                    </div>
                  </div>
                  <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(dataset.id);
                      }}
                    >
                      <Trash2 size={16} style={{ color: 'var(--danger)' }} />
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between text-sm">
                  <span style={{ color: 'var(--text-muted)' }}>
                    {dataset.columns?.length || 0} columns
                  </span>
                  <span style={{ color: 'var(--text-muted)' }}>
                    {dataset.last_refresh 
                      ? formatDistanceToNow(dataset.last_refresh)
                      : 'Never refreshed'
                    }
                  </span>
                </div>
                <div className="mt-4 flex gap-2">
                  <Button 
                    variant="outline" 
                    size="sm" 
                    className="flex-1"
                    onClick={() => handlePreview(dataset)}
                  >
                    <Eye size={14} className="mr-1" />
                    Preview
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => {}}
                  >
                    <RefreshCw size={14} />
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Preview Dialog */}
      <Dialog open={!!selectedDataset} onOpenChange={() => setSelectedDataset(null)}>
        <DialogContent 
          className="max-w-4xl max-h-[80vh] overflow-auto"
          style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}
        >
          <DialogHeader>
            <DialogTitle style={{ color: 'var(--text-primary)' }}>
              {selectedDataset?.name} - Preview
            </DialogTitle>
          </DialogHeader>
          {previewData && (
            <div className="overflow-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                    {previewData.columns.map((col: any) => (
                      <th 
                        key={col.name} 
                        className="text-left p-2 font-medium"
                        style={{ color: 'var(--text-secondary)' }}
                      >
                        {col.name}
                        <span className="ml-1 text-xs" style={{ color: 'var(--text-muted)' }}>
                          ({col.type})
                        </span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {previewData.data.map((row: any, i: number) => (
                    <tr 
                      key={i} 
                      style={{ borderBottom: '1px solid var(--border-color)' }}
                    >
                      {previewData.columns.map((col: any) => (
                        <td 
                          key={col.name} 
                          className="p-2"
                          style={{ color: 'var(--text-primary)' }}
                        >
                          {row[col.name] ?? '-'}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
