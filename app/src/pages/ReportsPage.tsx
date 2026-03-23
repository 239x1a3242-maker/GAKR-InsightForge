import { useEffect, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { reportsApi } from '@/api/reports';
import { datasetsApi } from '@/api/datasets';
import type { Report, Dataset } from '@/types';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import {
  FileBarChart, Plus, Trash2, Search,
  Layout, Edit, Clock, Eye, Pencil,
} from 'lucide-react';
import { formatDistanceToNow } from '@/utils/format';

export function ReportsPage() {
  const navigate = useNavigate();
  const [reports, setReports] = useState<Report[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');

  // Create dialog
  const [createOpen, setCreateOpen] = useState(false);
  const [newName, setNewName] = useState('');
  const [selectedDataset, setSelectedDataset] = useState('');
  const [isCreating, setIsCreating] = useState(false);

  // Rename dialog
  const [renameOpen, setRenameOpen] = useState(false);
  const [renameId, setRenameId] = useState('');
  const [renameName, setRenameName] = useState('');
  const [isRenaming, setIsRenaming] = useState(false);

  const loadData = useCallback(async () => {
    try {
      const [r, ds] = await Promise.all([reportsApi.getReports(), datasetsApi.getDatasets()]);
      setReports(r);
      setDatasets(ds);
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  const handleCreate = async () => {
    if (!newName || !selectedDataset) return;
    setIsCreating(true);
    try {
      const report = await reportsApi.createReport({ name: newName, dataset_id: selectedDataset });
      setCreateOpen(false);
      setNewName(''); setSelectedDataset('');
      navigate(`/reports/${report.id}`);
    } catch (e) { console.error(e); }
    finally { setIsCreating(false); }
  };

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm('Delete this report?')) return;
    await reportsApi.deleteReport(id);
    loadData();
  };

  const openRename = (report: Report, e: React.MouseEvent) => {
    e.stopPropagation();
    setRenameId(report.id);
    setRenameName(report.name);
    setRenameOpen(true);
  };

  const handleRename = async () => {
    if (!renameName.trim()) return;
    setIsRenaming(true);
    try {
      await reportsApi.updateReport(renameId, { name: renameName });
      setRenameOpen(false);
      loadData();
    } catch (e) { console.error(e); }
    finally { setIsRenaming(false); }
  };

  const filtered = reports.filter(r => r.name.toLowerCase().includes(searchQuery.toLowerCase()));

  if (isLoading) return (
    <div className="flex items-center justify-center h-64">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2" style={{ borderColor: 'var(--accent)' }} />
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-heading text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>Reports</h1>
          <p style={{ color: 'var(--text-secondary)' }}>Build and manage your data reports</p>
        </div>
        <Button onClick={() => setCreateOpen(true)} style={{ background: 'var(--accent)' }}>
          <Plus size={16} className="mr-2" /> New Report
        </Button>
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2" size={16} style={{ color: 'var(--text-muted)' }} />
        <Input placeholder="Search reports..." value={searchQuery} onChange={e => setSearchQuery(e.target.value)}
          className="pl-10" style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }} />
      </div>

      {/* Report grid */}
      {filtered.length === 0 ? (
        <Card style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
          <CardContent className="flex flex-col items-center justify-center py-16">
            <FileBarChart size={56} style={{ color: 'var(--text-muted)' }} className="mb-4" />
            <h3 className="text-lg font-medium mb-2" style={{ color: 'var(--text-primary)' }}>No reports yet</h3>
            <p className="mb-4 text-sm" style={{ color: 'var(--text-muted)' }}>Create a report to visualize your data with charts</p>
            <Button onClick={() => setCreateOpen(true)} style={{ background: 'var(--accent)' }}>
              <Plus size={16} className="mr-2" /> Create Report
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered.map(report => (
            <Card key={report.id} className="group transition-all hover:shadow-lg"
              style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
              <CardContent className="p-5">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg" style={{ background: 'rgba(59,130,246,0.15)' }}>
                      <Layout size={20} style={{ color: 'var(--accent)' }} />
                    </div>
                    <div>
                      <p className="font-semibold text-sm" style={{ color: 'var(--text-primary)' }}>{report.name}</p>
                      <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
                        {report.page_count || 0} pages
                      </p>
                    </div>
                  </div>
                  {/* Action buttons — always visible */}
                  <div className="flex gap-1">
                    <Button variant="ghost" size="icon" className="h-7 w-7" title="View"
                      onClick={() => navigate(`/reports/${report.id}?mode=view`)}>
                      <Eye size={13} style={{ color: 'var(--text-muted)' }} />
                    </Button>
                    <Button variant="ghost" size="icon" className="h-7 w-7" title="Edit"
                      onClick={() => navigate(`/reports/${report.id}`)}>
                      <Edit size={13} style={{ color: 'var(--text-muted)' }} />
                    </Button>
                    <Button variant="ghost" size="icon" className="h-7 w-7" title="Rename"
                      onClick={e => openRename(report, e)}>
                      <Pencil size={13} style={{ color: 'var(--text-muted)' }} />
                    </Button>
                    <Button variant="ghost" size="icon" className="h-7 w-7" title="Delete"
                      onClick={e => handleDelete(report.id, e)}>
                      <Trash2 size={13} style={{ color: 'var(--danger)' }} />
                    </Button>
                  </div>
                </div>
                <div className="flex items-center text-xs pt-3 border-t" style={{ borderColor: 'var(--border-color)' }}>
                  <span className="flex items-center gap-1" style={{ color: 'var(--text-muted)' }}>
                    <Clock size={11} /> {formatDistanceToNow(report.updated_at)}
                  </span>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Create Dialog */}
      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
          <DialogHeader>
            <DialogTitle style={{ color: 'var(--text-primary)' }}>Create New Report</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <div>
              <Label style={{ color: 'var(--text-secondary)' }}>Report Name</Label>
              <Input value={newName} onChange={e => setNewName(e.target.value)}
                placeholder="My Sales Report" className="mt-1"
                style={{ background: 'var(--bg-tertiary)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }} />
            </div>
            <div>
              <Label style={{ color: 'var(--text-secondary)' }}>Dataset</Label>
              <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                <SelectTrigger className="mt-1" style={{ background: 'var(--bg-tertiary)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}>
                  <SelectValue placeholder="Select a dataset" />
                </SelectTrigger>
                <SelectContent style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
                  {datasets.map(ds => <SelectItem key={ds.id} value={ds.id}>{ds.name}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <Button onClick={handleCreate} disabled={!newName || !selectedDataset || isCreating}
              className="w-full" style={{ background: 'var(--accent)' }}>
              {isCreating ? 'Creating...' : 'Create Report'}
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Rename Dialog */}
      <Dialog open={renameOpen} onOpenChange={setRenameOpen}>
        <DialogContent style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
          <DialogHeader>
            <DialogTitle style={{ color: 'var(--text-primary)' }}>Rename Report</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <div>
              <Label style={{ color: 'var(--text-secondary)' }}>New Name</Label>
              <Input value={renameName} onChange={e => setRenameName(e.target.value)}
                className="mt-1"
                style={{ background: 'var(--bg-tertiary)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}
                onKeyDown={e => e.key === 'Enter' && handleRename()} />
            </div>
            <Button onClick={handleRename} disabled={!renameName.trim() || isRenaming}
              className="w-full" style={{ background: 'var(--accent)' }}>
              {isRenaming ? 'Saving...' : 'Rename'}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
