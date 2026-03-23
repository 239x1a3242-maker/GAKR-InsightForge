import { useState } from 'react';
import type { Dataset } from '@/types';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  GitCompare, CheckCircle, AlertCircle
} from 'lucide-react';
import { ChartRenderer } from '@/components/charts/ChartRenderer';

interface DatasetComparisonProps {
  datasets: Dataset[];
}

export function DatasetComparison({ datasets }: DatasetComparisonProps) {
  const [dataset1, setDataset1] = useState<string>('');
  const [dataset2, setDataset2] = useState<string>('');
  const [comparisonType, setComparisonType] = useState('summary');
  const [comparisonResult, setComparisonResult] = useState<any>(null);
  const [isComparing, setIsComparing] = useState(false);

  const handleCompare = async () => {
    if (!dataset1 || !dataset2) return;

    setIsComparing(true);
    try {
      const ds1 = datasets.find(d => d.id === dataset1);
      const ds2 = datasets.find(d => d.id === dataset2);
      
      setComparisonResult({
        datasets: [dataset1, dataset2],
        comparison_type: comparisonType,
        summaries: {
          [dataset1]: {
            name: ds1?.name,
            row_count: ds1?.row_count,
            column_count: ds1?.columns?.length || 0
          },
          [dataset2]: {
            name: ds2?.name,
            row_count: ds2?.row_count,
            column_count: ds2?.columns?.length || 0
          }
        },
        similarities: {
          row_ratio: Math.min(ds1?.row_count || 0, ds2?.row_count || 0) / Math.max(ds1?.row_count || 1, ds2?.row_count || 1),
          column_overlap: 0.5
        },
        differences: [
          {
            type: 'size',
            description: `Row count differs by ${Math.abs((ds1?.row_count || 0) - (ds2?.row_count || 0)).toLocaleString()} rows`
          }
        ]
      });
    } catch (error) {
      console.error('Comparison failed:', error);
    } finally {
      setIsComparing(false);
    }
  };

  const selectedDatasets = datasets.filter(d => d.id === dataset1 || d.id === dataset2);

  return (
    <div className="space-y-6">
      <Card style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2" style={{ color: 'var(--text-primary)' }}>
            <GitCompare size={20} style={{ color: 'var(--accent)' }} />
            Compare Datasets
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-sm" style={{ color: 'var(--text-secondary)' }}>Dataset 1</label>
              <Select value={dataset1} onValueChange={setDataset1}>
                <SelectTrigger 
                  className="mt-1"
                  style={{ background: 'var(--bg-tertiary)', borderColor: 'var(--border-color)' }}
                >
                  <SelectValue placeholder="Select dataset" />
                </SelectTrigger>
                <SelectContent style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
                  {datasets.map((ds) => (
                    <SelectItem key={ds.id} value={ds.id}>{ds.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <label className="text-sm" style={{ color: 'var(--text-secondary)' }}>Dataset 2</label>
              <Select value={dataset2} onValueChange={setDataset2}>
                <SelectTrigger 
                  className="mt-1"
                  style={{ background: 'var(--bg-tertiary)', borderColor: 'var(--border-color)' }}
                >
                  <SelectValue placeholder="Select dataset" />
                </SelectTrigger>
                <SelectContent style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
                  {datasets.map((ds) => (
                    <SelectItem key={ds.id} value={ds.id}>{ds.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div>
            <label className="text-sm" style={{ color: 'var(--text-secondary)' }}>Comparison Type</label>
            <Select value={comparisonType} onValueChange={setComparisonType}>
              <SelectTrigger 
                className="mt-1"
                style={{ background: 'var(--bg-tertiary)', borderColor: 'var(--border-color)' }}
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
                <SelectItem value="summary">Summary Statistics</SelectItem>
                <SelectItem value="distribution">Distribution Comparison</SelectItem>
                <SelectItem value="correlation">Correlation Analysis</SelectItem>
                <SelectItem value="trend">Trend Analysis</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <Button 
            onClick={handleCompare}
            disabled={!dataset1 || !dataset2 || isComparing}
            className="w-full"
            style={{ background: 'var(--accent)' }}
          >
            <GitCompare size={16} className="mr-2" />
            {isComparing ? 'Comparing...' : 'Compare'}
          </Button>
        </CardContent>
      </Card>

      {/* Comparison Results */}
      {comparisonResult && (
        <Tabs defaultValue="overview" className="w-full">
          <TabsList style={{ background: 'var(--bg-secondary)' }}>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="statistics">Statistics</TabsTrigger>
            <TabsTrigger value="differences">Differences</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              {selectedDatasets.map((ds) => (
                <Card key={ds.id} style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
                  <CardHeader>
                    <CardTitle className="text-lg" style={{ color: 'var(--text-primary)' }}>
                      {ds.name}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <div className="flex justify-between">
                      <span style={{ color: 'var(--text-muted)' }}>Rows</span>
                      <span style={{ color: 'var(--text-primary)' }}>{ds.row_count.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span style={{ color: 'var(--text-muted)' }}>Columns</span>
                      <span style={{ color: 'var(--text-primary)' }}>{ds.columns?.length || 0}</span>
                    </div>
                    <div className="flex justify-between">
                      <span style={{ color: 'var(--text-muted)' }}>Size</span>
                      <span style={{ color: 'var(--text-primary)' }}>
                        {(ds.size_bytes / 1024 / 1024).toFixed(2)} MB
                      </span>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* Similarity Score */}
            <Card style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
              <CardHeader>
                <CardTitle className="text-lg" style={{ color: 'var(--text-primary)' }}>
                  Similarity Score
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-4">
                  <div className="flex-1">
                    <div 
                      className="h-4 rounded-full"
                      style={{ background: 'var(--bg-tertiary)' }}
                    >
                      <div 
                        className="h-full rounded-full transition-all"
                        style={{ 
                          width: `${(comparisonResult.similarities?.row_ratio || 0) * 100}%`,
                          background: (comparisonResult.similarities?.row_ratio || 0) > 0.7 ? 'var(--success)' :
                                     (comparisonResult.similarities?.row_ratio || 0) > 0.4 ? 'var(--warning)' : 'var(--danger)'
                        }}
                      />
                    </div>
                  </div>
                  <span className="font-semibold" style={{ color: 'var(--text-primary)' }}>
                    {((comparisonResult.similarities?.row_ratio || 0) * 100).toFixed(0)}%
                  </span>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="statistics">
            <Card style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
              <CardContent className="p-6">
                <ChartRenderer
                  type="bar"
                  data={selectedDatasets.map(ds => ({
                    name: ds.name,
                    rows: ds.row_count,
                    columns: ds.columns?.length || 0
                  }))}
                  config={{
                    x_field: 'name',
                    y_fields: ['rows', 'columns'],
                    stacked: false,
                    show_legend: true
                  }}
                  height={300}
                />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="differences">
            <Card style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
              <CardContent className="p-6 space-y-3">
                {comparisonResult.differences?.map((diff: any, idx: number) => (
                  <div 
                    key={idx}
                    className="flex items-center gap-3 p-3 rounded-lg"
                    style={{ background: 'var(--bg-tertiary)' }}
                  >
                    <AlertCircle size={18} style={{ color: 'var(--warning)' }} />
                    <span style={{ color: 'var(--text-primary)' }}>{diff.description}</span>
                  </div>
                ))}
                {(!comparisonResult.differences || comparisonResult.differences.length === 0) && (
                  <div className="text-center py-8" style={{ color: 'var(--text-muted)' }}>
                    <CheckCircle size={48} className="mx-auto mb-2" style={{ color: 'var(--success)' }} />
                    <p>No significant differences found</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
}
