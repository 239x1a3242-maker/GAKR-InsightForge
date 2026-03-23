import { useEffect, useState, useRef, useCallback } from 'react';
import { useParams, useNavigate, useSearchParams } from 'react-router-dom';
import { ResizableBox } from 'react-resizable';
import 'react-resizable/css/styles.css';
import { reportsApi } from '@/api/reports';
import { datasetsApi } from '@/api/datasets';
import { queryApi } from '@/api/query';
import type { Report, Dataset } from '@/types';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import {
  Save, ArrowLeft, Download,
  BarChart3, LineChart, AreaChart, PieChart,
  Table, TrendingUp, X, Settings2, Eye, Edit,
  ScatterChart, Activity, GitBranch, Layers,
  AlignLeft, Maximize2, Grid, Sun, Droplets,
  Radio, Gauge, BarChart2, Map, Globe,
  Sigma, BoxSelect, Waves, Filter, Triangle,
} from 'lucide-react';
import {
  PlotlyBarChart, PlotlyGroupedBarChart, PlotlyStackedBarChart, PlotlyStackedBar100Chart,
  PlotlyHorizontalBarChart, PlotlyLineChart, PlotlyMultiLineChart, PlotlySplineChart,
  PlotlyStepLineChart, PlotlyAreaChart, PlotlyComboChart,
  PlotlyPieChart, PlotlyDonutChart, PlotlyScatterChart, PlotlyBubbleChart,
  PlotlyHistogram, PlotlyBoxPlot, PlotlyViolinPlot,
  PlotlyTreemapChart, PlotlySunburstChart, PlotlyWaterfallChart, PlotlyFunnelChart,
  PlotlyHeatmap, PlotlyCalendarHeatmap, PlotlyRadarChart,
  PlotlyGaugeChart, PlotlyProgressChart, PlotlyParallelCoords,
  PlotlyChoroplethMap, PlotlyBubbleMap,
  PlotlyKPI, PlotlyDataTable,
} from '@/utils/plotlyCharts';

// Chart type registry — 32 types with icons
const CHART_TYPES = [
  // Bar family
  { type: 'bar',          label: 'Bar Chart',          Icon: BarChart3,   group: 'Bar' },
  { type: 'grouped_bar',  label: 'Grouped Bar',         Icon: BarChart2,   group: 'Bar' },
  { type: 'stacked_bar',  label: 'Stacked Bar',         Icon: Layers,      group: 'Bar' },
  { type: 'stacked_bar_100', label: '100% Stacked Bar', Icon: Maximize2,   group: 'Bar' },
  { type: 'horizontal_bar', label: 'Horizontal Bar',    Icon: AlignLeft,   group: 'Bar' },
  // Line family
  { type: 'line',         label: 'Line Chart',          Icon: LineChart,   group: 'Line' },
  { type: 'multi_line',   label: 'Multi-Line',          Icon: Activity,    group: 'Line' },
  { type: 'spline',       label: 'Spline Chart',        Icon: GitBranch,   group: 'Line' },
  { type: 'step_line',    label: 'Step Line',           Icon: Filter,      group: 'Line' },
  { type: 'area',         label: 'Area Chart',          Icon: AreaChart,   group: 'Line' },
  { type: 'combo',        label: 'Combo Chart',         Icon: BarChart3,   group: 'Line' },
  // Circular
  { type: 'pie',          label: 'Pie Chart',           Icon: PieChart,    group: 'Circular' },
  { type: 'donut',        label: 'Donut Chart',         Icon: Radio,       group: 'Circular' },
  // Scatter / Distribution
  { type: 'scatter',      label: 'Scatter Plot',        Icon: ScatterChart, group: 'Scatter' },
  { type: 'bubble',       label: 'Bubble Chart',        Icon: Droplets,    group: 'Scatter' },
  { type: 'histogram',    label: 'Histogram',           Icon: BarChart3,   group: 'Scatter' },
  { type: 'box',          label: 'Box Plot',            Icon: BoxSelect,   group: 'Scatter' },
  { type: 'violin',       label: 'Violin Plot',         Icon: Waves,       group: 'Scatter' },
  // Hierarchical
  { type: 'treemap',      label: 'Treemap',             Icon: Grid,        group: 'Hierarchical' },
  { type: 'sunburst',     label: 'Sunburst',            Icon: Sun,         group: 'Hierarchical' },
  // Flow
  { type: 'waterfall',    label: 'Waterfall',           Icon: Triangle,    group: 'Flow' },
  { type: 'funnel',       label: 'Funnel Chart',        Icon: Filter,      group: 'Flow' },
  // Geo
  { type: 'choropleth',   label: 'Choropleth Map',      Icon: Map,         group: 'Geo' },
  { type: 'bubble_map',   label: 'Bubble Map',          Icon: Globe,       group: 'Geo' },
  // Matrix / Polar
  { type: 'heatmap',      label: 'Heatmap',             Icon: Grid,        group: 'Matrix' },
  { type: 'calendar_heatmap', label: 'Calendar Heatmap', Icon: Grid,       group: 'Matrix' },
  { type: 'radar',        label: 'Radar Chart',         Icon: Radio,       group: 'Polar' },
  // KPI / Indicators
  { type: 'gauge',        label: 'Gauge Chart',         Icon: Gauge,       group: 'KPI' },
  { type: 'progress',     label: 'Progress Bar',        Icon: Sigma,       group: 'KPI' },
  { type: 'kpi',          label: 'KPI Card',            Icon: TrendingUp,  group: 'KPI' },
  // Multi-dim
  { type: 'parallel',     label: 'Parallel Coords',     Icon: Activity,    group: 'Multi' },
  // Table
  { type: 'table',        label: 'Data Table',          Icon: Table,       group: 'Table' },
];

// Widget stores size as percentage of canvas width (20–100) and px height
interface ChartWidget {
  id: string;
  type: string;
  title: string;
  dataset_id: string;
  x_axis: string;
  y_axis: string;
  z_axis: string;
  wPct: number;  // 20–100 (% of canvas)
  h: number;     // px
}

const DEF_PCT = 45;   // default ~2 per row
const DEF_H   = 320;
const MIN_PCT  = 20;  // minimum 20% (5 per row)
const MAX_PCT  = 100;
const MIN_H    = 200;
const MAX_H    = 800;

export function ReportBuilderPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [report, setReport] = useState<Report | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [dataCache, setDataCache] = useState<Record<string, Record<string, any>[]>>({});
  const [widgets, setWidgets] = useState<ChartWidget[]>([]);
  const [selectedWidget, setSelectedWidget] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [dragOver, setDragOver] = useState(false);
  const [dragType, setDragType] = useState<string | null>(null);
  const [isDirty, setIsDirty] = useState(false);
  const [viewMode, setViewMode] = useState(searchParams.get('mode') === 'view');
  const canvasRef = useRef<HTMLDivElement>(null);
  const loadedDatasetsRef = useRef<Set<string>>(new Set());

  // Unsaved changes — warn on browser close/refresh
  useEffect(() => {
    const handler = (e: BeforeUnloadEvent) => {
      if (isDirty) { e.preventDefault(); e.returnValue = ''; }
    };
    window.addEventListener('beforeunload', handler);
    return () => window.removeEventListener('beforeunload', handler);
  }, [isDirty]);

  useEffect(() => {
    const load = async () => {
      if (!id) return;
      try {
        const [r, ds] = await Promise.all([reportsApi.getReport(id), datasetsApi.getDatasets()]);
        setReport(r);
        setDatasets(ds);
        const existing: ChartWidget[] = [];
        for (const page of r.pages || []) {
          for (const v of page.visuals || []) {
            existing.push({
              id: v.id || `w_${Date.now()}_${Math.random()}`,
              type: v.visual_type,
              title: v.title || 'Chart',
              dataset_id: v.data_fields?.dataset_id || r.dataset_id || '',
              x_axis: v.data_fields?.x_axis || '',
              y_axis: v.data_fields?.y_axis || '',
              z_axis: v.data_fields?.z_axis || '',
              wPct: v.data_fields?.wPct || DEF_PCT,
              h: v.data_fields?.h || DEF_H,
            });
          }
        }
        setWidgets(existing);
        const uniqueDs = [...new Set(existing.map(w => w.dataset_id).filter(Boolean))];
        for (const dsId of uniqueDs) {
          loadedDatasetsRef.current.add(dsId);
          queryApi.executeQuery({ dataset_id: dsId, limit: 500 })
            .then(res => setDataCache(prev => ({ ...prev, [dsId]: res.data || (res as any).rows || [] })))
            .catch(console.error);
        }
      } catch (e) { console.error(e); }
      finally { setIsLoading(false); }
    };
    load();
  }, [id]);

  const loadDataForDataset = useCallback(async (dataset_id: string) => {
    if (!dataset_id || loadedDatasetsRef.current.has(dataset_id)) return;
    loadedDatasetsRef.current.add(dataset_id);
    try {
      const res = await queryApi.executeQuery({ dataset_id, limit: 500 });
      setDataCache(prev => ({ ...prev, [dataset_id]: res.data || (res as any).rows || [] }));
    } catch (e) {
      loadedDatasetsRef.current.delete(dataset_id);
      console.error(e);
    }
  }, []);

  const getDatasetColumns = (dataset_id: string) =>
    datasets.find(d => d.id === dataset_id)?.columns || [];

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (!dragType) return;
    const newWidget: ChartWidget = {
      id: `w_${Date.now()}`,
      type: dragType,
      title: CHART_TYPES.find(c => c.type === dragType)?.label || 'Chart',
      dataset_id: report?.dataset_id || datasets[0]?.id || '',
      x_axis: '', y_axis: '', z_axis: '',
      wPct: DEF_PCT, h: DEF_H,
    };
    setWidgets(prev => [...prev, newWidget]);
    setSelectedWidget(newWidget.id);
    setDragType(null);
    setIsDirty(true);
    if (newWidget.dataset_id) loadDataForDataset(newWidget.dataset_id);
  };

  const updateWidget = (wid: string, updates: Partial<ChartWidget>) => {
    setWidgets(prev => prev.map(w => w.id === wid ? { ...w, ...updates } : w));
    setIsDirty(true);
    if (updates.dataset_id) loadDataForDataset(updates.dataset_id);
  };

  const deleteWidget = (wid: string) => {
    setWidgets(prev => prev.filter(w => w.id !== wid));
    if (selectedWidget === wid) setSelectedWidget(null);
    setIsDirty(true);
  };

  const handleSave = async () => {
    if (!report) return;
    setIsSaving(true);
    try {
      const visuals = widgets.map(w => ({
        id: w.id,
        visual_type: w.type,
        title: w.title,
        x: 0, y: 0, width: 6, height: 4,
        chart_config: {},
        data_fields: {
          dataset_id: w.dataset_id,
          x_axis: w.x_axis, y_axis: w.y_axis, z_axis: w.z_axis,
          wPct: w.wPct, h: w.h,
        },
      }));
      const pages = report.pages?.length
        ? [{ ...report.pages[0], visuals }]
        : [{ id: 'page_1', name: 'Page 1', visuals, filters: [] }];
      await reportsApi.updateReport(report.id, { name: report.name, pages });
      setIsDirty(false);
    } catch (e) { console.error(e); }
    finally { setIsSaving(false); }
  };

  const handleBack = () => {
    if (isDirty && !confirm('You have unsaved changes. Leave without saving?')) return;
    navigate('/reports');
  };

  const pxToPct = (px: number): number => {
    const cw = canvasRef.current?.clientWidth || 900;
    return Math.max(MIN_PCT, Math.min(MAX_PCT, Math.round((px / cw) * 100)));
  };

  const renderChartContent = (widget: ChartWidget, h: number) => {
    const rawData = dataCache[widget.dataset_id] || [];
    const { x_axis, y_axis, z_axis, type } = widget;
    const chartH = h - 44;

    if (widget.dataset_id && !loadedDatasetsRef.current.has(widget.dataset_id))
      return <div className="flex items-center justify-center h-full text-sm" style={{ color: 'var(--text-muted)' }}>Loading data...</div>;
    if (!rawData.length)
      return <div className="flex items-center justify-center h-full text-sm" style={{ color: 'var(--text-muted)' }}>{widget.dataset_id ? 'No data available' : 'Select a dataset'}</div>;

    if (type === 'kpi') {
      const yCol = y_axis || Object.keys(rawData[0])[0];
      return <PlotlyKPI data={rawData} y={yCol} title={widget.title} height={chartH} />;
    }
    if (type === 'gauge') {
      const yCol = y_axis || Object.keys(rawData[0])[0];
      return <PlotlyGaugeChart data={rawData} y={yCol} title={widget.title} height={chartH} />;
    }
    if (type === 'progress') {
      const yCol = y_axis || Object.keys(rawData[0])[0];
      return <PlotlyProgressChart data={rawData} y={yCol} title={widget.title} height={chartH} />;
    }
    if (type === 'table')
      return <PlotlyDataTable data={rawData.slice(0, 100)} height={chartH} />;
    if (type === 'parallel')
      return <PlotlyParallelCoords data={rawData} x={x_axis || ''} y={y_axis || ''} z={z_axis} height={chartH} />;

    if (!x_axis)
      return <div className="flex items-center justify-center h-full text-sm" style={{ color: 'var(--text-muted)' }}>Select X axis in the panel</div>;

    const yCol = y_axis || Object.keys(rawData[0]).find(k => k !== x_axis) || x_axis;

    switch (type) {
      case 'bar':             return <PlotlyBarChart          data={rawData} x={x_axis} y={yCol} z={z_axis} height={chartH} />;
      case 'grouped_bar':     return <PlotlyGroupedBarChart   data={rawData} x={x_axis} y={yCol} z={z_axis} height={chartH} />;
      case 'stacked_bar':     return <PlotlyStackedBarChart   data={rawData} x={x_axis} y={yCol} z={z_axis} height={chartH} />;
      case 'stacked_bar_100': return <PlotlyStackedBar100Chart data={rawData} x={x_axis} y={yCol} z={z_axis} height={chartH} />;
      case 'horizontal_bar':  return <PlotlyHorizontalBarChart data={rawData} x={x_axis} y={yCol} height={chartH} />;
      case 'line':            return <PlotlyLineChart         data={rawData} x={x_axis} y={yCol} z={z_axis} height={chartH} />;
      case 'multi_line':      return <PlotlyMultiLineChart    data={rawData} x={x_axis} y={yCol} z={z_axis} height={chartH} />;
      case 'spline':          return <PlotlySplineChart       data={rawData} x={x_axis} y={yCol} height={chartH} />;
      case 'step_line':       return <PlotlyStepLineChart     data={rawData} x={x_axis} y={yCol} height={chartH} />;
      case 'area':            return <PlotlyAreaChart         data={rawData} x={x_axis} y={yCol} z={z_axis} height={chartH} />;
      case 'combo':           return <PlotlyComboChart        data={rawData} x={x_axis} y={yCol} z={z_axis} height={chartH} />;
      case 'pie':             return <PlotlyPieChart          data={rawData} x={x_axis} y={yCol} height={chartH} />;
      case 'donut':           return <PlotlyDonutChart        data={rawData} x={x_axis} y={yCol} height={chartH} />;
      case 'scatter':         return <PlotlyScatterChart      data={rawData} x={x_axis} y={yCol} z={z_axis} height={chartH} />;
      case 'bubble':          return <PlotlyBubbleChart       data={rawData} x={x_axis} y={yCol} z={z_axis} height={chartH} />;
      case 'histogram':       return <PlotlyHistogram         data={rawData} x={x_axis} y={yCol} height={chartH} />;
      case 'box':             return <PlotlyBoxPlot           data={rawData} x={x_axis} y={yCol} z={z_axis} height={chartH} />;
      case 'violin':          return <PlotlyViolinPlot        data={rawData} x={x_axis} y={yCol} z={z_axis} height={chartH} />;
      case 'treemap':         return <PlotlyTreemapChart      data={rawData} x={x_axis} y={yCol} z={z_axis} height={chartH} />;
      case 'sunburst':        return <PlotlySunburstChart     data={rawData} x={x_axis} y={yCol} z={z_axis} height={chartH} />;
      case 'waterfall':       return <PlotlyWaterfallChart    data={rawData} x={x_axis} y={yCol} height={chartH} />;
      case 'funnel':          return <PlotlyFunnelChart       data={rawData} x={x_axis} y={yCol} height={chartH} />;
      case 'heatmap':         return <PlotlyHeatmap           data={rawData} x={x_axis} y={yCol} z={z_axis} height={chartH} />;
      case 'calendar_heatmap': return <PlotlyCalendarHeatmap  data={rawData} x={x_axis} y={yCol} height={chartH} />;
      case 'radar':           return <PlotlyRadarChart        data={rawData} x={x_axis} y={yCol} z={z_axis} height={chartH} />;
      case 'choropleth':      return <PlotlyChoroplethMap     data={rawData} x={x_axis} y={yCol} height={chartH} />;
      case 'bubble_map':      return <PlotlyBubbleMap         data={rawData} x={x_axis} y={yCol} height={chartH} />;
      default: return null;
    }
  };

  const selectedW = widgets.find(w => w.id === selectedWidget);

  if (isLoading) return (
    <div className="flex items-center justify-center h-64">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2" style={{ borderColor: 'var(--accent)' }} />
    </div>
  );
  if (!report) return (
    <div className="flex flex-col items-center justify-center h-64">
      <p style={{ color: 'var(--text-muted)' }}>Report not found</p>
      <Button onClick={() => navigate('/reports')} className="mt-4">Back to Reports</Button>
    </div>
  );

  return (
    <div className="h-[calc(100vh-80px)] flex flex-col gap-3">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 rounded-lg flex-shrink-0"
        style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="icon" onClick={handleBack}>
            <ArrowLeft size={18} />
          </Button>
          <h1 className="font-semibold" style={{ color: 'var(--text-primary)' }}>{report.name}</h1>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => window.print()}
            style={{ borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}>
            <Download size={15} className="mr-1" /> Export PDF
          </Button>
          {viewMode ? (
            <Button size="sm" onClick={() => setViewMode(false)} style={{ background: 'var(--accent)' }}>
              <Edit size={15} className="mr-1" /> Edit
            </Button>
          ) : (
            <>
              <Button variant="outline" size="sm" onClick={() => setViewMode(true)}
                style={{ borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}>
                <Eye size={15} className="mr-1" /> View
              </Button>
              <Button size="sm" onClick={handleSave} disabled={isSaving} style={{ background: 'var(--accent)' }}>
                <Save size={15} className="mr-1" /> {isSaving ? 'Saving...' : 'Save'}
              </Button>
            </>
          )}
        </div>
      </div>

      <div className="flex flex-1 gap-3 overflow-hidden">
        {/* Left sidebar — edit mode only */}
        {!viewMode && (
          <div className="w-52 flex-shrink-0 rounded-lg p-3 overflow-auto"
            style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
            <p className="text-xs font-semibold uppercase mb-2" style={{ color: 'var(--text-muted)' }}>Chart Types</p>
            {/* Grouped icon picker — hover shows name */}
            {Array.from(new Set(CHART_TYPES.map(c => c.group))).map(group => (
              <div key={group} className="mb-2">
                <p className="text-xs px-0.5 mb-1" style={{ color: 'var(--text-muted)', opacity: 0.55, fontSize: 10 }}>{group}</p>
                <div className="grid grid-cols-4 gap-1">
                  {CHART_TYPES.filter(c => c.group === group).map(({ type, label, Icon }) => (
                    <div key={type}
                      draggable
                      onDragStart={() => setDragType(type)}
                      title={label}
                      className="flex items-center justify-center rounded cursor-grab active:cursor-grabbing transition-all hover:scale-110"
                      style={{ background: 'var(--bg-tertiary)', border: '1px solid var(--border-color)', height: 36 }}>
                      <Icon size={15} style={{ color: 'var(--accent)' }} />
                    </div>
                  ))}
                </div>
              </div>
            ))}

            {selectedW && (
              <div className="mt-3 pt-3" style={{ borderTop: '1px solid var(--border-color)' }}>
                <p className="text-xs font-semibold uppercase mb-3 flex items-center gap-1" style={{ color: 'var(--text-muted)' }}>
                  <Settings2 size={12} /> Configure
                </p>
                <div className="space-y-3">
                  <div>
                    <Label className="text-xs" style={{ color: 'var(--text-secondary)' }}>Title</Label>
                    <Input value={selectedW.title} onChange={e => updateWidget(selectedW.id, { title: e.target.value })}
                      className="mt-1 h-7 text-xs"
                      style={{ background: 'var(--bg-tertiary)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }} />
                  </div>
                  <div>
                    <Label className="text-xs" style={{ color: 'var(--text-secondary)' }}>Dataset</Label>
                    <Select value={selectedW.dataset_id}
                      onValueChange={v => updateWidget(selectedW.id, { dataset_id: v, x_axis: '', y_axis: '', z_axis: '' })}>
                      <SelectTrigger className="mt-1 h-7 text-xs"
                        style={{ background: 'var(--bg-tertiary)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}>
                        <SelectValue placeholder="Dataset" />
                      </SelectTrigger>
                      <SelectContent style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
                        {datasets.map(ds => <SelectItem key={ds.id} value={ds.id}>{ds.name}</SelectItem>)}
                      </SelectContent>
                    </Select>
                  </div>
                  {selectedW.dataset_id && (() => {
                    const cols = getDatasetColumns(selectedW.dataset_id);
                    return (
                      <>
                        {(['x_axis', 'y_axis', 'z_axis'] as const).map((axis, i) => (
                          <div key={axis}>
                            <Label className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                              {axis === 'x_axis' ? 'X Axis' : axis === 'y_axis' ? 'Y Axis' : 'Z Axis (optional)'}
                            </Label>
                            <Select value={selectedW[axis] || '__none__'}
                              onValueChange={v => updateWidget(selectedW.id, { [axis]: v === '__none__' ? '' : v } as any)}>
                              <SelectTrigger className="mt-1 h-7 text-xs"
                                style={{ background: 'var(--bg-tertiary)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}>
                                <SelectValue placeholder={axis.replace('_', ' ')} />
                              </SelectTrigger>
                              <SelectContent style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
                                {i === 2 && <SelectItem value="__none__">None</SelectItem>}
                                {cols.map(c => <SelectItem key={c.name} value={c.name}>{c.name}</SelectItem>)}
                              </SelectContent>
                            </Select>
                          </div>
                        ))}
                      </>
                    );
                  })()}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Canvas */}
        <div
          ref={canvasRef}
          className="flex-1 rounded-lg p-4 overflow-auto"
          style={{
            background: 'var(--bg-primary)',
            border: dragOver ? '2px dashed var(--accent)' : '1px solid var(--border-color)',
          }}
          onDragOver={e => { e.preventDefault(); if (!viewMode) setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          {widgets.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full" style={{ color: 'var(--text-muted)' }}>
              <BarChart3 size={56} className="mb-4 opacity-30" />
              <p className="text-lg mb-1">
                {viewMode ? 'No charts in this report' : 'Drag chart types here to build your report'}
              </p>
              {viewMode && (
                <Button size="sm" onClick={() => setViewMode(false)} className="mt-3" style={{ background: 'var(--accent)' }}>
                  <Edit size={14} className="mr-1" /> Edit Report
                </Button>
              )}
              {!viewMode && <p className="text-sm opacity-60">Up to 5 charts per row, unlimited rows</p>}
            </div>
          ) : (
            <div className="flex flex-wrap gap-3 content-start" style={{ minHeight: '100%' }}>
              {widgets.map(widget => {
                const canvasW = canvasRef.current?.clientWidth || 900;
                const wPx = Math.round((widget.wPct / 100) * canvasW);
                const minPx = Math.round((MIN_PCT / 100) * canvasW);
                const maxPx = Math.round((MAX_PCT / 100) * canvasW);

                return (
                  <ResizableBox
                    key={widget.id}
                    width={wPx}
                    height={widget.h}
                    minConstraints={[minPx, MIN_H]}
                    maxConstraints={[maxPx, MAX_H]}
                    resizeHandles={viewMode ? [] : ['n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw']}
                    onResizeStop={(_, { size }) => {
                      updateWidget(widget.id, { wPct: pxToPct(size.width), h: size.height });
                    }}
                    style={{ position: 'relative', flexShrink: 0 }}
                  >
                    <div
                      className="rounded-lg h-full flex flex-col transition-all"
                      style={{
                        background: 'var(--bg-secondary)',
                        border: !viewMode && selectedWidget === widget.id
                          ? '2px solid var(--accent)'
                          : '1px solid var(--border-color)',
                        overflow: 'hidden',
                        cursor: viewMode ? 'default' : 'pointer',
                      }}
                      onClick={() => !viewMode && setSelectedWidget(widget.id)}
                    >
                      {/* Header */}
                      <div className="flex items-center justify-between px-3 py-2 flex-shrink-0"
                        style={{ borderBottom: '1px solid var(--border-color)' }}>
                        <span className="text-sm font-medium truncate" style={{ color: 'var(--text-primary)' }}>
                          {widget.title}
                        </span>
                        {!viewMode && (
                          <button
                            onClick={e => { e.stopPropagation(); deleteWidget(widget.id); }}
                            className="p-1 rounded hover:bg-red-500/10 flex-shrink-0">
                            <X size={13} style={{ color: 'var(--danger, #ef4444)' }} />
                          </button>
                        )}
                      </div>
                      {/* Chart — fills remaining height */}
                      <div className="flex-1 min-h-0">
                        {renderChartContent(widget, widget.h)}
                      </div>
                    </div>
                  </ResizableBox>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
