import { useEffect, useState, useCallback, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ResizableBox } from 'react-resizable';
import 'react-resizable/css/styles.css';
import { reportsApi } from '@/api/reports';
import { datasetsApi } from '@/api/datasets';
import { queryApi } from '@/api/query';
import type { Dashboard, Dataset } from '@/types';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import {
  ArrowLeft, Download, GripVertical,
  BarChart3, LineChart, AreaChart, PieChart,
  Table, TrendingUp, X, Settings2, Edit, Eye,
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

// Percentage-based sizing: 20–100% of canvas width
const DEF_PCT = 45;
const DEF_H   = 300;
const MIN_PCT  = 20;
const MAX_PCT  = 100;
const MIN_H    = 200;
const MAX_H    = 800;

const CHART_TYPES = [
  { type: 'bar',          label: 'Bar Chart',          Icon: BarChart3,   group: 'Bar' },
  { type: 'grouped_bar',  label: 'Grouped Bar',         Icon: BarChart2,   group: 'Bar' },
  { type: 'stacked_bar',  label: 'Stacked Bar',         Icon: Layers,      group: 'Bar' },
  { type: 'stacked_bar_100', label: '100% Stacked Bar', Icon: Maximize2,   group: 'Bar' },
  { type: 'horizontal_bar', label: 'Horizontal Bar',    Icon: AlignLeft,   group: 'Bar' },
  { type: 'line',         label: 'Line Chart',          Icon: LineChart,   group: 'Line' },
  { type: 'multi_line',   label: 'Multi-Line',          Icon: Activity,    group: 'Line' },
  { type: 'spline',       label: 'Spline Chart',        Icon: GitBranch,   group: 'Line' },
  { type: 'step_line',    label: 'Step Line',           Icon: Filter,      group: 'Line' },
  { type: 'area',         label: 'Area Chart',          Icon: AreaChart,   group: 'Line' },
  { type: 'combo',        label: 'Combo Chart',         Icon: BarChart3,   group: 'Line' },
  { type: 'pie',          label: 'Pie Chart',           Icon: PieChart,    group: 'Circular' },
  { type: 'donut',        label: 'Donut Chart',         Icon: Radio,       group: 'Circular' },
  { type: 'scatter',      label: 'Scatter Plot',        Icon: ScatterChart, group: 'Scatter' },
  { type: 'bubble',       label: 'Bubble Chart',        Icon: Droplets,    group: 'Scatter' },
  { type: 'histogram',    label: 'Histogram',           Icon: BarChart3,   group: 'Scatter' },
  { type: 'box',          label: 'Box Plot',            Icon: BoxSelect,   group: 'Scatter' },
  { type: 'violin',       label: 'Violin Plot',         Icon: Waves,       group: 'Scatter' },
  { type: 'treemap',      label: 'Treemap',             Icon: Grid,        group: 'Hierarchical' },
  { type: 'sunburst',     label: 'Sunburst',            Icon: Sun,         group: 'Hierarchical' },
  { type: 'waterfall',    label: 'Waterfall',           Icon: Triangle,    group: 'Flow' },
  { type: 'funnel',       label: 'Funnel Chart',        Icon: Filter,      group: 'Flow' },
  { type: 'choropleth',   label: 'Choropleth Map',      Icon: Map,         group: 'Geo' },
  { type: 'bubble_map',   label: 'Bubble Map',          Icon: Globe,       group: 'Geo' },
  { type: 'heatmap',      label: 'Heatmap',             Icon: Grid,        group: 'Matrix' },
  { type: 'calendar_heatmap', label: 'Calendar Heatmap', Icon: Grid,       group: 'Matrix' },
  { type: 'radar',        label: 'Radar Chart',         Icon: Radio,       group: 'Polar' },
  { type: 'gauge',        label: 'Gauge Chart',         Icon: Gauge,       group: 'KPI' },
  { type: 'progress',     label: 'Progress Bar',        Icon: Sigma,       group: 'KPI' },
  { type: 'kpi',          label: 'KPI Card',            Icon: TrendingUp,  group: 'KPI' },
  { type: 'parallel',     label: 'Parallel Coords',     Icon: Activity,    group: 'Multi' },
  { type: 'table',        label: 'Data Table',          Icon: Table,       group: 'Table' },
];

interface DashWidget {
  id: string;
  type: string;
  title: string;
  dataset_id: string;
  x_axis: string;
  y_axis: string;
  z_axis: string;
  wPct: number;  // 20–100 % of canvas
  h: number;     // px
}

export function DashboardViewPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [dataCache, setDataCache] = useState<Record<string, Record<string, any>[]>>({});
  const [widgets, setWidgets] = useState<DashWidget[]>([]);
  const [selectedWidget, setSelectedWidget] = useState<string | null>(null);
  const [editMode, setEditMode] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [dragOver, setDragOver] = useState(false);
  const [dragType, setDragType] = useState<string | null>(null);
  const loadedDatasetsRef = useRef<Set<string>>(new Set());
  const canvasRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const load = async () => {
      if (!id) return;
      try {
        const [dash, ds] = await Promise.all([reportsApi.getDashboard(id), datasetsApi.getDatasets()]);
        setDashboard(dash);
        setDatasets(ds);
        const existing: DashWidget[] = (dash.widgets || []).map((w: any) => ({
          id: w.id,
          type: w.type || w.config?.type || 'bar',
          title: w.title || 'Widget',
          dataset_id: w.config?.dataset_id || '',
          x_axis: w.config?.x_axis || '',
          y_axis: w.config?.y_axis || '',
          z_axis: w.config?.z_axis || '',
          wPct: w.config?.wPct || DEF_PCT,
          h: w.config?.h || DEF_H,
        }));
        setWidgets(existing);
        // preload data for existing widgets
        const uniqueDs = [...new Set(existing.map((w: DashWidget) => w.dataset_id).filter(Boolean))];
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

  const getDatasetColumns = (dataset_id: string) => {
    return datasets.find(d => d.id === dataset_id)?.columns || [];
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (!dragType || !editMode) return;
    const newWidget: DashWidget = {
      id: `w_${Date.now()}`,
      type: dragType,
      title: CHART_TYPES.find(c => c.type === dragType)?.label || 'Widget',
      dataset_id: datasets[0]?.id || '',
      x_axis: '', y_axis: '', z_axis: '',
      wPct: DEF_PCT, h: DEF_H,
    };
    setWidgets(prev => [...prev, newWidget]);
    setSelectedWidget(newWidget.id);
    setDragType(null);
    if (newWidget.dataset_id) loadDataForDataset(newWidget.dataset_id);
  };

  const updateWidget = (wid: string, updates: Partial<DashWidget>) => {
    setWidgets(prev => prev.map(w => w.id === wid ? { ...w, ...updates } : w));
    if (updates.dataset_id) loadDataForDataset(updates.dataset_id);
  };

  const deleteWidget = (wid: string) => {
    setWidgets(prev => prev.filter(w => w.id !== wid));
    if (selectedWidget === wid) setSelectedWidget(null);
  };

  const handleSave = async () => {
    if (!dashboard) return;
    setIsSaving(true);
    try {
      const apiWidgets = widgets.map(w => ({
        id: w.id,
        type: w.type,
        title: w.title,
        x: 0, y: 0, width: 4, height: 4,
        config: { dataset_id: w.dataset_id, x_axis: w.x_axis, y_axis: w.y_axis, z_axis: w.z_axis, type: w.type, wPct: w.wPct, h: w.h },
      }));
      await reportsApi.updateDashboard(dashboard.id, { name: dashboard.name, widgets: apiWidgets });
      setEditMode(false);
    } catch (e) { console.error(e); }
    finally { setIsSaving(false); }
  };

  const renderChart = (widget: DashWidget) => {
    const rawData = dataCache[widget.dataset_id] || [];
    const { x_axis, y_axis, z_axis, type } = widget;
    const H = widget.h - 44; // subtract header

    if (widget.dataset_id && !loadedDatasetsRef.current.has(widget.dataset_id)) {
      return <div className="flex items-center justify-center h-full text-xs" style={{ color: 'var(--text-muted)' }}>Loading...</div>;
    }
    if (rawData.length === 0) {
      return <div className="flex items-center justify-center h-full text-xs" style={{ color: 'var(--text-muted)' }}>{widget.dataset_id ? 'No data' : 'Select a dataset'}</div>;
    }

    if (type === 'kpi') {
      const yCol = y_axis || Object.keys(rawData[0])[0];
      return <PlotlyKPI data={rawData} y={yCol} title={widget.title} height={H} />;
    }
    if (type === 'gauge') {
      const yCol = y_axis || Object.keys(rawData[0])[0];
      return <PlotlyGaugeChart data={rawData} y={yCol} title={widget.title} height={H} />;
    }
    if (type === 'progress') {
      const yCol = y_axis || Object.keys(rawData[0])[0];
      return <PlotlyProgressChart data={rawData} y={yCol} title={widget.title} height={H} />;
    }
    if (type === 'table') return <PlotlyDataTable data={rawData.slice(0, 50)} height={H} />;
    if (type === 'parallel') return <PlotlyParallelCoords data={rawData} x={x_axis || ''} y={y_axis || ''} z={z_axis} height={H} />;

    if (!x_axis) {
      return <div className="flex items-center justify-center h-full text-xs" style={{ color: 'var(--text-muted)' }}>{editMode ? 'Configure axes in panel' : 'No axes configured'}</div>;
    }

    const yCol = y_axis || Object.keys(rawData[0]).find(k => k !== x_axis) || x_axis;

    switch (type) {
      case 'bar':             return <PlotlyBarChart          data={rawData} x={x_axis} y={yCol} z={z_axis} height={H} />;
      case 'grouped_bar':     return <PlotlyGroupedBarChart   data={rawData} x={x_axis} y={yCol} z={z_axis} height={H} />;
      case 'stacked_bar':     return <PlotlyStackedBarChart   data={rawData} x={x_axis} y={yCol} z={z_axis} height={H} />;
      case 'stacked_bar_100': return <PlotlyStackedBar100Chart data={rawData} x={x_axis} y={yCol} z={z_axis} height={H} />;
      case 'horizontal_bar':  return <PlotlyHorizontalBarChart data={rawData} x={x_axis} y={yCol} height={H} />;
      case 'line':            return <PlotlyLineChart         data={rawData} x={x_axis} y={yCol} z={z_axis} height={H} />;
      case 'multi_line':      return <PlotlyMultiLineChart    data={rawData} x={x_axis} y={yCol} z={z_axis} height={H} />;
      case 'spline':          return <PlotlySplineChart       data={rawData} x={x_axis} y={yCol} height={H} />;
      case 'step_line':       return <PlotlyStepLineChart     data={rawData} x={x_axis} y={yCol} height={H} />;
      case 'area':            return <PlotlyAreaChart         data={rawData} x={x_axis} y={yCol} z={z_axis} height={H} />;
      case 'combo':           return <PlotlyComboChart        data={rawData} x={x_axis} y={yCol} z={z_axis} height={H} />;
      case 'pie':             return <PlotlyPieChart          data={rawData} x={x_axis} y={yCol} height={H} />;
      case 'donut':           return <PlotlyDonutChart        data={rawData} x={x_axis} y={yCol} height={H} />;
      case 'scatter':         return <PlotlyScatterChart      data={rawData} x={x_axis} y={yCol} z={z_axis} height={H} />;
      case 'bubble':          return <PlotlyBubbleChart       data={rawData} x={x_axis} y={yCol} z={z_axis} height={H} />;
      case 'histogram':       return <PlotlyHistogram         data={rawData} x={x_axis} y={yCol} height={H} />;
      case 'box':             return <PlotlyBoxPlot           data={rawData} x={x_axis} y={yCol} z={z_axis} height={H} />;
      case 'violin':          return <PlotlyViolinPlot        data={rawData} x={x_axis} y={yCol} z={z_axis} height={H} />;
      case 'treemap':         return <PlotlyTreemapChart      data={rawData} x={x_axis} y={yCol} z={z_axis} height={H} />;
      case 'sunburst':        return <PlotlySunburstChart     data={rawData} x={x_axis} y={yCol} z={z_axis} height={H} />;
      case 'waterfall':       return <PlotlyWaterfallChart    data={rawData} x={x_axis} y={yCol} height={H} />;
      case 'funnel':          return <PlotlyFunnelChart       data={rawData} x={x_axis} y={yCol} height={H} />;
      case 'heatmap':         return <PlotlyHeatmap           data={rawData} x={x_axis} y={yCol} z={z_axis} height={H} />;
      case 'calendar_heatmap': return <PlotlyCalendarHeatmap  data={rawData} x={x_axis} y={yCol} height={H} />;
      case 'radar':           return <PlotlyRadarChart        data={rawData} x={x_axis} y={yCol} z={z_axis} height={H} />;
      case 'choropleth':      return <PlotlyChoroplethMap     data={rawData} x={x_axis} y={yCol} height={H} />;
      case 'bubble_map':      return <PlotlyBubbleMap         data={rawData} x={x_axis} y={yCol} height={H} />;
      default: return null;
    }
  };

  const selectedW = widgets.find(w => w.id === selectedWidget);

  const pxToPct = (px: number): number => {
    const cw = canvasRef.current?.clientWidth || 900;
    return Math.max(MIN_PCT, Math.min(MAX_PCT, Math.round((px / cw) * 100)));
  };

  if (isLoading) return (
    <div className="flex items-center justify-center h-64">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2" style={{ borderColor: 'var(--accent)' }} />
    </div>
  );

  if (!dashboard) return (
    <div className="flex flex-col items-center justify-center h-64">
      <p style={{ color: 'var(--text-muted)' }}>Dashboard not found</p>
      <Button onClick={() => navigate('/reports')} className="mt-4">Back</Button>
    </div>
  );

  return (
    <div className="h-[calc(100vh-80px)] flex flex-col gap-3 print:h-auto">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 rounded-lg print:hidden"
        style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="icon" onClick={() => navigate('/reports')}>
            <ArrowLeft size={18} />
          </Button>
          <h1 className="font-semibold" style={{ color: 'var(--text-primary)' }}>{dashboard.name}</h1>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => window.print()}
            style={{ borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}>
            <Download size={15} className="mr-1" /> Export PDF
          </Button>
          {editMode ? (
            <>
              <Button variant="outline" size="sm" onClick={() => setEditMode(false)}
                style={{ borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}>
                <Eye size={15} className="mr-1" /> View
              </Button>
              <Button size="sm" onClick={handleSave} disabled={isSaving} style={{ background: 'var(--accent)' }}>
                {isSaving ? 'Saving...' : 'Save'}
              </Button>
            </>
          ) : (
            <Button size="sm" onClick={() => setEditMode(true)} style={{ background: 'var(--accent)' }}>
              <Edit size={15} className="mr-1" /> Edit
            </Button>
          )}
        </div>
      </div>

      <div className="flex flex-1 gap-3 overflow-hidden print:block">
        {/* Left panel (edit mode only) */}
        {editMode && (
          <div className="w-52 flex-shrink-0 rounded-lg p-3 overflow-auto print:hidden"
            style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
            <p className="text-xs font-semibold uppercase mb-2" style={{ color: 'var(--text-muted)' }}>Chart Types</p>
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

            {selectedW && editMode && (
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
                    <Select value={selectedW.dataset_id} onValueChange={v => updateWidget(selectedW.id, { dataset_id: v, x_axis: '', y_axis: '', z_axis: '' })}>
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
          style={{ background: 'var(--bg-primary)', border: dragOver ? '2px dashed var(--accent)' : '1px solid var(--border-color)' }}
          onDragOver={e => { e.preventDefault(); if (editMode) setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}>
          {widgets.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full print:hidden" style={{ color: 'var(--text-muted)' }}>
              <BarChart3 size={56} className="mb-4 opacity-30" />
              <p className="text-lg mb-1">{editMode ? 'Drag chart types here' : 'No widgets yet'}</p>
              {!editMode && <Button size="sm" onClick={() => setEditMode(true)} className="mt-3" style={{ background: 'var(--accent)' }}>
                <Edit size={14} className="mr-1" /> Edit Dashboard
              </Button>}
            </div>
          ) : (
            <div className="flex flex-wrap gap-4 content-start" style={{ minHeight: '100%' }}>
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
                    resizeHandles={editMode ? ['n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw'] : []}
                    onResizeStop={(_, { size }) => updateWidget(widget.id, { wPct: pxToPct(size.width), h: size.height })}
                    style={{ position: 'relative', flexShrink: 0 }}
                  >
                    <div
                      className="rounded-lg h-full flex flex-col transition-all"
                      style={{
                        background: 'var(--bg-secondary)',
                        border: selectedWidget === widget.id && editMode ? '2px solid var(--accent)' : '1px solid var(--border-color)',
                        cursor: editMode ? 'pointer' : 'default',
                        overflow: 'hidden',
                      }}
                      onClick={() => editMode && setSelectedWidget(widget.id)}
                    >
                      <div className="flex items-center justify-between px-3 py-2 flex-shrink-0"
                        style={{ borderBottom: '1px solid var(--border-color)' }}>
                        <div className="flex items-center gap-2">
                          {editMode && <GripVertical size={13} style={{ color: 'var(--text-muted)' }} />}
                          <span className="text-sm font-medium truncate" style={{ color: 'var(--text-primary)' }}>{widget.title}</span>
                        </div>
                        {editMode && (
                          <button onClick={e => { e.stopPropagation(); deleteWidget(widget.id); }}
                            className="p-1 rounded hover:bg-red-500/10 print:hidden flex-shrink-0">
                            <X size={13} style={{ color: 'var(--danger, #ef4444)' }} />
                          </button>
                        )}
                      </div>
                      <div className="flex-1 min-h-0">
                        {renderChart(widget)}
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
