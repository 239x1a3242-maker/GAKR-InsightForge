import { useMemo } from 'react';
import {
  BarChart, Bar, LineChart, Line, AreaChart, Area, PieChart, Pie, Cell,
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ComposedChart, RadarChart, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Radar, FunnelChart, Funnel, LabelList
} from 'recharts';
import { formatNumber, formatCurrency, formatPercent } from '@/utils/format';

const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899', '#14B8A6', '#6366F1', '#84CC16', '#06B6D4'];

interface ChartRendererProps {
  type: string;
  data: any[];
  config: {
    x_field?: string;
    y_field?: string;
    y_fields?: string[];
    title?: string;
    stacked?: boolean;
    show_legend?: boolean;
    show_grid?: boolean;
    format?: 'number' | 'currency' | 'percent';
    currency?: string;
  };
  height?: number;
}

export function ChartRenderer({ type, data, config, height = 300 }: ChartRendererProps) {
  const { x_field, y_field, y_fields, title, stacked, show_legend = true, show_grid = true, format } = config;

  const formatValue = (value: number) => {
    if (format === 'currency') return formatCurrency(value, config.currency);
    if (format === 'percent') return formatPercent(value);
    return formatNumber(value);
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload) return null;
    return (
      <div 
        className="p-3 rounded-lg shadow-lg"
        style={{ 
          background: 'var(--bg-secondary)', 
          border: '1px solid var(--border-color)' 
        }}
      >
        <p className="font-medium mb-2" style={{ color: 'var(--text-primary)' }}>{label}</p>
        {payload.map((entry: any, idx: number) => (
          <p key={idx} className="text-sm" style={{ color: entry.color }}>
            {entry.name}: {formatValue(entry.value)}
          </p>
        ))}
      </div>
    );
  };

  const chartData = useMemo(() => {
    if (!data || data.length === 0) return [];
    return data;
  }, [data]);

  if (!chartData.length) {
    return (
      <div className="flex items-center justify-center h-full" style={{ color: 'var(--text-muted)' }}>
        No data available
      </div>
    );
  }

  const safeYField = y_field || chartData[0] ? Object.keys(chartData[0])[1] || 'value' : 'value';
  const safeXField = x_field || chartData[0] ? Object.keys(chartData[0])[0] || 'name' : 'name';

  switch (type) {
    case 'bar':
      return (
        <ResponsiveContainer width="100%" height={height}>
          <BarChart data={chartData}>
            {show_grid && <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />}
            <XAxis dataKey={safeXField} stroke="var(--text-secondary)" fontSize={11} tickLine={false} />
            <YAxis stroke="var(--text-secondary)" fontSize={11} tickLine={false} tickFormatter={formatValue} />
            <Tooltip content={<CustomTooltip />} />
            {show_legend && <Tooltip />}
            {y_fields ? (
              y_fields.map((field, idx) => (
                <Bar 
                  key={field} 
                  dataKey={field} 
                  fill={COLORS[idx % COLORS.length]} 
                  radius={[4, 4, 0, 0]}
                  stackId={stacked ? 'stack' : undefined}
                />
              ))
            ) : (
              <Bar dataKey={safeYField} fill={COLORS[0]} radius={[4, 4, 0, 0]} />
            )}
          </BarChart>
        </ResponsiveContainer>
      );

    case 'line':
      return (
        <ResponsiveContainer width="100%" height={height}>
          <LineChart data={chartData}>
            {show_grid && <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />}
            <XAxis dataKey={safeXField} stroke="var(--text-secondary)" fontSize={11} tickLine={false} />
            <YAxis stroke="var(--text-secondary)" fontSize={11} tickLine={false} tickFormatter={formatValue} />
            <Tooltip content={<CustomTooltip />} />
            {y_fields ? (
              y_fields.map((field, idx) => (
                <Line 
                  key={field}
                  type="monotone" 
                  dataKey={field} 
                  stroke={COLORS[idx % COLORS.length]}
                  strokeWidth={2}
                  dot={false}
                />
              ))
            ) : (
              <Line 
                type="monotone" 
                dataKey={safeYField} 
                stroke={COLORS[0]}
                strokeWidth={2}
                dot={false}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      );

    case 'area':
      return (
        <ResponsiveContainer width="100%" height={height}>
          <AreaChart data={chartData}>
            {show_grid && <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />}
            <XAxis dataKey={safeXField} stroke="var(--text-secondary)" fontSize={11} tickLine={false} />
            <YAxis stroke="var(--text-secondary)" fontSize={11} tickLine={false} tickFormatter={formatValue} />
            <Tooltip content={<CustomTooltip />} />
            {y_fields ? (
              y_fields.map((field, idx) => (
                <Area 
                  key={field}
                  type="monotone" 
                  dataKey={field} 
                  stroke={COLORS[idx % COLORS.length]}
                  fill={COLORS[idx % COLORS.length]}
                  fillOpacity={0.3}
                  stackId={stacked ? 'stack' : undefined}
                />
              ))
            ) : (
              <Area 
                type="monotone" 
                dataKey={safeYField} 
                stroke={COLORS[0]}
                fill={COLORS[0]}
                fillOpacity={0.3}
              />
            )}
          </AreaChart>
        </ResponsiveContainer>
      );

    case 'pie':
    case 'donut':
      return (
        <ResponsiveContainer width="100%" height={height}>
          <PieChart>
            <Pie
              data={chartData}
              dataKey={safeYField}
              nameKey={safeXField}
              cx="50%"
              cy="50%"
              outerRadius={type === 'donut' ? '70%' : '80%'}
              innerRadius={type === 'donut' ? '50%' : '0%'}
              paddingAngle={2}
            >
              {chartData.map((_, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
          </PieChart>
        </ResponsiveContainer>
      );

    case 'scatter':
      return (
        <ResponsiveContainer width="100%" height={height}>
          <ScatterChart>
            {show_grid && <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />}
            <XAxis dataKey={safeXField} stroke="var(--text-secondary)" fontSize={11} tickLine={false} />
            <YAxis dataKey={safeYField} stroke="var(--text-secondary)" fontSize={11} tickLine={false} tickFormatter={formatValue} />
            <Tooltip content={<CustomTooltip />} />
            <Scatter data={chartData} fill={COLORS[0]} />
          </ScatterChart>
        </ResponsiveContainer>
      );

    case 'radar':
      return (
        <ResponsiveContainer width="100%" height={height}>
          <RadarChart data={chartData}>
            <PolarGrid stroke="var(--border-color)" />
            <PolarAngleAxis dataKey={safeXField} tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} />
            <PolarRadiusAxis tick={{ fill: 'var(--text-muted)', fontSize: 10 }} />
            <Radar 
              name={safeYField} 
              dataKey={safeYField} 
              stroke={COLORS[0]} 
              fill={COLORS[0]} 
              fillOpacity={0.3} 
            />
            <Tooltip content={<CustomTooltip />} />
          </RadarChart>
        </ResponsiveContainer>
      );

    case 'composed':
      return (
        <ResponsiveContainer width="100%" height={height}>
          <ComposedChart data={chartData}>
            {show_grid && <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />}
            <XAxis dataKey={safeXField} stroke="var(--text-secondary)" fontSize={11} tickLine={false} />
            <YAxis stroke="var(--text-secondary)" fontSize={11} tickLine={false} tickFormatter={formatValue} />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey={y_fields?.[0] || safeYField} fill={COLORS[0]} radius={[4, 4, 0, 0]} />
            <Line type="monotone" dataKey={y_fields?.[1] || safeYField} stroke={COLORS[1]} strokeWidth={2} />
          </ComposedChart>
        </ResponsiveContainer>
      );

    case 'funnel':
      return (
        <ResponsiveContainer width="100%" height={height}>
          <FunnelChart>
            <Tooltip content={<CustomTooltip />} />
            <Funnel
              dataKey={safeYField}
              data={chartData}
              isAnimationActive
              nameKey={safeXField}
            >
              {chartData.map((_, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
              <LabelList position="inside" fill="#fff" stroke="none" dataKey={safeXField} />
            </Funnel>
          </FunnelChart>
        </ResponsiveContainer>
      );

    case 'table':
      const columns = chartData.length > 0 ? Object.keys(chartData[0]) : [];
      return (
        <div className="overflow-auto h-full">
          <table className="w-full text-sm">
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                {columns.map((col) => (
                  <th key={col} className="text-left p-2 font-medium" style={{ color: 'var(--text-secondary)' }}>
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {chartData.slice(0, 100).map((row, i) => (
                <tr key={i} style={{ borderBottom: '1px solid var(--border-color)' }}>
                  {columns.map((col) => (
                    <td key={col} className="p-2" style={{ color: 'var(--text-primary)' }}>
                      {typeof row[col] === 'number' ? formatValue(row[col]) : row[col]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );

    case 'kpi':
      const kpiValue = chartData[0]?.[safeYField] || 0;
      const previousValue = chartData[0]?.previous || 0;
      const change = previousValue ? ((kpiValue - previousValue) / previousValue) * 100 : 0;
      
      return (
        <div className="flex flex-col items-center justify-center h-full">
          <span className="text-4xl font-bold" style={{ color: 'var(--text-primary)' }}>
            {formatValue(kpiValue)}
          </span>
          {change !== 0 && (
            <span 
              className="text-sm mt-2 flex items-center"
              style={{ color: change > 0 ? 'var(--success)' : 'var(--danger)' }}
            >
              {change > 0 ? '↑' : '↓'} {Math.abs(change).toFixed(1)}%
            </span>
          )}
          {title && (
            <span className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>
              {title}
            </span>
          )}
        </div>
      );

    default:
      return (
        <div className="flex items-center justify-center h-full" style={{ color: 'var(--text-muted)' }}>
          Chart type &quot;{type}&quot; not supported
        </div>
      );
  }
}
