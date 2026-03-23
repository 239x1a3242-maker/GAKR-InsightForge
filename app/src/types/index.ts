// User types
export interface User {
  id: string;
  email: string;
  name: string;
  is_active: boolean;
  email_verified: boolean;
  timezone: string;
  language: string;
  avatar_url?: string;
  last_login?: string;
  created_at: string;
}

export interface LoginCredentials {
  email: string;
  password: string;
}

export interface RegisterData {
  email: string;
  password: string;
  name: string;
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  user: User;
}

// Workspace types
export interface Workspace {
  id: string;
  name: string;
  slug: string;
  description?: string;
  is_active: boolean;
  settings: Record<string, any>;
  created_at: string;
  updated_at: string;
  member_count: number;
}

// Dataset types
export interface DatasetColumn {
  name: string;
  type: string;
  dtype?: string; // added in reference
  nullable: boolean;
  description?: string;
  sample_values?: any[];
  unique_count?: number; // added for dataset detail page
}

export interface Dataset {
  id: string;
  name: string;
  description?: string;
  owner_id: string;
  workspace_id?: string;
  source_type: string;
  columns?: DatasetColumn[];
  row_count: number;
  column_count?: number; // used by dashboard
  size_bytes: number;
  refresh_type: string;
  refresh_schedule?: string;
  last_refresh?: string;
  refresh_status: string;
  is_active: boolean;
  is_shared: boolean;
  created_at: string;
  updated_at: string;
}

export interface DatasetCreate {
  name: string;
  description?: string;
  source_type: string;
  source_config: Record<string, any>;
  refresh_type?: string;
  refresh_schedule?: string;
}

export interface DataPreview {
  columns: DatasetColumn[];
  data: Record<string, any>[];
  total_rows: number;
  page: number;
  page_size: number;
}

// Dataset profiling and quality
export interface DatasetProfile {
  total_rows: number;
  total_columns: number;
  numeric_columns: number;
  categorical_columns: number;
  memory_usage_mb: number;
  missing_percentage: number;
  duplicate_rows: number;
  insights: any[];
  column_stats: Record<string, any>;
}

export interface DataQualityIssue {
  type: string;
  column?: string;
  severity: 'low' | 'medium' | 'high';
  description: string;
  suggestion?: string;
}

export interface DataQualityResponse {
  issues: DataQualityIssue[];
}

// Query types
export interface Filter {
  field: string;
  operator: 'eq' | 'neq' | 'gt' | 'gte' | 'lt' | 'lte' | 'in' | 'not_in' | 'contains' | 'starts_with' | 'ends_with' | 'is_null' | 'is_not_null' | 'between';
  value?: any;
}

export interface Aggregation {
  field: string;
  function: 'sum' | 'avg' | 'min' | 'max' | 'count' | 'count_distinct' | 'stdev' | 'var' | 'median';
  alias?: string;
}

export interface Sort {
  field: string;
  direction: 'asc' | 'desc';
}

export interface QueryRequest {
  dataset_id: string;
  table_name?: string;
  fields?: string[];
  group_by?: string[];
  aggregations?: Aggregation[];
  filters?: Filter[];
  sort?: Sort[];
  page?: number;
  page_size?: number;
  limit?: number;
  offset?: number;
}

export interface QueryResponse {
  columns: { name: string; type: string }[] | string[];
  data: Record<string, any>[];
  rows?: Record<string, any>[];
  total_rows: number;
  page: number;
  page_size: number;
  execution_time_ms: number;
}

// Report types
export interface VisualConfig {
  id?: string;
  visual_type: string;
  title?: string;
  x: number;
  y: number;
  width: number;
  height: number;
  chart_config: Record<string, any>;
  data_fields: Record<string, any>;
  formatting?: Record<string, any>;
  conditional_formatting?: any[];
}

export interface PageConfig {
  id: string;
  name: string;
  visuals: VisualConfig[];
  filters: any[];
  background?: string;
}

export interface Report {
  id: string;
  name: string;
  description?: string;
  owner_id: string;
  workspace_id?: string;
  dataset_id: string;
  pages: PageConfig[];
  layout_config: Record<string, any>;
  theme: string;
  is_active: boolean;
  is_shared: boolean;
  page_count?: number;
  created_at: string;
  updated_at: string;
}

export interface ReportCreate {
  name: string;
  description?: string;
  dataset_id: string;
  pages?: PageConfig[];
  layout_config?: Record<string, any>;
  theme?: string;
}

// Dashboard types
export interface DashboardWidget {
  id: string;
  type: string;
  report_id?: string;
  visual_id?: string;
  x: number;
  y: number;
  width: number;
  height: number;
  title?: string;
  config: Record<string, any>;
}

export interface Dashboard {
  id: string;
  name: string;
  description?: string;
  owner_id: string;
  workspace_id?: string;
  widgets: DashboardWidget[];
  filters: any[];
  layout_config: Record<string, any>;
  theme: string;
  is_active: boolean;
  is_shared: boolean;
  is_favorite: boolean;
  created_at: string;
  updated_at: string;
}

export interface DashboardCreate {
  name: string;
  description?: string;
  widgets?: DashboardWidget[];
  filters?: any[];
  layout_config?: Record<string, any>;
  theme?: string;
  is_favorite?: boolean;
}

// Chart types
export type ChartType = 
  | 'bar' | 'line' | 'area' | 'pie' | 'donut' 
  | 'scatter' | 'bubble' | 'heatmap' | 'table'
  | 'kpi' | 'gauge' | 'funnel' | 'treemap' | 'waterfall';

export interface ChartConfig {
  type: ChartType;
  title?: string;
  xAxis?: { field: string; title?: string };
  yAxis?: { field: string; title?: string; aggregation?: string };
  color?: { field: string };
  size?: { field: string };
  legend?: { show: boolean; position?: string };
  tooltip?: { show: boolean };
}

// ML types
export interface MLModel {
  id: string;
  name: string;
  description?: string;
  dataset_id: string;
  target_column: string;
  problem_type: string;
  algorithm: string;
  metrics: Record<string, number>;
  cv_scores: number[];
  feature_columns: string[];
  feature_importance: Record<string, number>;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface TrainingJob {
  job_id: string;
  dataset_id: string;
  target_column: string;
  problem_type: string;
  status: string;
  best_algorithm?: string;
  best_model_id?: string;
  all_results: any[];
  feature_importance?: Record<string, number>;
}
