import apiClient from './client';

export interface DescriptiveAnalytics {
  dataset_id: string;
  total_rows: number;
  total_columns: number;
  numeric_columns: number;
  categorical_columns: number;
  date_columns: number;
  memory_usage_mb: number;
  missing_summary: {
    total_missing: number;
    missing_percentage: number;
    columns_with_missing: number;
  };
  correlation_matrix?: Record<string, Record<string, number>>;
  distributions: Record<string, {
    bins: string[];
    counts: number[];
  }>;
  column_stats: Record<string, any>;
  insights: string[];
  ai_insight?: {
    insight: string;
    confidence: number;
  };
}

export interface DiagnosticAnalytics {
  dataset_id: string;
  target_column?: string;
  feature_importance?: Record<string, number>;
  correlations: Record<string, Array<{ feature: string; correlation: number }>>;
  outliers: {
    total_outliers: number;
    outlier_percentage: number;
  };
  segments: Array<{
    name: string;
    size: number;
    mean_values: Record<string, number>;
  }>;
  root_causes: string[];
  ai_insight?: {
    insight: string;
    confidence: number;
  };
}

export interface AlgorithmResult {
  name: string;
  metrics: Record<string, number>;
  training_time_ms: number;
  rank: number;
}

export interface PredictiveAnalytics {
  dataset_id: string;
  target_column: string;
  task_type: 'classification' | 'regression';
  best_algorithm: string;
  algorithms: AlgorithmResult[];
  feature_importance: Record<string, number>;
  cross_validation_score: number;
  ai_insight?: {
    insight: string;
    confidence: number;
  };
}

export interface Recommendation {
  action: string;
  impact: string;
  confidence: number;
  supporting_data: Record<string, any>;
}

export interface PrescriptiveAnalytics {
  dataset_id: string;
  target_column: string;
  recommendations: Recommendation[];
  scenarios: Array<{
    name: string;
    target_value: number;
    probability: number;
  }>;
  what_if_analysis: {
    variable: string;
    current_value: number;
    impact_per_unit: number;
    suggested_increase: number;
  };
  ai_insight?: {
    insight: string;
    confidence: number;
  };
}

export interface AskAIRequest {
  question: string;
  context?: string;
}

export interface AskAIResponse {
  question: string;
  answer: string;
  confidence: number;
  suggested_followups: string[];
  generated_at: string;
  tokens_used: number;
}

export const analysisApi = {
  descriptive: async (datasetId: string, columns?: string[]): Promise<DescriptiveAnalytics> => {
    const response = await apiClient.post(`/api/analysis/descriptive/${datasetId}`, { columns });
    return response.data;
  },

  diagnostic: async (datasetId: string, targetColumn?: string): Promise<DiagnosticAnalytics> => {
    const response = await apiClient.post(`/api/analysis/diagnostic/${datasetId}`, null, {
      params: { target_column: targetColumn },
    });
    return response.data;
  },

  predictive: async (datasetId: string, targetColumn: string): Promise<PredictiveAnalytics> => {
    const response = await apiClient.post(`/api/analysis/predictive/${datasetId}`, null, {
      params: { target_column: targetColumn },
    });
    return response.data;
  },

  prescriptive: async (datasetId: string, targetColumn: string): Promise<PrescriptiveAnalytics> => {
    const response = await apiClient.post(`/api/analysis/prescriptive/${datasetId}`, null, {
      params: { target_column: targetColumn },
    });
    return response.data;
  },

  askAI: async (datasetId: string, question: string): Promise<AskAIResponse> => {
    const response = await apiClient.post(`/api/analysis/ask-ai/${datasetId}`, { question });
    return response.data;
  },
};
