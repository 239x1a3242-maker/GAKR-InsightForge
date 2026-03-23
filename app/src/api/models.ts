import apiClient from './client';

export interface TrainingConfig {
  max_runtime_secs?: number;
  nfolds?: number;
  max_models?: number;
  enable_stacking?: boolean;
  seed?: number;
}

export interface TrainingRequest {
  dataset_id: string;
  targets: string[];
  config: TrainingConfig;
}

export interface TrainingResponse {
  job_id: string;
  status: string;
  message: string;
  model_ids: string[];
}

export interface TrainingStatus {
  job_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'unknown';
  progress: number;
  current_step: string;
  message?: string;
  model_ids?: string[];
  error?: string;
  created_at: string;
  updated_at?: string;
  completed_at?: string;
}

export interface LeaderboardEntry {
  model_id: string;
  algorithm: string;
  auc?: number;
  logloss?: number;
  mean_per_class_error?: number;
  rmse?: number;
  mse?: number;
  mae?: number;
  rmsle?: number;
  training_time_ms: number;
}

export interface ModelVersion {
  id: string;
  registry_id: string;
  version_number: number;
  status: 'staging' | 'production' | 'archived';
  model_name: string;
  best_algorithm?: string;
  leaderboard: LeaderboardEntry[];
  training_duration_seconds: number;
  training_rows: number;
  feature_columns?: string[]; // added for predictions page
  created_at: string;
  promoted_at?: string;
}

export interface ModelRegistry {
  id: string;
  dataset_id: string;
  target_column: string;
  task_type: 'classification' | 'regression';
  production_version_id?: string;
  production_version?: ModelVersion;
  version_count: number;
  versions?: ModelVersion[]; // may be provided by API
  created_at: string;
  updated_at: string;
}

export interface ClassificationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  auc?: number;
  logloss?: number;
  confusion_matrix: number[][];
  roc_curve?: { fpr: number[]; tpr: number[] };
  pr_curve?: { precision: number[]; recall: number[] };
}

export interface RegressionMetrics {
  r2: number;
  rmse: number;
  mae: number;
  mse: number;
  mape?: number;
  residual_mean: number;
  residual_std: number;
  prediction_vs_actual: Array<{ actual: number; predicted: number }>;
}

export interface ModelMetrics {
  id: string;
  version_id: string;
  task_type: string;
  train_rows: number;
  test_rows: number;
  classification?: ClassificationMetrics;
  regression?: RegressionMetrics;
  feature_importance: Record<string, number>;
}

export interface ShapGlobal {
  feature_names: string[];
  shap_values: number[];
  base_value: number;
  feature_importance: Record<string, number>;
}

export interface ShapLocal {
  prediction: number;
  base_value: number;
  shap_values: Record<string, number>;
  feature_contributions: Array<{
    feature: string;
    value: any;
    contribution: number;
  }>;
}

export interface DriftReport {
  model_id: string;
  psi_score: number;
  drift_detected: boolean;
  threshold: number;
  feature_psi: Record<string, number>;
  recommendation: string;
}

export const modelsApi = {
  train: async (request: TrainingRequest): Promise<TrainingResponse> => {
    const response = await apiClient.post('/api/ml/train', request);
    return response.data;
  },

  getTrainingStatus: async (jobId: string): Promise<TrainingStatus> => {
    const response = await apiClient.get(`/api/ml/training-jobs/${jobId}`);
    return response.data;
  },

  list: async (datasetId?: string): Promise<ModelRegistry[]> => {
    const response = await apiClient.get('/api/ml/models', { params: { dataset_id: datasetId } });
    return response.data;
  },

  getVersions: async (registryId: string): Promise<ModelVersion[]> => {
    const response = await apiClient.get(`/api/ml/models/${registryId}/versions`);
    return response.data;
  },

  getMetrics: async (versionId: string): Promise<ModelMetrics> => {
    const response = await apiClient.get(`/api/ml/versions/${versionId}/metrics`);
    return response.data;
  },

  getLeaderboard: async (versionId: string): Promise<LeaderboardEntry[]> => {
    const response = await apiClient.get(`/api/ml/versions/${versionId}/leaderboard`);
    return response.data;
  },

  promote: async (versionId: string, comment?: string): Promise<{ success: boolean; message: string; new_version: number }> => {
    const response = await apiClient.post(`/api/ml/versions/${versionId}/promote`, { comment });
    return response.data;
  },

  archive: async (versionId: string): Promise<{ message: string }> => {
    const response = await apiClient.post(`/api/ml/versions/${versionId}/archive`);
    return response.data;
  },

  getShapGlobal: async (versionId: string): Promise<ShapGlobal> => {
    const response = await apiClient.get(`/api/ml/versions/${versionId}/shap/global`);
    return response.data;
  },

  getShapLocal: async (versionId: string, features: Record<string, any>): Promise<ShapLocal> => {
    const response = await apiClient.post(`/api/ml/versions/${versionId}/shap/local`, { features });
    return response.data;
  },

  checkDrift: async (versionId: string, batchData: Record<string, any>[]): Promise<DriftReport> => {
    const response = await apiClient.post(`/api/ml/versions/${versionId}/drift-check`, batchData);
    return response.data;
  },
};
