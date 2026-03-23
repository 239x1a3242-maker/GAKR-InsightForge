import apiClient from './client';

export const mlApi = {
  trainModel: async (data: {
    dataset_id: string;
    target_columns: string[];
    feature_columns?: string[];
    problem_type?: string;
    algorithms?: string[];
    test_size?: number;
    cv_folds?: number;
    hyperparameter_tuning?: boolean;
    auto_feature_engineering?: boolean;
    feature_selection_method?: string;
  }): Promise<any> => {
    const response = await apiClient.post('/api/ml/train', data);
    return response.data;
  },

  predict: async (
    modelIds: string[],
    data: any[],
    returnConfidence?: boolean
  ): Promise<any> => {
    const response = await apiClient.post('/api/ml/predict', {
      model_ids: modelIds,
      data,
      return_confidence: returnConfidence,
    });
    return response.data;
  },

  detectAnomalies: async (data: {
    dataset_id: string;
    columns: string[];
    sensitivity?: string;
    contamination?: number;
    algorithm?: string;
  }): Promise<any> => {
    const response = await apiClient.post('/api/ml/anomaly-detection', data);
    return response.data;
  },

  clusterData: async (data: {
    dataset_id: string;
    columns: string[];
    n_clusters?: number;
    algorithm?: string;
    auto_select_k?: boolean;
  }): Promise<any> => {
    const response = await apiClient.post('/api/ml/clustering', data);
    return response.data;
  },

  analyzeFeatureImportance: async (data: {
    dataset_id: string;
    target_column: string;
    feature_columns: string[];
    method?: string;
  }): Promise<any> => {
    const response = await apiClient.post('/api/ml/feature-importance', data);
    return response.data;
  },

  compareDatasets: async (data: {
    dataset_ids: string[];
    columns?: string[];
    comparison_type?: string;
  }): Promise<any> => {
    const response = await apiClient.post('/api/ml/compare-datasets', data);
    return response.data;
  },

  getModels: async (): Promise<any[]> => {
    try {
      const response = await apiClient.get('/api/ml/models');
      // Handle both array and paginated response formats
      const data = Array.isArray(response.data) ? response.data : (response.data.items || []);
      return data;
    } catch (error) {
      console.error('Failed to fetch models:', error);
      return [];
    }
  },

  getTrainingJob: async (jobId: string): Promise<any> => {
    const response = await apiClient.get(`/api/ml/training-jobs/${jobId}`);
    return response.data;
  },

  deleteModel: async (modelId: string): Promise<void> => {
    await apiClient.delete(`/api/ml/models/${modelId}`);
  },
};
