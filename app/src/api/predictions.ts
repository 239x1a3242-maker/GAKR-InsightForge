import apiClient from './client';

export interface PredictionRequest {
  features: Record<string, any>;
  return_shap?: boolean;
}

export interface PredictionResult {
  prediction: any;
  confidence?: number;
  probabilities?: Record<string, number>;
  shap_values?: Record<string, number>;
}

export interface PredictionResponse {
  model_id: string;
  model_version: number;
  predictions: PredictionResult[];
  prediction_time_ms: number;
  drift_detected?: boolean;
  drift_score?: number;
}

export interface BatchPredictionRequest {
  records: Record<string, any>[];
  return_confidence?: boolean;
}

export interface BatchPredictionResponse {
  model_id: string;
  model_version: number;
  predictions: PredictionResult[];
  total_records: number;
  prediction_time_ms: number;
  drift_summary?: {
    psi_score: number;
    drift_detected: boolean;
  };
}

export const predictionsApi = {
  predict: async (modelId: string, request: PredictionRequest): Promise<PredictionResponse> => {
    const response = await apiClient.post(`/api/ml/predict`, {
      model_ids: [modelId],
      data: [request.features],
      return_confidence: request.return_shap,
    });
    return response.data;
  },

  predictBatch: async (modelId: string, request: BatchPredictionRequest): Promise<BatchPredictionResponse> => {
    const response = await apiClient.post(`/api/ml/predict`, {
      model_ids: [modelId],
      data: request.records,
      return_confidence: request.return_confidence,
    });
    return response.data;
  },
};
