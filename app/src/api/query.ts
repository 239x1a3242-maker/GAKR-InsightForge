import apiClient from './client';
import type { QueryRequest, QueryResponse } from '@/types';

export const queryApi = {
  executeQuery: async (request: QueryRequest): Promise<QueryResponse> => {
    const response = await apiClient.post('/api/query/execute', request);
    return response.data;
  },

  getDistinctValues: async (
    datasetId: string,
    column: string,
    limit: number = 1000,
    search?: string
  ): Promise<{ column: string; values: string[]; count: number }> => {
    const response = await apiClient.get(`/api/query/${datasetId}/distinct/${column}`, {
      params: { limit, search },
    });
    return response.data;
  },

  getTimeIntelligence: async (data: {
    dataset_id: string;
    date_column: string;
    measure: string;
    comparison_type: string;
    current_period?: { start_date: string; end_date: string };
  }): Promise<any> => {
    const response = await apiClient.post('/api/query/time-intelligence', data);
    return response.data;
  },
};
