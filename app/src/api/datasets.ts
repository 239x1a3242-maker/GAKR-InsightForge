import apiClient from './client';
import type { Dataset, DatasetCreate, DataPreview } from '@/types';

export const datasetsApi = {
  list: async (workspaceId?: string): Promise<{ items: Dataset[]; total: number }> => {
    const params = workspaceId ? { workspace_id: workspaceId } : {};
    const response = await apiClient.get('/api/datasets', { params });
    return response.data;
  },

  getDatasets: async (workspaceId?: string): Promise<Dataset[]> => {
    const result = await datasetsApi.list(workspaceId);
    return result.items;
  },

  getDataset: async (id: string): Promise<Dataset> => {
    const response = await apiClient.get(`/api/datasets/${id}`);
    return response.data;
  },

  getProfile: async (id: string): Promise<any> => {
    const response = await apiClient.get(`/api/datasets/${id}/profile`);
    return response.data;
  },

  getQuality: async (id: string): Promise<any> => {
    const response = await apiClient.get(`/api/datasets/${id}/quality`);
    return response.data;
  },

  createDataset: async (data: DatasetCreate): Promise<Dataset> => {
    const response = await apiClient.post('/api/datasets', data);
    return response.data;
  },

  updateDataset: async (id: string, data: Partial<DatasetCreate>): Promise<Dataset> => {
    const response = await apiClient.put(`/api/datasets/${id}`, data);
    return response.data;
  },

  deleteDataset: async (id: string): Promise<void> => {
    await apiClient.delete(`/api/datasets/${id}`);
  },

  uploadFile: async (file: File, name?: string, workspaceId?: string): Promise<any> => {
    const formData = new FormData();
    formData.append('file', file);
    if (name) formData.append('name', name);
    if (workspaceId) formData.append('workspace_id', workspaceId);

    const response = await apiClient.post('/api/datasets/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  getPreview: async (id: string, page: number = 1, pageSize: number = 100): Promise<DataPreview> => {
    const response = await apiClient.get(`/api/datasets/${id}/preview`, {
      params: { page, page_size: pageSize },
    });
    return response.data;
  },

  getSchema: async (id: string): Promise<any> => {
    const response = await apiClient.get(`/api/datasets/${id}/schema`);
    return response.data;
  },

  refreshDataset: async (id: string): Promise<any> => {
    const response = await apiClient.post(`/api/datasets/${id}/refresh`);
    return response.data;
  },

  // Relationships
  createRelationship: async (datasetId: string, data: any): Promise<any> => {
    const response = await apiClient.post(`/api/datasets/${datasetId}/relationships`, data);
    return response.data;
  },

  // Calculated Fields
  createCalculatedField: async (datasetId: string, data: any): Promise<any> => {
    const response = await apiClient.post(`/api/datasets/${datasetId}/calculated-fields`, data);
    return response.data;
  },

  // Measures
  createMeasure: async (datasetId: string, data: any): Promise<any> => {
    const response = await apiClient.post(`/api/datasets/${datasetId}/measures`, data);
    return response.data;
  },
};
