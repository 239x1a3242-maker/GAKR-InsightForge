import apiClient from './client';
import type { Report, ReportCreate, Dashboard, DashboardCreate } from '@/types';

export const reportsApi = {
  // Reports
  getReports: async (workspaceId?: string, datasetId?: string): Promise<Report[]> => {
    const params: any = {};
    if (workspaceId) params.workspace_id = workspaceId;
    if (datasetId) params.dataset_id = datasetId;
    const response = await apiClient.get('/api/reports', { params });
    // backend returns { items: Report[], total, ... }
    return response.data.items || [];
  },

  getReport: async (id: string): Promise<Report> => {
    const response = await apiClient.get(`/api/reports/${id}`);
    return response.data;
  },

  createReport: async (data: ReportCreate): Promise<Report> => {
    const response = await apiClient.post('/api/reports', data);
    return response.data;
  },

  updateReport: async (id: string, data: Partial<ReportCreate>): Promise<Report> => {
    const response = await apiClient.put(`/api/reports/${id}`, data);
    return response.data;
  },

  deleteReport: async (id: string): Promise<void> => {
    await apiClient.delete(`/api/reports/${id}`);
  },

  addPage: async (reportId: string, pageName: string): Promise<any> => {
    const response = await apiClient.post(`/api/reports/${reportId}/pages`, null, {
      params: { page_name: pageName },
    });
    return response.data;
  },

  deletePage: async (reportId: string, pageId: string): Promise<void> => {
    await apiClient.delete(`/api/reports/${reportId}/pages/${pageId}`);
  },

  // Dashboards
  getDashboards: async (workspaceId?: string): Promise<Dashboard[]> => {
    const params = workspaceId ? { workspace_id: workspaceId } : {};
    const response = await apiClient.get('/api/dashboards', { params });
    return response.data.items || [];
  },

  getDashboard: async (id: string): Promise<Dashboard> => {
    const response = await apiClient.get(`/api/dashboards/${id}`);
    return response.data;
  },

  createDashboard: async (data: DashboardCreate): Promise<Dashboard> => {
    const response = await apiClient.post('/api/dashboards', data);
    return response.data;
  },

  updateDashboard: async (id: string, data: Partial<DashboardCreate>): Promise<Dashboard> => {
    const response = await apiClient.put(`/api/dashboards/${id}`, data);
    return response.data;
  },

  deleteDashboard: async (id: string): Promise<void> => {
    await apiClient.delete(`/api/dashboards/${id}`);
  },

  shareDashboard: async (id: string, data: { password?: string; expires_in_days?: number }): Promise<any> => {
    const response = await apiClient.post(`/api/dashboards/${id}/share`, data);
    return response.data;
  },

  // Export
  export: async (data: { format: string; report_id?: string; dashboard_id?: string }): Promise<any> => {
    const response = await apiClient.post('/api/reports/export', data);
    return response.data;
  },
};
