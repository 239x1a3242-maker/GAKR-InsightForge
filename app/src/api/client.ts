import axios, { type AxiosInstance, type AxiosError, type InternalAxiosRequestConfig } from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config: InternalAxiosRequestConfig) => {
        const token = localStorage.getItem('bi_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error: AxiosError) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response: any) => response,
      (error: AxiosError) => {
        if (error.response?.status === 401) {
          localStorage.removeItem('bi_token');
          localStorage.removeItem('bi_user');
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  get instance(): AxiosInstance {
    return this.client;
  }
}

export const apiClient = new ApiClient().instance;
export default apiClient;
