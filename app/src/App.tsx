import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from '@/contexts/AuthContext';
import { ThemeProvider } from '@/contexts/ThemeContext';
import { PrivateRoute } from '@/components/layout/PrivateRoute';
import { AuthPage } from '@/pages/AuthPage';
import { HomePage } from '@/pages/HomePage';
import { DatasetsPage } from '@/pages/DatasetsPage';
import { DatasetDetailPage } from '@/pages/DatasetDetailPage';
import { AnalysisPage } from '@/pages/analysis/AnalysisPage';
import { ReportsPage } from '@/pages/ReportsPage';
import { ReportBuilderPage } from '@/pages/ReportBuilderPage';
import { DashboardViewPage } from '@/pages/DashboardViewPage';
import { DashboardPage } from '@/pages/DashboardPage';
import { MLPage } from '@/pages/MLPage';
import { TrainingPage } from '@/pages/ml/TrainingPage';
import { PredictionsPage } from '@/pages/ml/PredictionsPage';
import { ModelsPage } from '@/pages/ml/ModelsPage';
import { ExplainabilityPage } from '@/pages/ml/ExplainabilityPage';
import { DriftPage } from '@/pages/ml/DriftPage';

function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<AuthPage />} />
            <Route path="/" element={<PrivateRoute><HomePage /></PrivateRoute>} />
            <Route path="/data" element={<PrivateRoute><DatasetsPage /></PrivateRoute>} />
            <Route path="/datasets" element={<PrivateRoute><DatasetsPage /></PrivateRoute>} />
            <Route path="/datasets/:id" element={<PrivateRoute><DatasetDetailPage /></PrivateRoute>} />
            <Route path="/analysis" element={<PrivateRoute><AnalysisPage /></PrivateRoute>} />
            <Route path="/analysis/:datasetId" element={<PrivateRoute><AnalysisPage /></PrivateRoute>} />
            <Route path="/reports" element={<PrivateRoute><ReportsPage /></PrivateRoute>} />
            <Route path="/reports/:id" element={<PrivateRoute><ReportBuilderPage /></PrivateRoute>} />
            <Route path="/dashboard" element={<PrivateRoute><DashboardPage /></PrivateRoute>} />
            <Route path="/dashboards" element={<Navigate to="/reports?tab=dashboards" replace />} />
            <Route path="/dashboards/:id" element={<PrivateRoute><DashboardViewPage /></PrivateRoute>} />
            <Route path="/ml" element={<PrivateRoute><MLPage /></PrivateRoute>} />
            <Route path="/ml/training" element={<PrivateRoute><TrainingPage /></PrivateRoute>} />
            <Route path="/ml/predictions" element={<PrivateRoute><PredictionsPage /></PrivateRoute>} />
            <Route path="/ml/models" element={<PrivateRoute><ModelsPage /></PrivateRoute>} />
            <Route path="/ml/explainability" element={<PrivateRoute><ExplainabilityPage /></PrivateRoute>} />
            <Route path="/ml/drift" element={<PrivateRoute><DriftPage /></PrivateRoute>} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </ThemeProvider>
  );
}

export default App;
