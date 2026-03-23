import { useState } from 'react';
import { MLTrainingPanel } from '@/components/ml/MLTrainingPanel';
import { MLPredictionPanel } from '@/components/ml/MLPredictionPanel';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { 
  Brain, Sparkles, TrendingUp,
  AlertTriangle, CheckCircle, Database
} from 'lucide-react';

export function MLPage() {
  const [activeTab, setActiveTab] = useState('training');
  const [trainedModels, setTrainedModels] = useState<any[]>([]);

  const handleTrainingComplete = (result: any) => {
    // Extract model IDs from training result
    const models = Object.entries(result.target_results || {}).map(([target, data]: [string, any]) => ({
      id: data.best_model_id,
      name: `Model for ${target}`,
      target_column: target,
      algorithm: data.best_algorithm,
      metrics: data.metrics
    }));
    setTrainedModels(prev => [...prev, ...models]);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-heading text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>
            Machine Learning
          </h1>
          <p style={{ color: 'var(--text-secondary)' }}>
            Train models, make predictions, and analyze your data
          </p>
        </div>
        <div className="flex gap-2">
          <Badge variant="outline" style={{ borderColor: 'var(--border-color)' }}>
            <CheckCircle size={14} className="mr-1" style={{ color: 'var(--success)' }} />
            {trainedModels.length} Models
          </Badge>
        </div>
      </div>

      {/* Info Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
          <CardContent className="p-4 flex items-center gap-3">
            <div 
              className="p-3 rounded-lg"
              style={{ background: 'var(--accent)20' }}
            >
              <Brain size={24} style={{ color: 'var(--accent)' }} />
            </div>
            <div>
              <p className="font-medium" style={{ color: 'var(--text-primary)' }}>AutoML</p>
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                Automatic algorithm selection
              </p>
            </div>
          </CardContent>
        </Card>

        <Card style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
          <CardContent className="p-4 flex items-center gap-3">
            <div 
              className="p-3 rounded-lg"
              style={{ background: 'var(--success)20' }}
            >
              <TrendingUp size={24} style={{ color: 'var(--success)' }} />
            </div>
            <div>
              <p className="font-medium" style={{ color: 'var(--text-primary)' }}>Predictions</p>
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                Make predictions on new data
              </p>
            </div>
          </CardContent>
        </Card>

        <Card style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
          <CardContent className="p-4 flex items-center gap-3">
            <div 
              className="p-3 rounded-lg"
              style={{ background: 'var(--warning)20' }}
            >
              <AlertTriangle size={24} style={{ color: 'var(--warning)' }} />
            </div>
            <div>
              <p className="font-medium" style={{ color: 'var(--text-primary)' }}>Anomaly Detection</p>
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                Find outliers in your data
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList style={{ background: 'var(--bg-secondary)' }}>
          <TabsTrigger value="training" className="flex items-center gap-2">
            <Brain size={16} />
            Model Training
          </TabsTrigger>
          <TabsTrigger value="prediction" className="flex items-center gap-2">
            <Sparkles size={16} />
            Predictions
          </TabsTrigger>
          <TabsTrigger value="models" className="flex items-center gap-2">
            <Database size={16} />
            My Models
          </TabsTrigger>
        </TabsList>

        <TabsContent value="training" className="mt-6">
          <MLTrainingPanel onTrainingComplete={handleTrainingComplete} />
        </TabsContent>

        <TabsContent value="prediction" className="mt-6">
          <MLPredictionPanel 
            modelIds={trainedModels.map(m => m.id)}
          />
        </TabsContent>

        <TabsContent value="models" className="mt-6">
          {trainedModels.length === 0 ? (
            <Card style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
              <CardContent className="flex flex-col items-center justify-center py-16">
                <Brain size={64} style={{ color: 'var(--text-muted)' }} className="mb-4" />
                <h3 className="text-lg font-medium mb-2" style={{ color: 'var(--text-primary)' }}>
                  No models yet
                </h3>
                <p className="text-center mb-6 max-w-md" style={{ color: 'var(--text-muted)' }}>
                  Train your first machine learning model to start making predictions
                </p>
                <Button 
                  onClick={() => setActiveTab('training')}
                  style={{ background: 'var(--accent)' }}
                >
                  <Brain size={16} className="mr-2" />
                  Train Model
                </Button>
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {trainedModels.map((model) => (
                <Card 
                  key={model.id}
                  style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}
                >
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg" style={{ color: 'var(--text-primary)' }}>
                        {model.name}
                      </CardTitle>
                      <Badge style={{ background: 'var(--accent)' }}>
                        {model.algorithm}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span style={{ color: 'var(--text-muted)' }}>Target</span>
                        <span style={{ color: 'var(--text-primary)' }}>{model.target_column}</span>
                      </div>
                      {model.metrics && Object.entries(model.metrics).slice(0, 3).map(([key, value]: [string, any]) => (
                        <div key={key} className="flex justify-between text-sm">
                          <span style={{ color: 'var(--text-muted)' }}>{key.toUpperCase()}</span>
                          <span style={{ color: 'var(--text-primary)' }}>
                            {typeof value === 'number' ? value.toFixed(4) : value}
                          </span>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
