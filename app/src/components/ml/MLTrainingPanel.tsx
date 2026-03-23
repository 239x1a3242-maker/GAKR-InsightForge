import { useState, useEffect } from 'react';
import { mlApi, datasetsApi } from '@/api';
import type { Dataset } from '@/types';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  Brain, Play, Loader2, CheckCircle, 
  Settings, ChevronDown, ChevronUp 
} from 'lucide-react';

interface MLTrainingPanelProps {
  datasetId?: string;
  onTrainingComplete?: (result: any) => void;
}

export function MLTrainingPanel({ datasetId: initialDatasetId, onTrainingComplete }: MLTrainingPanelProps) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>(initialDatasetId || '');
  const [columns, setColumns] = useState<string[]>([]);
  const [targetColumns, setTargetColumns] = useState<string[]>([]);
  const [featureColumns, setFeatureColumns] = useState<string[]>([]);
  const [problemType, setProblemType] = useState('auto');
  const [isTraining, setIsTraining] = useState(false);
  const [trainingResult, setTrainingResult] = useState<any>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [cvFolds, setCvFolds] = useState(5);
  const [testSize, setTestSize] = useState(0.2);
  const [hyperparameterTuning, setHyperparameterTuning] = useState(false);
  const [autoFeatureEngineering, setAutoFeatureEngineering] = useState(true);
  const [featureSelectionMethod, setFeatureSelectionMethod] = useState('all');

  useEffect(() => {
    loadDatasets();
  }, []);

  useEffect(() => {
    if (selectedDataset) {
      loadDatasetColumns(selectedDataset);
    }
  }, [selectedDataset]);

  const loadDatasets = async () => {
    try {
      const data = await datasetsApi.getDatasets();
      setDatasets(data);
    } catch (error) {
      console.error('Failed to load datasets:', error);
    }
  };

  const loadDatasetColumns = async (datasetId: string) => {
    try {
      const schema = await datasetsApi.getSchema(datasetId);
      if (schema && schema.columns && Array.isArray(schema.columns)) {
        const cols = schema.columns.map((c: any) => c.name);
        setColumns(cols);
      } else {
        setColumns([]);
      }
    } catch (error) {
      console.error('Failed to load columns:', error);
      setColumns([]);
    }
  };

  const handleTargetToggle = (col: string) => {
    setTargetColumns(prev => 
      prev.includes(col) 
        ? prev.filter(c => c !== col)
        : [...prev, col]
    );
  };

  const handleFeatureToggle = (col: string) => {
    setFeatureColumns(prev => 
      prev.includes(col) 
        ? prev.filter(c => c !== col)
        : [...prev, col]
    );
  };

  const handleSelectAllFeatures = () => {
    const availableFeatures = columns.filter(c => !targetColumns.includes(c));
    setFeatureColumns(availableFeatures);
  };

  const handleDeselectAllFeatures = () => {
    setFeatureColumns([]);
  };

  const handleTrain = async () => {
    if (targetColumns.length === 0) return;

    setIsTraining(true);
    try {
      const result = await mlApi.trainModel({
        dataset_id: selectedDataset,
        target_columns: targetColumns,
        feature_columns: featureColumns.length > 0 ? featureColumns : undefined,
        problem_type: problemType as any,
        test_size: testSize,
        cv_folds: cvFolds,
        hyperparameter_tuning: hyperparameterTuning,
        auto_feature_engineering: autoFeatureEngineering,
        feature_selection_method: featureSelectionMethod as any,
      });
      setTrainingResult(result);
      onTrainingComplete?.(result);
    } catch (error) {
      console.error('Training failed:', error);
    } finally {
      setIsTraining(false);
    }
  };

  const availableFeatures = columns.filter(c => !targetColumns.includes(c));

  return (
    <div className="space-y-6">
      <Card style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2" style={{ color: 'var(--text-primary)' }}>
            <Brain size={20} style={{ color: 'var(--accent)' }} />
            ML Model Training
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Dataset Selection */}
          {!initialDatasetId && (
            <div>
              <Label style={{ color: 'var(--text-secondary)' }}>Dataset</Label>
              <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                <SelectTrigger 
                  className="mt-1"
                  style={{ background: 'var(--bg-tertiary)', borderColor: 'var(--border-color)' }}
                >
                  <SelectValue placeholder="Select a dataset" />
                </SelectTrigger>
                <SelectContent style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
                  {datasets.map((ds) => (
                    <SelectItem key={ds.id} value={ds.id}>{ds.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          {selectedDataset && (
            <>
              {/* Target Columns */}
              <div>
                <Label style={{ color: 'var(--text-secondary)' }}>Target Columns (to predict)</Label>
                <div className="mt-2 flex flex-wrap gap-2">
                  {columns.map((col) => (
                    <Badge
                      key={col}
                      variant={targetColumns.includes(col) ? 'default' : 'outline'}
                      className="cursor-pointer"
                      style={{
                        background: targetColumns.includes(col) ? 'var(--accent)' : 'transparent',
                        borderColor: 'var(--border-color)',
                        color: targetColumns.includes(col) ? 'white' : 'var(--text-secondary)'
                      }}
                      onClick={() => handleTargetToggle(col)}
                    >
                      {col}
                    </Badge>
                  ))}
                </div>
                <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                  Click to select one or more target columns
                </p>
              </div>

              {/* Feature Columns */}
              {targetColumns.length > 0 && (
                <div>
                  <div className="flex items-center justify-between">
                    <Label style={{ color: 'var(--text-secondary)' }}>Feature Columns</Label>
                    <div className="flex gap-2">
                      <Button variant="ghost" size="sm" onClick={handleSelectAllFeatures}>
                        Select All
                      </Button>
                      <Button variant="ghost" size="sm" onClick={handleDeselectAllFeatures}>
                        Clear
                      </Button>
                    </div>
                  </div>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {availableFeatures.map((col) => (
                      <Badge
                        key={col}
                        variant={featureColumns.includes(col) ? 'default' : 'outline'}
                        className="cursor-pointer"
                        style={{
                          background: featureColumns.includes(col) ? 'var(--success)' : 'transparent',
                          borderColor: 'var(--border-color)',
                          color: featureColumns.includes(col) ? 'white' : 'var(--text-secondary)'
                        }}
                        onClick={() => handleFeatureToggle(col)}
                      >
                        {col}
                      </Badge>
                    ))}
                  </div>
                  <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                    {featureColumns.length === 0 
                      ? 'Auto-select features (recommended)' 
                      : `${featureColumns.length} features selected`
                    }
                  </p>
                </div>
              )}

              {/* Problem Type */}
              <div>
                <Label style={{ color: 'var(--text-secondary)' }}>Problem Type</Label>
                <Select value={problemType} onValueChange={setProblemType}>
                  <SelectTrigger 
                    className="mt-1"
                    style={{ background: 'var(--bg-tertiary)', borderColor: 'var(--border-color)' }}
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
                    <SelectItem value="auto">Auto-detect</SelectItem>
                    <SelectItem value="regression">Regression</SelectItem>
                    <SelectItem value="classification">Classification</SelectItem>
                    <SelectItem value="time_series">Time Series</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Advanced Options */}
              <div>
                <Button 
                  variant="ghost" 
                  className="w-full justify-between"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                >
                  <span className="flex items-center gap-2">
                    <Settings size={16} />
                    Advanced Options
                  </span>
                  {showAdvanced ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                </Button>
                
                {showAdvanced && (
                  <div className="mt-4 space-y-4 p-4 rounded-lg" style={{ background: 'var(--bg-tertiary)' }}>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <Label style={{ color: 'var(--text-secondary)' }}>CV Folds</Label>
                        <Input
                          type="number"
                          min={2}
                          max={10}
                          value={cvFolds}
                          onChange={(e) => setCvFolds(Number(e.target.value))}
                          className="mt-1"
                          style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}
                        />
                      </div>
                      <div>
                        <Label style={{ color: 'var(--text-secondary)' }}>Test Size</Label>
                        <Input
                          type="number"
                          min={0.1}
                          max={0.5}
                          step={0.05}
                          value={testSize}
                          onChange={(e) => setTestSize(Number(e.target.value))}
                          className="mt-1"
                          style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}
                        />
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      <Checkbox 
                        checked={hyperparameterTuning}
                        onCheckedChange={(checked) => setHyperparameterTuning(checked as boolean)}
                      />
                      <Label style={{ color: 'var(--text-secondary)' }}>
                        Enable hyperparameter tuning
                      </Label>
                    </div>

                    <div className="flex items-center gap-2">
                      <Checkbox 
                        checked={autoFeatureEngineering}
                        onCheckedChange={(checked) => setAutoFeatureEngineering(checked as boolean)}
                      />
                      <Label style={{ color: 'var(--text-secondary)' }}>
                        Auto feature engineering
                      </Label>
                    </div>

                    <div>
                      <Label style={{ color: 'var(--text-secondary)' }}>Feature Selection</Label>
                      <Select value={featureSelectionMethod} onValueChange={setFeatureSelectionMethod}>
                        <SelectTrigger 
                          className="mt-1"
                          style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}
                        >
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
                          <SelectItem value="all">Use all features</SelectItem>
                          <SelectItem value="mutual_info">Mutual Information</SelectItem>
                          <SelectItem value="f_score">F-Score</SelectItem>
                          <SelectItem value="correlation">Correlation</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                )}
              </div>

              {/* Train Button */}
              <Button 
                onClick={handleTrain}
                disabled={targetColumns.length === 0 || isTraining}
                className="w-full"
                style={{ background: 'var(--accent)' }}
              >
                {isTraining ? (
                  <>
                    <Loader2 size={16} className="mr-2 animate-spin" />
                    Training...
                  </>
                ) : (
                  <>
                    <Play size={16} className="mr-2" />
                    Train Model
                  </>
                )}
              </Button>
            </>
          )}
        </CardContent>
      </Card>

      {/* Training Results */}
      {trainingResult && (
        <Card style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
          <CardHeader>
            <CardTitle className="flex items-center gap-2" style={{ color: 'var(--text-primary)' }}>
              <CheckCircle size={20} style={{ color: 'var(--success)' }} />
              Training Complete
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="text-sm" style={{ color: 'var(--text-muted)' }}>
              Training time: {trainingResult.total_training_time_seconds?.toFixed(2)}s
            </div>
            
            {Object.entries(trainingResult.target_results || {}).map(([target, result]: [string, any]) => (
              <div key={target} className="p-4 rounded-lg" style={{ background: 'var(--bg-tertiary)' }}>
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium" style={{ color: 'var(--text-primary)' }}>{target}</h4>
                  <Badge style={{ background: 'var(--accent)' }}>{result.best_algorithm}</Badge>
                </div>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-3">
                  {Object.entries(result.metrics || {}).map(([metric, value]: [string, any]) => (
                    <div key={metric}>
                      <p className="text-xs uppercase" style={{ color: 'var(--text-muted)' }}>{metric}</p>
                      <p className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
                        {typeof value === 'number' ? value.toFixed(4) : value}
                      </p>
                    </div>
                  ))}
                </div>

                <div className="mt-3">
                  <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>
                    CV Score: {(result.cv_mean * 100).toFixed(1)}% ± {(result.cv_std * 100).toFixed(1)}%
                  </p>
                  <Progress value={result.cv_mean * 100} className="h-2" />
                </div>

                {result.feature_importance && Object.keys(result.feature_importance).length > 0 && (
                  <div className="mt-4">
                    <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>Top Features</p>
                    <div className="space-y-1">
                      {Object.entries(result.feature_importance)
                        .sort(([,a]: any, [,b]: any) => b - a)
                        .slice(0, 5)
                        .map(([feature, importance]: [string, any]) => (
                          <div key={feature} className="flex items-center gap-2">
                            <span className="text-xs w-24 truncate" style={{ color: 'var(--text-secondary)' }}>{feature}</span>
                            <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ background: 'var(--bg-secondary)' }}>
                              <div 
                                className="h-full rounded-full"
                                style={{ 
                                  width: `${importance * 100}%`,
                                  background: 'var(--accent)'
                                }}
                              />
                            </div>
                            <span className="text-xs w-12 text-right" style={{ color: 'var(--text-muted)' }}>
                              {(importance * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
