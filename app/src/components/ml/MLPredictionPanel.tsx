import { useState, useEffect } from 'react';
import { mlApi } from '@/api';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Checkbox } from '@/components/ui/checkbox';
import { Badge } from '@/components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { 
  Sparkles, Play, Loader2, Download, BarChart3,
  ChevronDown, ChevronUp
} from 'lucide-react';
import { formatNumber } from '@/utils/format';

interface MLPredictionPanelProps {
  modelIds?: string[];
  sampleData?: any[];
}

export function MLPredictionPanel({ modelIds: initialModelIds, sampleData }: MLPredictionPanelProps) {
  const [models, setModels] = useState<any[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>(initialModelIds || []);
  const [predictionData, setPredictionData] = useState<any[]>(sampleData || []);
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [returnConfidence, setReturnConfidence] = useState(true);
  const [showDataInput, setShowDataInput] = useState(false);
  const [dataInput, setDataInput] = useState('');

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      const data: any = await mlApi.getModels();
      if (Array.isArray(data)) {
        setModels(data);
      } else if (data && Array.isArray(data.items)) {
        setModels(data.items);
      } else {
        setModels([]);
      }
    } catch (error) {
      console.error('Failed to load models:', error);
      setModels([]);
    }
  };

  const handleModelToggle = (modelId: string) => {
    setSelectedModels(prev => 
      prev.includes(modelId)
        ? prev.filter(id => id !== modelId)
        : [...prev, modelId]
    );
  };

  const handleDataInputChange = (value: string) => {
    setDataInput(value);
    try {
      const parsed = JSON.parse(value);
      if (Array.isArray(parsed)) {
        setPredictionData(parsed);
      }
    } catch {
      // Invalid JSON
    }
  };

  const handlePredict = async () => {
    if (selectedModels.length === 0 || predictionData.length === 0) return;

    setIsPredicting(true);
    try {
      const result = await mlApi.predict(
        selectedModels,
        predictionData,
        returnConfidence
      );
      setPredictionResult(result);
    } catch (error) {
      console.error('Prediction failed:', error);
    } finally {
      setIsPredicting(false);
    }
  };

  const handleDownloadResults = () => {
    if (!predictionResult) return;
    
    const dataStr = JSON.stringify(predictionResult, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'predictions.json';
    link.click();
  };

  return (
    <div className="space-y-6">
      <Card style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2" style={{ color: 'var(--text-primary)' }}>
            <Sparkles size={20} style={{ color: 'var(--accent)' }} />
            ML Predictions
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Model Selection */}
          <div>
            <Label style={{ color: 'var(--text-secondary)' }}>Select Models</Label>
            <div className="mt-2 space-y-2">
              {models.length === 0 ? (
                <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                  No trained models available. Train a model first.
                </p>
              ) : (
                models.map((model) => (
                  <div 
                    key={model.id}
                    className="flex items-center gap-3 p-2 rounded-lg cursor-pointer transition-colors"
                    style={{ 
                      background: selectedModels.includes(model.id) ? 'var(--accent)20' : 'var(--bg-tertiary)',
                      border: `1px solid ${selectedModels.includes(model.id) ? 'var(--accent)' : 'var(--border-color)'}`
                    }}
                    onClick={() => handleModelToggle(model.id)}
                  >
                    <div className="flex-1">
                      <p className="font-medium text-sm" style={{ color: 'var(--text-primary)' }}>
                        {model.name}
                      </p>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        Target: {model.target_column} | Algorithm: {model.algorithm}
                      </p>
                    </div>
                    <Badge 
                      variant={selectedModels.includes(model.id) ? 'default' : 'outline'}
                      style={{ 
                        background: selectedModels.includes(model.id) ? 'var(--accent)' : 'transparent'
                      }}
                    >
                      {selectedModels.includes(model.id) ? 'Selected' : 'Select'}
                    </Badge>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Data Input */}
          <div>
            <div className="flex items-center justify-between">
              <Label style={{ color: 'var(--text-secondary)' }}>Input Data</Label>
              <Button variant="ghost" size="sm" onClick={() => setShowDataInput(!showDataInput)}>
                {showDataInput ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </Button>
            </div>
            
            {showDataInput ? (
              <div className="mt-2">
                <textarea
                  value={dataInput}
                  onChange={(e) => handleDataInputChange(e.target.value)}
                  placeholder={`[\n  {"feature1": 1, "feature2": 2},\n  {"feature1": 3, "feature2": 4}\n]`}
                  className="w-full h-32 p-3 rounded-md text-sm font-mono"
                  style={{ 
                    background: 'var(--bg-tertiary)', 
                    border: '1px solid var(--border-color)',
                    color: 'var(--text-primary)',
                    resize: 'vertical'
                  }}
                />
                <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                  Enter data as JSON array
                </p>
              </div>
            ) : (
              <div 
                className="mt-2 p-3 rounded-lg"
                style={{ background: 'var(--bg-tertiary)' }}
              >
                <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                  {predictionData.length} rows ready for prediction
                </p>
              </div>
            )}
          </div>

          {/* Options */}
          <div className="flex gap-4">
            <div className="flex items-center gap-2">
              <Checkbox 
                checked={returnConfidence}
                onCheckedChange={(checked) => setReturnConfidence(checked as boolean)}
              />
              <Label className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                Return confidence scores
              </Label>
            </div>
          </div>

          {/* Predict Button */}
          <Button 
            onClick={handlePredict}
            disabled={selectedModels.length === 0 || predictionData.length === 0 || isPredicting}
            className="w-full"
            style={{ background: 'var(--accent)' }}
          >
            {isPredicting ? (
              <>
                <Loader2 size={16} className="mr-2 animate-spin" />
                Predicting...
              </>
            ) : (
              <>
                <Play size={16} className="mr-2" />
                Run Prediction
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Prediction Results */}
      {predictionResult && (
        <Card style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="flex items-center gap-2" style={{ color: 'var(--text-primary)' }}>
              <BarChart3 size={20} style={{ color: 'var(--success)' }} />
              Prediction Results
            </CardTitle>
            <Button variant="outline" size="sm" onClick={handleDownloadResults}>
              <Download size={14} className="mr-1" />
              Download
            </Button>
          </CardHeader>
          <CardContent>
            <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
              Prediction time: {predictionResult.prediction_time_ms}ms | 
              Input rows: {predictionResult.input_rows}
            </p>

            {Object.entries(predictionResult.predictions || {}).map(([target, result]: [string, any]) => (
              <div key={target} className="mb-6">
                <h4 className="font-medium mb-2" style={{ color: 'var(--text-primary)' }}>
                  {target}
                </h4>
                
                <div className="overflow-auto">
                  <Table>
                    <TableHeader>
                      <TableRow style={{ borderBottom: '1px solid var(--border-color)' }}>
                        <TableHead style={{ color: 'var(--text-secondary)' }}>Row</TableHead>
                        <TableHead style={{ color: 'var(--text-secondary)' }}>Prediction</TableHead>
                        {result.confidence && (
                          <TableHead style={{ color: 'var(--text-secondary)' }}>Confidence</TableHead>
                        )}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {result.predictions.slice(0, 20).map((pred: any, idx: number) => (
                        <TableRow key={idx} style={{ borderBottom: '1px solid var(--border-color)' }}>
                          <TableCell style={{ color: 'var(--text-muted)' }}>{idx + 1}</TableCell>
                          <TableCell style={{ color: 'var(--text-primary)' }}>
                            {typeof pred === 'number' ? formatNumber(pred) : pred}
                          </TableCell>
                          {result.confidence && (
                            <TableCell>
                              <div className="flex items-center gap-2">
                                <div 
                                  className="w-16 h-2 rounded-full"
                                  style={{ background: 'var(--bg-tertiary)' }}
                                >
                                  <div 
                                    className="h-full rounded-full"
                                    style={{ 
                                      width: `${(result.confidence[idx] * 100).toFixed(0)}%`,
                                      background: result.confidence[idx] > 0.8 ? 'var(--success)' : 
                                                  result.confidence[idx] > 0.5 ? 'var(--warning)' : 'var(--danger)'
                                    }}
                                  />
                                </div>
                                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                                  {(result.confidence[idx] * 100).toFixed(0)}%
                                </span>
                              </div>
                            </TableCell>
                          )}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>

                {result.predictions.length > 20 && (
                  <p className="text-sm mt-2 text-center" style={{ color: 'var(--text-muted)' }}>
                    Showing 20 of {result.predictions.length} predictions
                  </p>
                )}
              </div>
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

// Label component for the form
function Label({ children, className, style }: { children: React.ReactNode; className?: string; style?: React.CSSProperties }) {
  return (
    <label className={`text-sm font-medium ${className}`} style={style}>
      {children}
    </label>
  );
}
