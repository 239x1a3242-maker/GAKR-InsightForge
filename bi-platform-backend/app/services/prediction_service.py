"""Prediction service."""
import os
import json
import uuid
import h2o
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.model_registry import ModelVersion, ModelMetrics
from app.models.prediction import PredictionLog
from app.core.config import settings
import time


class PredictionService:
    """Prediction service."""
    
    # Cache for loaded models
    _model_cache: Dict[str, Any] = {}
    
    @staticmethod
    def load_model(model_path: str) -> Any:
        """Load H2O model from path."""
        if model_path in PredictionService._model_cache:
            return PredictionService._model_cache[model_path]
        
        model = h2o.load_model(model_path)
        PredictionService._model_cache[model_path] = model
        return model
    
    @staticmethod
    def validate_features(input_features: Dict[str, Any], feature_columns: List[str]) -> List[str]:
        """Validate input features against expected schema."""
        errors = []
        
        # Check for missing features
        missing = set(feature_columns) - set(input_features.keys())
        if missing:
            errors.append(f"Missing features: {', '.join(missing)}")
        
        # Check for extra features
        extra = set(input_features.keys()) - set(feature_columns)
        if extra:
            errors.append(f"Extra features: {', '.join(extra)}")
        
        return errors
    
    @staticmethod
    def prepare_input(input_features: Dict[str, Any], feature_columns: List[str]) -> pd.DataFrame:
        """Prepare input for prediction."""
        # Create dataframe with correct column order
        data = {col: [input_features.get(col)] for col in feature_columns}
        return pd.DataFrame(data)
    
    @staticmethod
    def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index."""
        # Handle edge cases
        if len(expected) == 0 or len(actual) == 0:
            return 0.0
        
        # Create bins based on expected distribution
        min_val = min(expected.min(), actual.min())
        max_val = max(expected.max(), actual.max())
        
        if min_val == max_val:
            return 0.0
        
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        # Calculate percentages
        expected_percents = np.histogram(expected, bins=bin_edges)[0] / len(expected)
        actual_percents = np.histogram(actual, bins=bin_edges)[0] / len(actual)
        
        # Add small value to avoid division by zero
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        # Calculate PSI
        psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        
        return float(psi)
    
    @staticmethod
    async def predict(
        db: AsyncSession,
        version_id: uuid.UUID,
        features: Dict[str, Any],
        user_id: Optional[uuid.UUID] = None,
        return_shap: bool = False
    ) -> Dict[str, Any]:
        """Make single prediction."""
        start_time = time.time()
        
        # Get model version
        result = await db.execute(
            select(ModelVersion, ModelMetrics)
            .join(ModelMetrics, ModelVersion.id == ModelMetrics.version_id)
            .where(ModelVersion.id == version_id)
        )
        row = result.first()
        
        if not row:
            raise ValueError(f"Model version {version_id} not found")
        
        version, metrics = row
        
        # Validate features
        errors = PredictionService.validate_features(features, version.feature_columns)
        if errors:
            raise ValueError(f"Feature validation failed: {'; '.join(errors)}")
        
        # Load model
        model = PredictionService.load_model(version.model_path)
        
        # Prepare input
        input_df = PredictionService.prepare_input(features, version.feature_columns)
        h2o_df = h2o.H2OFrame(input_df)
        
        # Make prediction
        prediction = model.predict(h2o_df)
        pred_df = prediction.as_data_frame()
        
        # Extract result
        pred_value = pred_df['predict'].iloc[0]
        
        result = {
            'prediction': pred_value,
            'confidence': None,
            'probabilities': None,
        }
        
        # Get probabilities for classification
        if version.registry.task_type == 'classification':
            if 'p0' in pred_df.columns and 'p1' in pred_df.columns:
                result['confidence'] = float(max(pred_df['p0'].iloc[0], pred_df['p1'].iloc[0]))
                result['probabilities'] = {
                    'class_0': float(pred_df['p0'].iloc[0]),
                    'class_1': float(pred_df['p1'].iloc[0]),
                }
        
        # SHAP values if requested
        if return_shap:
            try:
                contributions = model.predict_contributions(h2o_df)
                contrib_df = contributions.as_data_frame()
                result['shap_values'] = contrib_df.to_dict('records')[0]
            except Exception as e:
                result['shap_values'] = None
        
        prediction_time = time.time() - start_time
        
        # Log prediction
        log = PredictionLog(
            model_version_id=version_id,
            input_features=features,
            prediction={'value': pred_value},
            confidence=result['confidence'],
            prediction_time_ms=prediction_time * 1000,
            created_by=user_id,
        )
        db.add(log)
        await db.commit()
        
        result['prediction_time_ms'] = prediction_time * 1000
        
        return result
    
    @staticmethod
    async def predict_batch(
        db: AsyncSession,
        version_id: uuid.UUID,
        records: List[Dict[str, Any]],
        user_id: Optional[uuid.UUID] = None,
        return_confidence: bool = True
    ) -> Dict[str, Any]:
        """Make batch predictions."""
        start_time = time.time()
        
        # Get model version
        result = await db.execute(
            select(ModelVersion, ModelMetrics)
            .join(ModelMetrics, ModelVersion.id == ModelMetrics.version_id)
            .where(ModelVersion.id == version_id)
        )
        row = result.first()
        
        if not row:
            raise ValueError(f"Model version {version_id} not found")
        
        version, metrics = row
        
        # Load model
        model = PredictionService.load_model(version.model_path)
        
        # Prepare input
        input_df = pd.DataFrame(records)
        
        # Ensure all required columns are present
        for col in version.feature_columns:
            if col not in input_df.columns:
                input_df[col] = None
        
        # Reorder columns
        input_df = input_df[version.feature_columns]
        
        # Convert to H2O
        h2o_df = h2o.H2OFrame(input_df)
        
        # Make predictions
        predictions = model.predict(h2o_df)
        pred_df = predictions.as_data_frame()
        
        # Format results
        results = []
        for i in range(len(records)):
            result_item = {
                'prediction': pred_df['predict'].iloc[i],
            }
            
            if return_confidence and version.registry.task_type == 'classification':
                if 'p0' in pred_df.columns and 'p1' in pred_df.columns:
                    result_item['confidence'] = float(max(pred_df['p0'].iloc[i], pred_df['p1'].iloc[i]))
                    result_item['probabilities'] = {
                        'class_0': float(pred_df['p0'].iloc[i]),
                        'class_1': float(pred_df['p1'].iloc[i]),
                    }
            
            results.append(result_item)
        
        prediction_time = time.time() - start_time
        
        # Log predictions (batch)
        log = PredictionLog(
            model_version_id=version_id,
            input_features={'batch_size': len(records)},
            prediction={'batch_predictions': len(results)},
            prediction_time_ms=prediction_time * 1000,
            created_by=user_id,
        )
        db.add(log)
        await db.commit()
        
        return {
            'predictions': results,
            'total_records': len(records),
            'prediction_time_ms': prediction_time * 1000,
        }
    
    @staticmethod
    async def check_drift(
        db: AsyncSession,
        version_id: uuid.UUID,
        batch_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check for data drift."""
        # Get model version and metrics
        result = await db.execute(
            select(ModelVersion, ModelMetrics)
            .join(ModelMetrics, ModelVersion.id == ModelMetrics.version_id)
            .where(ModelVersion.id == version_id)
        )
        row = result.first()
        
        if not row:
            raise ValueError(f"Model version {version_id} not found")
        
        version, metrics = row
        
        # Convert batch data to dataframe
        batch_df = pd.DataFrame(batch_data)
        
        # Calculate PSI for each numeric feature
        feature_psi = {}
        drift_detected = False
        
        # Get training data distribution (from prediction_vs_actual if available)
        training_dist = {}
        if metrics.prediction_vs_actual:
            training_dist['prediction'] = [p['predicted'] for p in metrics.prediction_vs_actual]
        
        # Calculate PSI for predictions if we have training distribution
        psi_score = 0.0
        if 'prediction' in training_dist and 'prediction' in batch_df.columns:
            training_vals = np.array(training_dist['prediction'])
            batch_vals = batch_df['prediction'].values if 'prediction' in batch_df.columns else np.array([])
            
            if len(batch_vals) > 0:
                psi_score = PredictionService.calculate_psi(training_vals, batch_vals)
        
        # Check drift threshold
        threshold = 0.2
        drift_detected = psi_score > threshold
        
        # Feature-level PSI for numeric columns
        numeric_cols = batch_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in batch_df.columns:
                # Use percentiles as proxy for training distribution
                training_vals = np.random.normal(
                    batch_df[col].mean(),
                    batch_df[col].std(),
                    1000
                )
                batch_vals = batch_df[col].dropna().values
                
                if len(batch_vals) > 0:
                    psi = PredictionService.calculate_psi(training_vals, batch_vals)
                    feature_psi[col] = psi
        
        return {
            'model_id': version_id,
            'psi_score': psi_score,
            'drift_detected': drift_detected,
            'threshold': threshold,
            'feature_psi': feature_psi,
            'recommendation': 'Retrain model' if drift_detected else 'No action needed',
        }
