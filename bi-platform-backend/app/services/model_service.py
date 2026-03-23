"""Model service with H2O AutoML."""
import os
import json
import hashlib
import uuid
import h2o
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc
from app.models.model_registry import ModelRegistry, ModelVersion, ModelMetrics, ModelStatus
from app.models.dataset import Dataset
from app.services.dataset_service import DatasetService
from app.core.config import settings
import joblib


# Initialize H2O
h2o.init(ip=settings.H2O_IP, port=settings.H2O_PORT, strict_version_check=False)


class ModelService:
    """Model service for training and registry."""
    
    @staticmethod
    def compute_schema_hash(columns: List[str], dtypes: Dict[str, str]) -> str:
        """Compute hash of schema for validation."""
        schema_str = json.dumps({'columns': sorted(columns), 'dtypes': dtypes}, sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]
    
    @staticmethod
    async def get_or_create_registry(
        db: AsyncSession,
        dataset_id: uuid.UUID,
        target_column: str,
        task_type: str
    ) -> ModelRegistry:
        """Get or create model registry entry."""
        result = await db.execute(
            select(ModelRegistry).where(
                and_(
                    ModelRegistry.dataset_id == dataset_id,
                    ModelRegistry.target_column == target_column
                )
            )
        )
        registry = result.scalar_one_or_none()
        
        if not registry:
            registry = ModelRegistry(
                dataset_id=dataset_id,
                target_column=target_column,
                task_type=task_type,
            )
            db.add(registry)
            await db.commit()
            await db.refresh(registry)
        
        return registry
    
    @staticmethod
    async def get_next_version_number(db: AsyncSession, registry_id: uuid.UUID) -> int:
        """Get next version number for registry."""
        result = await db.execute(
            select(ModelVersion).where(
                ModelVersion.registry_id == registry_id
            ).order_by(desc(ModelVersion.version_number))
        )
        latest = result.scalar_one_or_none()
        return (latest.version_number + 1) if latest else 1
    
    @staticmethod
    def prepare_data(df: pd.DataFrame, target_column: str, feature_columns: Optional[List[str]] = None) -> tuple:
        """Prepare data for training."""
        # Remove rows with missing target
        df = df.dropna(subset=[target_column])
        
        # Select features
        if feature_columns:
            features = [f for f in feature_columns if f in df.columns and f != target_column]
        else:
            features = [c for c in df.columns if c != target_column]
        
        # Remove columns with all missing values
        features = [f for f in features if df[f].notna().any()]
        
        X = df[features].copy()
        y = df[target_column].copy()
        
        return X, y, features
    
    @staticmethod
    def train_h2o_automl(
        df: pd.DataFrame,
        target_column: str,
        task_type: str,
        config: Dict[str, Any],
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Train H2O AutoML model."""
        # Prepare data
        X, y, features = ModelService.prepare_data(df, target_column, feature_columns)
        
        # Create training dataframe
        train_df = X.copy()
        train_df[target_column] = y
        
        # Convert to H2O frame
        h2o_df = h2o.H2OFrame(train_df)
        
        # Set target type
        if task_type == 'classification':
            h2o_df[target_column] = h2o_df[target_column].asfactor()
        
        # Split data
        train, test = h2o_df.split_frame(ratios=[0.8], seed=config.get('seed', 42))
        
        # Configure algorithms
        include_algos = ['GLM', 'DRF', 'GBM', 'XGBoost', 'XRT']
        if config.get('enable_stacking', True):
            include_algos.append('StackedEnsemble')
        
        # Train AutoML
        aml = h2o.automl.H2OAutoML(
            max_runtime_secs=config.get('max_runtime_secs', 600),
            nfolds=config.get('nfolds', 5),
            max_models=config.get('max_models', 20),
            include_algos=include_algos,
            seed=config.get('seed', 42),
            sort_metric='AUC' if task_type == 'classification' else 'RMSE',
        )
        
        aml.train(y=target_column, training_frame=train, leaderboard_frame=test)
        
        # Get best model
        best_model = aml.leader
        model_name = best_model.model_id
        
        # Save model
        os.makedirs(settings.MODEL_DIR, exist_ok=True)
        model_path = h2o.save_model(best_model, path=settings.MODEL_DIR, force=True)
        
        # Get leaderboard
        leaderboard = aml.leaderboard.as_data_frame()
        leaderboard_entries = []
        
        for idx, row in leaderboard.head(10).iterrows():
            entry = {
                'model_id': row['model_id'],
                'algorithm': row['model_id'].split('_')[0] if '_' in row['model_id'] else row['model_id'],
                'training_time_ms': int(row.get('training_time_ms', 0)),
            }
            
            # Add metrics based on task type
            if task_type == 'classification':
                entry['auc'] = float(row.get('auc', 0)) if 'auc' in row else None
                entry['logloss'] = float(row.get('logloss', 0)) if 'logloss' in row else None
                entry['mean_per_class_error'] = float(row.get('mean_per_class_error', 0)) if 'mean_per_class_error' in row else None
            else:
                entry['rmse'] = float(row.get('rmse', 0)) if 'rmse' in row else None
                entry['mse'] = float(row.get('mse', 0)) if 'mse' in row else None
                entry['mae'] = float(row.get('mae', 0)) if 'mae' in row else None
                entry['rmsle'] = float(row.get('rmsle', 0)) if 'rmsle' in row else None
            
            leaderboard_entries.append(entry)
        
        # Get feature importance
        try:
            importance = best_model.varimp(use_pandas=True)
            feature_importance = dict(zip(importance['variable'], importance['relative_importance']))
        except:
            feature_importance = {f: 1.0 / len(features) for f in features}
        
        # Get predictions on test set
        predictions = best_model.predict(test)
        
        # Calculate metrics
        y_true = test[target_column].as_data_frame().values.flatten()
        y_pred = predictions['predict'].as_data_frame().values.flatten()
        
        metrics = ModelService.calculate_metrics(y_true, y_pred, task_type)
        
        return {
            'model_name': model_name,
            'model_path': model_path,
            'features': features,
            'leaderboard': leaderboard_entries,
            'feature_importance': feature_importance,
            'metrics': metrics,
            'training_rows': len(train),
            'test_rows': len(test),
        }
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, Any]:
        """Calculate model metrics."""
        from sklearn import metrics as sk_metrics
        
        if task_type == 'classification':
            # Ensure integer labels
            if y_true.dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_true = le.fit_transform(y_true)
                y_pred = le.transform(y_pred)
            
            # Classification metrics
            result = {
                'accuracy': float(sk_metrics.accuracy_score(y_true, y_pred)),
                'precision': float(sk_metrics.precision_score(y_true, y_pred, average='weighted', zero_division=0)),
                'recall': float(sk_metrics.recall_score(y_true, y_pred, average='weighted', zero_division=0)),
                'f1_score': float(sk_metrics.f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            }
            
            # Binary classification specific
            if len(np.unique(y_true)) == 2:
                try:
                    result['auc'] = float(sk_metrics.roc_auc_score(y_true, y_pred))
                    
                    # ROC curve data
                    fpr, tpr, _ = sk_metrics.roc_curve(y_true, y_pred)
                    result['roc_curve'] = {
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist(),
                    }
                    
                    # PR curve data
                    precision, recall, _ = sk_metrics.precision_recall_curve(y_true, y_pred)
                    result['pr_curve'] = {
                        'precision': precision.tolist(),
                        'recall': recall.tolist(),
                    }
                except:
                    pass
            
            # Confusion matrix
            result['confusion_matrix'] = sk_metrics.confusion_matrix(y_true, y_pred).tolist()
            
        else:
            # Regression metrics
            result = {
                'r2': float(sk_metrics.r2_score(y_true, y_pred)),
                'rmse': float(np.sqrt(sk_metrics.mean_squared_error(y_true, y_pred))),
                'mae': float(sk_metrics.mean_absolute_error(y_true, y_pred)),
                'mse': float(sk_metrics.mean_squared_error(y_true, y_pred)),
            }
            
            # MAPE
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
            result['mape'] = float(mape)
            
            # Residuals
            residuals = y_true - y_pred
            result['residual_mean'] = float(np.mean(residuals))
            result['residual_std'] = float(np.std(residuals))
            
            # Prediction vs actual (sample)
            sample_size = min(1000, len(y_true))
            indices = np.random.choice(len(y_true), sample_size, replace=False)
            result['prediction_vs_actual'] = [
                {'actual': float(y_true[i]), 'predicted': float(y_pred[i])}
                for i in indices
            ]
        
        return result
    
    @staticmethod
    async def train_model(
        db: AsyncSession,
        dataset_id: uuid.UUID,
        target_column: str,
        config: Dict[str, Any],
        task_type: str,
    ) -> ModelVersion:
        """Train a model for a target column."""
        # Get dataset
        result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
        dataset = result.scalar_one_or_none()
        
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Load data
        df = DatasetService.load_dataset(dataset.file_path)
        
        # Get or create registry
        registry = await ModelService.get_or_create_registry(db, dataset_id, target_column, task_type)
        
        # Get next version number
        version_number = await ModelService.get_next_version_number(db, registry.id)
        
        # Train model
        import time
        start_time = time.time()
        
        training_result = ModelService.train_h2o_automl(
            df, target_column, task_type, config
        )
        
        training_duration = time.time() - start_time
        
        # Compute schema hash
        schema_hash = ModelService.compute_schema_hash(
            training_result['features'],
            {col['name']: col['dtype'] for col in dataset.columns}
        )
        
        # Create version
        version = ModelVersion(
            registry_id=registry.id,
            version_number=version_number,
            status=ModelStatus.STAGING,
            model_path=training_result['model_path'],
            model_name=training_result['model_name'],
            schema_hash=schema_hash,
            feature_columns=training_result['features'],
            training_config=config,
            leaderboard=training_result['leaderboard'],
            best_algorithm=training_result['leaderboard'][0]['algorithm'] if training_result['leaderboard'] else None,
            training_duration_seconds=training_duration,
            training_rows=training_result['training_rows'],
        )
        
        db.add(version)
        await db.commit()
        await db.refresh(version)
        
        # Create metrics
        metrics_data = training_result['metrics']
        
        if task_type == 'classification':
            metrics = ModelMetrics(
                version_id=version.id,
                train_rows=training_result['training_rows'],
                test_rows=training_result['test_rows'],
                accuracy=metrics_data.get('accuracy'),
                precision=metrics_data.get('precision'),
                recall=metrics_data.get('recall'),
                f1_score=metrics_data.get('f1_score'),
                auc=metrics_data.get('auc'),
                logloss=metrics_data.get('logloss'),
                confusion_matrix=metrics_data.get('confusion_matrix'),
                roc_curve=metrics_data.get('roc_curve'),
                pr_curve=metrics_data.get('pr_curve'),
                feature_importance=training_result['feature_importance'],
            )
        else:
            metrics = ModelMetrics(
                version_id=version.id,
                train_rows=training_result['training_rows'],
                test_rows=training_result['test_rows'],
                r2=metrics_data.get('r2'),
                rmse=metrics_data.get('rmse'),
                mae=metrics_data.get('mae'),
                mse=metrics_data.get('mse'),
                mape=metrics_data.get('mape'),
                residual_mean=metrics_data.get('residual_mean'),
                residual_std=metrics_data.get('residual_std'),
                prediction_vs_actual=metrics_data.get('prediction_vs_actual'),
                feature_importance=training_result['feature_importance'],
            )
        
        db.add(metrics)
        await db.commit()
        
        return version
    
    @staticmethod
    async def promote_version(db: AsyncSession, version_id: uuid.UUID) -> ModelVersion:
        """Promote model version to production."""
        result = await db.execute(select(ModelVersion).where(ModelVersion.id == version_id))
        version = result.scalar_one_or_none()
        
        if not version:
            raise ValueError(f"Version {version_id} not found")
        
        # Archive current production version
        result = await db.execute(
            select(ModelRegistry).where(ModelRegistry.id == version.registry_id)
        )
        registry = result.scalar_one()
        
        if registry.production_version_id:
            result = await db.execute(
                select(ModelVersion).where(ModelVersion.id == registry.production_version_id)
            )
            old_prod = result.scalar_one()
            old_prod.status = ModelStatus.ARCHIVED
            old_prod.archived_at = datetime.now(timezone.utc)
        
        # Promote new version
        version.status = ModelStatus.PRODUCTION
        version.promoted_at = datetime.now(timezone.utc)
        
        # Update registry
        registry.production_version_id = version.id
        
        await db.commit()
        await db.refresh(version)
        
        return version
    
    @staticmethod
    async def get_model_versions(db: AsyncSession, registry_id: uuid.UUID) -> List[ModelVersion]:
        """Get all versions for a registry."""
        result = await db.execute(
            select(ModelVersion)
            .where(ModelVersion.registry_id == registry_id)
            .order_by(desc(ModelVersion.version_number))
        )
        return result.scalars().all()
    
    @staticmethod
    async def get_production_model(db: AsyncSession, target_column: str, dataset_id: uuid.UUID = None) -> Optional[ModelVersion]:
        """Get production model for a target column."""
        query = select(ModelVersion).join(ModelRegistry).where(
            and_(
                ModelRegistry.target_column == target_column,
                ModelVersion.status == ModelStatus.PRODUCTION
            )
        )
        
        if dataset_id:
            query = query.where(ModelRegistry.dataset_id == dataset_id)
        
        result = await db.execute(query)
        return result.scalar_one_or_none()
