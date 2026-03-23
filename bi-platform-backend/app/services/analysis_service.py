"""Analysis service for analytics intelligence."""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from app.services.dataset_service import DatasetService
from app.models.dataset import Dataset
from app.models.analysis import AnalysisResult
from sqlalchemy import select
import uuid


class AnalysisService:
    """Analysis service for descriptive, diagnostic, predictive, and prescriptive analytics."""
    
    @staticmethod
    async def descriptive_analytics(
        db: AsyncSession,
        dataset_id: uuid.UUID,
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate descriptive analytics."""
        # Get dataset
        result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
        dataset = result.scalar_one_or_none()
        
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Load data
        df = DatasetService.load_dataset(dataset.file_path)
        
        if columns:
            df = df[columns]
        
        # Basic statistics
        numeric_df = df.select_dtypes(include=[np.number])
        
        analysis = {
            'dataset_id': str(dataset_id),
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(numeric_df.columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'date_columns': len(df.select_dtypes(include=['datetime64']).columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
        }
        
        # Missing values summary
        missing = df.isnull().sum()
        analysis['missing_summary'] = {
            'total_missing': int(missing.sum()),
            'missing_percentage': float(missing.sum() / (len(df) * len(df.columns)) * 100),
            'columns_with_missing': int((missing > 0).sum()),
        }
        
        # Correlation matrix
        if len(numeric_df.columns) > 1:
            corr = numeric_df.corr().round(3)
            analysis['correlation_matrix'] = corr.to_dict()
        
        # Distributions
        distributions = {}
        for col in numeric_df.columns:
            hist, bins = np.histogram(df[col].dropna(), bins=20)
            distributions[col] = {
                'bins': [f"{b:.2f}" for b in bins[:-1]],
                'counts': hist.tolist(),
            }
        
        analysis['distributions'] = distributions
        
        # Column statistics
        column_stats = {}
        for col in df.columns:
            if col in numeric_df.columns:
                column_stats[col] = {
                    'type': 'numeric',
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'skewness': float(df[col].skew()),
                }
            else:
                column_stats[col] = {
                    'type': 'categorical',
                    'unique_count': int(df[col].nunique()),
                    'top_values': df[col].value_counts().head(5).to_dict(),
                }
        
        analysis['column_stats'] = column_stats
        
        # Generate insights
        insights = []
        
        # High correlation pairs
        if len(numeric_df.columns) > 1:
            high_corr = []
            for i, col1 in enumerate(numeric_df.columns):
                for col2 in numeric_df.columns[i+1:]:
                    corr_val = numeric_df[col1].corr(numeric_df[col2])
                    if abs(corr_val) > 0.8:
                        high_corr.append(f"{col1} ↔ {col2}: {corr_val:.2f}")
            if high_corr:
                insights.append(f"Strong correlations detected: {'; '.join(high_corr[:3])}")
        
        # Missing values
        if analysis['missing_summary']['missing_percentage'] > 10:
            insights.append(f"Dataset has {analysis['missing_summary']['missing_percentage']:.1f}% missing values")
        
        # Skewed distributions
        for col, stats in column_stats.items():
            if stats.get('type') == 'numeric' and abs(stats.get('skewness', 0)) > 2:
                insights.append(f"'{col}' has highly skewed distribution (skewness: {stats['skewness']:.2f})")
        
        analysis['insights'] = insights
        
        # Save result
        result = AnalysisResult(
            dataset_id=dataset_id,
            analysis_type='descriptive',
            summary=analysis,
        )
        db.add(result)
        await db.commit()
        
        return analysis
    
    @staticmethod
    async def diagnostic_analytics(
        db: AsyncSession,
        dataset_id: uuid.UUID,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate diagnostic analytics."""
        # Get dataset
        result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
        dataset = result.scalar_one_or_none()
        
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Load data
        df = DatasetService.load_dataset(dataset.file_path)
        numeric_df = df.select_dtypes(include=[np.number])
        
        analysis = {
            'dataset_id': str(dataset_id),
            'target_column': target_column,
        }
        
        # Feature importance (correlation with target)
        if target_column and target_column in numeric_df.columns:
            correlations = {}
            for col in numeric_df.columns:
                if col != target_column:
                    corr = numeric_df[col].corr(numeric_df[target_column])
                    correlations[col] = abs(corr) if not pd.isna(corr) else 0
            
            analysis['feature_importance'] = dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Correlations
        correlations = {}
        for col in numeric_df.columns:
            col_corrs = []
            for other_col in numeric_df.columns:
                if col != other_col:
                    corr = numeric_df[col].corr(numeric_df[other_col])
                    if abs(corr) > 0.5:
                        col_corrs.append({'feature': other_col, 'correlation': round(corr, 3)})
            if col_corrs:
                correlations[col] = sorted(col_corrs, key=lambda x: abs(x['correlation']), reverse=True)[:5]
        
        analysis['correlations'] = correlations
        
        # Outlier detection
        outliers = {}
        if len(numeric_df.columns) > 0:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(numeric_df.dropna())
            
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            outlier_labels = iso_forest.fit_predict(scaled)
            
            outliers = {
                'total_outliers': int((outlier_labels == -1).sum()),
                'outlier_percentage': float((outlier_labels == -1).sum() / len(outlier_labels) * 100),
            }
        
        analysis['outliers'] = outliers
        
        # Segments (simple quartile-based)
        segments = []
        if len(numeric_df.columns) > 0:
            first_col = numeric_df.columns[0]
            q1 = numeric_df[first_col].quantile(0.25)
            q3 = numeric_df[first_col].quantile(0.75)
            
            low_segment = numeric_df[numeric_df[first_col] <= q1]
            high_segment = numeric_df[numeric_df[first_col] >= q3]
            
            segments = [
                {
                    'name': f'Low {first_col}',
                    'size': len(low_segment),
                    'mean_values': low_segment.mean().to_dict(),
                },
                {
                    'name': f'High {first_col}',
                    'size': len(high_segment),
                    'mean_values': high_segment.mean().to_dict(),
                },
            ]
        
        analysis['segments'] = segments
        
        # Root causes
        root_causes = []
        
        if target_column and target_column in numeric_df.columns:
            # Find features most correlated with target
            for col in numeric_df.columns:
                if col != target_column:
                    corr = numeric_df[col].corr(numeric_df[target_column])
                    if abs(corr) > 0.7:
                        direction = "positively" if corr > 0 else "negatively"
                        root_causes.append(f"'{col}' is strongly {direction} correlated with '{target_column}' (r={corr:.2f})")
        
        if not root_causes:
            root_causes.append("No strong correlations found with target variable")
        
        analysis['root_causes'] = root_causes
        
        # Save result
        result = AnalysisResult(
            dataset_id=dataset_id,
            analysis_type='diagnostic',
            summary=analysis,
        )
        db.add(result)
        await db.commit()
        
        return analysis
    
    @staticmethod
    async def predictive_analytics(
        db: AsyncSession,
        dataset_id: uuid.UUID,
        target_column: str
    ) -> Dict[str, Any]:
        """Generate predictive analytics using quick model training."""
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder
        
        # Get dataset
        result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
        dataset = result.scalar_one_or_none()
        
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Load data
        df = DatasetService.load_dataset(dataset.file_path)
        
        # Detect task type
        task_detection = await DatasetService.detect_task_type(df, target_column)
        task_type = task_detection['task_type']
        
        # Prepare data
        numeric_df = df.select_dtypes(include=[np.number]).fillna(0)
        
        if target_column not in numeric_df.columns:
            raise ValueError(f"Target column '{target_column}' must be numeric")
        
        X = numeric_df.drop(columns=[target_column])
        y = numeric_df[target_column]
        
        # Encode target for classification
        if task_type == 'classification':
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
        
        # Train quick models
        algorithms = [
            ('RandomForest', RandomForestClassifier if task_type == 'classification' else RandomForestRegressor),
        ]
        
        results = []
        for name, ModelClass in algorithms:
            model = ModelClass(n_estimators=50, random_state=42, n_jobs=-1)
            
            if task_type == 'classification':
                scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
                metrics = {
                    'accuracy': float(scores.mean()),
                    'std': float(scores.std()),
                }
            else:
                scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                metrics = {
                    'r2': float(scores.mean()),
                    'std': float(scores.std()),
                }
            
            results.append({
                'name': name,
                'metrics': metrics,
                'training_time_ms': 0,  # Quick estimate
                'rank': 1,
            })
        
        # Feature importance
        model = RandomForestClassifier(n_estimators=50, random_state=42) if task_type == 'classification' else RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        feature_importance = dict(zip(X.columns, model.feature_importances_))
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        analysis = {
            'dataset_id': str(dataset_id),
            'target_column': target_column,
            'task_type': task_type,
            'best_algorithm': results[0]['name'],
            'algorithms': results,
            'feature_importance': feature_importance,
            'cross_validation_score': results[0]['metrics'].get('accuracy') or results[0]['metrics'].get('r2'),
        }
        
        # Save result
        result = AnalysisResult(
            dataset_id=dataset_id,
            analysis_type='predictive',
            summary=analysis,
        )
        db.add(result)
        await db.commit()
        
        return analysis
    
    @staticmethod
    async def prescriptive_analytics(
        db: AsyncSession,
        dataset_id: uuid.UUID,
        target_column: str
    ) -> Dict[str, Any]:
        """Generate prescriptive analytics with recommendations."""
        # Get predictive analysis first
        predictive = await AnalysisService.predictive_analytics(db, dataset_id, target_column)
        
        # Get dataset
        result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
        dataset = result.scalar_one_or_none()
        
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Load data
        df = DatasetService.load_dataset(dataset.file_path)
        numeric_df = df.select_dtypes(include=[np.number])
        
        analysis = {
            'dataset_id': str(dataset_id),
            'target_column': target_column,
        }
        
        # Generate recommendations based on feature importance
        recommendations = []
        
        if predictive.get('feature_importance'):
            top_features = list(predictive['feature_importance'].keys())[:3]
            
            for feature in top_features:
                if feature in numeric_df.columns:
                    mean_val = numeric_df[feature].mean()
                    std_val = numeric_df[feature].std()
                    
                    recommendations.append({
                        'action': f"Optimize '{feature}'",
                        'impact': f"High impact on '{target_column}'",
                        'confidence': 0.85,
                        'supporting_data': {
                            'current_mean': round(mean_val, 2),
                            'std': round(std_val, 2),
                            'suggested_range': [round(mean_val - std_val, 2), round(mean_val + std_val, 2)],
                        },
                    })
        
        analysis['recommendations'] = recommendations
        
        # Scenarios
        scenarios = []
        if target_column in numeric_df.columns:
            target_mean = numeric_df[target_column].mean()
            target_std = numeric_df[target_column].std()
            
            scenarios = [
                {
                    'name': 'Optimistic',
                    'target_value': round(target_mean + target_std, 2),
                    'probability': 0.25,
                },
                {
                    'name': 'Expected',
                    'target_value': round(target_mean, 2),
                    'probability': 0.5,
                },
                {
                    'name': 'Pessimistic',
                    'target_value': round(target_mean - target_std, 2),
                    'probability': 0.25,
                },
            ]
        
        analysis['scenarios'] = scenarios
        
        # What-if analysis
        what_if = {}
        if predictive.get('feature_importance'):
            top_feature = list(predictive['feature_importance'].keys())[0]
            if top_feature in numeric_df.columns:
                feature_vals = numeric_df[top_feature]
                what_if = {
                    'variable': top_feature,
                    'current_value': round(feature_vals.mean(), 2),
                    'impact_per_unit': round(predictive['feature_importance'][top_feature] * 100, 2),
                    'suggested_increase': round(feature_vals.std(), 2),
                }
        
        analysis['what_if_analysis'] = what_if
        
        # Save result
        result = AnalysisResult(
            dataset_id=dataset_id,
            analysis_type='prescriptive',
            summary=analysis,
        )
        db.add(result)
        await db.commit()
        
        return analysis
