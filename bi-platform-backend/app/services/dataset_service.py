"""Dataset service."""
import os
import uuid
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, BinaryIO
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from app.models.dataset import Dataset, DatasetProfile
from app.core.config import settings
import aiofiles


class DatasetService:
    """Dataset service."""
    
    @staticmethod
    async def save_uploaded_file(file: BinaryIO, filename: str) -> str:
        """Save uploaded file and return path."""
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        file_id = str(uuid.uuid4())
        ext = os.path.splitext(filename)[1].lower()
        file_path = os.path.join(settings.UPLOAD_DIR, f"{file_id}{ext}")
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = file.read()
            await f.write(content)
        
        return file_path
    
    @staticmethod
    def load_dataset(file_path: str) -> pd.DataFrame:
        """Load dataset from file."""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.csv':
            return pd.read_csv(file_path)
        elif ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif ext == '.json':
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    @staticmethod
    def detect_schema(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect schema from dataframe."""
        columns = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            
            # Simplify dtype
            if dtype.startswith('int') or dtype.startswith('float'):
                simplified_type = 'numeric'
            elif dtype == 'object':
                simplified_type = 'string'
            elif dtype.startswith('datetime'):
                simplified_type = 'datetime'
            elif dtype == 'bool':
                simplified_type = 'boolean'
            else:
                simplified_type = 'string'
            
            columns.append({
                'name': col,
                'type': simplified_type,
                'dtype': dtype,
                'nullable': df[col].isnull().any(),
                'unique_count': int(df[col].nunique()),
            })
        
        return columns
    
    @staticmethod
    def generate_profile(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate dataset profile."""
        profile = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
        }
        
        # Column type counts
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        profile['numeric_columns'] = len(numeric_cols)
        profile['categorical_columns'] = len(categorical_cols)
        profile['date_columns'] = len(date_cols)
        
        # Missing values
        missing_counts = df.isnull().sum()
        profile['total_missing_values'] = int(missing_counts.sum())
        profile['missing_percentage'] = float(missing_counts.sum() / (len(df) * len(df.columns)) * 100)
        
        # Duplicate rows
        profile['duplicate_rows'] = int(df.duplicated().sum())
        
        # Constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        profile['constant_columns'] = constant_cols
        
        # Column statistics
        column_stats = {}
        for col in df.columns:
            stats = {}
            
            if col in numeric_cols:
                stats['mean'] = float(df[col].mean()) if not df[col].isnull().all() else None
                stats['median'] = float(df[col].median()) if not df[col].isnull().all() else None
                stats['std'] = float(df[col].std()) if not df[col].isnull().all() else None
                stats['min'] = float(df[col].min()) if not df[col].isnull().all() else None
                stats['max'] = float(df[col].max()) if not df[col].isnull().all() else None
                stats['skewness'] = float(df[col].skew()) if not df[col].isnull().all() else None
                stats['kurtosis'] = float(df[col].kurtosis()) if not df[col].isnull().all() else None
                stats['missing_count'] = int(df[col].isnull().sum())
                stats['missing_percentage'] = float(df[col].isnull().sum() / len(df) * 100)
            else:
                stats['unique_count'] = int(df[col].nunique())
                stats['top_values'] = df[col].value_counts().head(10).to_dict()
                stats['missing_count'] = int(df[col].isnull().sum())
                stats['missing_percentage'] = float(df[col].isnull().sum() / len(df) * 100)
            
            column_stats[col] = stats
        
        profile['column_stats'] = column_stats
        
        # Correlation matrix (numeric only)
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().round(4)
            # Convert to dict, handling NaN values
            corr_dict = {}
            for col in corr_matrix.columns:
                corr_dict[col] = {}
                for idx in corr_matrix.index:
                    val = corr_matrix.loc[idx, col]
                    corr_dict[col][idx] = None if pd.isna(val) else float(val)
            profile['correlation_matrix'] = corr_dict
        else:
            profile['correlation_matrix'] = {}
        
        # Generate insights
        insights = []
        
        # High correlation warning
        if len(numeric_cols) > 1:
            high_corr_pairs = []
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    corr_val = df[col1].corr(df[col2])
                    if abs(corr_val) > 0.9:
                        high_corr_pairs.append(f"{col1} ↔ {col2} ({corr_val:.2f})")
            if high_corr_pairs:
                insights.append({
                    'type': 'high_correlation',
                    'severity': 'medium',
                    'message': f"High correlation detected: {', '.join(high_corr_pairs[:3])}",
                    'suggestion': 'Consider removing one of each highly correlated pair'
                })
        
        # High missing values
        high_missing = [(col, stats['missing_percentage']) 
                        for col, stats in column_stats.items() 
                        if stats.get('missing_percentage', 0) > 50]
        if high_missing:
            insights.append({
                'type': 'high_missing',
                'severity': 'high',
                'message': f"Columns with >50% missing: {', '.join([c for c, _ in high_missing[:3]])}",
                'suggestion': 'Consider imputation or dropping these columns'
            })
        
        # Imbalanced target suggestion
        for col in categorical_cols:
            if df[col].nunique() == 2:
                value_counts = df[col].value_counts(normalize=True)
                if value_counts.min() < 0.1:
                    insights.append({
                        'type': 'imbalanced_class',
                        'severity': 'medium',
                        'message': f"'{col}' appears to be an imbalanced binary target",
                        'suggestion': 'Consider using class weights or resampling techniques'
                    })
        
        profile['insights'] = insights
        
        return profile
    
    @staticmethod
    async def detect_task_type(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Detect task type for target column."""
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        target = df[target_column]
        unique_count = target.nunique()
        total_count = len(target)
        unique_ratio = unique_count / total_count
        
        # Detect dtype
        dtype = str(target.dtype)
        
        # Classification detection
        if dtype == 'object' or dtype == 'category':
            task_type = 'classification'
            reason = f"Target is categorical (dtype: {dtype})"
            suggested_algorithms = ['RandomForest', 'GBM', 'XGBoost', 'GLM']
        elif dtype.startswith('int') and unique_count < 20 and unique_ratio < 0.05:
            task_type = 'classification'
            reason = f"Target is integer with {unique_count} unique values ({unique_ratio*100:.1f}% of data)"
            suggested_algorithms = ['RandomForest', 'GBM', 'XGBoost', 'GLM']
        else:
            task_type = 'regression'
            reason = f"Target is numeric with {unique_count} unique values"
            suggested_algorithms = ['GBM', 'XGBoost', 'RandomForest', 'GLM']
        
        return {
            'target_column': target_column,
            'task_type': task_type,
            'confidence': 0.95 if task_type == 'classification' else 0.9,
            'reason': reason,
            'unique_count': int(unique_count),
            'unique_ratio': float(unique_ratio),
            'suggested_algorithms': suggested_algorithms,
        }
    
    @staticmethod
    async def create_dataset(
        db: AsyncSession,
        name: str,
        description: Optional[str],
        file_path: str,
        created_by: uuid.UUID
    ) -> Dataset:
        """Create dataset record."""
        # Load and analyze
        df = DatasetService.load_dataset(file_path)
        columns = DatasetService.detect_schema(df)
        
        # Get file info
        file_size = os.path.getsize(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()
        file_type = 'csv' if file_ext == '.csv' else 'xlsx' if file_ext in ['.xlsx', '.xls'] else 'json'
        
        # Create dataset
        dataset = Dataset(
            name=name,
            description=description,
            file_path=file_path,
            file_type=file_type,
            file_size_bytes=file_size,
            row_count=len(df),
            column_count=len(columns),
            columns=columns,
            created_by=created_by,
        )
        
        db.add(dataset)
        await db.commit()
        await db.refresh(dataset)
        
        # Generate and save profile
        profile_data = DatasetService.generate_profile(df)
        profile = DatasetProfile(
            dataset_id=dataset.id,
            **profile_data
        )
        db.add(profile)
        await db.commit()
        
        return dataset
    
    @staticmethod
    async def get_dataset(db: AsyncSession, dataset_id: uuid.UUID) -> Optional[Dataset]:
        """Get dataset by ID."""
        result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_datasets(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[Dataset]:
        """Get all datasets."""
        result = await db.execute(
            select(Dataset).order_by(Dataset.created_at.desc()).offset(skip).limit(limit)
        )
        return result.scalars().all()
    
    @staticmethod
    async def delete_dataset(db: AsyncSession, dataset_id: uuid.UUID) -> bool:
        """Delete dataset."""
        dataset = await DatasetService.get_dataset(db, dataset_id)
        if not dataset:
            return False
        
        # Delete file
        try:
            if os.path.exists(dataset.file_path):
                os.remove(dataset.file_path)
        except Exception:
            pass
        
        # Delete from database
        await db.execute(delete(Dataset).where(Dataset.id == dataset_id))
        await db.commit()
        
        return True
