"""
Advanced Data Service: Universal loader → Auto-profiling → ML-ready preprocessing
Supports CSV, Excel, JSON, TSV, Parquet, Feather, XML - FIXED COLUMN ISSUE
Production-grade with schema validation, data quality checks, and comprehensive logging
"""

import pandas as pd
import numpy as np
import os
import logging
import warnings
from typing import Dict, Tuple, Any, Optional, List
from pathlib import Path
from collections import Counter
import re
import hashlib
import json
from datetime import datetime

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional profiling libraries (pipeline still works if missing)
try:
    from ydata_profiling import ProfileReport
except ImportError:
    ProfileReport = None

try:
    import sweetviz as sv
except ImportError:
    sv = None

FORMAT_LOADERS = {
    "csv": lambda p: load_csv_with_encoding_detection(p),
    "tsv": lambda p: load_csv_with_encoding_detection(p, sep="\t"),
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "json": pd.read_json,
    "parquet": pd.read_parquet,
    "feather": pd.read_feather,
    "xml": pd.read_xml,
}


def load_csv_with_encoding_detection(file_path: str, sep: str = ",") -> pd.DataFrame:
    """Load CSV/TSV with automatic encoding detection and fallback strategies"""
    encodings_to_try = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, sep=sep, encoding=encoding, low_memory=False)
            logger.info(f"Successfully loaded {file_path} with encoding: {encoding}")
            return df
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise ValueError(f"File {file_path} is empty")
        except Exception as e:
            logger.warning(f"Failed to load {file_path} with encoding {encoding}: {e}")
            continue
    
    raise ValueError(f"Could not load {file_path} with any of the attempted encodings: {encodings_to_try}")


def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of file for data integrity verification"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def detect_column_types_deterministic(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Deterministically detect and classify column types"""
    type_classification = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "boolean": [],
        "text": [],
        "unknown": []
    }
    
    for col in df.columns:
        # Check for boolean first (most specific)
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 2 and set(str(v).lower() for v in unique_vals).issubset({'true', 'false', '1', '0', 'yes', 'no', 'y', 'n'}):
            type_classification["boolean"].append(col)
        # Check for datetime
        elif pd.api.types.is_datetime64_any_dtype(df[col]) or df[col].astype(str).str.match(r'\d{4}-\d{2}-\d{2}').any():
            try:
                pd.to_datetime(df[col], errors='raise')
                type_classification["datetime"].append(col)
            except:
                pass
        # Check for numeric
        elif pd.api.types.is_numeric_dtype(df[col]):
            type_classification["numeric"].append(col)
        # Check for categorical (low cardinality object columns)
        elif df[col].dtype == 'object':
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio <= 0.05 or df[col].nunique() <= 20:  # Low cardinality
                type_classification["categorical"].append(col)
            else:
                type_classification["text"].append(col)
        else:
            type_classification["unknown"].append(col)
    
    return type_classification


class DataValidationError(Exception):
    """Custom exception for data validation failures"""
    pass


class DataQualityWarning(Warning):
    """Custom warning for data quality issues"""
    pass


def validate_schema_stability(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate schema stability and detect potential issues"""
    validation_results = {
        "is_valid": True,
        "warnings": [],
        "errors": [],
        "schema_info": {}
    }
    
    # Check for duplicate column names
    if df.columns.duplicated().any():
        validation_results["errors"].append("Duplicate column names detected")
        validation_results["is_valid"] = False
    
    # Check for columns with all missing values
    all_missing_cols = df.columns[df.isna().all()].tolist()
    if all_missing_cols:
        validation_results["errors"].append(f"Columns with all missing values: {all_missing_cols}")
        validation_results["is_valid"] = False
    
    # Check for columns with single unique values (constant columns)
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        validation_results["warnings"].append(f"Constant columns detected (may be removed): {constant_cols}")
    
    # Check for high cardinality categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.8:  # More than 80% unique values
            validation_results["warnings"].append(f"High cardinality column: {col} ({df[col].nunique()} unique values)")
    
    # Check for mixed data types in object columns
    for col in cat_cols:
        non_null_values = df[col].dropna()
        if len(non_null_values) > 0:
            type_counts = non_null_values.apply(type).value_counts()
            if len(type_counts) > 1:
                validation_results["warnings"].append(f"Mixed data types in column {col}: {dict(type_counts)}")
    
    validation_results["schema_info"] = {
        "total_columns": len(df.columns),
        "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
        "categorical_columns": len(cat_cols),
        "datetime_columns": len(df.select_dtypes(include=['datetime64']).columns),
        "constant_columns": constant_cols,
        "all_missing_columns": all_missing_cols
    }
    
    return validation_results


def validate_missing_data(df: pd.DataFrame, missing_thresholds: Dict[str, float] = None) -> Dict[str, Any]:
    """Validate missing data patterns and enforce thresholds"""
    if missing_thresholds is None:
        missing_thresholds = {"column_max_missing_pct": 50.0, "row_max_missing_pct": 80.0}
    
    validation_results = {
        "is_valid": True,
        "warnings": [],
        "errors": [],
        "missing_stats": {}
    }
    
    # Column-wise missing data
    col_missing_pct = (df.isna().sum() / len(df) * 100).round(2)
    high_missing_cols = col_missing_pct[col_missing_pct > missing_thresholds["column_max_missing_pct"]]
    
    if not high_missing_cols.empty:
        validation_results["errors"].append(
            f"Columns exceed missing data threshold ({missing_thresholds['column_max_missing_pct']}%): "
            f"{high_missing_cols.to_dict()}"
        )
        validation_results["is_valid"] = False
    
    # Row-wise missing data
    row_missing_pct = (df.isna().sum(axis=1) / len(df.columns) * 100)
    high_missing_rows_pct = (row_missing_pct > missing_thresholds["row_max_missing_pct"]).sum() / len(df) * 100
    
    if high_missing_rows_pct > 10:  # More than 10% of rows have high missing data
        validation_results["warnings"].append(
            f"{high_missing_rows_pct:.1f}% of rows exceed missing data threshold ({missing_thresholds['row_max_missing_pct']}%)"
        )
    
    # Check for systematic missing patterns
    missing_corr = df.isna().corr()
    high_corr_missing = []
    for i in range(len(missing_corr.columns)):
        for j in range(i+1, len(missing_corr.columns)):
            if abs(missing_corr.iloc[i, j]) > 0.8:  # High correlation in missing patterns
                high_corr_missing.append((missing_corr.columns[i], missing_corr.columns[j], missing_corr.iloc[i, j]))
    
    if high_corr_missing:
        validation_results["warnings"].append(f"Systematic missing patterns detected: {high_corr_missing[:3]}")  # Show first 3
    
    validation_results["missing_stats"] = {
        "total_missing": int(df.isna().sum().sum()),
        "total_missing_pct": round(df.isna().sum().sum() / df.size * 100, 2),
        "columns_with_missing": int((col_missing_pct > 0).sum()),
        "column_missing_pct": col_missing_pct.to_dict(),
        "rows_with_missing": int((row_missing_pct > 0).sum()),
        "row_missing_pct": row_missing_pct.describe().to_dict()
    }
    
    return validation_results


def detect_outliers(df: pd.DataFrame, method: str = "iqr", threshold: float = 1.5) -> Dict[str, Any]:
    """Detect outliers using various statistical methods"""
    outlier_results = {
        "outlier_counts": {},
        "outlier_percentages": {},
        "method_used": method
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        values = df[col].dropna()
        if len(values) < 10:  # Skip columns with too few values
            continue
            
        if method == "iqr":
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = values[(values < lower_bound) | (values > upper_bound)]
        elif method == "zscore":
            z_scores = np.abs((values - values.mean()) / values.std())
            outliers = values[z_scores > threshold]
        elif method == "mad":
            median = values.median()
            mad = np.median(np.abs(values - median))
            modified_z = 0.6745 * (values - median) / mad
            outliers = values[np.abs(modified_z) > threshold]
        else:
            continue
            
        outlier_results["outlier_counts"][col] = len(outliers)
        outlier_results["outlier_percentages"][col] = round(len(outliers) / len(values) * 100, 2)
    
    return outlier_results


def validate_target_leakage(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """Validate for potential target leakage issues"""
    leakage_results = {
        "potential_leakage": [],
        "warnings": [],
        "high_correlation_features": []
    }
    
    if target_column not in df.columns:
        return leakage_results
    
    target = df[target_column]
    
    # Check for perfect correlation (potential data leakage)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col == target_column:
            continue
        correlation = abs(df[col].corr(target))
        if correlation > 0.95:  # Very high correlation
            leakage_results["potential_leakage"].append({
                "feature": col,
                "correlation": correlation,
                "warning": "Extremely high correlation with target - possible data leakage"
            })
    
    # Check for ID-like columns that might correlate with target
    for col in df.columns:
        if col == target_column:
            continue
        
        # Check if column looks like an ID (sequential numbers, unique values)
        if df[col].dtype in ['int64', 'float64']:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.9:  # >90% unique values
                leakage_results["warnings"].append(f"ID-like column detected: {col} ({unique_ratio:.1%} unique)")
    
    # Check for timestamp columns that might indicate temporal leakage
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    if len(datetime_cols) > 0:
        leakage_results["warnings"].append(f"Datetime columns present: {datetime_cols.tolist()} - ensure no future information leakage")
    
    return leakage_results


def validate_class_distribution(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """Validate class distribution for classification tasks"""
    distribution_results = {
        "class_counts": {},
        "class_percentages": {},
        "imbalance_ratio": 1.0,
        "warnings": []
    }
    
    if target_column not in df.columns:
        return distribution_results
    
    target = df[target_column]
    value_counts = target.value_counts()
    
    distribution_results["class_counts"] = value_counts.to_dict()
    distribution_results["class_percentages"] = (value_counts / len(target) * 100).round(2).to_dict()
    
    if len(value_counts) > 1:
        distribution_results["imbalance_ratio"] = value_counts.max() / value_counts.min()
        
        # Check for severe imbalance
        if distribution_results["imbalance_ratio"] > 10:
            distribution_results["warnings"].append(f"Severe class imbalance detected (ratio: {distribution_results['imbalance_ratio']:.1f})")
        
        # Check for rare classes
        min_class_pct = value_counts.min() / len(target) * 100
        if min_class_pct < 1.0:
            distribution_results["warnings"].append(f"Rare class detected (<1% of data): {min_class_pct:.2f}%")
    
    return distribution_results


def comprehensive_data_validation(
    df: pd.DataFrame, 
    target_column: Optional[str] = None,
    validation_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run comprehensive data validation suite"""
    if validation_config is None:
        validation_config = {
            "missing_thresholds": {"column_max_missing_pct": 50.0, "row_max_missing_pct": 80.0},
            "outlier_method": "iqr",
            "outlier_threshold": 1.5,
            "fail_on_warnings": False
        }
    
    logger.info(f"Starting comprehensive data validation for dataset with shape {df.shape}")
    
    validation_report = {
        "overall_status": "PASS",
        "validation_timestamp": pd.Timestamp.now().isoformat(),
        "dataset_info": {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2)
        },
        "validation_results": {},
        "warnings": [],
        "errors": []
    }
    
    # Schema validation
    schema_validation = validate_schema_stability(df)
    validation_report["validation_results"]["schema"] = schema_validation
    
    if not schema_validation["is_valid"]:
        validation_report["overall_status"] = "FAIL"
        validation_report["errors"].extend(schema_validation["errors"])
    
    validation_report["warnings"].extend(schema_validation["warnings"])
    
    # Missing data validation
    missing_validation = validate_missing_data(df, validation_config["missing_thresholds"])
    validation_report["validation_results"]["missing_data"] = missing_validation
    
    if not missing_validation["is_valid"]:
        validation_report["overall_status"] = "FAIL"
        validation_report["errors"].extend(missing_validation["errors"])
    
    validation_report["warnings"].extend(missing_validation["warnings"])
    
    # Outlier detection
    outlier_detection = detect_outliers(df, validation_config["outlier_method"], validation_config["outlier_threshold"])
    validation_report["validation_results"]["outliers"] = outlier_detection
    
    # Target-specific validations
    if target_column:
        # Target leakage validation
        leakage_validation = validate_target_leakage(df, target_column)
        validation_report["validation_results"]["target_leakage"] = leakage_validation
        validation_report["warnings"].extend(leakage_validation["warnings"])
        
        if leakage_validation["potential_leakage"]:
            validation_report["warnings"].extend([item["warning"] for item in leakage_validation["potential_leakage"]])
        
        # Class distribution validation (for classification)
        if df[target_column].dtype in ['object', 'category'] or df[target_column].nunique() <= 20:
            class_validation = validate_class_distribution(df, target_column)
            validation_report["validation_results"]["class_distribution"] = class_validation
            validation_report["warnings"].extend(class_validation["warnings"])
    
    # Dataset size warnings
    if len(df) < 100:
        validation_report["warnings"].append(f"Very small dataset ({len(df)} rows) - results may not be reliable")
    elif len(df) < 1000:
        validation_report["warnings"].append(f"Small dataset ({len(df)} rows) - consider collecting more data")
    
    # Feature-to-row ratio warning
    feature_count = len(df.columns) - (1 if target_column else 0)
    if feature_count > len(df):
        validation_report["warnings"].append(f"More features ({feature_count}) than samples ({len(df)}) - risk of overfitting")
    
    # Final status determination
    if validation_config.get("fail_on_warnings", False) and validation_report["warnings"]:
        validation_report["overall_status"] = "FAIL"
        validation_report["errors"].append("Validation failed due to warnings (fail_on_warnings=True)")
    
    logger.info(f"Data validation completed with status: {validation_report['overall_status']}")
    if validation_report["errors"]:
        logger.error(f"Validation errors: {validation_report['errors']}")
    if validation_report["warnings"]:
        logger.warning(f"Validation warnings: {validation_report['warnings']}")
    
    return validation_report

def load_dataset(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    ext = Path(file_path).suffix[1:].lower()
    if ext not in FORMAT_LOADERS:
        raise ValueError(f"Unsupported format: .{ext}")
    df = FORMAT_LOADERS[ext](file_path)
    if df.empty:
        raise ValueError("Empty dataset loaded")
    return df

def generate_profile_summary(df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
    """Generate comprehensive dataset profile with health checks"""
    logger.info(f"Generating profile summary for dataset with shape {df.shape}")
    
    # Basic statistics
    type_classification = detect_column_types_deterministic(df)
    
    summary = {
        "shape": list(df.shape),
        "rows": df.shape[0],
        "columns": df.shape[1],
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        "data_types": df.dtypes.astype(str).to_dict(),
        "column_types": {k: len(v) for k, v in type_classification.items()},
        "missing_count": int(df.isna().sum().sum()),
        "missing_pct": round(df.isna().sum().sum() / df.size * 100, 2) if df.size > 0 else 0,
        "duplicates": int(df.duplicated().sum()),
        "duplicate_pct": round(df.duplicated().sum() / len(df) * 100, 2) if len(df) > 0 else 0,
    }
    
    # Column-wise missing statistics
    missing_by_col = df.isna().sum()
    summary["missing_by_column"] = missing_by_col.to_dict()
    summary["missing_pct_by_column"] = (missing_by_col / len(df) * 100).round(2).to_dict()
    
    # Cardinality information
    cardinality = df.nunique()
    summary["cardinality"] = cardinality.to_dict()
    summary["cardinality_ratio"] = (cardinality / len(df)).round(4).to_dict()
    
    # Numeric statistics
    if type_classification["numeric"]:
        numeric_stats = df[type_classification["numeric"]].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        summary["numeric_stats"] = numeric_stats.round(4).to_dict()
        
        # Additional numeric health checks
        numeric_health = {}
        for col in type_classification["numeric"]:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                numeric_health[col] = {
                    "zeros": int((col_data == 0).sum()),
                    "negatives": int((col_data < 0).sum()),
                    "infinites": int(np.isinf(col_data).sum()),
                    "range": [float(col_data.min()), float(col_data.max())],
                    "iqr": float(col_data.quantile(0.75) - col_data.quantile(0.25)),
                    "skewness": float(col_data.skew()),
                    "kurtosis": float(col_data.kurtosis()),
                }
        summary["numeric_health"] = numeric_health
    
    # Categorical statistics
    if type_classification["categorical"]:
        cat_stats = {}
        for col in type_classification["categorical"]:
            value_counts = df[col].value_counts()
            cat_stats[col] = {
                "unique_values": int(df[col].nunique()),
                "most_frequent": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                "most_frequent_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "least_frequent": str(value_counts.index[-1]) if len(value_counts) > 0 else None,
                "least_frequent_count": int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
                "top_5_values": value_counts.head(5).to_dict(),
            }
        summary["categorical_stats"] = cat_stats
    
    # Dataset health warnings
    health_warnings = []
    
    # Size warnings
    if df.shape[0] < 100:
        health_warnings.append("CRITICAL: Dataset has very few samples (<100) - results may not be reliable")
    elif df.shape[0] < 1000:
        health_warnings.append("WARNING: Small dataset (<1000 samples) - consider collecting more data")
    
    # Missing data warnings
    if summary["missing_pct"] > 30:
        health_warnings.append(f"CRITICAL: High missing data rate ({summary['missing_pct']}%)")
    elif summary["missing_pct"] > 10:
        health_warnings.append(f"WARNING: Moderate missing data rate ({summary['missing_pct']}%)")
    
    # Duplicate warnings
    if summary["duplicate_pct"] > 5:
        health_warnings.append(f"WARNING: High duplicate rate ({summary['duplicate_pct']}%)")
    
    # Feature-to-sample ratio warning
    feature_count = df.shape[1] - (1 if target_column else 0)
    if feature_count > df.shape[0]:
        health_warnings.append(f"WARNING: More features ({feature_count}) than samples ({df.shape[0]}) - risk of overfitting")
    
    # Target-specific checks
    if target_column and target_column in df.columns:
        target_info = analyze_target_variable(df, target_column)
        summary["target_analysis"] = target_info
        
        if "warnings" in target_info:
            health_warnings.extend(target_info["warnings"])
    
    summary["health_warnings"] = health_warnings
    summary["health_score"] = max(0, 100 - len(health_warnings) * 10)  # Simple health score
    
    logger.info(f"Profile summary generated with {len(health_warnings)} health warnings")
    return summary


def analyze_target_variable(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """Analyze target variable for classification/regression suitability"""
    target = df[target_column]
    analysis = {
        "dtype": str(target.dtype),
        "missing_count": int(target.isna().sum()),
        "unique_values": int(target.nunique()),
        "warnings": []
    }
    
    # Check for missing target values
    if analysis["missing_count"] > 0:
        analysis["warnings"].append(f"Target column has {analysis['missing_count']} missing values")
    
    # Classification analysis
    if target.dtype == 'object' or target.nunique() <= 20:
        analysis["task_type"] = "classification"
        value_counts = target.value_counts()
        analysis["class_counts"] = value_counts.to_dict()
        analysis["class_percentages"] = (value_counts / len(target) * 100).round(2).to_dict()
        
        # Class balance analysis
        if len(value_counts) > 1:
            imbalance_ratio = value_counts.max() / value_counts.min()
            analysis["imbalance_ratio"] = float(imbalance_ratio)
            
            if imbalance_ratio > 10:
                analysis["warnings"].append(f"Severe class imbalance (ratio: {imbalance_ratio:.1f})")
            elif imbalance_ratio > 5:
                analysis["warnings"].append(f"Moderate class imbalance (ratio: {imbalance_ratio:.1f})")
            
            # Rare classes
            min_class_pct = value_counts.min() / len(target) * 100
            if min_class_pct < 1.0:
                analysis["warnings"].append(f"Rare class detected (<1% of data): {min_class_pct:.2f}%")
    
    else:
        analysis["task_type"] = "regression"
        analysis["target_stats"] = target.describe().to_dict()
        
        # Check for constant target
        if target.nunique() <= 1:
            analysis["warnings"].append("Target variable is constant - cannot train regression model")
    
    return analysis

def clean_data(df: pd.DataFrame, cleaning_config: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Comprehensive data cleaning with explicit handling and audit trail"""
    if cleaning_config is None:
        cleaning_config = {
            "remove_duplicates": True,
            "handle_missing": "auto",  # auto, drop, impute
            "normalize_categories": True,
            "normalize_dates": True,
            "remove_constants": True,
            "outlier_method": "iqr",
            "outlier_threshold": 1.5
        }
    
    logger.info(f"Starting data cleaning on dataset with shape {df.shape}")
    cleaning_audit = {
        "original_shape": df.shape,
        "transformations": [],
        "removed_columns": [],
        "imputed_values": {},
        "timestamp": datetime.now().isoformat()
    }
    
    df_clean = df.copy()
    
    # 1. Remove duplicates
    if cleaning_config["remove_duplicates"]:
        original_count = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_count = original_count - len(df_clean)
        if removed_count > 0:
            cleaning_audit["transformations"].append(f"Removed {removed_count} duplicate rows")
            logger.info(f"Removed {removed_count} duplicate rows")
    
    # 2. Handle missing values explicitly
    missing_stats = df_clean.isna().sum()
    cols_with_missing = missing_stats[missing_stats > 0]
    
    if not cols_with_missing.empty:
        if cleaning_config["handle_missing"] == "drop":
            # Drop columns with too much missing data
            cols_to_drop = cols_with_missing[cols_with_missing / len(df_clean) > 0.5].index.tolist()
            if cols_to_drop:
                df_clean = df_clean.drop(columns=cols_to_drop)
                cleaning_audit["removed_columns"].extend(cols_to_drop)
                cleaning_audit["transformations"].append(f"Dropped columns with >50% missing: {cols_to_drop}")
                logger.info(f"Dropped columns with high missing rates: {cols_to_drop}")
        
        elif cleaning_config["handle_missing"] == "impute":
            # Impute missing values
            for col in cols_with_missing.index:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    # Numeric imputation
                    imputer = SimpleImputer(strategy='median')
                    df_clean[col] = imputer.fit_transform(df_clean[[col]]).ravel()
                    cleaning_audit["imputed_values"][col] = f"median imputation ({missing_stats[col]} values)"
                else:
                    # Categorical imputation
                    imputer = SimpleImputer(strategy='most_frequent')
                    df_clean[col] = imputer.fit_transform(df_clean[[col]]).ravel()
                    cleaning_audit["imputed_values"][col] = f"most_frequent imputation ({missing_stats[col]} values)"
            
            if cleaning_audit["imputed_values"]:
                cleaning_audit["transformations"].append(f"Imputed missing values in {len(cleaning_audit['imputed_values'])} columns")
                logger.info(f"Imputed missing values in {len(cleaning_audit['imputed_values'])} columns")
    
    # 3. Normalize categorical values
    if cleaning_config["normalize_categories"]:
        type_classification = detect_column_types_deterministic(df_clean)
        for col in type_classification["categorical"]:
            original_unique = df_clean[col].nunique()
            # Strip whitespace and convert to lowercase for consistency
            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
            # Remove extra whitespace within strings
            df_clean[col] = df_clean[col].str.replace(r'\s+', ' ', regex=True)
            new_unique = df_clean[col].nunique()
            if new_unique != original_unique:
                cleaning_audit["transformations"].append(f"Normalized categorical column '{col}': {original_unique} → {new_unique} unique values")
                logger.info(f"Normalized categorical column '{col}': {original_unique} → {new_unique} unique values")
    
    # 4. Normalize datetime columns
    if cleaning_config["normalize_dates"]:
        type_classification = detect_column_types_deterministic(df_clean)
        for col in type_classification["datetime"]:
            try:
                # Attempt to parse various date formats
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                # Check for successful parsing
                parsed_count = df_clean[col].notna().sum()
                original_count = len(df_clean)
                if parsed_count < original_count:
                    failed_pct = ((original_count - parsed_count) / original_count) * 100
                    cleaning_audit["transformations"].append(f"Date normalization for '{col}': {parsed_count}/{original_count} successful ({failed_pct:.1f}% failed)")
                    logger.warning(f"Date parsing failed for {failed_pct:.1f}% of values in column '{col}'")
            except Exception as e:
                logger.warning(f"Could not normalize dates in column '{col}': {e}")
    
    # 5. Remove constant columns
    if cleaning_config["remove_constants"]:
        constant_cols = []
        for col in df_clean.columns:
            if df_clean[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            df_clean = df_clean.drop(columns=constant_cols)
            cleaning_audit["removed_columns"].extend(constant_cols)
            cleaning_audit["transformations"].append(f"Removed constant columns: {constant_cols}")
            logger.info(f"Removed constant columns: {constant_cols}")
    
    # 6. Outlier detection (flagged, not removed)
    if cleaning_config["outlier_method"]:
        outlier_results = detect_outliers(df_clean, cleaning_config["outlier_method"], cleaning_config["outlier_threshold"])
        total_outliers = sum(outlier_results["outlier_counts"].values())
        if total_outliers > 0:
            cleaning_audit["transformations"].append(f"Detected {total_outliers} potential outliers (not removed)")
            cleaning_audit["outlier_summary"] = outlier_results
            logger.info(f"Detected {total_outliers} potential outliers across {len(outlier_results['outlier_counts'])} columns")
    
    cleaning_audit["final_shape"] = df_clean.shape
    cleaning_audit["rows_removed"] = cleaning_audit["original_shape"][0] - df_clean.shape[0]
    cleaning_audit["columns_removed"] = len(cleaning_audit["removed_columns"])
    
    logger.info(f"Data cleaning completed. Shape: {cleaning_audit['original_shape']} → {cleaning_audit['final_shape']}")
    
    return df_clean, cleaning_audit

def preprocess_data(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    preprocessing_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Advanced data preprocessing with feature engineering and audit trail"""
    if preprocessing_config is None:
        preprocessing_config = {
            "test_size": 0.2,
            "random_state": 42,
            "scaling": "standard",  # standard, minmax, robust, none
            "encoding": "onehot",  # onehot, ordinal, none
            "feature_selection": "auto",  # auto, none
            "rename_features": True
        }
    
    logger.info(f"Starting data preprocessing on dataset with shape {df.shape}")
    preprocessing_audit = {
        "original_shape": df.shape,
        "transformations": [],
        "feature_engineering": {},
        "timestamp": datetime.now().isoformat()
    }
    
    # Separate target if specified
    if target_column and target_column in df.columns:
        logger.info(f"Separating target column: {target_column}")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        preprocessing_audit["transformations"].append(f"Separated target column '{target_column}'")
    else:
        X = df.copy()
        y = None
        if target_column:
            logger.warning(f"Target column '{target_column}' not found in dataset")
    
    # Detect column types
    type_classification = detect_column_types_deterministic(X)
    logger.info(f"Detected column types: { {k: len(v) for k, v in type_classification.items()} }")
    
    # Feature pruning: remove high-missing or constant features
    cols_to_remove = []
    
    # Remove columns with all missing values
    all_missing = X.columns[X.isna().all()].tolist()
    if all_missing:
        cols_to_remove.extend(all_missing)
        preprocessing_audit["transformations"].append(f"Removed all-missing columns: {all_missing}")
    
    # Remove constant columns
    for col in X.columns:
        if col not in cols_to_remove and X[col].nunique() <= 1:
            cols_to_remove.append(col)
    
    if cols_to_remove:
        X = X.drop(columns=cols_to_remove)
        preprocessing_audit["transformations"].append(f"Removed {len(cols_to_remove)} problematic columns: {cols_to_remove}")
        logger.info(f"Removed {len(cols_to_remove)} problematic columns")
    
    # Re-detect types after pruning
    type_classification = detect_column_types_deterministic(X)
    
    # Build preprocessing pipelines
    transformers = []
    
    # Numeric preprocessing
    if type_classification["numeric"]:
        if preprocessing_config["scaling"] == "standard":
            scaler = StandardScaler()
        elif preprocessing_config["scaling"] == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif preprocessing_config["scaling"] == "robust":
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            scaler = None
        
        if scaler:
            numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", scaler),
            ])
            transformers.append(("num", numeric_pipeline, type_classification["numeric"]))
            preprocessing_audit["transformations"].append(f"Applied {preprocessing_config['scaling']} scaling to {len(type_classification['numeric'])} numeric columns")
    
    # Categorical preprocessing
    if type_classification["categorical"]:
        if preprocessing_config["encoding"] == "onehot":
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop='if_binary')
        elif preprocessing_config["encoding"] == "ordinal":
            from sklearn.preprocessing import OrdinalEncoder
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        else:
            encoder = None
        
        if encoder:
            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", encoder),
            ])
            transformers.append(("cat", categorical_pipeline, type_classification["categorical"]))
            preprocessing_audit["transformations"].append(f"Applied {preprocessing_config['encoding']} encoding to {len(type_classification['categorical'])} categorical columns")
    
    # Boolean preprocessing (treat as numeric 0/1)
    if type_classification["boolean"]:
        bool_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ])
        transformers.append(("bool", bool_pipeline, type_classification["boolean"]))
        preprocessing_audit["transformations"].append(f"Processed {len(type_classification['boolean'])} boolean columns")
    
    if not transformers:
        raise ValueError("No valid columns found for preprocessing")
    
    # Create and fit preprocessor
    preprocessor = ColumnTransformer(transformers, remainder='drop')
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"f_{i}" for i in range(X_processed.shape[1])]
    
    # Feature renaming for consistency
    if preprocessing_config["rename_features"]:
        original_names = feature_names.copy()
        # Clean feature names: remove special chars, standardize
        feature_names = [re.sub(r'[^\w]', '_', name).lower() for name in feature_names]
        if feature_names != original_names:
            preprocessing_audit["transformations"].append("Renamed features for consistency")
    
    preprocessing_audit["final_feature_count"] = len(feature_names)
    preprocessing_audit["feature_types"] = {k: len(v) for k, v in type_classification.items()}
    
    result = {
        "X_processed": X_processed,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "column_types": type_classification,
        "preprocessing_audit": preprocessing_audit,
    }
    
    # Train/test split if target exists
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, 
            test_size=preprocessing_config["test_size"], 
            random_state=preprocessing_config["random_state"],
            stratify=y if hasattr(y, 'nunique') and y.nunique() <= 20 else None
        )
        result.update({
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        })
        preprocessing_audit["transformations"].append(f"Created train/test split: {len(X_train)}/{len(X_test)} samples")
        logger.info(f"Created train/test split: {len(X_train)}/{len(X_test)} samples")
    
    logger.info(f"Preprocessing completed. Final shape: {X_processed.shape}")
    return result

def process_dataset(
    file_path: str,
    target_column: Optional[str] = None,
    generate_html_report: bool = False,
    validation_config: Optional[Dict[str, Any]] = None,
    cleaning_config: Optional[Dict[str, Any]] = None,
    preprocessing_config: Optional[Dict[str, Any]] = None,
    fail_on_validation_error: bool = True
) -> Dict[str, Any]:
    """End-to-end dataset processing with comprehensive validation, cleaning, and preprocessing"""
    start_time = datetime.now()
    logger.info(f"🚀 Starting comprehensive dataset processing: {file_path}")
    
    audit_trail = {
        "file_path": str(file_path),
        "file_hash": calculate_file_hash(file_path),
        "processing_start": start_time.isoformat(),
        "steps_completed": [],
        "warnings": [],
        "errors": []
    }
    
    try:
        # Step 1: Data Ingestion
        logger.info("📥 Step 1: Data Ingestion")
        df_raw = load_dataset(file_path)
        audit_trail["steps_completed"].append("data_ingestion")
        audit_trail["raw_shape"] = df_raw.shape
        audit_trail["raw_columns"] = df_raw.columns.tolist()
        logger.info(f"✅ Loaded dataset: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
        
        # Step 2: Comprehensive Data Validation
        logger.info("🔍 Step 2: Data Validation")
        validation_report = comprehensive_data_validation(df_raw, target_column, validation_config)
        audit_trail["steps_completed"].append("data_validation")
        audit_trail["validation_status"] = validation_report["overall_status"]
        
        # Fail fast on validation errors if requested
        if fail_on_validation_error and validation_report["overall_status"] == "FAIL":
            error_msg = f"❌ Data validation failed: {validation_report['errors']}"
            audit_trail["errors"].extend(validation_report["errors"])
            logger.error(error_msg)
            raise DataValidationError(error_msg)
        
        # Collect validation warnings
        audit_trail["warnings"].extend(validation_report["warnings"])
        
        # Step 3: Data Cleaning
        logger.info("🧹 Step 3: Data Cleaning")
        df_clean, cleaning_audit = clean_data(df_raw, cleaning_config)
        audit_trail["steps_completed"].append("data_cleaning")
        audit_trail["cleaning_audit"] = cleaning_audit
        logger.info(f"✅ Data cleaned: {cleaning_audit['original_shape']} → {cleaning_audit['final_shape']}")
        
        # Step 4: Dataset Profiling
        logger.info("📊 Step 4: Dataset Profiling")
        profile_summary = generate_profile_summary(df_clean, target_column)
        audit_trail["steps_completed"].append("dataset_profiling")
        audit_trail["profile_summary"] = profile_summary
        logger.info(f"✅ Profile generated with health score: {profile_summary.get('health_score', 'N/A')}")
        
        # Step 5: Data Preprocessing
        logger.info("⚙️ Step 5: Data Preprocessing")
        processed_data = preprocess_data(df_clean, target_column, preprocessing_config)
        audit_trail["steps_completed"].append("data_preprocessing")
        audit_trail["preprocessing_audit"] = processed_data.get("preprocessing_audit", {})
        logger.info(f"✅ Preprocessing completed: {processed_data['X_processed'].shape[1]} features generated")
        
        # Step 6: Generate Reports (optional)
        eda_report_path = None
        if generate_html_report:
            logger.info("📈 Step 6: Generating EDA Report")
            eda_report_path = f"eda_report_{Path(file_path).stem}_{int(start_time.timestamp())}.html"
            if generate_eda_report(df_clean, eda_report_path):
                audit_trail["steps_completed"].append("eda_report_generation")
                logger.info(f"✅ EDA report generated: {eda_report_path}")
            else:
                logger.warning("❌ EDA report generation failed")
        
        # Final audit trail completion
        end_time = datetime.now()
        audit_trail["processing_end"] = end_time.isoformat()
        audit_trail["total_duration_seconds"] = (end_time - start_time).total_seconds()
        audit_trail["final_status"] = "SUCCESS"
        
        logger.info(f"🎉 Dataset processing completed successfully in {audit_trail['total_duration_seconds']:.2f} seconds")
        
        # Compile final result
        result = {
            "raw_dataframe": df_raw,
            "cleaned_dataframe": df_clean,
            "processed_data": processed_data,
            "profile_summary": profile_summary,
            "validation_report": validation_report,
            "cleaning_audit": cleaning_audit,
            "audit_trail": audit_trail,
            "preprocessor": processed_data["preprocessor"],
        }
        
        if eda_report_path:
            result["eda_report_path"] = eda_report_path
        
        return result
        
    except Exception as e:
        # Error handling and audit trail completion
        end_time = datetime.now()
        audit_trail["processing_end"] = end_time.isoformat()
        audit_trail["total_duration_seconds"] = (end_time - start_time).total_seconds()
        audit_trail["final_status"] = "FAILED"
        audit_trail["fatal_error"] = str(e)
        
        logger.error(f"❌ Dataset processing failed: {e}")
        raise
