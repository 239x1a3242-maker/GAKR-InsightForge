"""
Advanced Data Service: Universal loader → Auto-profiling → ML-ready preprocessing
Supports CSV, Excel, JSON, TSV, Parquet, Feather, XML
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple, Any, Optional
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

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
    "csv": pd.read_csv,
    "tsv": lambda p: pd.read_csv(p, sep="\t"),
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "json": pd.read_json,
    "parquet": pd.read_parquet,
    "feather": pd.read_feather,
    "xml": pd.read_xml,
}


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


def detect_column_types(df: pd.DataFrame) -> Tuple[list, list, list]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    return numeric_cols, categorical_cols, datetime_cols


def generate_profile_summary(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols, categorical_cols, datetime_cols = detect_column_types(df)
    summary: Dict[str, Any] = {
        "shape": list(df.shape),
        "rows": df.shape[0],
        "columns": df.shape[1],
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        "missing_count": int(df.isna().sum().sum()),
        "missing_pct": round(df.isna().sum().sum() / df.size * 100, 2),
        "duplicates": int(df.duplicated().sum()),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
    }
    if numeric_cols:
        summary["describe_numeric"] = df[numeric_cols].describe().round(2).to_dict()
    if categorical_cols:
        summary["describe_categorical"] = df[categorical_cols].describe(include="object").to_dict()
    return summary


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().reset_index(drop=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    constant_cols = [c for c in numeric_cols if df[c].nunique() <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)
    return df


def preprocess_data(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    test_size: float = 0.2,
) -> Dict[str, Any]:
    numeric_cols, categorical_cols, _ = detect_column_types(df)

    if target_column and target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df.copy()
        y = None

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])

    X_processed = preprocessor.fit_transform(X)

    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"f_{i}" for i in range(X_processed.shape[1])]

    result: Dict[str, Any] = {
        "X_processed": X_processed,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
    }

    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=42
        )
        result.update(
            {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            }
        )

    return result


def generate_eda_report(df: pd.DataFrame, output_path: str) -> bool:
    if sv is None:
        # Sweetviz not installed; skip EDA report
        return False
    try:
        report = sv.analyze(df)
        report.show_html(output_path, open_browser=False)
        return True
    except Exception:
        return False



def process_dataset(
    file_path: str,
    target_column: Optional[str] = None,
    generate_html_report: bool = False,
) -> Dict[str, Any]:
    df_raw = load_dataset(file_path)
    df_clean = clean_data(df_raw)
    profile_summary = generate_profile_summary(df_clean)
    processed_data = preprocess_data(df_clean, target_column)

    result: Dict[str, Any] = {
        "raw_dataframe": df_raw,
        "cleaned_dataframe": df_clean,
        "processed_data": processed_data,
        "profile_summary": profile_summary,
        "preprocessor": processed_data["preprocessor"],
    }

    if generate_html_report:
        eda_path = f"eda_report_{Path(file_path).stem}.html"
        if generate_eda_report(df_clean, eda_path):
            result["eda_report_path"] = eda_path

    return result
