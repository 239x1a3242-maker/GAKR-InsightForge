"""Datasets API (File-based)."""
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from typing import List, Optional, Dict, Any
import csv
import json
from pathlib import Path
import pandas as pd

from app.api.auth import get_current_user
from app.core.filedb import DatasetDB, UPLOADS_DIR

router = APIRouter()


def load_dataset_data(file_path: Path) -> tuple[List[Dict[str, Any]], int]:
    """Load dataset from CSV, Excel, or JSON file with multiple encoding fallbacks."""
    try:
        suffix = file_path.suffix.lower()
        if suffix == '.csv':
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            for enc in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    break
                except (UnicodeDecodeError, Exception):
                    continue
            if df is None:
                raise ValueError("Could not decode CSV file")
        elif suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                import json as _json
                data = _json.load(f)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.json_normalize(data)
            else:
                raise ValueError("Unsupported JSON structure")
        else:
            raise ValueError("Unsupported file format")

        rows = df.to_dict('records')
        return rows, len(rows)
    except Exception as e:
        raise ValueError(f"Failed to read file: {str(e)}")


@router.post("/upload")
async def upload_dataset(
    name: str = Form(...),
    description: Optional[str] = Form(None),
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Upload a dataset."""
    # Validate file type
    allowed_types = ['.csv', '.json', '.xlsx']
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )
    
    try:
        # Save file to uploads directory
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        user_upload_dir = UPLOADS_DIR / current_user["id"] / "datasets"
        user_upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        file_path = user_upload_dir / file.filename
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Load and parse data
        rows, row_count = load_dataset_data(file_path)
        
        # Create dataset record
        dataset = DatasetDB.create(
            user_id=current_user["id"],
            name=name or file.filename,
            filename=file.filename,
            rows=row_count,
            columns=len(rows[0]) if rows else 0,
            file_size=len(content)
        )
        
        # Save dataset data
        DatasetDB.save_data(dataset["id"], rows)
        
        # construct response matching frontend dataset schema
        # determine source_type from filename
        file_ext2 = Path(dataset.get("filename", "")).suffix.lower().lstrip('.')
        source_type = file_ext2 if file_ext2 in ['csv', 'json', 'xlsx'] else 'csv'
        
        # build column metadata
        columns_meta = _build_columns_meta(rows)
        
        return {
            "id": dataset["id"],
            "name": dataset["name"],
            "description": dataset.get("description", ""),
            "owner_id": current_user["id"],
            "source_type": source_type,
            "columns": columns_meta,
            "row_count": dataset.get("rows", 0),
            "column_count": len(columns_meta),
            "size_bytes": dataset.get("file_size", 0),
            "refresh_type": "manual",
            "last_refresh": dataset.get("updated_at"),
            "refresh_status": "completed",
            "is_active": True,
            "is_shared": False,
            "created_at": dataset.get("created_at", ""),
            "updated_at": dataset.get("updated_at", ""),
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process dataset: {str(e)}"
        )


@router.get("")
async def list_datasets(
    skip: int = 0,
    limit: int = 100,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """List all datasets for current user."""
    user_datasets = DatasetDB.get_by_user(current_user["id"])
    
    # Apply pagination
    paginated = user_datasets[skip:skip + limit]
    
    # Transform datasets to match frontend expectations
    formatted_datasets = []
    for dataset in paginated:
        # Determine source_type from filename
        file_ext = Path(dataset.get("filename", "")).suffix.lower().lstrip('.')
        source_type = file_ext if file_ext in ['csv', 'json', 'xlsx'] else 'csv'
        
        # Get column names from saved data
        data_rows = DatasetDB.get_data(dataset["id"]) or []
        columns = _build_columns_meta(data_rows)
        
        formatted_datasets.append({
            "id": dataset["id"],
            "name": dataset["name"],
            "description": dataset.get("description", ""),
            "owner_id": dataset.get("user_id", ""),
            "source_type": source_type,
            "columns": columns,
            "row_count": dataset.get("rows", 0),
            "column_count": len(columns),
            "size_bytes": dataset.get("file_size", 0),
            "refresh_type": "manual",
            "last_refresh": dataset.get("updated_at", ""),
            "refresh_status": "completed",
            "is_active": True,
            "is_shared": False,
            "created_at": dataset.get("created_at", ""),
            "updated_at": dataset.get("updated_at", "")
        })
    
    return {
        "items": formatted_datasets,
        "total": len(user_datasets),
        "skip": skip,
        "limit": limit
    }


@router.get("/{dataset_id}")
async def get_dataset(
    dataset_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get dataset by ID."""
    dataset = DatasetDB.get_by_id(dataset_id)
    
    if not dataset or dataset.get("user_id") != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    # Determine source_type from filename
    file_ext = Path(dataset.get("filename", "")).suffix.lower().lstrip('.')
    source_type = file_ext if file_ext in ['csv', 'json', 'xlsx'] else 'csv'
    
    # Get column names from saved data
    data_rows = DatasetDB.get_data(dataset["id"]) or []
    columns = _build_columns_meta(data_rows)
    
    return {
        "id": dataset["id"],
        "name": dataset["name"],
        "description": dataset.get("description", ""),
        "owner_id": dataset.get("user_id", ""),
        "source_type": source_type,
        "columns": columns,
        "row_count": dataset.get("rows", 0),
        "column_count": len(columns),
        "size_bytes": dataset.get("file_size", 0),
        "refresh_type": "manual",
        "last_refresh": dataset.get("updated_at", ""),
        "refresh_status": "completed",
        "is_active": True,
        "is_shared": False,
        "created_at": dataset.get("created_at", ""),
        "updated_at": dataset.get("updated_at", "")
    }


@router.get("/{dataset_id}/preview")
async def get_dataset_preview(
    dataset_id: str,
    page: int = 1,
    page_size: int = 10,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get dataset preview with pagination matching frontend contract."""
    dataset = DatasetDB.get_by_id(dataset_id)
    
    if not dataset or dataset.get("user_id") != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    rows = DatasetDB.get_data(dataset_id) or []
    total_rows = len(rows)
    # compute slice based on page/page_size
    start = (page - 1) * page_size
    end = start + page_size
    preview_rows = rows[start:end]

    columns = _build_columns_meta(rows)

    return {
        "columns": columns,
        "data": preview_rows,
        "total_rows": total_rows,
        "page": page,
        "page_size": page_size
    }


@router.get("/{dataset_id}/data")
async def get_dataset_data(
    dataset_id: str,
    skip: int = 0,
    limit: int = 100,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get full dataset data with pagination."""
    dataset = DatasetDB.get_by_id(dataset_id)
    
    if not dataset or dataset.get("user_id") != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    rows = DatasetDB.get_data(dataset_id)
    paginated = rows[skip:skip + limit] if rows else []
    
    return {
        "id": dataset_id,
        "name": dataset["name"],
        "rows": paginated,
        "total_rows": len(rows) if rows else 0,
        "skip": skip,
        "limit": limit
    }


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Delete dataset."""
    dataset = DatasetDB.get_by_id(dataset_id)
    
    if not dataset or dataset.get("user_id") != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    DatasetDB.delete(dataset_id)
    
    return {"message": "Dataset deleted successfully"}


@router.get("/{dataset_id}/schema")
async def get_dataset_schema(
    dataset_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get dataset schema (column information)."""
    dataset = DatasetDB.get_by_id(dataset_id)
    
    if not dataset or dataset.get("user_id") != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    data_rows = DatasetDB.get_data(dataset_id) or []
    columns = _build_columns_meta(data_rows)
    
    return {
        "id": dataset_id,
        "columns": columns,
        "total_columns": len(columns),
        "total_rows": len(data_rows)
    }


@router.get("/{dataset_id}/profile")
async def get_dataset_profile(
    dataset_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get dataset profile (statistical info)."""
    dataset = DatasetDB.get_by_id(dataset_id)
    
    if not dataset or dataset.get("user_id") != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    data_rows = DatasetDB.get_data(dataset_id) or []
    if not data_rows:
        return {
            "total_rows": 0,
            "total_columns": 0,
            "numeric_columns": 0,
            "categorical_columns": 0,
            "memory_usage_mb": 0,
            "missing_percentage": 0,
            "duplicate_rows": 0,
            "insights": [],
            "column_stats": {}
        }

    import numpy as np
    df = pd.DataFrame(data_rows)
    numeric_df = df.select_dtypes(include=[np.number])
    categorical_df = df.select_dtypes(exclude=[np.number])

    total_cells = len(df) * len(df.columns)
    total_missing = int(df.isna().sum().sum())
    missing_pct = round(total_missing / max(total_cells, 1) * 100, 2)
    duplicate_rows = int(df.duplicated().sum())

    # Column stats
    column_stats: Dict[str, Any] = {}
    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        stat: Dict[str, Any] = {
            "dtype": str(series.dtype),
            "missing": missing,
            "missing_percentage": round(missing / len(df) * 100, 2) if len(df) > 0 else 0,
            "unique": int(series.nunique()),
        }
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean) > 0:
                stat.update({
                    "mean": round(float(clean.mean()), 4),
                    "std": round(float(clean.std()), 4),
                    "min": round(float(clean.min()), 4),
                    "max": round(float(clean.max()), 4),
                    "median": round(float(clean.median()), 4),
                })
        else:
            top_vals = series.value_counts().head(5).to_dict()
            stat["top_values"] = {str(k): int(v) for k, v in top_vals.items()}
        column_stats[col] = stat

    # Insights
    insights = []
    if missing_pct > 5:
        insights.append({"message": f"{missing_pct}% of values are missing", "severity": "medium", "suggestion": "Consider imputing or removing columns with high missing rates"})
    if duplicate_rows > 0:
        insights.append({"message": f"{duplicate_rows} duplicate rows detected", "severity": "low", "suggestion": "Consider deduplicating the dataset"})
    high_missing = [c for c, s in column_stats.items() if s.get("missing_percentage", 0) > 30]
    if high_missing:
        insights.append({"message": f"Columns with >30% missing: {', '.join(high_missing[:5])}", "severity": "high", "suggestion": "Consider dropping these columns"})

    # Correlation matrix (numeric only)
    correlation_matrix = None
    if len(numeric_df.columns) >= 2:
        corr = numeric_df.iloc[:, :20].corr().round(4)
        correlation_matrix = {col: corr[col].to_dict() for col in corr.columns}

    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "numeric_columns": len(numeric_df.columns),
        "categorical_columns": len(categorical_df.columns),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 4),
        "missing_percentage": missing_pct,
        "duplicate_rows": duplicate_rows,
        "insights": insights,
        "column_stats": column_stats,
        "correlation_matrix": correlation_matrix,
    }


@router.get("/{dataset_id}/quality")
async def get_dataset_quality(
    dataset_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get dataset quality assessment."""
    dataset = DatasetDB.get_by_id(dataset_id)
    
    if not dataset or dataset.get("user_id") != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    return {
        "issues": [],
        "quality_score": 100,
        "recommendations": []
    }


@router.post("/{dataset_id}/refresh")
async def refresh_dataset(
    dataset_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Refresh dataset data."""
    dataset = DatasetDB.get_by_id(dataset_id)
    
    if not dataset or dataset.get("user_id") != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    return {"message": "Dataset refresh started", "status": "pending"}


@router.post("/{dataset_id}/relationships")
async def create_relationship(
    dataset_id: str,
    data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Create relationship between datasets."""
    dataset = DatasetDB.get_by_id(dataset_id)
    
    if not dataset or dataset.get("user_id") != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    return {"id": "rel_placeholder", "status": "created"}


@router.post("/{dataset_id}/calculated-fields")
async def create_calculated_field(
    dataset_id: str,
    data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Create calculated field."""
    dataset = DatasetDB.get_by_id(dataset_id)
    
    if not dataset or dataset.get("user_id") != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    return {"id": "cf_placeholder", "name": data.get("name", ""), "status": "created"}


@router.post("/{dataset_id}/measures")
async def create_measure(
    dataset_id: str,
    data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Create measure."""
    dataset = DatasetDB.get_by_id(dataset_id)
    
    if not dataset or dataset.get("user_id") != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    return {"id": "m_placeholder", "name": data.get("name", ""), "status": "created"}


def _detect_column_type(series: pd.Series) -> str:
    """Detect column type for frontend display."""
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    # Try datetime detection
    if series.dtype == object:
        sample = series.dropna().head(20)
        try:
            pd.to_datetime(sample, errors='raise')
            return "datetime"
        except Exception:
            pass
    return "string"


def _build_columns_meta(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build column metadata with proper type detection."""
    if not rows:
        return []
    df = pd.DataFrame(rows[:100])  # sample for type detection
    return [
        {
            "name": col,
            "type": _detect_column_type(df[col]),
            "nullable": bool(df[col].isna().any()),
            "description": "",
        }
        for col in df.columns
    ]


@router.post("/{dataset_id}/clean")
async def clean_dataset(
    dataset_id: str,
    data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Clean dataset and save as new dataset."""
    import numpy as np

    dataset = DatasetDB.get_by_id(dataset_id)
    if not dataset or dataset.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    rows = DatasetDB.get_data(dataset_id) or []
    if not rows:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Dataset has no data")

    df = pd.DataFrame(rows)
    original_rows = len(df)

    remove_missing = data.get("remove_missing", False)
    fill_strategy = data.get("fill_missing_strategy", "")
    remove_duplicates = data.get("remove_duplicates", False)
    remove_outliers = data.get("remove_outliers", False)
    new_name = data.get("new_name", f"{dataset.get('name', 'dataset')}_cleaned")

    # Apply cleaning steps
    if remove_missing:
        df = df.dropna()

    if fill_strategy in ("mean", "median", "mode"):
        for col in df.select_dtypes(include=[np.number]).columns:
            if fill_strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif fill_strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif fill_strategy == "mode":
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])

    if remove_duplicates:
        df = df.drop_duplicates()

    if remove_outliers:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    cleaned_rows = len(df)
    rows_removed = original_rows - cleaned_rows

    # Save as new dataset
    new_dataset = DatasetDB.create(
        user_id=current_user["id"],
        name=new_name,
        filename=f"{new_name}.csv",
        rows=cleaned_rows,
        columns=len(df.columns),
        file_size=0,
    )
    DatasetDB.save_data(new_dataset["id"], df.to_dict("records"))

    return {
        "original_rows": original_rows,
        "cleaned_rows": cleaned_rows,
        "rows_removed": rows_removed,
        "new_dataset_id": new_dataset["id"],
        "new_dataset_name": new_name,
    }

