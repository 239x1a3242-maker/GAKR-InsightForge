"""Query API (File-based, pandas-based)."""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from app.api.auth import get_current_user
from app.core.filedb import DatasetDB

router = APIRouter()


class QueryFilter(BaseModel):
    column: str
    operator: str  # eq, neq, gt, lt, gte, lte, contains, in
    value: Any


class QuerySort(BaseModel):
    column: str
    direction: str = "asc"


class QueryRequest(BaseModel):
    dataset_id: str
    columns: Optional[List[str]] = None
    fields: Optional[List[str]] = None  # frontend alias for columns
    filters: Optional[List[QueryFilter]] = None
    sort: Optional[List[QuerySort]] = None
    limit: int = 1000
    offset: int = 0
    page: Optional[int] = None
    page_size: Optional[int] = None
    aggregations: Optional[List[Dict[str, Any]]] = None
    group_by: Optional[List[str]] = None


def _apply_filter(df, f: QueryFilter):
    import pandas as pd
    col = f.column
    val = f.value
    op = f.operator

    if col not in df.columns:
        return df

    series = df[col]

    if op == "eq":
        return df[series == val]
    elif op == "neq":
        return df[series != val]
    elif op == "gt":
        return df[pd.to_numeric(series, errors='coerce') > float(val)]
    elif op == "lt":
        return df[pd.to_numeric(series, errors='coerce') < float(val)]
    elif op == "gte":
        return df[pd.to_numeric(series, errors='coerce') >= float(val)]
    elif op == "lte":
        return df[pd.to_numeric(series, errors='coerce') <= float(val)]
    elif op == "contains":
        return df[series.astype(str).str.contains(str(val), case=False, na=False)]
    elif op == "in":
        return df[series.isin(val if isinstance(val, list) else [val])]
    return df


@router.post("/execute")
async def execute_query(
    request: QueryRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    import pandas as pd
    import numpy as np
    import time

    dataset = DatasetDB.get_by_id(request.dataset_id)
    if not dataset or dataset.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    rows = DatasetDB.get_data(request.dataset_id) or []
    if not rows:
        return {"columns": [], "data": [], "rows": [], "total_rows": 0, "execution_time_ms": 0, "page": 1, "page_size": request.limit}

    start = time.time()
    df = pd.DataFrame(rows)

    # Resolve limit/offset from page/page_size if provided
    limit = request.limit
    offset = request.offset
    if request.page_size:
        limit = request.page_size
    if request.page and request.page_size:
        offset = (request.page - 1) * request.page_size

    # Select columns (support both 'columns' and 'fields' from frontend)
    col_select = request.columns or request.fields
    if col_select:
        valid_cols = [c for c in col_select if c in df.columns]
        if valid_cols:
            df = df[valid_cols]

    # Apply filters
    if request.filters:
        for f in request.filters:
            df = _apply_filter(df, f)

    # Group by + aggregations
    if request.group_by and request.aggregations:
        valid_groups = [c for c in request.group_by if c in df.columns]
        if valid_groups:
            agg_dict = {}
            for agg in request.aggregations:
                col = agg.get("column")
                func = agg.get("function", "sum")
                if col and col in df.columns:
                    agg_dict[col] = func
            if agg_dict:
                df = df.groupby(valid_groups).agg(agg_dict).reset_index()

    # Sort
    if request.sort:
        sort_cols = [s.column for s in request.sort if s.column in df.columns]
        sort_asc = [s.direction == "asc" for s in request.sort if s.column in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=sort_asc)

    total_rows = len(df)

    # Paginate
    df = df.iloc[offset: offset + limit]

    # Convert to JSON-safe format
    df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
    result_rows = df.to_dict("records")

    elapsed_ms = int((time.time() - start) * 1000)

    return {
        "columns": list(df.columns),
        "data": result_rows,
        "rows": result_rows,  # backward compat
        "total_rows": total_rows,
        "execution_time_ms": elapsed_ms,
        "page": (offset // limit) + 1 if limit else 1,
        "page_size": limit,
        "offset": offset,
        "limit": limit,
    }


@router.get("/{dataset_id}/distinct/{column}")
async def get_distinct_values(
    dataset_id: str,
    column: str,
    limit: int = 1000,
    search: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    dataset = DatasetDB.get_by_id(dataset_id)
    if not dataset or dataset.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    rows = DatasetDB.get_data(dataset_id) or []
    if not rows:
        return {"column": column, "values": [], "count": 0}

    import pandas as pd
    df = pd.DataFrame(rows)
    if column not in df.columns:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Column '{column}' not found")

    values = df[column].dropna().astype(str).unique().tolist()
    if search:
        values = [v for v in values if search.lower() in v.lower()]

    values = sorted(values)[:limit]
    return {"column": column, "values": values, "count": len(values)}


@router.post("/time-intelligence")
async def time_intelligence(
    data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    import pandas as pd

    dataset_id = data.get("dataset_id")
    date_column = data.get("date_column")
    measure = data.get("measure")

    dataset = DatasetDB.get_by_id(dataset_id)
    if not dataset or dataset.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    rows = DatasetDB.get_data(dataset_id) or []
    if not rows:
        return {"current_period": {}, "previous_period": {}, "change_pct": 0}

    df = pd.DataFrame(rows)
    if date_column not in df.columns or measure not in df.columns:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid date_column or measure")

    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df[measure] = pd.to_numeric(df[measure], errors='coerce')
    df = df.dropna(subset=[date_column, measure])

    if df.empty:
        return {"current_period": {}, "previous_period": {}, "change_pct": 0}

    df = df.sort_values(date_column)
    mid = len(df) // 2
    current = df.iloc[mid:]
    previous = df.iloc[:mid]

    current_sum = float(current[measure].sum())
    previous_sum = float(previous[measure].sum())
    change_pct = ((current_sum - previous_sum) / previous_sum * 100) if previous_sum != 0 else 0

    return {
        "current_period": {"sum": round(current_sum, 4), "count": len(current)},
        "previous_period": {"sum": round(previous_sum, 4), "count": len(previous)},
        "change_pct": round(change_pct, 2),
    }
