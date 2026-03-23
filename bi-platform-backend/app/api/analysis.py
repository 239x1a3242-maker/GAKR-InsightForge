"""Analysis API (File-based, pandas-based)."""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from app.api.auth import get_current_user
from app.core.filedb import DatasetDB

router = APIRouter()


class DescriptiveRequest(BaseModel):
    columns: Optional[List[str]] = None


class AskAIRequest(BaseModel):
    question: str
    context: Optional[str] = None


def _load_df(dataset_id: str, user_id: str):
    """Load dataset as DataFrame, raising 404 if not found."""
    import pandas as pd
    dataset = DatasetDB.get_by_id(dataset_id)
    if not dataset or dataset.get("user_id") != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    rows = DatasetDB.get_data(dataset_id) or []
    return dataset, pd.DataFrame(rows)


@router.post("/descriptive/{dataset_id}")
async def descriptive_analysis(
    dataset_id: str,
    request: DescriptiveRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    import pandas as pd
    import numpy as np

    dataset, df = _load_df(dataset_id, current_user["id"])

    if request.columns:
        df = df[[c for c in request.columns if c in df.columns]]

    numeric_df = df.select_dtypes(include=[np.number])
    categorical_df = df.select_dtypes(exclude=[np.number])

    # Column stats
    column_stats: Dict[str, Any] = {}
    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        missing_pct = round(missing / len(df) * 100, 2) if len(df) > 0 else 0
        stat: Dict[str, Any] = {
            "dtype": str(series.dtype),
            "missing": missing,
            "missing_percentage": missing_pct,
            "unique": int(series.nunique()),
        }
        if pd.api.types.is_numeric_dtype(series):
            stat.update({
                "mean": round(float(series.mean()), 4) if not series.isna().all() else None,
                "std": round(float(series.std()), 4) if not series.isna().all() else None,
                "min": round(float(series.min()), 4) if not series.isna().all() else None,
                "max": round(float(series.max()), 4) if not series.isna().all() else None,
                "median": round(float(series.median()), 4) if not series.isna().all() else None,
                "q25": round(float(series.quantile(0.25)), 4) if not series.isna().all() else None,
                "q75": round(float(series.quantile(0.75)), 4) if not series.isna().all() else None,
            })
        else:
            top_vals = series.value_counts().head(5).to_dict()
            stat["top_values"] = {str(k): int(v) for k, v in top_vals.items()}
        column_stats[col] = stat

    # Distributions for numeric columns
    distributions: Dict[str, Any] = {}
    for col in numeric_df.columns[:10]:  # limit to 10 cols
        series = numeric_df[col].dropna()
        if len(series) == 0:
            continue
        counts, bin_edges = np.histogram(series, bins=20)
        distributions[col] = {
            "bins": [f"{round(float(b), 2)}" for b in bin_edges[:-1]],
            "counts": counts.tolist(),
        }

    # Correlation matrix (numeric only, limit to 20 cols)
    correlation_matrix = None
    if len(numeric_df.columns) >= 2:
        corr = numeric_df.iloc[:, :20].corr().round(4)
        correlation_matrix = {col: corr[col].to_dict() for col in corr.columns}

    # Insights
    insights = []
    total_missing = int(df.isna().sum().sum())
    if total_missing > 0:
        pct = round(total_missing / (len(df) * len(df.columns)) * 100, 1)
        insights.append(f"{total_missing} missing values ({pct}% of all cells)")

    high_missing_cols = [c for c, s in column_stats.items() if s.get("missing_percentage", 0) > 20]
    if high_missing_cols:
        insights.append(f"Columns with >20% missing: {', '.join(high_missing_cols[:5])}")

    low_variance_cols = []
    for col in numeric_df.columns:
        if numeric_df[col].std() == 0:
            low_variance_cols.append(col)
    if low_variance_cols:
        insights.append(f"Zero-variance columns (constant): {', '.join(low_variance_cols[:5])}")

    return {
        "dataset_id": dataset_id,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "numeric_columns": len(numeric_df.columns),
        "categorical_columns": len(categorical_df.columns),
        "date_columns": 0,
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 4),
        "missing_summary": {
            "total_missing": total_missing,
            "missing_percentage": round(total_missing / max(len(df) * len(df.columns), 1) * 100, 2),
            "columns_with_missing": int((df.isna().sum() > 0).sum()),
        },
        "correlation_matrix": correlation_matrix,
        "distributions": distributions,
        "column_stats": column_stats,
        "insights": insights,
    }


@router.post("/diagnostic/{dataset_id}")
async def diagnostic_analysis(
    dataset_id: str,
    target_column: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    import pandas as pd
    import numpy as np

    dataset, df = _load_df(dataset_id, current_user["id"])
    numeric_df = df.select_dtypes(include=[np.number])

    # Outlier detection using IQR
    total_outliers = 0
    for col in numeric_df.columns:
        q1 = numeric_df[col].quantile(0.25)
        q3 = numeric_df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = ((numeric_df[col] < q1 - 1.5 * iqr) | (numeric_df[col] > q3 + 1.5 * iqr)).sum()
        total_outliers += int(outliers)

    # Correlations with target
    correlations: Dict[str, List[Dict[str, Any]]] = {}
    if target_column and target_column in df.columns:
        target = pd.to_numeric(df[target_column], errors='coerce')
        for col in numeric_df.columns:
            if col == target_column:
                continue
            corr = float(numeric_df[col].corr(target))
            if not pd.isna(corr):
                correlations.setdefault(target_column, []).append({"feature": col, "correlation": round(corr, 4)})
        if target_column in correlations:
            correlations[target_column].sort(key=lambda x: abs(x["correlation"]), reverse=True)

    return {
        "dataset_id": dataset_id,
        "target_column": target_column,
        "correlations": correlations,
        "outliers": {
            "total_outliers": total_outliers,
            "outlier_percentage": round(total_outliers / max(len(df) * len(numeric_df.columns), 1) * 100, 2),
        },
        "segments": [],
        "root_causes": [],
    }


@router.post("/predictive/{dataset_id}")
async def predictive_analysis(
    dataset_id: str,
    target_column: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    import pandas as pd
    import numpy as np

    dataset, df = _load_df(dataset_id, current_user["id"])

    if not target_column or target_column not in df.columns:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="target_column is required")

    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.metrics import accuracy_score, r2_score

    features = [c for c in df.columns if c != target_column]
    X = df[features].copy()
    y = df[target_column].copy()

    mask = y.notna()
    X, y = X[mask], y[mask]

    for col in X.columns:
        if X[col].dtype == object:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    task_type = "classification" if y.dtype == object or y.nunique() <= 20 else "regression"
    if task_type == "classification":
        y = LabelEncoder().fit_transform(y.astype(str))

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.2, random_state=42)

    algorithms_results = []
    if task_type == "classification":
        candidates = [
            ("RandomForest", RandomForestClassifier(n_estimators=50, random_state=42)),
            ("LogisticRegression", LogisticRegression(max_iter=300, random_state=42)),
        ]
        score_fn = accuracy_score
    else:
        candidates = [
            ("RandomForest", RandomForestRegressor(n_estimators=50, random_state=42)),
            ("LinearRegression", LinearRegression()),
        ]
        score_fn = r2_score

    best_score = -999
    best_name = None
    best_model = None

    for name, model in candidates:
        try:
            t0 = time.time() if False else __import__('time').time()
            model.fit(X_train, y_train)
            score = float(score_fn(y_test, model.predict(X_test)))
            elapsed = int((__import__('time').time() - t0) * 1000)
            algorithms_results.append({"name": name, "metrics": {"score": round(score, 4)}, "training_time_ms": elapsed, "rank": 0})
            if score > best_score:
                best_score = score
                best_name = name
                best_model = model
        except Exception:
            continue

    # Rank
    algorithms_results.sort(key=lambda x: x["metrics"]["score"], reverse=True)
    for i, r in enumerate(algorithms_results):
        r["rank"] = i + 1

    # Feature importance
    feature_importance = {}
    if best_model and hasattr(best_model, 'feature_importances_'):
        fi = best_model.feature_importances_
        total = fi.sum() or 1
        feature_importance = {features[i]: round(float(fi[i] / total), 4) for i in range(len(features))}

    cv_score = 0.0
    if best_model:
        try:
            cv = cross_val_score(best_model, X_s, y, cv=3, scoring='accuracy' if task_type == 'classification' else 'r2')
            cv_score = float(cv.mean())
        except Exception:
            pass

    return {
        "dataset_id": dataset_id,
        "target_column": target_column,
        "task_type": task_type,
        "best_algorithm": best_name or "N/A",
        "algorithms": algorithms_results,
        "feature_importance": feature_importance,
        "cross_validation_score": round(cv_score, 4),
    }


@router.post("/prescriptive/{dataset_id}")
async def prescriptive_analysis(
    dataset_id: str,
    target_column: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    dataset, df = _load_df(dataset_id, current_user["id"])

    if not target_column or target_column not in df.columns:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="target_column is required")

    import pandas as pd
    import numpy as np

    numeric_df = df.select_dtypes(include=[np.number])
    target_series = pd.to_numeric(df[target_column], errors='coerce').dropna()

    recommendations = []
    for col in numeric_df.columns:
        if col == target_column:
            continue
        corr = float(numeric_df[col].corr(target_series))
        if abs(corr) > 0.3:
            direction = "increase" if corr > 0 else "decrease"
            recommendations.append({
                "action": f"{direction.capitalize()} {col}",
                "impact": f"Correlation with {target_column}: {round(corr, 3)}",
                "confidence": round(abs(corr), 3),
                "supporting_data": {"correlation": round(corr, 3)},
            })

    recommendations.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "dataset_id": dataset_id,
        "target_column": target_column,
        "recommendations": recommendations[:10],
        "scenarios": [],
        "what_if_analysis": {
            "variable": recommendations[0]["action"].split()[-1] if recommendations else "",
            "current_value": 0,
            "impact_per_unit": recommendations[0]["confidence"] if recommendations else 0,
            "suggested_increase": 1,
        },
    }


@router.post("/ask-ai/{dataset_id}")
async def ask_ai(
    dataset_id: str,
    request: AskAIRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    import pandas as pd
    import numpy as np
    from datetime import datetime

    dataset, df = _load_df(dataset_id, current_user["id"])

    # Build a simple context-aware answer without LLM
    question_lower = request.question.lower()
    answer = ""

    if any(w in question_lower for w in ["row", "record", "size", "how many"]):
        answer = f"The dataset '{dataset.get('name')}' has {len(df):,} rows and {len(df.columns)} columns."
    elif any(w in question_lower for w in ["column", "field", "feature"]):
        answer = f"The dataset has {len(df.columns)} columns: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}."
    elif any(w in question_lower for w in ["missing", "null", "empty"]):
        missing = df.isna().sum().sum()
        answer = f"There are {missing:,} missing values across all columns ({round(missing / max(len(df) * len(df.columns), 1) * 100, 1)}% of all cells)."
    elif any(w in question_lower for w in ["average", "mean", "avg"]):
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            means = numeric_df.mean().round(2).to_dict()
            top = list(means.items())[:5]
            answer = "Column averages: " + ", ".join([f"{k}: {v}" for k, v in top])
        else:
            answer = "No numeric columns found to compute averages."
    elif any(w in question_lower for w in ["max", "maximum", "highest"]):
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            maxes = numeric_df.max().round(2).to_dict()
            top = list(maxes.items())[:5]
            answer = "Column maximums: " + ", ".join([f"{k}: {v}" for k, v in top])
        else:
            answer = "No numeric columns found."
    elif any(w in question_lower for w in ["min", "minimum", "lowest"]):
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            mins = numeric_df.min().round(2).to_dict()
            top = list(mins.items())[:5]
            answer = "Column minimums: " + ", ".join([f"{k}: {v}" for k, v in top])
        else:
            answer = "No numeric columns found."
    else:
        answer = (
            f"Dataset '{dataset.get('name')}' summary: {len(df):,} rows, {len(df.columns)} columns. "
            f"Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}. "
            f"Missing values: {df.isna().sum().sum():,}."
        )

    return {
        "question": request.question,
        "answer": answer,
        "confidence": 0.85,
        "suggested_followups": [
            "What are the missing values?",
            "What are the column averages?",
            "How many rows does this dataset have?",
        ],
        "generated_at": datetime.utcnow().isoformat(),
        "tokens_used": len(answer.split()),
    }
