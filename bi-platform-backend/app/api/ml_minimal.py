"""Machine Learning API (File-based, sklearn-based AutoML)."""
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import uuid
import time
from datetime import datetime

from app.api.auth import get_current_user
from app.core.filedb import ModelDB, DatasetDB

router = APIRouter()


# ── Pydantic models ──────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    dataset_id: str
    target_columns: List[str]
    feature_columns: Optional[List[str]] = None
    problem_type: str = "auto"
    test_size: float = 0.2
    cv_folds: int = 5
    hyperparameter_tuning: bool = False
    auto_feature_engineering: bool = True
    feature_selection_method: str = "all"


class PredictRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_ids: List[str]
    data: List[Dict[str, Any]]
    return_confidence: bool = True


# ── Background training ──────────────────────────────────────────────────────

def _run_training(model_id: str, dataset_id: str, target_columns: List[str],
                  feature_columns: Optional[List[str]], problem_type: str,
                  test_size: float, cv_folds: int, hyperparameter_tuning: bool,
                  auto_feature_engineering: bool, feature_selection_method: str,
                  user_id: str):
    """Run actual sklearn training in background."""
    try:
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.ensemble import (
            RandomForestClassifier, RandomForestRegressor,
            GradientBoostingClassifier, GradientBoostingRegressor,
            ExtraTreesClassifier, ExtraTreesRegressor,
            AdaBoostClassifier, AdaBoostRegressor,
        )
        from sklearn.linear_model import (
            LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
        )
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import (
            accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
            r2_score, mean_squared_error, mean_absolute_error
        )
        import pickle
        from pathlib import Path

        # Load dataset
        dataset = DatasetDB.get_by_id(dataset_id)
        if not dataset:
            ModelDB.update(model_id, status="failed", error="Dataset not found")
            return

        rows = DatasetDB.get_data(dataset_id)
        if not rows:
            ModelDB.update(model_id, status="failed", error="Dataset has no data")
            return

        df = pd.DataFrame(rows)
        
        # Validate target columns exist
        missing_targets = [t for t in target_columns if t not in df.columns]
        if missing_targets:
            ModelDB.update(model_id, status="failed", error=f"Target columns not found: {', '.join(missing_targets)}")
            return

        # Determine features
        if feature_columns:
            features = [c for c in feature_columns if c in df.columns and c not in target_columns]
        else:
            features = [c for c in df.columns if c not in target_columns]
        
        if not features:
            ModelDB.update(model_id, status="failed", error="No valid feature columns available")
            return

        target_results = {}
        start_time = time.time()
        last_features = list(features)

        for target in target_columns:
            if target not in df.columns:
                continue

            X = df[features].copy()
            y = df[target].copy()

            # Drop rows with missing target
            mask = y.notna()
            X, y = X[mask], y[mask]
            
            if len(X) < 10:
                target_results[target] = {
                    "error": f"Insufficient data after removing missing values (only {len(X)} rows)"
                }
                continue

            # Encode categoricals in X using LabelEncoder
            encoders = {}
            for col in X.columns:
                if X[col].dtype == object or str(X[col].dtype) == 'category':
                    X[col] = X[col].fillna('MISSING').astype(str)
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
                    encoders[col] = le

            # Impute numeric missing values with median (from reference)
            imputers = {}
            numeric_cols = X.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                imp = SimpleImputer(strategy='median')
                X[col] = imp.fit_transform(X[[col]]).ravel()
                imputers[col] = imp

            # Determine task type
            task = problem_type
            if task == "auto":
                unique_vals = y.nunique()
                is_numeric = pd.api.types.is_numeric_dtype(y)
                if not is_numeric:
                    task = "classification"
                elif unique_vals <= 20 or (unique_vals / len(y) < 0.05 and unique_vals <= 30):
                    task = "classification"
                else:
                    task = "regression"

            # Encode target for classification
            target_encoder = None
            if task == "classification":
                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(y.astype(str))
            else:
                y = pd.to_numeric(y, errors='coerce').fillna(0).values

            # Use StratifiedKFold for classification (from reference)
            if task == "classification":
                # Check if we have enough samples per class for stratification
                unique_classes, class_counts = np.unique(y, return_counts=True)
                min_class_count = class_counts.min()
                if min_class_count < 2:
                    target_results[target] = {
                        "error": f"Insufficient samples for stratification (min class has {min_class_count} samples)"
                    }
                    continue
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                cv = StratifiedKFold(n_splits=min(cv_folds, 5), shuffle=True, random_state=42)
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                cv = KFold(n_splits=min(cv_folds, 5), shuffle=True, random_state=42)

            # Scale features
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # Train multiple algorithms and pick best (expanded from reference)
            if task == "classification":
                candidates = {
                    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                    "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
                    "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
                    "DecisionTree": DecisionTreeClassifier(random_state=42),
                    "NaiveBayes": GaussianNB(),
                }
                cv_scoring = 'accuracy'
            else:
                candidates = {
                    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
                    "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                    "AdaBoost": AdaBoostRegressor(n_estimators=50, random_state=42),
                    "Ridge": Ridge(random_state=42),
                    "Lasso": Lasso(random_state=42),
                    "ElasticNet": ElasticNet(random_state=42),
                    "DecisionTree": DecisionTreeRegressor(random_state=42),
                }
                cv_scoring = 'r2'

            best_score = -999
            best_name = None
            best_model = None

            for name, model in candidates.items():
                try:
                    model.fit(X_train_s, y_train)
                    if task == "classification":
                        score = accuracy_score(y_test, model.predict(X_test_s))
                    else:
                        score = r2_score(y_test, model.predict(X_test_s))
                    if score > best_score:
                        best_score = score
                        best_name = name
                        best_model = model
                except Exception as e:
                    print(f"Algorithm {name} failed: {e}")
                    continue

            if best_model is None:
                target_results[target] = {"error": "All algorithms failed to train"}
                continue

            # Compute metrics
            y_pred = best_model.predict(X_test_s)
            if task == "classification":
                n_classes = len(set(y_test))
                avg = 'weighted' if n_classes > 2 else 'binary'
                metrics = {
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "f1": float(f1_score(y_test, y_pred, average=avg, zero_division=0)),
                    "precision": float(precision_score(y_test, y_pred, average=avg, zero_division=0)),
                    "recall": float(recall_score(y_test, y_pred, average=avg, zero_division=0)),
                }
                # ROC AUC for binary (from reference)
                if n_classes == 2 and hasattr(best_model, 'predict_proba'):
                    try:
                        metrics["auc"] = float(roc_auc_score(y_test, best_model.predict_proba(X_test_s)[:, 1]))
                    except Exception:
                        pass
            else:
                mse = mean_squared_error(y_test, y_pred)
                # MAPE (from reference)
                with np.errstate(divide='ignore', invalid='ignore'):
                    mape = float(np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1))) * 100)
                    if np.isnan(mape) or np.isinf(mape):
                        mape = 0.0
                metrics = {
                    "r2": float(r2_score(y_test, y_pred)),
                    "rmse": float(mse ** 0.5),
                    "mae": float(mean_absolute_error(y_test, y_pred)),
                    "mape": mape,
                }

            # CV score using proper CV splitter
            try:
                cv_scores = cross_val_score(best_model, X_train_s, y_train, cv=cv, scoring=cv_scoring)
                cv_mean = float(cv_scores.mean())
                cv_std = float(cv_scores.std())
            except Exception as e:
                print(f"CV scoring failed: {e}")
                cv_mean = best_score
                cv_std = 0.0

            # Feature importance
            feature_importance = {}
            if hasattr(best_model, 'feature_importances_'):
                fi = best_model.feature_importances_
                total = fi.sum() or 1
                feature_importance = {features[i]: float(fi[i] / total) for i in range(len(features))}
            elif hasattr(best_model, 'coef_'):
                coef = np.abs(best_model.coef_[0] if best_model.coef_.ndim > 1 else best_model.coef_)
                total = coef.sum() or 1
                feature_importance = {features[i]: float(coef[i] / total) for i in range(len(features))}

            # Save model artifact
            models_dir = Path(__file__).parent.parent / "data" / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            artifact = {
                "model": best_model,
                "scaler": scaler,
                "encoders": encoders,
                "imputers": imputers,
                "target_encoder": target_encoder,
                "features": features,
                "task": task,
            }
            artifact_path = models_dir / f"{model_id}_{target}.pkl"
            with open(artifact_path, "wb") as f:
                pickle.dump(artifact, f)

            target_results[target] = {
                "best_algorithm": best_name,
                "best_model_id": f"{model_id}_{target}",
                "task_type": task,
                "metrics": metrics,
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "feature_importance": feature_importance,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
            }

        if not target_results:
            ModelDB.update(model_id, status="failed", error="No targets were successfully trained")
            return

        total_time = time.time() - start_time
        ModelDB.update(
            model_id,
            status="completed",
            target_results=target_results,
            total_training_time_seconds=total_time,
            target_columns=target_columns,
            feature_columns=last_features,
            dataset_id=dataset_id,
            completed_at=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"Training failed: {error_msg}")
        print(traceback.format_exc())
        ModelDB.update(model_id, status="failed", error=error_msg)

@router.get("/models")
async def list_models(
    skip: int = 0,
    limit: int = 100,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    models = ModelDB.get_by_user(current_user["id"])
    paginated = models[skip: skip + limit]
    return {"items": paginated, "total": len(models), "skip": skip, "limit": limit}


@router.get("/models/{model_id}")
async def get_model(
    model_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    from pathlib import Path
    models = ModelDB.get_by_user(current_user["id"])
    model = next((m for m in models if m.get("id") == model_id), None)
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    return model


@router.post("/train")
async def train_model(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    # Validate dataset exists
    dataset = DatasetDB.get_by_id(request.dataset_id)
    if not dataset or dataset.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    model_id = str(uuid.uuid4())
    model = ModelDB.create(
        user_id=current_user["id"],
        name=f"Model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        algorithm="AutoML",
        dataset_id=request.dataset_id,
        status="training",
    )
    model_id = model["id"]

    background_tasks.add_task(
        _run_training,
        model_id=model_id,
        dataset_id=request.dataset_id,
        target_columns=request.target_columns,
        feature_columns=request.feature_columns,
        problem_type=request.problem_type,
        test_size=request.test_size,
        cv_folds=request.cv_folds,
        hyperparameter_tuning=request.hyperparameter_tuning,
        auto_feature_engineering=request.auto_feature_engineering,
        feature_selection_method=request.feature_selection_method,
        user_id=current_user["id"],
    )

    return {
        "id": model_id,
        "status": "training",
        "progress": 0,
        "message": "Model training started",
        "target_columns": request.target_columns,
    }


@router.post("/predict")
async def predict(
    request: PredictRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    import pickle
    import pandas as pd
    from pathlib import Path

    if not request.model_ids or not request.data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="model_ids and data are required")

    models_dir = Path(__file__).parent.parent / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    predictions_by_target: Dict[str, Any] = {}
    start_time = time.time()

    for model_id_target in request.model_ids:
        artifact_path = models_dir / f"{model_id_target}.pkl"
        if not artifact_path.exists():
            predictions_by_target[model_id_target] = {"error": f"Model artifact not found: {model_id_target}.pkl"}
            continue

        try:
            with open(artifact_path, "rb") as f:
                artifact = pickle.load(f)

            model = artifact["model"]
            scaler = artifact["scaler"]
            encoders = artifact["encoders"]
            imputers = artifact.get("imputers", {})
            target_encoder = artifact.get("target_encoder")
            features = artifact["features"]

            df = pd.DataFrame(request.data)
            
            # Check for missing required features
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                predictions_by_target[model_id_target] = {
                    "error": f"Missing required features: {', '.join(missing_features)}"
                }
                continue
            
            # Preprocess features
            for col in features:
                if col in encoders:
                    df[col] = df[col].fillna('MISSING').astype(str)
                    le = encoders[col]
                    df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    df[col] = le.transform(df[col])
                elif col in imputers:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = imputers[col].transform(df[[col]]).ravel()
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            X = scaler.transform(df[features])
            preds = model.predict(X)

            if target_encoder is not None:
                preds = target_encoder.inverse_transform(preds)

            confidence = None
            if request.return_confidence and hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                confidence = proba.max(axis=1).tolist()

            # Extract target name from artifact filename
            target_name = model_id_target.split('_', 1)[1] if '_' in model_id_target else model_id_target
            predictions_by_target[target_name] = {
                "predictions": preds.tolist(),
                "confidence": confidence,
            }
        except Exception as e:
            import traceback
            predictions_by_target[model_id_target] = {
                "error": f"{type(e).__name__}: {str(e)}",
                "traceback": traceback.format_exc()
            }

    if not predictions_by_target:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No valid model artifacts found for the provided model_ids"
        )

    return {
        "predictions": predictions_by_target,
        "prediction_time_ms": int((time.time() - start_time) * 1000),
        "input_rows": len(request.data),
    }


@router.get("/training-jobs/{job_id}")
async def get_training_job(
    job_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    models = ModelDB.get_by_user(current_user["id"])
    model = next((m for m in models if m.get("id") == job_id), None)
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Training job not found")
    return model


@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    import pickle
    from pathlib import Path
    models = ModelDB.get_by_user(current_user["id"])
    model = next((m for m in models if m.get("id") == model_id), None)
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    # Remove artifact files
    models_dir = Path(__file__).parent.parent / "data" / "models"
    for f in models_dir.glob(f"{model_id}*.pkl"):
        f.unlink(missing_ok=True)

    # Remove from index
    all_models = ModelDB.get_all()
    from app.core.filedb import FileDB
    FileDB.save(ModelDB.MODELS_INDEX, [m for m in all_models if m.get("id") != model_id])
    return {"message": "Model deleted"}


@router.post("/anomaly-detection")
async def detect_anomalies(
    data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    import pandas as pd
    import numpy as np

    dataset_id = data.get("dataset_id")
    columns = data.get("columns", [])
    contamination = float(data.get("contamination", 0.05))

    dataset = DatasetDB.get_by_id(dataset_id)
    if not dataset or dataset.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    rows = DatasetDB.get_data(dataset_id) or []
    if not rows:
        return {"anomalies": [], "scores": [], "total_anomalies": 0}

    df = pd.DataFrame(rows)
    if columns:
        df = df[[c for c in columns if c in df.columns]]

    # Use numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return {"anomalies": [], "scores": [], "total_anomalies": 0}

    numeric_df = numeric_df.fillna(numeric_df.mean())

    try:
        from sklearn.ensemble import IsolationForest
        clf = IsolationForest(contamination=contamination, random_state=42)
        scores = clf.fit_predict(numeric_df)
        anomaly_indices = [int(i) for i, s in enumerate(scores) if s == -1]
        anomaly_scores = clf.score_samples(numeric_df).tolist()
        return {
            "anomalies": anomaly_indices,
            "scores": anomaly_scores,
            "total_anomalies": len(anomaly_indices),
            "contamination": contamination,
        }
    except Exception as e:
        return {"anomalies": [], "scores": [], "total_anomalies": 0, "error": str(e)}


@router.post("/clustering")
async def cluster_data(
    data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    import pandas as pd
    import numpy as np

    dataset_id = data.get("dataset_id")
    columns = data.get("columns", [])
    n_clusters = int(data.get("n_clusters", 3))

    dataset = DatasetDB.get_by_id(dataset_id)
    if not dataset or dataset.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    rows = DatasetDB.get_data(dataset_id) or []
    if not rows:
        return {"clusters": [], "labels": [], "n_clusters": n_clusters}

    df = pd.DataFrame(rows)
    if columns:
        df = df[[c for c in columns if c in df.columns]]

    numeric_df = df.select_dtypes(include=[np.number]).fillna(0)
    if numeric_df.empty:
        return {"clusters": [], "labels": [], "n_clusters": n_clusters}

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(numeric_df)
        km = KMeans(n_clusters=min(n_clusters, len(X)), random_state=42, n_init=10)
        labels = km.fit_predict(X)
        cluster_sizes = {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
        return {
            "labels": labels.tolist(),
            "n_clusters": n_clusters,
            "cluster_sizes": cluster_sizes,
            "inertia": float(km.inertia_),
        }
    except Exception as e:
        return {"clusters": [], "labels": [], "n_clusters": n_clusters, "error": str(e)}


@router.post("/feature-importance")
async def feature_importance(
    data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    import pandas as pd
    import numpy as np

    dataset_id = data.get("dataset_id")
    target_column = data.get("target_column")
    feature_columns = data.get("feature_columns", [])

    dataset = DatasetDB.get_by_id(dataset_id)
    if not dataset or dataset.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    rows = DatasetDB.get_data(dataset_id) or []
    if not rows or not target_column:
        return {"features": {}, "importance_scores": []}

    df = pd.DataFrame(rows)
    if feature_columns:
        features = [c for c in feature_columns if c in df.columns]
    else:
        features = [c for c in df.columns if c != target_column]

    X = df[features].copy()
    y = df[target_column].copy()

    from sklearn.preprocessing import LabelEncoder
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        if y.dtype == object or y.nunique() <= 20:
            le = LabelEncoder()
            y_enc = le.fit_transform(y.astype(str))
            model = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            y_enc = pd.to_numeric(y, errors='coerce').fillna(0)
            model = RandomForestRegressor(n_estimators=50, random_state=42)

        model.fit(X, y_enc)
        fi = model.feature_importances_
        total = fi.sum() or 1
        importance = {features[i]: float(fi[i] / total) for i in range(len(features))}
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        return {"features": sorted_importance, "importance_scores": list(sorted_importance.values())}
    except Exception as e:
        return {"features": {}, "importance_scores": [], "error": str(e)}


@router.post("/compare-datasets")
async def compare_datasets(
    data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    import pandas as pd
    import numpy as np

    dataset_ids = data.get("dataset_ids", [])
    if len(dataset_ids) < 2:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Need at least 2 dataset IDs")

    summaries = {}
    for ds_id in dataset_ids:
        dataset = DatasetDB.get_by_id(ds_id)
        if not dataset or dataset.get("user_id") != current_user["id"]:
            continue
        rows = DatasetDB.get_data(ds_id) or []
        df = pd.DataFrame(rows)
        summaries[ds_id] = {
            "name": dataset.get("name"),
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "numeric_columns": int(df.select_dtypes(include=[np.number]).shape[1]),
        }

    # Find common columns
    all_cols = [set(s["columns"]) for s in summaries.values()]
    common_cols = list(set.intersection(*all_cols)) if all_cols else []

    return {
        "summaries": summaries,
        "common_columns": common_cols,
        "differences": [
            {"type": "columns", "description": f"Datasets share {len(common_cols)} common columns"}
        ],
    }
