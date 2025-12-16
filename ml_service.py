"""
ML Service:
- Auto-detects ML task (regression / classification / unsupervised)
- Trains multiple models
- Evaluates with task-specific metrics
- Returns best model + leaderboard + feature_names
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

from sklearn.linear_model import (
    LinearRegression, LogisticRegression,
    Ridge, Lasso, ElasticNet
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    silhouette_score
)

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier

import joblib
import os


def detect_task(y: Optional[pd.Series]) -> str:
    if y is None:
        return "unsupervised"
    if y.nunique() <= 20 and (str(y.dtype).startswith("int") or y.dtype == "object"):
        return "classification"
    return "regression"


# Note: distance-based models need scaled features; scaling is done in data_service.
REGRESSION_MODELS = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "ElasticNet": ElasticNet(),
    "DecisionTree": DecisionTreeRegressor(),
    "RandomForest": RandomForestRegressor(),
    "GradientBoosting": GradientBoostingRegressor(),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(),
    "MLP": MLPRegressor(max_iter=500),
    "XGBoost": xgb.XGBRegressor(eval_metric="rmse", tree_method="hist", n_estimators=200, max_depth=6),
    "LightGBM": lgb.LGBMRegressor(),
    "CatBoost": CatBoostRegressor(verbose=False),
}

CLASSIFICATION_MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=500),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "SVC": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "MLP": MLPClassifier(max_iter=500),
    "XGBoost": xgb.XGBClassifier(eval_metric="logloss", tree_method="hist", n_estimators=200, max_depth=6),
    "LightGBM": lgb.LGBMClassifier(),
    "CatBoost": CatBoostClassifier(verbose=False),
}

UNSUPERVISED_MODELS = {
    "KMeans": KMeans(n_clusters=3, random_state=42),
}


def evaluate_regression(y_true, y_pred) -> Dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def evaluate_classification(y_true, y_pred) -> Dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def evaluate_clustering(X, labels) -> Dict[str, float]:
    return {
        "SilhouetteScore": float(silhouette_score(X, labels)),
    }


def train_models(processed_data: Dict[str, Any]) -> Dict[str, Any]:
    X_train = processed_data.get("X_train")
    X_test = processed_data.get("X_test")
    y_train = processed_data.get("y_train")
    y_test = processed_data.get("y_test")
    X_all = processed_data.get("X_processed")
    feature_names: List[str] = processed_data.get("feature_names", [])

    task = detect_task(y_train if y_train is not None else None)
    results: Dict[str, Dict[str, float]] = {}
    best_model = None
    best_model_name = None
    best_score = -np.inf

    if task == "regression":
        models = REGRESSION_MODELS
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                metrics = evaluate_regression(y_test, preds)
                results[name] = metrics
                if metrics["R2"] >= best_score:
                    best_score = metrics["R2"]
                    best_model = model
                    best_model_name = name
            except Exception as e:
                results[name] = {"error": str(e)}

    elif task == "classification":
        models = CLASSIFICATION_MODELS
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                if hasattr(model, "predict_proba"):
                    _ = model.predict_proba(X_test)
                metrics = evaluate_classification(y_test, preds)
                results[name] = metrics
                if metrics["F1"] >= best_score:
                    best_score = metrics["F1"]
                    best_model = model
                    best_model_name = name
            except Exception as e:
                results[name] = {"error": str(e)}

    else:
        if X_all.shape[0] < 50:
            raise ValueError("Too few samples for reliable clustering (need >= 50).")
        models = UNSUPERVISED_MODELS
        for name, model in models.items():
            try:
                labels = model.fit_predict(X_all)
                metrics = evaluate_clustering(X_all, labels)
                results[name] = metrics
                best_model = model
                best_model_name = name
            except Exception as e:
                results[name] = {"error": str(e)}

    leaderboard = []
    if task == "regression":
        for name, m in results.items():
            if "R2" in m:
                leaderboard.append({"model": name, "primary_metric": "R2", "score": m["R2"], **m})
    elif task == "classification":
        for name, m in results.items():
            if "F1" in m:
                leaderboard.append({"model": name, "primary_metric": "F1", "score": m["F1"], **m})
    else:
        for name, m in results.items():
            if "SilhouetteScore" in m:
                leaderboard.append({"model": name, "primary_metric": "SilhouetteScore", "score": m["SilhouetteScore"], **m})

    leaderboard = sorted(leaderboard, key=lambda x: x["score"], reverse=True)

    return {
        "task": task,
        "metrics": results,
        "best_model": best_model,
        "best_model_name": best_model_name,
        "leaderboard": leaderboard,
        "feature_names": feature_names,
    }


def save_model(model: Any, task: str, model_name: str, path: str = "backend/models/trained_model.pkl") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"model": model, "task": task, "model_name": model_name}
    joblib.dump(payload, path)
    return path
