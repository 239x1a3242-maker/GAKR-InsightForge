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
    silhouette_score, confusion_matrix
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier

import joblib
import os


def detect_task(y: Optional[pd.Series]) -> str:
    print("Detecting task...")
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
    print("Evaluating regression model...")
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def evaluate_classification(y_true, y_pred) -> Dict[str, float]:
    print("Evaluating classification model...")
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def evaluate_classification_with_cv(model, X, y, cv_folds=5) -> Dict[str, Any]:
    """Evaluate classification model with cross-validation and confusion matrix"""
    print(f"Evaluating classification model with {cv_folds}-fold CV...")
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42), scoring='f1_weighted')
    
    # Fit model on full training data for confusion matrix
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Basic metrics
    metrics = evaluate_classification(y, y_pred)
    
    return {
        **metrics,
        "CV_Mean_F1": float(np.mean(cv_scores)),
        "CV_Std_F1": float(np.std(cv_scores)),
        "CV_Scores": cv_scores.tolist(),
        "Confusion_Matrix": cm.tolist(),
        "Dataset_Size": len(y),
        "CV_Folds": cv_folds,
    }


def evaluate_clustering(X, labels) -> Dict[str, float]:
    print("Evaluating clustering model...")
    return {
        "SilhouetteScore": float(silhouette_score(X, labels)),
    }


def train_models(processed_data: Dict[str, Any]) -> Dict[str, Any]:
    print("Training models...")
    X_train = processed_data.get("X_train")
    X_test = processed_data.get("X_test")
    y_train = processed_data.get("y_train")
    y_test = processed_data.get("y_test")
    X_all = processed_data.get("X_processed")
    feature_names: List[str] = processed_data.get("feature_names", [])

    task = detect_task(y_train if y_train is not None else None)
    results: Dict[str, Dict[str, Any]] = {}
    best_model = None
    best_model_name = None
    best_score = -np.inf
    dataset_warnings = []

    # Dataset size warnings
    dataset_size = len(X_all) if X_all is not None else len(X_train)
    if dataset_size < 50:
        dataset_warnings.append("⚠ Dataset is very small (<50 samples); results may not generalize well.")
    elif dataset_size < 100:
        dataset_warnings.append("⚠ Dataset is small (<100 samples); consider collecting more data for robust results.")

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
        models = CLASSIFICATION_MODELS.copy()
        
        # For small datasets, prefer simpler models and use cross-validation
        is_small_dataset = len(X_train) < 100
        
        if is_small_dataset:
            # Remove complex models that may overfit on small data
            models_to_remove = ["LightGBM", "XGBoost", "MLP", "GradientBoosting"]
            for model_name in models_to_remove:
                models.pop(model_name, None)
            dataset_warnings.append("ℹ Using simplified model selection for small dataset (preferring LogisticRegression, DecisionTree, RandomForest).")
        
        for name, model in models.items():
            try:
                if is_small_dataset and len(X_train) >= 10:
                    # Use cross-validation for small datasets
                    cv_metrics = evaluate_classification_with_cv(model, X_train, y_train, cv_folds=min(5, len(X_train)))
                    results[name] = cv_metrics
                    score = cv_metrics["CV_Mean_F1"]
                else:
                    # Standard evaluation for larger datasets
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    metrics = evaluate_classification(y_test, preds)
                    results[name] = metrics
                    score = metrics["F1"]
                
                if score >= best_score:
                    best_score = score
                    best_model = model
                    best_model_name = name
            except Exception as e:
                results[name] = {"error": str(e)}

    else:  # unsupervised
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
            if "F1" in m or "CV_Mean_F1" in m:
                primary_score = m.get("CV_Mean_F1", m.get("F1", 0))
                leaderboard.append({"model": name, "primary_metric": "F1 (CV)" if "CV_Mean_F1" in m else "F1", "score": primary_score, **m})
    else:
        for name, m in results.items():
            if "SilhouetteScore" in m:
                leaderboard.append({"model": name, "primary_metric": "SilhouetteScore", "score": m["SilhouetteScore"], **m})

    leaderboard = sorted(leaderboard, key=lambda x: x["score"], reverse=True)
    print("Training completed.")

    return {
        "task": task,
        "metrics": results,
        "best_model": best_model,
        "best_model_name": best_model_name,
        "leaderboard": leaderboard,
        "feature_names": feature_names,
        "dataset_warnings": dataset_warnings,
    }


def save_model(model: Any, task: str, model_name: str, path: str = "backend/models/trained_model.pkl") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"model": model, "task": task, "model_name": model_name}
    joblib.dump(payload, path)
    print(f"Saving model to {path}...")
    return path
