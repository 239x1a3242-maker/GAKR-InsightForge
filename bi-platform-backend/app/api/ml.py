"""Machine Learning API routes."""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import uuid
import joblib
import os
import time
import json

from app.core.database import get_db
from app.models.user import User
from app.models.dataset import Dataset
from app.schemas.ml import (
    TrainingRequest, TrainingResponse, AlgorithmResult, SingleTargetResult,
    ModelInfo, PredictionRequest, PredictionResponse, SinglePredictionResult,
    ForecastRequest, ForecastResponse,
    AnomalyDetectionRequest, AnomalyDetectionResponse, AnomalyResult,
    ClusteringRequest, ClusteringResponse,
    FeatureImportanceRequest, FeatureImportanceResponse,
    DatasetComparisonRequest, DatasetComparisonResponse,
    ModelComparisonRequest, ModelComparisonResponse
)
from app.api.auth import get_current_user
from app.api.datasets import load_dataset_file

router = APIRouter(prefix="/ml", tags=["Machine Learning"])

# Model storage
MODEL_DIR = "/tmp/bi_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Store training jobs
training_jobs: Dict[str, Dict] = {}


class EnhancedMLPipeline:
    """Enhanced Machine Learning Pipeline with multi-target support."""
    
    def __init__(self):
        self.models = {}
        self.preprocessors = {}
    
    def detect_problem_type(self, y: pd.Series) -> str:
        """Auto-detect problem type."""
        n_unique = y.nunique()
        
        if pd.api.types.is_datetime64_any_dtype(y):
            return "time_series"
        
        if n_unique == 2:
            return "classification"
        elif n_unique <= 20 and pd.api.types.is_integer_dtype(y):
            return "classification"
        else:
            return "regression"
    
    def get_algorithms(self, problem_type: str):
        """Get algorithms for problem type."""
        from sklearn.ensemble import (
            RandomForestRegressor, RandomForestClassifier,
            GradientBoostingRegressor, GradientBoostingClassifier,
            ExtraTreesRegressor, ExtraTreesClassifier,
            AdaBoostRegressor, AdaBoostClassifier
        )
        from sklearn.linear_model import (
            LinearRegression, Ridge, Lasso, ElasticNet,
            LogisticRegression, SGDClassifier
        )
        from sklearn.svm import SVR, SVC
        from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        
        if problem_type == "regression":
            return {
                "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                "gradient_boosting": GradientBoostingRegressor(random_state=42),
                "linear": LinearRegression(),
                "ridge": Ridge(random_state=42),
                "lasso": Lasso(random_state=42),
                "elastic_net": ElasticNet(random_state=42),
                "svr": SVR(),
                "knn": KNeighborsRegressor(),
                "extra_trees": ExtraTreesRegressor(random_state=42, n_jobs=-1),
                "adaboost": AdaBoostRegressor(random_state=42),
                "decision_tree": DecisionTreeRegressor(random_state=42),
            }
        else:  # classification
            return {
                "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                "gradient_boosting": GradientBoostingClassifier(random_state=42),
                "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
                "svc": SVC(probability=True, random_state=42),
                "knn": KNeighborsClassifier(),
                "naive_bayes": GaussianNB(),
                "decision_tree": DecisionTreeClassifier(random_state=42),
                "extra_trees": ExtraTreesClassifier(random_state=42, n_jobs=-1),
                "adaboost": AdaBoostClassifier(random_state=42),
                "sgd": SGDClassifier(random_state=42),
            }
    
    def preprocess_data(self, df: pd.DataFrame, feature_cols: List[str], target_cols: List[str], 
                       handle_missing: str = "auto", scaling: str = "auto", encode_categorical: bool = True):
        """Preprocess data for training."""
        from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
        from sklearn.impute import SimpleImputer
        
        # Select columns
        X = df[feature_cols].copy()
        y = df[target_cols].copy() if target_cols else None
        
        preprocessing_info = {}
        
        # Handle missing values
        if handle_missing == "auto":
            # Use median for numeric, mode for categorical
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            categorical_cols = X.select_dtypes(include=['object']).columns
            
            if len(numeric_cols) > 0:
                imputer = SimpleImputer(strategy='median')
                X[numeric_cols] = imputer.fit_transform(X[numeric_cols])
            if len(categorical_cols) > 0:
                imputer = SimpleImputer(strategy='most_frequent')
                X[categorical_cols] = imputer.fit_transform(X[categorical_cols])
        elif handle_missing == "impute_mean":
            X = X.fillna(X.mean())
        elif handle_missing == "impute_median":
            X = X.fillna(X.median())
        elif handle_missing == "impute_mode":
            X = X.fillna(X.mode().iloc[0])
        elif handle_missing == "drop":
            X = X.dropna()
            if y is not None:
                y = y.loc[X.index]
        
        preprocessing_info['missing_handling'] = handle_missing
        
        # Encode categorical variables
        label_encoders = {}
        if encode_categorical:
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
        
        preprocessing_info['label_encoders'] = {k: list(v.classes_) for k, v in label_encoders.items()}
        
        # Encode target if categorical
        target_encoder = None
        if y is not None:
            for col in y.columns:
                if y[col].dtype == 'object':
                    le = LabelEncoder()
                    y[col] = le.fit_transform(y[col].astype(str))
                    target_encoder = le
        
        # Scale features
        scaler = None
        if scaling == "auto" or scaling == "standard":
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        elif scaling == "minmax":
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        elif scaling == "robust":
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        preprocessing_info['scaling'] = scaling
        
        return X, y, preprocessing_info, scaler, label_encoders, target_encoder
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = "all", max_features: Optional[int] = None):
        """Select features based on importance."""
        from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif, f_regression, f_classif
        
        if method == "all" or max_features is None or max_features >= X.shape[1]:
            return list(X.columns)
        
        # Detect problem type for y
        problem_type = self.detect_problem_type(y)
        
        if method == "mutual_info":
            if problem_type == "regression":
                selector = SelectKBest(mutual_info_regression, k=min(max_features, X.shape[1]))
            else:
                selector = SelectKBest(mutual_info_classif, k=min(max_features, X.shape[1]))
        elif method == "f_score":
            if problem_type == "regression":
                selector = SelectKBest(f_regression, k=min(max_features, X.shape[1]))
            else:
                selector = SelectKBest(f_classif, k=min(max_features, X.shape[1]))
        else:
            return list(X.columns)
        
        selector.fit(X, y)
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        return selected_features
    
    def train_model(self, X, y, algorithm, problem_type, cv_folds=5):
        """Train a single model."""
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train
        start_time = time.time()
        model = algorithm
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2' if problem_type == 'regression' else 'accuracy')
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        if problem_type == "regression":
            metrics = {
                "mse": float(mean_squared_error(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred)),
            }
        else:
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            }
        
        # Feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_.tolist()))
        elif hasattr(model, 'coef_'):
            feature_importance = dict(zip(X.columns, np.abs(model.coef_).tolist() if model.coef_.ndim == 1 else np.abs(model.coef_[0]).tolist()))
        
        return {
            "model": model,
            "metrics": metrics,
            "cv_scores": cv_scores.tolist(),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "training_time": training_time,
            "feature_importance": feature_importance,
            "y_test": y_test,
            "y_pred": y_pred
        }


pipeline = EnhancedMLPipeline()


@router.post("/train", response_model=TrainingResponse)
async def train_models(
    data: TrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Train ML models with AutoML - supports multiple targets."""
    start_time = time.time()
    
    # Get dataset
    result = await db.execute(
        select(Dataset).where(
            and_(
                Dataset.id == data.dataset_id,
                Dataset.is_active == True
            )
        )
    )
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if dataset.owner_id != current_user.id and not dataset.is_shared:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Load data
    df = await load_dataset_file(data.dataset_id)
    
    # Check row count
    if len(df) > 100000:
        df = df.sample(100000, random_state=42)
    
    # Auto-select features if not provided
    feature_cols = data.feature_columns
    if not feature_cols:
        feature_cols = [col for col in df.columns if col not in data.target_columns]
    
    # Preprocess data
    X, y, preprocessing_info, scaler, label_encoders, target_encoder = pipeline.preprocess_data(
        df, feature_cols, data.target_columns,
        handle_missing=data.handle_missing,
        scaling=data.scaling,
        encode_categorical=data.encode_categorical
    )
    
    # Create job
    job_id = str(uuid.uuid4())
    
    target_results = {}
    all_algorithm_results = []
    
    # Train models for each target
    for target_col in data.target_columns:
        y_target = y[target_col] if len(data.target_columns) > 1 else y
        
        # Detect problem type
        problem_type = data.problem_type
        if problem_type == "auto":
            problem_type = pipeline.detect_problem_type(y_target)
        
        # Get algorithms
        algorithms = pipeline.get_algorithms(problem_type)
        
        if data.algorithms:
            selected = [a.value.split('_')[-1] if '_' in a.value else a.value for a in data.algorithms]
            algorithms = {k: v for k, v in algorithms.items() if k in selected}
        
        # Feature selection
        selected_features = pipeline.select_features(
            X, y_target, 
            method=data.feature_selection_method.value,
            max_features=data.max_features
        )
        X_selected = X[selected_features]
        
        # Train models
        algorithm_results = []
        for name, algorithm in algorithms.items():
            try:
                result = pipeline.train_model(
                    X_selected, y_target, algorithm, problem_type, data.cv_folds
                )
                
                # Save model
                model_id = str(uuid.uuid4())
                model_path = os.path.join(MODEL_DIR, f"{model_id}.pkl")
                
                # Save model with metadata
                model_data = {
                    'model': result["model"],
                    'scaler': scaler,
                    'label_encoders': label_encoders,
                    'target_encoder': target_encoder,
                    'feature_columns': selected_features,
                    'target_column': target_col,
                    'problem_type': problem_type,
                    'algorithm': name,
                    'metrics': result["metrics"],
                    'preprocessing_info': preprocessing_info
                }
                joblib.dump(model_data, model_path)
                
                algorithm_results.append({
                    "algorithm": name,
                    "model_id": model_id,
                    "metrics": result["metrics"],
                    "cv_scores": result["cv_scores"],
                    "cv_mean": result["cv_mean"],
                    "cv_std": result["cv_std"],
                    "training_time_seconds": result["training_time"],
                    "feature_importance": result["feature_importance"],
                    "status": "success"
                })
            except Exception as e:
                algorithm_results.append({
                    "algorithm": name,
                    "status": "failed",
                    "error_message": str(e)
                })
        
        # Sort by CV mean score
        successful = [r for r in algorithm_results if r["status"] == "success"]
        if successful:
            best = max(successful, key=lambda x: x["cv_mean"])
            target_results[target_col] = SingleTargetResult(
                target_column=target_col,
                problem_type=problem_type,
                best_algorithm=best["algorithm"],
                best_model_id=best["model_id"],
                metrics=best["metrics"],
                cv_scores=best["cv_scores"],
                cv_mean=best["cv_mean"],
                cv_std=best["cv_std"],
                feature_importance=best["feature_importance"],
                training_time_seconds=best["training_time_seconds"]
            )
        
        all_algorithm_results.extend(algorithm_results)
    
    total_time = time.time() - start_time
    
    # Store job
    training_jobs[job_id] = {
        'status': 'completed',
        'target_results': target_results,
        'all_results': all_algorithm_results
    }
    
    return TrainingResponse(
        job_id=job_id,
        dataset_id=data.dataset_id,
        target_columns=data.target_columns,
        status="completed",
        target_results=target_results,
        best_algorithm=list(target_results.values())[0].best_algorithm if target_results else None,
        best_model_id=list(target_results.values())[0].best_model_id if target_results else None,
        all_results=all_algorithm_results,
        completed_at=datetime.now(timezone.utc),
        total_training_time_seconds=total_time,
        selected_features=feature_cols,
        preprocessing_info=preprocessing_info
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    data: PredictionRequest,
    current_user: User = Depends(get_current_user)
):
    """Make predictions with trained models - supports multiple targets."""
    start_time = time.time()
    
    predictions = {}
    
    for model_id in data.model_ids:
        # Load model
        model_path = os.path.join(MODEL_DIR, f"{model_id}.pkl")
        
        if not os.path.exists(model_path):
            continue
        
        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data.get('scaler')
        label_encoders = model_data.get('label_encoders', {})
        feature_columns = model_data['feature_columns']
        target_column = model_data['target_column']
        
        # Prepare data
        df = pd.DataFrame(data.data)
        
        # Select only required features
        X = df[[col for col in feature_columns if col in df.columns]].copy()
        
        # Apply label encoders
        for col, le in label_encoders.items():
            if col in X.columns:
                X[col] = X[col].astype(str)
                # Handle unseen categories
                X[col] = X[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                X[col] = le.transform(X[col])
        
        # Fill missing columns with 0
        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0
        
        X = X[feature_columns]
        
        # Apply scaler if available
        if scaler:
            X_scaled = scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Use only top features if specified
        if data.use_top_features_only and hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[-data.use_top_features_only:]
            X = X.iloc[:, top_indices]
        
        # Make predictions
        preds = model.predict(X).tolist()
        
        # Get confidence if available
        confidence = None
        if hasattr(model, 'predict_proba') and data.return_confidence:
            try:
                proba = model.predict_proba(X)
                confidence = proba.max(axis=1).tolist()
            except:
                pass
        
        # SHAP explanations if requested
        shap_values = None
        if data.explain:
            try:
                import shap
                explainer = shap.TreeExplainer(model) if hasattr(model, 'tree_') else shap.KernelExplainer(model.predict, X)
                shap_vals = explainer.shap_values(X)
                shap_values = [
                    dict(zip(feature_columns, vals.tolist() if hasattr(vals, 'tolist') else vals))
                    for vals in shap_vals
                ]
            except:
                pass
        
        predictions[target_column] = SinglePredictionResult(
            model_id=model_id,
            target_column=target_column,
            predictions=preds,
            confidence=confidence,
            shap_values=shap_values
        )
    
    prediction_time = int((time.time() - start_time) * 1000)
    
    return PredictionResponse(
        predictions=predictions,
        input_rows=len(data.data),
        prediction_time_ms=prediction_time
    )


@router.post("/anomaly-detection", response_model=AnomalyDetectionResponse)
async def detect_anomalies(
    data: AnomalyDetectionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Detect anomalies in data with multiple algorithms."""
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
    
    # Get dataset
    result = await db.execute(
        select(Dataset).where(
            and_(
                Dataset.id == data.dataset_id,
                Dataset.is_active == True
            )
        )
    )
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Load data
    df = await load_dataset_file(data.dataset_id)
    
    # Select columns
    X = df[data.columns].select_dtypes(include=[np.number]).fillna(0)
    
    # Detect anomalies
    contamination = data.contamination
    if data.sensitivity == "low":
        contamination = 0.05
    elif data.sensitivity == "high":
        contamination = 0.2
    
    # Select algorithm
    if data.algorithm == "isolation_forest":
        model = IsolationForest(contamination=contamination, random_state=42)
        labels = model.fit_predict(X)
        scores = model.decision_function(X)
    elif data.algorithm == "local_outlier_factor":
        model = LocalOutlierFactor(contamination=contamination, n_neighbors=20)
        labels = model.fit_predict(X)
        scores = model.negative_outlier_factor_.tolist()
    elif data.algorithm == "one_class_svm":
        model = OneClassSVM(nu=contamination)
        labels = model.fit_predict(X)
        scores = model.decision_function(X).tolist()
    else:
        model = IsolationForest(contamination=contamination, random_state=42)
        labels = model.fit_predict(X)
        scores = model.decision_function(X)
    
    # Get anomalies (label == -1)
    anomaly_indices = np.where(labels == -1)[0]
    threshold = np.percentile(scores, contamination * 100)
    
    anomalies = []
    for idx in anomaly_indices[:100]:  # Limit to 100 anomalies
        row = df.iloc[idx]
        anomalies.append(AnomalyResult(
            index=int(idx),
            values={col: row[col] for col in data.columns},
            score=float(scores[idx]),
            is_anomaly=True
        ))
    
    return AnomalyDetectionResponse(
        dataset_id=data.dataset_id,
        anomalies=anomalies,
        anomaly_count=len(anomaly_indices),
        anomaly_percent=len(anomaly_indices) / len(df) * 100,
        scores=scores.tolist() if hasattr(scores, 'tolist') else scores,
        threshold=float(threshold)
    )


@router.post("/clustering", response_model=ClusteringResponse)
async def cluster_data(
    data: ClusteringRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Perform clustering on data with automatic K selection."""
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    
    # Get dataset
    result = await db.execute(
        select(Dataset).where(
            and_(
                Dataset.id == data.dataset_id,
                Dataset.is_active == True
            )
        )
    )
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Load data
    df = await load_dataset_file(data.dataset_id)
    
    # Select and scale columns
    X = df[data.columns].select_dtypes(include=[np.number]).fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Auto-select K if not provided
    n_clusters = data.n_clusters
    if n_clusters is None and data.auto_select_k:
        best_score = -1
        best_k = 2
        for k in range(2, min(data.max_k + 1, len(X) // 10)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            if len(set(labels)) > 1:
                score = silhouette_score(X_scaled, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        n_clusters = best_k
    elif n_clusters is None:
        n_clusters = 3
    
    # Perform clustering
    inertia = None
    if data.algorithm == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        cluster_centers = scaler.inverse_transform(model.cluster_centers_).tolist()
        inertia = model.inertia_
    elif data.algorithm == "dbscan":
        model = DBSCAN(eps=0.5, min_samples=5)
        labels = model.fit_predict(X_scaled)
        cluster_centers = None
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    elif data.algorithm == "hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X_scaled)
        cluster_centers = None
    elif data.algorithm == "gmm":
        model = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = model.fit_predict(X_scaled)
        cluster_centers = scaler.inverse_transform(model.means_).tolist()
    else:
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        cluster_centers = scaler.inverse_transform(model.cluster_centers_).tolist()
        inertia = model.inertia_
    
    # Calculate silhouette score
    if len(set(labels)) > 1 and -1 not in labels:
        sil_score = silhouette_score(X_scaled, labels)
    else:
        sil_score = 0.0
    
    # Get cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {int(u): int(c) for u, c in zip(unique, counts) if u != -1}
    
    return ClusteringResponse(
        dataset_id=data.dataset_id,
        n_clusters=n_clusters,
        labels=labels.tolist(),
        cluster_sizes=cluster_sizes,
        cluster_centers=cluster_centers,
        silhouette_score=sil_score,
        inertia=inertia,
        algorithm=data.algorithm
    )


@router.post("/feature-importance", response_model=FeatureImportanceResponse)
async def analyze_feature_importance(
    data: FeatureImportanceRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Analyze feature importance using various methods."""
    from sklearn.feature_selection import mutual_info_regression, mutual_info_classif, f_regression, f_classif
    
    # Get dataset
    result = await db.execute(
        select(Dataset).where(
            and_(
                Dataset.id == data.dataset_id,
                Dataset.is_active == True
            )
        )
    )
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Load data
    df = await load_dataset_file(data.dataset_id)
    
    # Prepare data
    X = df[data.feature_columns].select_dtypes(include=[np.number]).fillna(0)
    y = df[data.target_column]
    
    # Encode target if categorical
    if y.dtype == 'object':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
    
    # Calculate importance
    if data.method == "mutual_info":
        if y.dtype == 'object' or len(np.unique(y)) <= 20:
            importances = mutual_info_classif(X, y)
        else:
            importances = mutual_info_regression(X, y)
    elif data.method == "f_score":
        if y.dtype == 'object' or len(np.unique(y)) <= 20:
            importances, _ = f_classif(X, y)
        else:
            importances, _ = f_regression(X, y)
    else:
        importances = mutual_info_regression(X, y)
    
    # Normalize to 0-1
    importances = importances / importances.sum() if importances.sum() > 0 else importances
    
    importance_dict = dict(zip(X.columns, importances.tolist()))
    
    return FeatureImportanceResponse(
        dataset_id=data.dataset_id,
        target_column=data.target_column,
        importances=importance_dict,
        method=data.method
    )


@router.post("/compare-datasets", response_model=DatasetComparisonResponse)
async def compare_datasets(
    data: DatasetComparisonRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Compare multiple datasets."""
    from scipy.stats import ks_2samp
    
    datasets_data = {}
    for ds_id in data.dataset_ids:
        result = await db.execute(
            select(Dataset).where(
                and_(
                    Dataset.id == ds_id,
                    Dataset.is_active == True
                )
            )
        )
        dataset = result.scalar_one_or_none()
        
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {ds_id} not found")
        
        df = await load_dataset_file(ds_id)
        datasets_data[ds_id] = df
    
    # Compare summaries
    summaries = {}
    for ds_id, df in datasets_data.items():
        summaries[ds_id] = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "numeric_summary": df.describe().to_dict()
        }
    
    # Calculate similarities
    similarities = {}
    differences = []
    
    if len(data.dataset_ids) == 2:
        ds1, ds2 = data.dataset_ids
        df1, df2 = datasets_data[ds1], datasets_data[ds2]
        
        # Compare column overlap
        common_cols = set(df1.columns) & set(df2.columns)
        similarities["column_overlap"] = len(common_cols) / max(len(df1.columns), len(df2.columns))
        
        # Compare distributions for common numeric columns
        for col in common_cols:
            if df1[col].dtype in ['int64', 'float64'] and df2[col].dtype in ['int64', 'float64']:
                try:
                    stat, p_value = ks_2samp(df1[col].dropna(), df2[col].dropna())
                    if p_value < 0.05:
                        differences.append({
                            "column": col,
                            "type": "distribution",
                            "description": f"Significant difference in distribution (p={p_value:.4f})"
                        })
                except:
                    pass
    
    return DatasetComparisonResponse(
        datasets=data.dataset_ids,
        comparison_type=data.comparison_type,
        results=summaries,
        similarities=similarities,
        differences=differences
    )


@router.get("/models", response_model=List[ModelInfo])
async def list_models(
    current_user: User = Depends(get_current_user)
):
    """List trained models for user."""
    models = []
    
    for filename in os.listdir(MODEL_DIR):
        if filename.endswith(".pkl"):
            model_id = filename[:-4]
            try:
                model_data = joblib.load(os.path.join(MODEL_DIR, filename))
                models.append(ModelInfo(
                    id=model_id,
                    name=f"Model {model_id[:8]}",
                    dataset_id=model_data.get('dataset_id', 'unknown'),
                    target_column=model_data.get('target_column', 'unknown'),
                    problem_type=model_data.get('problem_type', 'regression'),
                    algorithm=model_data.get('algorithm', 'unknown'),
                    metrics=model_data.get('metrics', {}),
                    cv_scores=[],
                    feature_columns=model_data.get('feature_columns', []),
                    feature_importance={},
                    status="ready",
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc)
                ))
            except:
                pass
    
    return models


@router.get("/training-jobs/{job_id}")
async def get_training_job(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get training job status."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    return training_jobs[job_id]


@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a trained model."""
    model_path = os.path.join(MODEL_DIR, f"{model_id}.pkl")
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    
    os.remove(model_path)
    
    return {"message": "Model deleted successfully"}
