"""Machine learning schemas."""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ProblemType(str, Enum):
    """ML problem types."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"
    ANOMALY_DETECTION = "anomaly_detection"
    AUTO = "auto"


class AlgorithmType(str, Enum):
    """ML algorithm types."""
    # Regression
    RANDOM_FOREST_REGRESSOR = "random_forest_regressor"
    GRADIENT_BOOSTING_REGRESSOR = "gradient_boosting_regressor"
    LINEAR_REGRESSION = "linear_regression"
    RIDGE_REGRESSION = "ridge_regression"
    LASSO_REGRESSION = "lasso_regression"
    ELASTIC_NET = "elastic_net"
    SVR = "svr"
    KNN_REGRESSOR = "knn_regressor"
    EXTRA_TREES_REGRESSOR = "extra_trees_regressor"
    ADABOOST_REGRESSOR = "adaboost_regressor"
    XGBOOST_REGRESSOR = "xgboost_regressor"
    LIGHTGBM_REGRESSOR = "lightgbm_regressor"
    
    # Classification
    RANDOM_FOREST_CLASSIFIER = "random_forest_classifier"
    GRADIENT_BOOSTING_CLASSIFIER = "gradient_boosting_classifier"
    LOGISTIC_REGRESSION = "logistic_regression"
    SVC = "svc"
    KNN_CLASSIFIER = "knn_classifier"
    NAIVE_BAYES = "naive_bayes"
    DECISION_TREE = "decision_tree"
    EXTRA_TREES_CLASSIFIER = "extra_trees_classifier"
    ADABOOST_CLASSIFIER = "adaboost_classifier"
    SGD_CLASSIFIER = "sgd_classifier"
    XGBOOST_CLASSIFIER = "xgboost_classifier"
    LIGHTGBM_CLASSIFIER = "lightgbm_classifier"


class FeatureSelectionMethod(str, Enum):
    """Feature selection methods."""
    ALL = "all"
    CORRELATION = "correlation"
    MUTUAL_INFO = "mutual_info"
    CHI2 = "chi2"
    RFE = "rfe"
    SELECT_K_BEST = "select_k_best"


class TrainingRequest(BaseModel):
    """Model training request schema."""
    dataset_id: str
    target_columns: List[str] = Field(..., min_length=1, description="One or more target columns to predict")
    feature_columns: Optional[List[str]] = Field(None, description="Features to use (auto-select if not provided)")
    
    problem_type: ProblemType = ProblemType.AUTO
    algorithms: Optional[List[AlgorithmType]] = None  # None = use all
    
    # Training parameters
    test_size: float = Field(0.2, ge=0.1, le=0.5)
    cv_folds: int = Field(5, ge=2, le=10)
    
    # Hyperparameter tuning
    hyperparameter_tuning: bool = False
    tuning_strategy: str = "grid"  # grid, random, bayesian
    max_iterations: int = Field(50, ge=10, le=500)
    
    # Feature engineering
    auto_feature_engineering: bool = True
    feature_selection_method: FeatureSelectionMethod = FeatureSelectionMethod.ALL
    max_features: Optional[int] = Field(None, ge=1, le=100)
    
    # Preprocessing
    handle_missing: str = "auto"  # auto, drop, impute_mean, impute_median, impute_mode
    scaling: str = "auto"  # auto, standard, minmax, robust, none
    encode_categorical: bool = True
    
    # Time series specific
    date_column: Optional[str] = None
    forecast_horizon: Optional[int] = Field(None, ge=1, le=365)
    
    # Multi-target specific
    train_separate_models: bool = True  # Train separate model for each target


class SingleTargetResult(BaseModel):
    """Result for a single target column."""
    target_column: str
    problem_type: str
    best_algorithm: str
    best_model_id: str
    metrics: Dict[str, float]
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    feature_importance: Dict[str, float]
    training_time_seconds: float


class AlgorithmResult(BaseModel):
    """Single algorithm training result."""
    algorithm: str
    status: str  # success, failed
    error_message: Optional[str] = None
    
    # Metrics
    metrics: Dict[str, float] = {}
    
    # Cross-validation scores
    cv_scores: List[float] = []
    cv_mean: float
    cv_std: float
    
    # Training time
    training_time_seconds: float
    
    # Hyperparameters used
    hyperparameters: Dict[str, Any] = {}


class TrainingResponse(BaseModel):
    """Model training response schema."""
    job_id: str
    dataset_id: str
    target_columns: List[str]
    
    status: str  # pending, running, completed, failed
    
    # Results for each target
    target_results: Dict[str, SingleTargetResult] = {}
    
    # Overall results
    best_algorithm: Optional[str] = None
    best_model_id: Optional[str] = None
    all_results: List[AlgorithmResult] = []
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_training_time_seconds: float = 0
    
    # Selected features
    selected_features: List[str] = []
    
    # Preprocessing info
    preprocessing_info: Dict[str, Any] = {}


class ModelInfo(BaseModel):
    """Trained model information."""
    id: str
    name: str
    description: Optional[str] = None
    
    dataset_id: str
    target_column: str
    problem_type: ProblemType
    algorithm: AlgorithmType
    
    # Performance
    metrics: Dict[str, float]
    cv_scores: List[float]
    
    # Features
    feature_columns: List[str]
    feature_importance: Dict[str, float]
    
    # Status
    status: str  # training, ready, deployed, archived
    
    created_at: datetime
    updated_at: datetime


class PredictionRequest(BaseModel):
    """Prediction request schema."""
    model_ids: List[str]  # Can predict with multiple models
    data: List[Dict[str, Any]]
    return_confidence: bool = False
    explain: bool = False
    use_top_features_only: Optional[int] = None  # Use only top N features


class SinglePredictionResult(BaseModel):
    """Prediction result for a single model."""
    model_id: str
    target_column: str
    predictions: List[Any]
    confidence: Optional[List[float]] = None
    explanations: Optional[List[Dict[str, Any]]] = None
    shap_values: Optional[List[Dict[str, float]]] = None


class PredictionResponse(BaseModel):
    """Prediction response schema."""
    predictions: Dict[str, SinglePredictionResult]  # Keyed by target column
    input_rows: int
    prediction_time_ms: int


class ForecastRequest(BaseModel):
    """Time series forecast request schema."""
    model_id: str
    periods: int = Field(..., ge=1, le=365)
    confidence_interval: float = Field(0.95, ge=0.8, le=0.99)
    include_history: bool = True


class ForecastResponse(BaseModel):
    """Time series forecast response schema."""
    model_id: str
    dates: List[str]
    forecast: List[float]
    lower_bound: List[float]
    upper_bound: List[float]
    history_dates: Optional[List[str]] = None
    history_values: Optional[List[float]] = None


class AnomalyDetectionRequest(BaseModel):
    """Anomaly detection request schema."""
    dataset_id: str
    columns: List[str]
    sensitivity: str = "medium"  # low, medium, high
    contamination: float = Field(0.1, ge=0.01, le=0.5)
    algorithm: str = "isolation_forest"  # isolation_forest, local_outlier_factor, one_class_svm


class AnomalyResult(BaseModel):
    """Single anomaly result."""
    index: int
    values: Dict[str, Any]
    score: float
    is_anomaly: bool


class AnomalyDetectionResponse(BaseModel):
    """Anomaly detection response schema."""
    dataset_id: str
    anomalies: List[AnomalyResult]
    anomaly_count: int
    anomaly_percent: float
    scores: List[float]
    threshold: float


class ClusteringRequest(BaseModel):
    """Clustering request schema."""
    dataset_id: str
    columns: List[str]
    n_clusters: Optional[int] = Field(None, ge=2, le=50)  # Auto if None
    algorithm: str = "kmeans"  # kmeans, dbscan, hierarchical, gmm
    auto_select_k: bool = True
    max_k: int = Field(10, ge=2, le=50)


class ClusteringResponse(BaseModel):
    """Clustering response schema."""
    dataset_id: str
    n_clusters: int
    labels: List[int]
    cluster_sizes: Dict[int, int]
    cluster_centers: Optional[List[List[float]]] = None
    silhouette_score: float
    inertia: Optional[float] = None
    algorithm: str


class FeatureImportanceRequest(BaseModel):
    """Feature importance analysis request."""
    dataset_id: str
    target_column: str
    feature_columns: List[str]
    method: str = "mutual_info"  # mutual_info, f_regression, chi2, permutation


class FeatureImportanceResponse(BaseModel):
    """Feature importance response."""
    dataset_id: str
    target_column: str
    importances: Dict[str, float]
    method: str


class DatasetComparisonRequest(BaseModel):
    """Dataset comparison request."""
    dataset_ids: List[str]
    columns: Optional[List[str]] = None
    comparison_type: str = "summary"  # summary, correlation, distribution, trend


class DatasetComparisonResponse(BaseModel):
    """Dataset comparison response."""
    datasets: List[str]
    comparison_type: str
    results: Dict[str, Any]
    similarities: Dict[str, float]
    differences: List[Dict[str, Any]]


class ModelComparisonRequest(BaseModel):
    """Model comparison request schema."""
    model_ids: List[str]
    test_dataset_id: Optional[str] = None


class ModelComparisonResponse(BaseModel):
    """Model comparison response schema."""
    models: List[ModelInfo]
    comparison_metrics: Dict[str, List[float]]
    best_model_id: str
    recommendation: str
