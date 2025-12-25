"""
Visualization Service:
- Consumes outputs from data_service.py and ml_service.py
- Produces visual artifacts (static figures, interactive figures)
- Recommends chart types based on data + task
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA

sns.set(style="whitegrid")


def plot_model_comparison(leaderboard: List[Dict[str, Any]], save_path: Optional[str] = None):
    print("Plotting model comparison...")
    if not leaderboard:
        raise ValueError("Empty leaderboard provided to plot_model_comparison")
    
    df = pd.DataFrame(leaderboard)
    plt.figure(figsize=(12, 8))
    
    # Check if we have CV data
    has_cv = any('CV_Mean_F1' in str(m) for m in leaderboard)
    
    if has_cv:
        # Plot with error bars for CV results
        means = []
        stds = []
        models = []
        for item in leaderboard:
            if 'CV_Mean_F1' in item:
                means.append(item['CV_Mean_F1'])
                stds.append(item.get('CV_Std_F1', 0))
                models.append(item['model'])
        
        if means:
            plt.barh(models, means, xerr=stds, capsize=5, alpha=0.7, color='skyblue')
            plt.xlabel("F1 Score (Cross-Validation Mean ± Std)")
            plt.title("Model Comparison with Cross-Validation\n(Higher is better)")
        else:
            # Fallback to regular barplot
            sns.barplot(data=df, x="score", y="model", orient="h")
            plt.xlabel("Score")
    else:
        sns.barplot(data=df, x="score", y="model", orient="h")
        plt.xlabel("Score")
    
    plt.ylabel("Model")
    plt.title(f"Model Comparison ({df['primary_metric'].iloc[0]})")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm: List[List[int]], class_names: Optional[List[str]] = None, save_path: Optional[str] = None):
    """Plot confusion matrix for classification results"""
    print("Plotting confusion matrix...")
    cm = np.array(cm)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names or [f'Class {i}' for i in range(cm.shape[1])],
                yticklabels=class_names or [f'Class {i}' for i in range(cm.shape[0])])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_actual_vs_predicted(y_true, y_pred, save_path: Optional[str] = None):
    print("Plotting Actual vs Predicted...")
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color="red", linestyle="--", label="Ideal")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_feature_importance(model: Any, feature_names: List[str], top_n: int = 20, save_path: Optional[str] = None):
    print("Plotting feature importance...")
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not support feature importance (no feature_importances_).")
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances.values, y=importances.index, orient="h")
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_clusters(X: np.ndarray, labels: np.ndarray, save_path: Optional[str] = None):
    print("Plotting clusters...")
    if X.shape[0] < 2:
        raise ValueError("Need at least 2 samples for PCA-based cluster visualization.")
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    df = pd.DataFrame({"PC1": X_2d[:, 0], "PC2": X_2d[:, 1], "Cluster": labels.astype(str)})
    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="Cluster", palette="tab10")
    plt.title("Cluster Visualization (PCA)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def interactive_metric_dashboard(leaderboard: List[Dict[str, Any]]):
    print("Creating interactive metric dashboard...")
    if not leaderboard:
        raise ValueError("Empty leaderboard provided to interactive_metric_dashboard")
    df = pd.DataFrame(leaderboard)
    fig = px.bar(
        df,
        x="score",
        y="model",
        orientation="h",
        color="score",
        title="Model Performance Dashboard",
        labels={"score": df["primary_metric"].iloc[0], "model": "Model"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig


def recommend_charts(df: pd.DataFrame, target: Optional[str]) -> Dict[str, List[str]]:
    print("Recommending chart types...")
    recommendations = {
        "kpi": [],
        "comparison": [],
        "composition": [],
        "trend": [],
        "relationship": [],
        "distribution": [],
        "clustering": [],
        "model": [],
    }

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()


    if target and target in numeric_cols:
        recommendations["kpi"].append("KPI Card")
        recommendations["distribution"].extend(["Histogram", "Box Plot"])

    if categorical_cols and target and target in numeric_cols:
        recommendations["comparison"].extend(["Bar Chart", "Grouped Bar", "Stacked Bar", "Column Chart"])
        recommendations["composition"].extend(["Pie Chart", "Donut Chart", "Treemap"])

    if datetime_cols and target and target in numeric_cols:
        recommendations["trend"].extend(["Line Chart", "Area Chart", "Stacked Area"])

    if len(numeric_cols) >= 2:
        recommendations["relationship"].extend(["Scatter Plot", "Bubble Chart", "Correlation Heatmap"])

    if numeric_cols and not recommendations["distribution"]:
        recommendations["distribution"].append("Histogram")

    return recommendations


def get_viz_plan(df: pd.DataFrame, target: Optional[str], ml_task: Optional[str] = None) -> Dict[str, Any]:
    print("Generating visualization plan...")
    rec = recommend_charts(df, target)
    if ml_task in ("regression", "classification"):
        rec["model"].extend(["Actual vs Predicted", "Model Comparison Bar Chart"])
        if ml_task == "classification":
            rec["model"].append("Confusion Matrix")
    elif ml_task == "unsupervised":
        rec["clustering"].extend(["PCA Scatter", "Cluster Plot"])
    return rec
