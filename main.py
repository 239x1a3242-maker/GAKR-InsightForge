"""
Main FastAPI app for AI Data Analysis Platform
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import numpy as np

from data_service import process_dataset
from ml_service import train_models, save_model
from viz_service import (
    plot_model_comparison,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_clusters,
    get_viz_plan,
)
from ai_insights_service import generate_structured_narrative

app = FastAPI(title="AI Data Analysis Platform")

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

TEMP_VIZ_DIR = Path("data/temp")
TEMP_VIZ_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/analyze/")
async def analyze_file(
    file: UploadFile = File(...),
    target_column: str = Form(None),
):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 1) DATA
        data_results = process_dataset(
            file_path=str(file_path),
            target_column=target_column,
            generate_html_report=True,
        )
        cleaned_df = data_results["cleaned_dataframe"]
        processed_data = data_results["processed_data"]
        profile_summary = data_results["profile_summary"]

        # 2) ML
        ml_results = train_models(processed_data)
        best_model = ml_results["best_model"]
        best_model_name = ml_results["best_model_name"]
        task_type = ml_results["task"]
        leaderboard = ml_results["leaderboard"]
        feature_names = ml_results.get("feature_names", processed_data.get("feature_names", []))

        save_model(best_model, task=task_type, model_name=best_model_name)

        y_pred_sample = None
        y_true_sample = None
        preds_full = None
        y_test = None

        if task_type != "unsupervised":
            X_test = processed_data["X_test"]
            y_test = processed_data["y_test"]
            preds_full = best_model.predict(X_test)
            y_pred_sample = preds_full[:20].tolist()
            y_true_sample = np.array(y_test)[:20].tolist()

        # 3) VIZ (images + recommendations)
        viz_results = {}

        model_comparison_path = TEMP_VIZ_DIR / "model_comparison.png"
        plot_model_comparison(leaderboard, save_path=str(model_comparison_path))
        viz_results["model_comparison"] = str(model_comparison_path)

        if best_model is not None and hasattr(best_model, "feature_importances_") and feature_names:
            fi_path = TEMP_VIZ_DIR / "feature_importance.png"
            plot_feature_importance(best_model, feature_names, save_path=str(fi_path))
            viz_results["feature_importance"] = str(fi_path)

        if task_type != "unsupervised" and preds_full is not None and y_test is not None:
            avp_path = TEMP_VIZ_DIR / "actual_vs_predicted.png"
            plot_actual_vs_predicted(y_test, preds_full, save_path=str(avp_path))
            viz_results["actual_vs_predicted"] = str(avp_path)

        if task_type == "unsupervised":
            X_all = processed_data["X_processed"]
            labels = best_model.fit_predict(X_all)
            cluster_path = TEMP_VIZ_DIR / "cluster_plot.png"
            plot_clusters(X_all, labels, save_path=str(cluster_path))
            viz_results["cluster_plot"] = str(cluster_path)

        viz_plan = get_viz_plan(cleaned_df, target_column, ml_task=task_type)
        viz_results["recommendations"] = viz_plan
        viz_results["chart_types"] = viz_plan  # for analysis_payload

        # 4) Build analysis_payload for AI
        analysis_payload = {
            "dataset_summary": {
                "rows": profile_summary.get("rows"),
                "columns": profile_summary.get("columns"),
                "memory_mb": profile_summary.get("memory_mb"),
            },
            "data_quality": {
                "missing_pct": profile_summary.get("missing_pct"),
                "duplicates": profile_summary.get("duplicates"),
            },
            "kpis": {},  # could be filled from numeric describe if desired
            "model_info": {
                "name": best_model_name,
                "task": task_type,
            },
            "model_metrics": ml_results["metrics"],
            "feature_importance": (
                dict(zip(feature_names, best_model.feature_importances_))
                if best_model is not None and hasattr(best_model, "feature_importances_") and feature_names
                else {}
            ),
            "predictions_summary": {
                "y_true_sample": y_true_sample,
                "y_pred_sample": y_pred_sample,
            },
            "anomalies": None,
            "risks": None,
            "charts": viz_results["chart_types"],
        }

        structured_narrative = generate_structured_narrative(analysis_payload)

        # 5) Final response
        response = {
            "success": True,
            "file_name": file.filename,
            "data_results": {
                "profile_summary": profile_summary,
                "cleaned_shape": list(cleaned_df.shape),
                "eda_report_path": data_results.get("eda_report_path"),
            },
            "ml_results": {
                "task": task_type,
                "best_model_name": best_model_name,
                "leaderboard": leaderboard,
                "metrics": ml_results["metrics"],
                "y_true_sample": y_true_sample,
                "y_pred_sample": y_pred_sample,
            },
            "viz_results": viz_results,
            "ai_results": {
                "structured_narrative": structured_narrative,
            },
        }

        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)
