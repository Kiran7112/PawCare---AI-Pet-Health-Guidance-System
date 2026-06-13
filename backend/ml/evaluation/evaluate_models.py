# ml/evaluation/evaluate_models.py
"""
Model Evaluation Module for PawCare+ ML pipeline.
Evaluates trained models on held-out evaluation datasets.
"""

import pandas as pd
import numpy as np
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

# Configure logging
logger = logging.getLogger(__name__)

# Feature definitions (must match training)
HEALTH_RISK_CATEGORICAL = ["pet_species", "weight_status", "living_situation", "exercise_level"]
HEALTH_RISK_NUMERICAL = ["age_years", "conditions_count", "allergies_count"]
HEALTH_RISK_TARGET = "health_risk_score"

CAPABILITY_CATEGORICAL = ["owner_experience", "vet_access", "owner_commitment"]
CAPABILITY_TARGET = "care_capability_score"


def evaluate_all_models(
    evaluation_dir: str,
    model_dir: str
) -> Dict[str, Any]:
    """
    Evaluate trained models on evaluation datasets.
    
    This function:
    1. Loads evaluation datasets for both models
    2. Loads trained models, scalers, and encoders
    3. Generates predictions
    4. Calculates comprehensive metrics
    5. Returns evaluation results for both models
    
    Args:
        evaluation_dir: Path to directory containing evaluation CSV files
                       Expected files: pet_health_risk_evaluation.csv, owner_care_capability_evaluation.csv
        model_dir: Path to directory containing saved model artifacts
        
    Returns:
        Dictionary containing:
            - models: Dictionary with evaluation results for each model
            - health_risk: Dictionary with metrics for health risk model
            - care_capability: Dictionary with metrics for care capability model
            - summary: Overall summary of evaluation
            - timestamp: ISO format timestamp
    """
    logger.info("=" * 60)
    logger.info("EVALUATING TRAINED MODELS")
    logger.info("=" * 60)
    logger.info(f"Evaluation directory: {evaluation_dir}")
    logger.info(f"Model directory: {model_dir}")
    
    results = {
        "models": {},
        "health_risk": {},
        "care_capability": {},
        "summary": {},
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    # ==========================================
    # EVALUATE HEALTH RISK MODEL
    # ==========================================
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATING HEALTH RISK MODEL")
    logger.info("=" * 50)
    
    try:
        health_risk_results = evaluate_health_risk_model(evaluation_dir, model_dir)
        results["health_risk"] = health_risk_results
        results["models"]["health_risk"] = health_risk_results
        logger.info(f"✓ Health risk model evaluation complete")
    except Exception as e:
        logger.error(f"❌ Failed to evaluate health risk model: {str(e)}")
        results["health_risk"] = {
            "status": "failed",
            "error": str(e)
        }
    
    # ==========================================
    # EVALUATE CARE CAPABILITY MODEL
    # ==========================================
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATING CARE CAPABILITY MODEL")
    logger.info("=" * 50)
    
    try:
        capability_results = evaluate_care_capability_model(evaluation_dir, model_dir)
        results["care_capability"] = capability_results
        results["models"]["care_capability"] = capability_results
        logger.info(f"✓ Care capability model evaluation complete")
    except Exception as e:
        logger.error(f"❌ Failed to evaluate care capability model: {str(e)}")
        results["care_capability"] = {
            "status": "failed",
            "error": str(e)
        }
    
    # ==========================================
    # GENERATE SUMMARY
    # ==========================================
    results["summary"] = generate_evaluation_summary(results)
    
    # ==========================================
    # SAVE RESULTS
    # ==========================================
    save_evaluation_results(results, model_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    
    return results


def evaluate_health_risk_model(
    evaluation_dir: str,
    model_dir: str
) -> Dict[str, Any]:
    """
    Evaluate the pet health risk model on evaluation dataset.
    
    Args:
        evaluation_dir: Directory containing evaluation CSV files
        model_dir: Directory containing saved model artifacts
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("\n📊 Evaluating Health Risk Model...")
    
    results = {
        "status": "failed",
        "test_samples": 0,
        "r2_score": None,
        "mae": None,
        "rmse": None,
        "mse": None,
        "additional_metrics": {},
        "feature_importance": None
    }
    
    # ==========================================
    # LOAD EVALUATION DATA
    # ==========================================
    eval_file = os.path.join(evaluation_dir, "pet_health_risk_evaluation.csv")
    
    if not os.path.exists(eval_file):
        error_msg = f"Evaluation file not found: {eval_file}"
        logger.error(error_msg)
        results["error"] = error_msg
        return results
    
    logger.info(f"Loading evaluation data from: {eval_file}")
    df_eval = pd.read_csv(eval_file)
    logger.info(f"Loaded {len(df_eval):,} evaluation samples")
    
    results["test_samples"] = len(df_eval)
    
    # ==========================================
    # LOAD MODEL ARTIFACTS
    # ==========================================
    logger.info("Loading model artifacts...")
    
    model_path = os.path.join(model_dir, "pet_health_risk_model.pkl")
    scaler_path = os.path.join(model_dir, "pet_health_risk_scaler.pkl")
    encoders_path = os.path.join(model_dir, "pet_health_risk_encoder.pkl")
    
    if not all(os.path.exists(p) for p in [model_path, scaler_path, encoders_path]):
        error_msg = "Missing model artifacts"
        logger.error(error_msg)
        results["error"] = error_msg
        return results
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)
    
    logger.info(f"Model type: {type(model).__name__}")
    
    # ==========================================
    # PREPARE FEATURES AND TARGET
    # ==========================================
    logger.info("Preparing features...")
    
    # Check for required columns
    required_cols = HEALTH_RISK_CATEGORICAL + HEALTH_RISK_NUMERICAL + [HEALTH_RISK_TARGET]
    missing_cols = [col for col in required_cols if col not in df_eval.columns]
    
    if missing_cols:
        error_msg = f"Missing required columns: {missing_cols}"
        logger.error(error_msg)
        results["error"] = error_msg
        return results
    
    # Extract features and target
    X_categorical = df_eval[HEALTH_RISK_CATEGORICAL].copy()
    X_numerical = df_eval[HEALTH_RISK_NUMERICAL].copy()
    y_true = df_eval[HEALTH_RISK_TARGET].copy()
    
    # Handle any missing values
    X_categorical = X_categorical.fillna('unknown')
    X_numerical = X_numerical.fillna(X_numerical.median())
    
    # Encode categorical features
    X_categorical_encoded = pd.DataFrame()
    
    for col in HEALTH_RISK_CATEGORICAL:
        encoder = encoders[col]
        
        # Handle unseen labels
        def encode_value(val):
            try:
                return encoder.transform([val])[0]
            except ValueError:
                # Use most common class for unseen labels
                return encoder.transform([encoder.classes_[0]])[0]
        
        X_categorical_encoded[col] = X_categorical[col].apply(encode_value)
    
    # Build feature matrix
    X = pd.concat([X_categorical_encoded, X_numerical], axis=1)
    
    logger.info(f"Feature matrix shape: {X.shape}")
    
    # ==========================================
    # SCALE FEATURES
    # ==========================================
    X_scaled = scaler.transform(X)
    
    # ==========================================
    # GENERATE PREDICTIONS
    # ==========================================
    logger.info("Generating predictions...")
    y_pred = model.predict(X_scaled)
    
    # Clip predictions to valid range
    y_pred = np.clip(y_pred, 0.0, 1.0)
    
    # ==========================================
    # CALCULATE METRICS
    # ==========================================
    logger.info("Calculating metrics...")
    
    # Basic metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    logger.info(f"  MSE:  {mse:.6f}")
    logger.info(f"  MAE:  {mae:.6f}")
    logger.info(f"  RMSE: {rmse:.6f}")
    logger.info(f"  R²:   {r2:.4f}")
    
    results.update({
        "status": "success",
        "r2_score": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
        "mse": float(mse)
    })
    
    # ==========================================
    # ADDITIONAL METRICS
    # ==========================================
    
    # Prediction error distribution
    errors = y_pred - y_true
    results["additional_metrics"]["error_distribution"] = {
        "mean_error": float(errors.mean()),
        "std_error": float(errors.std()),
        "min_error": float(errors.min()),
        "max_error": float(errors.max()),
        "percent_within_0_1": float((abs(errors) < 0.1).mean()),
        "percent_within_0_2": float((abs(errors) < 0.2).mean())
    }
    
    # Accuracy by risk category
    categories = pd.cut(y_true, bins=[0, 0.3, 0.6, 1.0], 
                        labels=["Low Risk", "Moderate Risk", "High Risk"])
    
    category_accuracy = {}
    for category in ["Low Risk", "Moderate Risk", "High Risk"]:
        mask = categories == category
        if mask.sum() > 0:
            category_accuracy[category] = {
                "count": int(mask.sum()),
                "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
                "rmse": float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
            }
    
    results["additional_metrics"]["accuracy_by_risk_category"] = category_accuracy
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        feature_names = HEALTH_RISK_CATEGORICAL + HEALTH_RISK_NUMERICAL
        results["feature_importance"] = {
            name: float(imp) 
            for name, imp in zip(feature_names, model.feature_importances_)
        }
    
    return results


def evaluate_care_capability_model(
    evaluation_dir: str,
    model_dir: str
) -> Dict[str, Any]:
    """
    Evaluate the owner care capability model on evaluation dataset.
    
    Args:
        evaluation_dir: Directory containing evaluation CSV files
        model_dir: Directory containing saved model artifacts
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("\n📊 Evaluating Care Capability Model...")
    
    results = {
        "status": "failed",
        "test_samples": 0,
        "r2_score": None,
        "mae": None,
        "rmse": None,
        "mse": None,
        "additional_metrics": {},
        "feature_importance": None
    }
    
    # ==========================================
    # LOAD EVALUATION DATA
    # ==========================================
    eval_file = os.path.join(evaluation_dir, "owner_care_capability_evaluation.csv")
    
    if not os.path.exists(eval_file):
        error_msg = f"Evaluation file not found: {eval_file}"
        logger.error(error_msg)
        results["error"] = error_msg
        return results
    
    logger.info(f"Loading evaluation data from: {eval_file}")
    df_eval = pd.read_csv(eval_file)
    logger.info(f"Loaded {len(df_eval):,} evaluation samples")
    
    results["test_samples"] = len(df_eval)
    
    # ==========================================
    # LOAD MODEL ARTIFACTS
    # ==========================================
    logger.info("Loading model artifacts...")
    
    model_path = os.path.join(model_dir, "owner_care_capability_model.pkl")
    scaler_path = os.path.join(model_dir, "owner_care_capability_scaler.pkl")
    encoders_path = os.path.join(model_dir, "owner_care_capability_encoder.pkl")
    
    if not all(os.path.exists(p) for p in [model_path, scaler_path, encoders_path]):
        error_msg = "Missing model artifacts"
        logger.error(error_msg)
        results["error"] = error_msg
        return results
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)
    
    logger.info(f"Model type: {type(model).__name__}")
    
    # ==========================================
    # PREPARE FEATURES AND TARGET
    # ==========================================
    logger.info("Preparing features...")
    
    # Check for required columns
    required_cols = CAPABILITY_CATEGORICAL + [CAPABILITY_TARGET]
    missing_cols = [col for col in required_cols if col not in df_eval.columns]
    
    if missing_cols:
        error_msg = f"Missing required columns: {missing_cols}"
        logger.error(error_msg)
        results["error"] = error_msg
        return results
    
    # Extract features and target
    X_categorical = df_eval[CAPABILITY_CATEGORICAL].copy()
    y_true = df_eval[CAPABILITY_TARGET].copy()
    
    # Handle any missing values
    X_categorical = X_categorical.fillna('unknown')
    
    # Encode categorical features
    X_encoded = pd.DataFrame()
    
    for col in CAPABILITY_CATEGORICAL:
        encoder = encoders[col]
        
        # Handle unseen labels
        def encode_value(val):
            try:
                return encoder.transform([val])[0]
            except ValueError:
                # Use most common class for unseen labels
                return encoder.transform([encoder.classes_[0]])[0]
        
        X_encoded[col] = X_categorical[col].apply(encode_value)
    
    logger.info(f"Feature matrix shape: {X_encoded.shape}")
    
    # ==========================================
    # SCALE FEATURES
    # ==========================================
    X_scaled = scaler.transform(X_encoded)
    
    # ==========================================
    # GENERATE PREDICTIONS
    # ==========================================
    logger.info("Generating predictions...")
    y_pred = model.predict(X_scaled)
    
    # Clip predictions to valid range
    y_pred = np.clip(y_pred, 0.0, 100.0)
    
    # ==========================================
    # CALCULATE METRICS
    # ==========================================
    logger.info("Calculating metrics...")
    
    # Basic metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    logger.info(f"  MSE:  {mse:.2f}")
    logger.info(f"  MAE:  {mae:.2f}")
    logger.info(f"  RMSE: {rmse:.2f}")
    logger.info(f"  R²:   {r2:.4f}")
    
    results.update({
        "status": "success",
        "r2_score": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
        "mse": float(mse)
    })
    
    # ==========================================
    # ADDITIONAL METRICS
    # ==========================================
    
    # Prediction error distribution
    errors = y_pred - y_true
    results["additional_metrics"]["error_distribution"] = {
        "mean_error": float(errors.mean()),
        "std_error": float(errors.std()),
        "min_error": float(errors.min()),
        "max_error": float(errors.max()),
        "percent_within_10": float((abs(errors) < 10).mean()),
        "percent_within_20": float((abs(errors) < 20).mean())
    }
    
    # Accuracy by capability level
    bins = [0, 33, 66, 100]
    labels = ["Low Capability", "Medium Capability", "High Capability"]
    categories = pd.cut(y_true, bins=bins, labels=labels, include_lowest=True)
    
    category_accuracy = {}
    for category in labels:
        mask = categories == category
        if mask.sum() > 0:
            category_accuracy[category] = {
                "count": int(mask.sum()),
                "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
                "rmse": float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
            }
    
    results["additional_metrics"]["accuracy_by_capability_level"] = category_accuracy
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        results["feature_importance"] = {
            name: float(imp) 
            for name, imp in zip(CAPABILITY_CATEGORICAL, model.feature_importances_)
        }
    
    return results


def generate_evaluation_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a summary of evaluation results.
    
    Args:
        results: Complete evaluation results dictionary
        
    Returns:
        Dictionary with summary information
    """
    summary = {
        "models_evaluated": [],
        "overall_status": "success",
        "comparison": {}
    }
    
    health = results.get("health_risk", {})
    capability = results.get("care_capability", {})
    
    if health.get("status") == "success":
        summary["models_evaluated"].append("health_risk")
    
    if capability.get("status") == "success":
        summary["models_evaluated"].append("care_capability")
    
    if not summary["models_evaluated"]:
        summary["overall_status"] = "failed"
    
    # Comparison if both models succeeded
    if "health_risk" in summary["models_evaluated"] and "care_capability" in summary["models_evaluated"]:
        summary["comparison"] = {
            "health_risk_r2": health.get("r2_score"),
            "care_capability_r2": capability.get("r2_score"),
            "better_model": "health_risk" if health.get("r2_score", 0) > capability.get("r2_score", 0) else "care_capability"
        }
    
    return summary


def save_evaluation_results(results: Dict[str, Any], model_dir: str) -> None:
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Evaluation results dictionary
        model_dir: Directory to save results
    """
    try:
        output_file = os.path.join(model_dir, "evaluation_results.json")
        
        # Create a serializable copy
        serializable_results = json.loads(
            json.dumps(results, default=str)
        )
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"\n💾 Evaluation results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to save evaluation results: {str(e)}")


def print_evaluation_report(results: Dict[str, Any]) -> None:
    """
    Print a formatted evaluation report.
    
    Args:
        results: Evaluation results dictionary
    """
    print("\n" + "=" * 70)
    print("MODEL EVALUATION REPORT")
    print("=" * 70)
    
    # Health Risk Model
    health = results.get("health_risk", {})
    if health.get("status") == "success":
        print("\n🔴 PET HEALTH RISK MODEL")
        print("-" * 40)
        print(f"Test samples: {health.get('test_samples', 0):,}")
        print(f"R² Score:     {health.get('r2_score', 0):.4f}")
        print(f"MAE:          {health.get('mae', 0):.4f}")
        print(f"RMSE:         {health.get('rmse', 0):.4f}")
        
        # Error distribution
        err_dist = health.get("additional_metrics", {}).get("error_distribution", {})
        print(f"\nError Distribution:")
        print(f"  Mean error:     {err_dist.get('mean_error', 0):.4f}")
        print(f"  Within 0.1:      {err_dist.get('percent_within_0_1', 0):.1%}")
        print(f"  Within 0.2:      {err_dist.get('percent_within_0_2', 0):.1%}")
        
        # Category accuracy
        cat_acc = health.get("additional_metrics", {}).get("accuracy_by_risk_category", {})
        if cat_acc:
            print(f"\nAccuracy by Risk Category:")
            for category, metrics in cat_acc.items():
                print(f"  {category}:")
                print(f"    MAE:  {metrics.get('mae', 0):.4f}")
                print(f"    RMSE: {metrics.get('rmse', 0):.4f}")
        
        # Feature importance
        fi = health.get("feature_importance", {})
        if fi:
            print(f"\nFeature Importance:")
            for feature, importance in sorted(fi.items(), key=lambda x: x[1], reverse=True):
                print(f"  {feature}: {importance:.3f}")
    
    # Care Capability Model
    capability = results.get("care_capability", {})
    if capability.get("status") == "success":
        print("\n🟡 OWNER CARE CAPABILITY MODEL")
        print("-" * 40)
        print(f"Test samples: {capability.get('test_samples', 0):,}")
        print(f"R² Score:     {capability.get('r2_score', 0):.4f}")
        print(f"MAE:          {capability.get('mae', 0):.2f}")
        print(f"RMSE:         {capability.get('rmse', 0):.2f}")
        
        # Error distribution
        err_dist = capability.get("additional_metrics", {}).get("error_distribution", {})
        print(f"\nError Distribution:")
        print(f"  Mean error:     {err_dist.get('mean_error', 0):.2f}")
        print(f"  Within 10 pts:   {err_dist.get('percent_within_10', 0):.1%}")
        print(f"  Within 20 pts:   {err_dist.get('percent_within_20', 0):.1%}")
        
        # Category accuracy
        cat_acc = capability.get("additional_metrics", {}).get("accuracy_by_capability_level", {})
        if cat_acc:
            print(f"\nAccuracy by Capability Level:")
            for category, metrics in cat_acc.items():
                print(f"  {category}:")
                print(f"    MAE:  {metrics.get('mae', 0):.2f}")
                print(f"    RMSE: {metrics.get('rmse', 0):.2f}")
        
        # Feature importance
        fi = capability.get("feature_importance", {})
        if fi:
            print(f"\nFeature Importance:")
            for feature, importance in sorted(fi.items(), key=lambda x: x[1], reverse=True):
                print(f"  {feature}: {importance:.3f}")
    
    print("\n" + "=" * 70)


# ==========================================
# COMMAND LINE INTERFACE
# ==========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--evaluation-dir", required=True, help="Directory containing evaluation CSV files")
    parser.add_argument("--model-dir", required=True, help="Directory containing saved model artifacts")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run evaluation
    results = evaluate_all_models(
        evaluation_dir=args.evaluation_dir,
        model_dir=args.model_dir
    )
    
    # Print report
    print_evaluation_report(results)