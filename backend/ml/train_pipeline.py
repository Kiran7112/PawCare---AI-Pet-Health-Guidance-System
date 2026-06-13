# ml/train_pipeline.py
"""
Train Pipeline Orchestrator for PawCare+ ML pipeline.
Orchestrates the complete ML training pipeline from data cleaning to model evaluation.
"""

import os
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import pipeline modules
from ml.data_cleaning.clean_health_risk_data import clean_health_risk_dataset
from ml.data_cleaning.clean_care_capability_data import clean_care_capability_dataset
from ml.train_model.train_health_risk_model import train_health_risk_model
from ml.train_model.train_care_capability_model import train_care_capability_model
from ml.evaluation.evaluate_models import evaluate_all_models


class PipelineStatus:
    """Pipeline step status constants"""
    STARTED = "started"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


def run_full_pipeline(
    training_dir: str,
    evaluation_dir: str,
    output_dir: str,
    processed_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Orchestrate complete ML training pipeline.
    
    This function runs the entire ML pipeline in three steps:
    1. Data Cleaning: Clean raw training datasets
    2. Model Training: Train models on cleaned data
    3. Model Evaluation: Evaluate models on evaluation datasets
    
    Args:
        training_dir: Path to directory containing raw training CSV files
                     Expected files: pet_health_risk_training.csv, owner_care_capability_training.csv
        evaluation_dir: Path to directory containing evaluation CSV files
                       Expected files: pet_health_risk_evaluation.csv, owner_care_capability_evaluation.csv
        output_dir: Path to save trained model pickle files
        processed_dir: Optional path to save cleaned data (if None, cleaned data not saved)
    
    Returns:
        Dictionary containing:
            - status: Overall pipeline status ("started", "success", "failed")
            - steps: Dictionary containing results for each pipeline step
            - timestamp: ISO format timestamp
            - message: Overall status message
    """
    pipeline_start = datetime.datetime.now()
    
    logger.info("=" * 70)
    logger.info("STARTING COMPLETE ML TRAINING PIPELINE")
    logger.info("=" * 70)
    
    # Initialize results dictionary
    results = {
        "status": PipelineStatus.STARTED,
        "timestamp": pipeline_start.isoformat(),
        "steps": {
            "data_cleaning": {"status": PipelineStatus.STARTED},
            "model_training": {"status": PipelineStatus.STARTED},
            "model_evaluation": {"status": PipelineStatus.STARTED}
        },
        "message": "Pipeline execution started"
    }
    
    # Validate directories
    if not _validate_directories(training_dir, evaluation_dir, output_dir, results):
        return results
    
    # ==========================================
    # STEP 1: DATA CLEANING
    # ==========================================
    logger.info("\n" + "=" * 50)
    logger.info("STEP 1: DATA CLEANING")
    logger.info("=" * 50)
    
    try:
        cleaning_results = _run_data_cleaning_step(
            training_dir, processed_dir, results
        )
        results["steps"]["data_cleaning"] = cleaning_results
        
        if cleaning_results["status"] != PipelineStatus.SUCCESS:
            logger.error("Data cleaning step failed, aborting pipeline")
            results["status"] = PipelineStatus.FAILED
            results["message"] = "Pipeline failed at data cleaning step"
            return results
            
    except Exception as e:
        logger.error(f"Unexpected error in data cleaning step: {str(e)}", exc_info=True)
        results["steps"]["data_cleaning"] = {
            "status": PipelineStatus.FAILED,
            "error": str(e)
        }
        results["status"] = PipelineStatus.FAILED
        results["message"] = f"Pipeline failed at data cleaning step: {str(e)}"
        return results
    
    # ==========================================
    # STEP 2: MODEL TRAINING
    # ==========================================
    logger.info("\n" + "=" * 50)
    logger.info("STEP 2: MODEL TRAINING")
    logger.info("=" * 50)
    
    try:
        training_results = _run_model_training_step(
            processed_dir if processed_dir else training_dir,
            output_dir,
            results
        )
        results["steps"]["model_training"] = training_results
        
        if training_results["status"] != PipelineStatus.SUCCESS:
            logger.error("Model training step failed, aborting pipeline")
            results["status"] = PipelineStatus.FAILED
            results["message"] = "Pipeline failed at model training step"
            return results
            
    except Exception as e:
        logger.error(f"Unexpected error in model training step: {str(e)}", exc_info=True)
        results["steps"]["model_training"] = {
            "status": PipelineStatus.FAILED,
            "error": str(e)
        }
        results["status"] = PipelineStatus.FAILED
        results["message"] = f"Pipeline failed at model training step: {str(e)}"
        return results
    
    # ==========================================
    # STEP 3: MODEL EVALUATION
    # ==========================================
    logger.info("\n" + "=" * 50)
    logger.info("STEP 3: MODEL EVALUATION")
    logger.info("=" * 50)
    
    try:
        evaluation_results = _run_model_evaluation_step(
            evaluation_dir, output_dir, results
        )
        results["steps"]["model_evaluation"] = evaluation_results
        
        if evaluation_results["status"] != PipelineStatus.SUCCESS:
            logger.warning("Model evaluation step had issues, but pipeline continuing")
            # Don't fail the whole pipeline for evaluation issues
            
    except Exception as e:
        logger.error(f"Unexpected error in model evaluation step: {str(e)}", exc_info=True)
        results["steps"]["model_evaluation"] = {
            "status": PipelineStatus.FAILED,
            "error": str(e)
        }
        # Continue despite evaluation errors
    
    # ==========================================
    # PIPELINE COMPLETE
    # ==========================================
    pipeline_end = datetime.datetime.now()
    duration = (pipeline_end - pipeline_start).total_seconds()
    
    logger.info("\n" + "=" * 70)
    logger.info(f"PIPELINE COMPLETED IN {duration:.2f} SECONDS")
    logger.info("=" * 70)
    
    results["status"] = PipelineStatus.SUCCESS
    results["message"] = f"Pipeline completed successfully in {duration:.2f} seconds"
    results["duration_seconds"] = duration
    results["completion_timestamp"] = pipeline_end.isoformat()
    
    # Save pipeline results to output directory
    _save_pipeline_results(results, output_dir)
    
    return results


# ==========================================
# STEP-SPECIFIC FUNCTIONS
# ==========================================

def _validate_directories(
    training_dir: str,
    evaluation_dir: str,
    output_dir: str,
    results: Dict[str, Any]
) -> bool:
    """
    Validate that required directories and files exist.
    
    Args:
        training_dir: Training data directory
        evaluation_dir: Evaluation data directory
        output_dir: Output directory for models
        results: Results dictionary to update on failure
        
    Returns:
        True if validation passes, False otherwise
    """
    logger.info("Validating directories and files...")
    
    # Check training directory
    if not os.path.exists(training_dir):
        error_msg = f"Training directory does not exist: {training_dir}"
        logger.error(error_msg)
        results["status"] = PipelineStatus.FAILED
        results["message"] = error_msg
        return False
    
    # Check evaluation directory
    if not os.path.exists(evaluation_dir):
        error_msg = f"Evaluation directory does not exist: {evaluation_dir}"
        logger.error(error_msg)
        results["status"] = PipelineStatus.FAILED
        results["message"] = error_msg
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory ready: {output_dir}")
    
    # Check required training files
    required_training_files = [
        "pet_health_risk_training.csv",
        "owner_care_capability_training.csv"
    ]
    
    missing_files = []
    for filename in required_training_files:
        filepath = os.path.join(training_dir, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
    
    if missing_files:
        error_msg = f"Missing required training files: {', '.join(missing_files)}"
        logger.error(error_msg)
        results["status"] = PipelineStatus.FAILED
        results["message"] = error_msg
        return False
    
    # Check required evaluation files
    required_eval_files = [
        "pet_health_risk_evaluation.csv",
        "owner_care_capability_evaluation.csv"
    ]
    
    missing_files = []
    for filename in required_eval_files:
        filepath = os.path.join(evaluation_dir, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
    
    if missing_files:
        logger.warning(f"Missing evaluation files: {', '.join(missing_files)}")
        # Don't fail, but note the issue
    
    logger.info("✓ Directory validation passed")
    return True


def _run_data_cleaning_step(
    training_dir: str,
    processed_dir: Optional[str],
    results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run the data cleaning step of the pipeline.
    
    Args:
        training_dir: Training data directory
        processed_dir: Optional directory to save cleaned data
        results: Main results dictionary
        
    Returns:
        Dictionary with cleaning step results
    """
    logger.info("Running data cleaning step...")
    
    step_results = {
        "status": PipelineStatus.STARTED,
        "health_risk_cleaning": {},
        "care_capability_cleaning": {}
    }
    
    # Create processed directory if provided and doesn't exist
    if processed_dir:
        os.makedirs(processed_dir, exist_ok=True)
        logger.info(f"Processed data will be saved to: {processed_dir}")
    
    # Clean health risk dataset
    logger.info("\n--- Cleaning Pet Health Risk Dataset ---")
    health_risk_path = os.path.join(training_dir, "pet_health_risk_training.csv")
    
    try:
        health_risk_cleaned = clean_health_risk_dataset(
            health_risk_path,
            processed_dir
        )
        
        step_results["health_risk_cleaning"] = {
            "status": PipelineStatus.SUCCESS,
            "input_file": health_risk_path,
            "output_file": health_risk_cleaned.get("output_file") if processed_dir else None,
            "original_shape": health_risk_cleaned.get("original_shape"),
            "cleaned_shape": health_risk_cleaned.get("cleaned_shape"),
            "removed_rows": health_risk_cleaned.get("removed_rows", 0)
        }
        logger.info(f"✓ Health risk dataset cleaned: {health_risk_cleaned.get('cleaned_shape')} records")
        
    except Exception as e:
        logger.error(f"Failed to clean health risk dataset: {str(e)}")
        step_results["health_risk_cleaning"] = {
            "status": PipelineStatus.FAILED,
            "input_file": health_risk_path,
            "error": str(e)
        }
        step_results["status"] = PipelineStatus.FAILED
        return step_results
    
    # Clean care capability dataset
    logger.info("\n--- Cleaning Owner Care Capability Dataset ---")
    capability_path = os.path.join(training_dir, "owner_care_capability_training.csv")
    
    try:
        capability_cleaned = clean_care_capability_dataset(
            capability_path,
            processed_dir
        )
        
        step_results["care_capability_cleaning"] = {
            "status": PipelineStatus.SUCCESS,
            "input_file": capability_path,
            "output_file": capability_cleaned.get("output_file") if processed_dir else None,
            "original_shape": capability_cleaned.get("original_shape"),
            "cleaned_shape": capability_cleaned.get("cleaned_shape"),
            "removed_rows": capability_cleaned.get("removed_rows", 0)
        }
        logger.info(f"✓ Care capability dataset cleaned: {capability_cleaned.get('cleaned_shape')} records")
        
    except Exception as e:
        logger.error(f"Failed to clean care capability dataset: {str(e)}")
        step_results["care_capability_cleaning"] = {
            "status": PipelineStatus.FAILED,
            "input_file": capability_path,
            "error": str(e)
        }
        step_results["status"] = PipelineStatus.FAILED
        return step_results
    
    step_results["status"] = PipelineStatus.SUCCESS
    logger.info("\n✓ Data cleaning step completed successfully")
    
    return step_results


def _run_model_training_step(
    data_dir: str,
    output_dir: str,
    results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run the model training step of the pipeline.
    
    Args:
        data_dir: Directory containing cleaned data
        output_dir: Directory to save trained models
        results: Main results dictionary
        
    Returns:
        Dictionary with training step results
    """
    logger.info("Running model training step...")
    
    step_results = {
        "status": PipelineStatus.STARTED,
        "health_risk_training": {},
        "care_capability_training": {},
        "output_directory": output_dir
    }
    
    # Determine data file paths
    health_risk_data = os.path.join(data_dir, "pet_health_risk_clean.csv")
    if not os.path.exists(health_risk_data):
        # Fall back to original training directory
        health_risk_data = os.path.join(
            os.path.dirname(data_dir), 
            "pet_health_risk_training.csv"
        )
        logger.warning(f"Cleaned data not found, using original: {health_risk_data}")
    
    capability_data = os.path.join(data_dir, "owner_care_capability_clean.csv")
    if not os.path.exists(capability_data):
        # Fall back to original training directory
        capability_data = os.path.join(
            os.path.dirname(data_dir),
            "owner_care_capability_training.csv"
        )
        logger.warning(f"Cleaned data not found, using original: {capability_data}")
    
    # Train health risk model
    logger.info("\n--- Training Pet Health Risk Model ---")
    try:
        health_risk_results = train_health_risk_model(
            health_risk_data,
            output_dir
        )
        
        step_results["health_risk_training"] = {
            "status": PipelineStatus.SUCCESS,
            "data_file": health_risk_data,
            "metrics": health_risk_results.get("metrics"),
            "model_file": health_risk_results.get("model_file"),
            "scaler_file": health_risk_results.get("scaler_file"),
            "encoder_file": health_risk_results.get("encoder_file")
        }
        logger.info(f"✓ Health risk model trained: {health_risk_results.get('metrics')}")
        
    except Exception as e:
        logger.error(f"Failed to train health risk model: {str(e)}")
        step_results["health_risk_training"] = {
            "status": PipelineStatus.FAILED,
            "data_file": health_risk_data,
            "error": str(e)
        }
        step_results["status"] = PipelineStatus.FAILED
        return step_results
    
    # Train care capability model
    logger.info("\n--- Training Owner Care Capability Model ---")
    try:
        capability_results = train_care_capability_model(
            capability_data,
            output_dir
        )
        
        step_results["care_capability_training"] = {
            "status": PipelineStatus.SUCCESS,
            "data_file": capability_data,
            "metrics": capability_results.get("metrics"),
            "model_file": capability_results.get("model_file"),
            "scaler_file": capability_results.get("scaler_file"),
            "encoder_file": capability_results.get("encoder_file")
        }
        logger.info(f"✓ Care capability model trained: {capability_results.get('metrics')}")
        
    except Exception as e:
        logger.error(f"Failed to train care capability model: {str(e)}")
        step_results["care_capability_training"] = {
            "status": PipelineStatus.FAILED,
            "data_file": capability_data,
            "error": str(e)
        }
        step_results["status"] = PipelineStatus.FAILED
        return step_results
    
    step_results["status"] = PipelineStatus.SUCCESS
    logger.info("\n✓ Model training step completed successfully")
    
    return step_results


def _run_model_evaluation_step(
    evaluation_dir: str,
    model_dir: str,
    results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run the model evaluation step of the pipeline.
    
    Args:
        evaluation_dir: Directory containing evaluation datasets
        model_dir: Directory containing trained models
        results: Main results dictionary
        
    Returns:
        Dictionary with evaluation step results
    """
    logger.info("Running model evaluation step...")
    
    step_results = {
        "status": PipelineStatus.STARTED,
        "evaluation_dir": evaluation_dir,
        "model_dir": model_dir
    }
    
    # Check if evaluation files exist
    health_risk_eval = os.path.join(evaluation_dir, "pet_health_risk_evaluation.csv")
    capability_eval = os.path.join(evaluation_dir, "owner_care_capability_evaluation.csv")
    
    missing_files = []
    if not os.path.exists(health_risk_eval):
        missing_files.append("pet_health_risk_evaluation.csv")
    if not os.path.exists(capability_eval):
        missing_files.append("owner_care_capability_evaluation.csv")
    
    if missing_files:
        logger.warning(f"Missing evaluation files: {', '.join(missing_files)}")
        step_results["status"] = PipelineStatus.SKIPPED
        step_results["message"] = f"Evaluation skipped: missing {', '.join(missing_files)}"
        return step_results
    
    # Run evaluation
    try:
        evaluation_results = evaluate_all_models(
            evaluation_dir,
            model_dir
        )
        
        step_results.update({
            "status": PipelineStatus.SUCCESS,
            "health_risk_evaluation": evaluation_results.get("pet_health_risk", {}),
            "care_capability_evaluation": evaluation_results.get("owner_care_capability", {}),
            "summary": evaluation_results.get("summary")
        })
        
        logger.info(f"✓ Model evaluation completed")
        
    except Exception as e:
        logger.error(f"Failed to evaluate models: {str(e)}")
        step_results["status"] = PipelineStatus.FAILED
        step_results["error"] = str(e)
    
    return step_results


def _save_pipeline_results(results: Dict[str, Any], output_dir: str) -> None:
    """
    Save pipeline results to a JSON file in the output directory.
    
    Args:
        results: Complete pipeline results dictionary
        output_dir: Directory to save results
    """
    try:
        results_path = os.path.join(output_dir, "pipeline_results.json")
        
        # Create a serializable copy
        serializable_results = json.loads(
            json.dumps(results, default=str)
        )
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Pipeline results saved to: {results_path}")
        
    except Exception as e:
        logger.error(f"Failed to save pipeline results: {str(e)}")


# ==========================================
# COMMAND LINE INTERFACE
# ==========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run complete ML training pipeline")
    parser.add_argument("--training-dir", required=True, help="Directory containing training CSV files")
    parser.add_argument("--evaluation-dir", required=True, help="Directory containing evaluation CSV files")
    parser.add_argument("--output-dir", required=True, help="Directory to save trained models")
    parser.add_argument("--processed-dir", help="Directory to save cleaned data (optional)")
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline_results = run_full_pipeline(
        training_dir=args.training_dir,
        evaluation_dir=args.evaluation_dir,
        output_dir=args.output_dir,
        processed_dir=args.processed_dir
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Status: {pipeline_results['status']}")
    print(f"Message: {pipeline_results.get('message')}")
    
    if pipeline_results['status'] == PipelineStatus.SUCCESS:
        print(f"Duration: {pipeline_results.get('duration_seconds', 0):.2f} seconds")
        
        # Print training metrics if available
        training = pipeline_results['steps'].get('model_training', {})
        if training.get('status') == PipelineStatus.SUCCESS:
            print("\n📊 Health Risk Model:")
            metrics = training.get('health_risk_training', {}).get('metrics', {})
            for key, value in metrics.items():
                print(f"  {key}: {value}")
            
            print("\n📊 Care Capability Model:")
            metrics = training.get('care_capability_training', {}).get('metrics', {})
            for key, value in metrics.items():
                print(f"  {key}: {value}")