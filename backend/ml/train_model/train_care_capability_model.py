# ml/train_model/train_care_capability_model.py
"""
Care Capability Model Training Module for PawCare+ ML pipeline.
Trains a regression model to predict owner care capability scores (0-100).
"""

import pandas as pd
import numpy as np
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json

# Configure logging
logger = logging.getLogger(__name__)

# Try to import xgboost if available
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed, falling back to RandomForest")

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature definitions
CATEGORICAL_FEATURES = [
    "owner_experience",
    "vet_access",
    "owner_commitment"
]

# No numerical features for this model
NUMERICAL_FEATURES = []

TARGET_FEATURE = "care_capability_score"

# Expected feature count
EXPECTED_FEATURES = len(CATEGORICAL_FEATURES)  # 3 features


def train_care_capability_model(
    training_file: str,
    output_dir: str
) -> Dict[str, Any]:
    """
    Train a regression model for owner care capability prediction.
    
    This function:
    1. Loads cleaned training data
    2. Encodes categorical features using LabelEncoder
    3. Scales features using StandardScaler
    4. Trains a regression model (XGBoost or RandomForest)
    5. Evaluates on train/test splits
    6. Saves model artifacts (model, scaler, encoders)
    
    Args:
        training_file: Path to cleaned training CSV (owner_care_capability_clean.csv)
        output_dir: Directory to save model artifacts
        
    Returns:
        Dictionary containing:
            - status: String "success" or "failed"
            - train_r2: R-squared on training data
            - test_r2: R-squared on test data
            - model_type: String describing the model
            - n_features: Number of features (should be 3)
            - training_samples: Number of training samples
            - test_samples: Number of test samples
            - metrics: Dictionary of additional metrics
            - feature_importance: Feature importance scores
            - artifacts: Paths to saved artifacts
    """
    logger.info("=" * 60)
    logger.info("TRAINING OWNER CARE CAPABILITY MODEL")
    logger.info("=" * 60)
    logger.info(f"Training file: {training_file}")
    logger.info(f"Output directory: {output_dir}")
    
    results = {
        "status": "failed",
        "train_r2": None,
        "test_r2": None,
        "model_type": None,
        "n_features": EXPECTED_FEATURES,
        "training_samples": None,
        "test_samples": None,
        "metrics": {},
        "feature_importance": {},
        "artifacts": {}
    }
    
    try:
        # ==========================================
        # STEP 1: LOAD DATA
        # ==========================================
        logger.info("\n📂 Loading training data...")
        df = pd.read_csv(training_file)
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Validate required columns
        required_cols = CATEGORICAL_FEATURES + [TARGET_FEATURE]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            error_msg = f"Missing required columns: {missing_cols}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # ==========================================
        # STEP 2: PREPARE FEATURES AND TARGET
        # ==========================================
        logger.info("\n🔧 Preparing features and target...")
        
        # Separate features and target
        X_categorical = df[CATEGORICAL_FEATURES].copy()
        y = df[TARGET_FEATURE].copy()
        
        logger.info(f"Categorical features: {len(CATEGORICAL_FEATURES)}")
        logger.info(f"Target: {TARGET_FEATURE}")
        
        # Check target range
        logger.info(f"Target range: [{y.min():.1f}, {y.max():.1f}]")
        logger.info(f"Target mean: {y.mean():.1f} ± {y.std():.1f}")
        
        # Check for any remaining NaN values
        if X_categorical.isnull().any().any():
            logger.warning("Found NaN in categorical features, filling with 'unknown'")
            X_categorical = X_categorical.fillna('unknown')
        
        # ==========================================
        # STEP 3: ENCODE CATEGORICAL FEATURES
        # ==========================================
        logger.info("\n🏷️ Encoding categorical features...")
        
        encoders = {}
        X_encoded = pd.DataFrame()
        
        for col in CATEGORICAL_FEATURES:
            logger.info(f"  Encoding {col}...")
            encoder = LabelEncoder()
            
            # Fit encoder on all values
            encoded_values = encoder.fit_transform(X_categorical[col])
            X_encoded[col] = encoded_values
            
            # Store encoder
            encoders[col] = encoder
            
            logger.info(f"    Classes: {list(encoder.classes_)}")
        
        # ==========================================
        # STEP 4: BUILD FEATURE MATRIX
        # ==========================================
        logger.info("\n🔢 Building feature matrix...")
        
        X = X_encoded
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Feature names: {list(X.columns)}")
        
        # Verify feature count
        if X.shape[1] != EXPECTED_FEATURES:
            logger.warning(f"Expected {EXPECTED_FEATURES} features, got {X.shape[1]}")
        
        # ==========================================
        # STEP 5: SPLIT INTO TRAIN/TEST
        # ==========================================
        logger.info("\n✂️ Splitting into train/test sets...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE,
            shuffle=True
        )
        
        logger.info(f"Training samples: {len(X_train):,}")
        logger.info(f"Test samples: {len(X_test):,}")
        
        results["training_samples"] = len(X_train)
        results["test_samples"] = len(X_test)
        
        # ==========================================
        # STEP 6: SCALE FEATURES
        # ==========================================
        logger.info("\n📊 Scaling features...")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info(f"Scaler mean: {scaler.mean_}")
        logger.info(f"Scaler scale: {scaler.scale_}")
        
        # ==========================================
        # STEP 7: CREATE AND TRAIN MODEL
        # ==========================================
        logger.info("\n🤖 Training regression model...")
        
        # Try XGBoost first, fallback to RandomForest
        if XGBOOST_AVAILABLE:
            logger.info("Using XGBoost Regressor")
            model = XGBRegressor(
                n_estimators=200,
                max_depth=4,  # Shallow trees for small feature set
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            model_type = "XGBRegressor"
        else:
            logger.info("Using RandomForest Regressor")
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            model_type = "RandomForestRegressor"
        
        # Train model
        model.fit(X_train_scaled, y_train)
        logger.info(f"✓ Model training complete")
        
        results["model_type"] = model_type
        
        # ==========================================
        # STEP 8: EVALUATE MODEL
        # ==========================================
        logger.info("\n📈 Evaluating model...")
        
        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        logger.info(f"Train R²: {train_r2:.4f}")
        logger.info(f"Test R²:  {test_r2:.4f}")
        logger.info(f"Train MAE: {train_mae:.2f}")
        logger.info(f"Test MAE:  {test_mae:.2f}")
        logger.info(f"Train RMSE: {train_rmse:.2f}")
        logger.info(f"Test RMSE:  {test_rmse:.2f}")
        
        # Store metrics
        results["train_r2"] = float(train_r2)
        results["test_r2"] = float(test_r2)
        results["metrics"] = {
            "train": {
                "r2": float(train_r2),
                "mae": float(train_mae),
                "rmse": float(train_rmse)
            },
            "test": {
                "r2": float(test_r2),
                "mae": float(test_mae),
                "rmse": float(test_rmse)
            }
        }
        
        # ==========================================
        # STEP 9: FEATURE IMPORTANCE
        # ==========================================
        logger.info("\n🔍 Calculating feature importance...")
        
        feature_names = list(X.columns)
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            
            feature_importance = {}
            for i, idx in enumerate(indices):
                feature_importance[feature_names[idx]] = float(importances[idx])
                logger.info(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
            
            results["feature_importance"] = feature_importance
        else:
            logger.warning("Model does not provide feature_importances_")
        
        # ==========================================
        # STEP 10: SAVE ARTIFACTS
        # ==========================================
        logger.info("\n💾 Saving model artifacts...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(output_dir, "owner_care_capability_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"✓ Model saved to: {model_path}")
        results["artifacts"]["model"] = model_path
        
        # Save scaler
        scaler_path = os.path.join(output_dir, "owner_care_capability_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"✓ Scaler saved to: {scaler_path}")
        results["artifacts"]["scaler"] = scaler_path
        
        # Save encoders
        encoders_path = os.path.join(output_dir, "owner_care_capability_encoder.pkl")
        with open(encoders_path, 'wb') as f:
            pickle.dump(encoders, f)
        logger.info(f"✓ Encoders saved to: {encoders_path}")
        results["artifacts"]["encoders"] = encoders_path
        
        # Save feature names for reference
        features_path = os.path.join(output_dir, "owner_care_capability_features.json")
        with open(features_path, 'w') as f:
            json.dump({
                "categorical_features": CATEGORICAL_FEATURES,
                "target": TARGET_FEATURE,
                "all_features": feature_names,
                "expected_categories": {
                    "owner_experience": ["novice", "experienced", "expert", "unknown"],
                    "vet_access": ["regular", "emergency only", "limited", "none", "unknown"],
                    "owner_commitment": ["casual", "dedicated", "obsessive", "unknown"]
                }
            }, f, indent=2)
        logger.info(f"✓ Feature config saved to: {features_path}")
        
        # ==========================================
        # STEP 11: TRAINING SUMMARY
        # ==========================================
        logger.info("\n" + "=" * 40)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 40)
        logger.info(f"Model: {model_type}")
        logger.info(f"Training samples: {len(X_train):,}")
        logger.info(f"Test samples: {len(X_test):,}")
        logger.info(f"Features: {EXPECTED_FEATURES}")
        logger.info(f"Train R²: {train_r2:.4f}")
        logger.info(f"Test R²:  {test_r2:.4f}")
        logger.info(f"Train-Test gap: {train_r2 - test_r2:.4f}")
        
        results["status"] = "success"
        logger.info("\n✅ Care capability model training completed successfully")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error training care capability model: {str(e)}", exc_info=True)
        results["status"] = "failed"
        results["error"] = str(e)
        return results


def load_care_capability_model(model_dir: str) -> Tuple[Any, StandardScaler, Dict[str, LabelEncoder]]:
    """
    Load trained care capability model and artifacts.
    
    Args:
        model_dir: Directory containing model artifacts
        
    Returns:
        Tuple of (model, scaler, encoders)
    """
    model_path = os.path.join(model_dir, "owner_care_capability_model.pkl")
    scaler_path = os.path.join(model_dir, "owner_care_capability_scaler.pkl")
    encoders_path = os.path.join(model_dir, "owner_care_capability_encoder.pkl")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)
    
    logger.info(f"Loaded model from {model_dir}")
    return model, scaler, encoders


def predict_care_capability(
    model: Any,
    scaler: StandardScaler,
    encoders: Dict[str, LabelEncoder],
    features: Dict[str, str]
) -> float:
    """
    Make a prediction using the trained model.
    
    Args:
        model: Trained model
        scaler: Fitted StandardScaler
        encoders: Dictionary of LabelEncoders
        features: Dictionary with feature values (owner_experience, vet_access, owner_commitment)
        
    Returns:
        Predicted care capability score (0-100)
    """
    # Prepare feature vector
    feature_values = []
    
    # Encode categorical features
    for col in CATEGORICAL_FEATURES:
        value = features.get(col, 'unknown')
        encoder = encoders[col]
        
        try:
            encoded = encoder.transform([value])[0]
        except ValueError:
            # Unseen label - use most common class
            encoded = encoder.transform([encoder.classes_[0]])[0]
        
        feature_values.append(encoded)
    
    # Scale features
    X = np.array([feature_values])
    X_scaled = scaler.transform(X)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    
    # Clip to valid range
    return float(np.clip(prediction, 0.0, 100.0))


# ==========================================
# COMMAND LINE INTERFACE
# ==========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train owner care capability model")
    parser.add_argument("training_file", help="Path to cleaned training CSV")
    parser.add_argument("output_dir", help="Directory to save model artifacts")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Train model
    results = train_care_capability_model(
        training_file=args.training_file,
        output_dir=args.output_dir
    )
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL TRAINING RESULTS")
    print("=" * 60)
    
    if results["status"] == "success":
        print(f"✅ Model training successful!")
        print(f"Model type: {results['model_type']}")
        print(f"Train R²: {results['train_r2']:.4f}")
        print(f"Test R²:  {results['test_r2']:.4f}")
        print(f"\nArtifacts saved to: {args.output_dir}")
        
        # Show feature importance
        if results["feature_importance"]:
            print("\n📊 Feature Importance:")
            for feature, importance in results["feature_importance"].items():
                print(f"  • {feature}: {importance:.4f}")
    else:
        print(f"❌ Model training failed: {results.get('error', 'Unknown error')}")