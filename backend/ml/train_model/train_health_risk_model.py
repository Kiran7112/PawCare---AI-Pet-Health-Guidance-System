# # ml/train_model/train_health_risk_model.py
# """
# Health Risk Model Training Module for PawCare+ ML pipeline.
# Trains a regression model to predict pet health risk scores (0-1).
# """

# import pandas as pd
# import numpy as np
# import logging
# import os
# import pickle
# import joblib
# from pathlib import Path
# from typing import Dict, Any, Optional, Tuple, List
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# import json

# # Configure logging
# logger = logging.getLogger(__name__)

# # Try to import xgboost if available
# try:
#     from xgboost import XGBRegressor
#     XGBOOST_AVAILABLE = True
# except ImportError:
#     XGBOOST_AVAILABLE = False
#     logger.warning("XGBoost not installed, falling back to RandomForest")

# # Model configuration
# RANDOM_STATE = 42
# TEST_SIZE = 0.2

# # Feature definitions
# CATEGORICAL_FEATURES = [
#     "pet_species",
#     "weight_status",
#     "living_situation",
#     "exercise_level"
# ]

# NUMERICAL_FEATURES = [
#     "age_years",
#     "conditions_count",
#     "allergies_count"
# ]

# TARGET_FEATURE = "health_risk_score"

# # Expected feature count
# EXPECTED_FEATURES = len(CATEGORICAL_FEATURES) + len(NUMERICAL_FEATURES)  # 7 features


# def train_health_risk_model(
#     training_file: str,
#     output_dir: str
# ) -> Dict[str, Any]:
#     """
#     Train a regression model for pet health risk prediction.
    
#     This function:
#     1. Loads cleaned training data
#     2. Encodes categorical features using LabelEncoder
#     3. Scales numerical features using StandardScaler
#     4. Trains a regression model (XGBoost or RandomForest)
#     5. Evaluates on train/test splits
#     6. Saves model artifacts (model, scaler, encoders)
    
#     Args:
#         training_file: Path to cleaned training CSV (pet_health_risk_clean.csv)
#         output_dir: Directory to save model artifacts
        
#     Returns:
#         Dictionary containing:
#             - status: String "success" or "failed"
#             - train_r2: R-squared on training data
#             - test_r2: R-squared on test data
#             - model_type: String describing the model
#             - n_features: Number of features (should be 7)
#             - training_samples: Number of training samples
#             - test_samples: Number of test samples
#             - metrics: Dictionary of additional metrics
#             - feature_importance: Feature importance scores
#             - artifacts: Paths to saved artifacts
#     """
#     logger.info("=" * 60)
#     logger.info("TRAINING PET HEALTH RISK MODEL")
#     logger.info("=" * 60)
#     logger.info(f"Training file: {training_file}")
#     logger.info(f"Output directory: {output_dir}")
    
#     results = {
#         "status": "failed",
#         "train_r2": None,
#         "test_r2": None,
#         "model_type": None,
#         "n_features": EXPECTED_FEATURES,
#         "training_samples": None,
#         "test_samples": None,
#         "metrics": {},
#         "feature_importance": {},
#         "artifacts": {}
#     }
    
#     try:
#         # ==========================================
#         # STEP 1: LOAD DATA
#         # ==========================================
#         logger.info("\n📂 Loading training data...")
#         df = pd.read_csv(training_file)
#         logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        
#         # Validate required columns
#         required_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TARGET_FEATURE]
#         missing_cols = [col for col in required_cols if col not in df.columns]
#         if missing_cols:
#             error_msg = f"Missing required columns: {missing_cols}"
#             logger.error(error_msg)
#             raise ValueError(error_msg)
        
#         # ==========================================
#         # STEP 2: PREPARE FEATURES AND TARGET
#         # ==========================================
#         logger.info("\n🔧 Preparing features and target...")
        
#         # Separate features and target
#         X_categorical = df[CATEGORICAL_FEATURES].copy()
#         X_numerical = df[NUMERICAL_FEATURES].copy()
#         y = df[TARGET_FEATURE].copy()
        
#         logger.info(f"Categorical features: {len(CATEGORICAL_FEATURES)}")
#         logger.info(f"Numerical features: {len(NUMERICAL_FEATURES)}")
#         logger.info(f"Target: {TARGET_FEATURE}")
        
#         # Check for any remaining NaN values
#         if X_categorical.isnull().any().any():
#             logger.warning("Found NaN in categorical features, filling with 'unknown'")
#             X_categorical = X_categorical.fillna('unknown')
        
#         if X_numerical.isnull().any().any():
#             logger.warning("Found NaN in numerical features, filling with median")
#             X_numerical = X_numerical.fillna(X_numerical.median())
        
#         # ==========================================
#         # STEP 3: ENCODE CATEGORICAL FEATURES
#         # ==========================================
#         logger.info("\n🏷️ Encoding categorical features...")
        
#         encoders = {}
#         X_categorical_encoded = pd.DataFrame()
        
#         for col in CATEGORICAL_FEATURES:
#             logger.info(f"  Encoding {col}...")
#             encoder = LabelEncoder()
            
#             # Fit encoder on all values
#             encoded_values = encoder.fit_transform(X_categorical[col])
#             X_categorical_encoded[col] = encoded_values
            
#             # Store encoder
#             encoders[col] = encoder
            
#             logger.info(f"    Classes: {list(encoder.classes_)}")
        
#         # ==========================================
#         # STEP 4: BUILD FEATURE MATRIX
#         # ==========================================
#         logger.info("\n🔢 Building feature matrix...")
        
#         # Combine encoded categorical and numerical features
#         X = pd.concat([X_categorical_encoded, X_numerical], axis=1)
        
#         logger.info(f"Feature matrix shape: {X.shape}")
#         logger.info(f"Feature names: {list(X.columns)}")
        
#         # Verify feature count
#         if X.shape[1] != EXPECTED_FEATURES:
#             logger.warning(f"Expected {EXPECTED_FEATURES} features, got {X.shape[1]}")
        
#         # ==========================================
#         # STEP 5: SPLIT INTO TRAIN/TEST
#         # ==========================================
#         logger.info("\n✂️ Splitting into train/test sets...")
        
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, 
#             test_size=TEST_SIZE, 
#             random_state=RANDOM_STATE,
#             shuffle=True
#         )
        
#         logger.info(f"Training samples: {len(X_train):,}")
#         logger.info(f"Test samples: {len(X_test):,}")
        
#         results["training_samples"] = len(X_train)
#         results["test_samples"] = len(X_test)
        
#         # ==========================================
#         # STEP 6: SCALE FEATURES
#         # ==========================================
#         logger.info("\n📊 Scaling features...")
        
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
        
#         logger.info(f"Scaler mean: {scaler.mean_[:5]}...")  # Show first 5
#         logger.info(f"Scaler scale: {scaler.scale_[:5]}...")
        
#         # ==========================================
#         # STEP 7: CREATE AND TRAIN MODEL
#         # ==========================================
#         logger.info("\n🤖 Training regression model...")
        
#         # Try XGBoost first, fallback to RandomForest
#         if XGBOOST_AVAILABLE:
#             logger.info("Using XGBoost Regressor")
#             model = XGBRegressor(
#                 n_estimators=200,
#                 max_depth=6,
#                 learning_rate=0.1,
#                 subsample=0.8,
#                 colsample_bytree=0.8,
#                 random_state=RANDOM_STATE,
#                 n_jobs=-1
#             )
#             model_type = "XGBRegressor"
#         else:
#             logger.info("Using RandomForest Regressor")
#             model = RandomForestRegressor(
#                 n_estimators=200,
#                 max_depth=10,
#                 min_samples_split=5,
#                 min_samples_leaf=2,
#                 random_state=RANDOM_STATE,
#                 n_jobs=-1
#             )
#             model_type = "RandomForestRegressor"
        
#         # Train model
#         model.fit(X_train_scaled, y_train)
#         logger.info(f"✓ Model training complete")
        
#         results["model_type"] = model_type
        
#         # ==========================================
#         # STEP 8: EVALUATE MODEL
#         # ==========================================
#         logger.info("\n📈 Evaluating model...")
        
#         # Predictions
#         y_train_pred = model.predict(X_train_scaled)
#         y_test_pred = model.predict(X_test_scaled)
        
#         # Calculate metrics
#         train_r2 = r2_score(y_train, y_train_pred)
#         test_r2 = r2_score(y_test, y_test_pred)
        
#         train_mae = mean_absolute_error(y_train, y_train_pred)
#         test_mae = mean_absolute_error(y_test, y_test_pred)
        
#         train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
#         test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
#         logger.info(f"Train R²: {train_r2:.4f}")
#         logger.info(f"Test R²:  {test_r2:.4f}")
#         logger.info(f"Train MAE: {train_mae:.4f}")
#         logger.info(f"Test MAE:  {test_mae:.4f}")
#         logger.info(f"Train RMSE: {train_rmse:.4f}")
#         logger.info(f"Test RMSE:  {test_rmse:.4f}")
        
#         # Store metrics
#         results["train_r2"] = float(train_r2)
#         results["test_r2"] = float(test_r2)
#         results["metrics"] = {
#             "train": {
#                 "r2": float(train_r2),
#                 "mae": float(train_mae),
#                 "rmse": float(train_rmse)
#             },
#             "test": {
#                 "r2": float(test_r2),
#                 "mae": float(test_mae),
#                 "rmse": float(test_rmse)
#             }
#         }
        
#         # ==========================================
#         # STEP 9: FEATURE IMPORTANCE
#         # ==========================================
#         logger.info("\n🔍 Calculating feature importance...")
        
#         feature_names = list(X.columns)
        
#         if hasattr(model, 'feature_importances_'):
#             importances = model.feature_importances_
            
#             # Sort features by importance
#             indices = np.argsort(importances)[::-1]
            
#             feature_importance = {}
#             for i, idx in enumerate(indices):
#                 feature_importance[feature_names[idx]] = float(importances[idx])
#                 logger.info(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
            
#             results["feature_importance"] = feature_importance
#         else:
#             logger.warning("Model does not provide feature_importances_")
        
#         # ==========================================
#         # STEP 10: SAVE ARTIFACTS
#         # ==========================================
#         logger.info("\n💾 Saving model artifacts...")
        
#         # Create output directory if it doesn't exist
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Save model
#         model_path = os.path.join(output_dir, "pet_health_risk_model.pkl")
#         with open(model_path, 'wb') as f:
#             pickle.dump(model, f)
#         logger.info(f"✓ Model saved to: {model_path}")
#         results["artifacts"]["model"] = model_path
        
#         # Save scaler
#         scaler_path = os.path.join(output_dir, "pet_health_risk_scaler.pkl")
#         with open(scaler_path, 'wb') as f:
#             pickle.dump(scaler, f)
#         logger.info(f"✓ Scaler saved to: {scaler_path}")
#         results["artifacts"]["scaler"] = scaler_path
        
#         # Save encoders
#         encoders_path = os.path.join(output_dir, "pet_health_risk_encoder.pkl")
#         with open(encoders_path, 'wb') as f:
#             pickle.dump(encoders, f)
#         logger.info(f"✓ Encoders saved to: {encoders_path}")
#         results["artifacts"]["encoders"] = encoders_path
        
#         # Save feature names for reference
#         features_path = os.path.join(output_dir, "pet_health_risk_features.json")
#         with open(features_path, 'w') as f:
#             json.dump({
#                 "categorical_features": CATEGORICAL_FEATURES,
#                 "numerical_features": NUMERICAL_FEATURES,
#                 "target": TARGET_FEATURE,
#                 "all_features": feature_names
#             }, f, indent=2)
#         logger.info(f"✓ Feature config saved to: {features_path}")
        
#         # ==========================================
#         # STEP 11: TRAINING SUMMARY
#         # ==========================================
#         logger.info("\n" + "=" * 40)
#         logger.info("TRAINING SUMMARY")
#         logger.info("=" * 40)
#         logger.info(f"Model: {model_type}")
#         logger.info(f"Training samples: {len(X_train):,}")
#         logger.info(f"Test samples: {len(X_test):,}")
#         logger.info(f"Features: {EXPECTED_FEATURES}")
#         logger.info(f"Train R²: {train_r2:.4f}")
#         logger.info(f"Test R²:  {test_r2:.4f}")
#         logger.info(f"Train-Test gap: {train_r2 - test_r2:.4f}")
        
#         results["status"] = "success"
#         logger.info("\n✅ Health risk model training completed successfully")
        
#         return results
        
#     except Exception as e:
#         logger.error(f"❌ Error training health risk model: {str(e)}", exc_info=True)
#         results["status"] = "failed"
#         results["error"] = str(e)
#         return results


# def load_health_risk_model(model_dir: str) -> Tuple[Any, StandardScaler, Dict[str, LabelEncoder]]:
#     """
#     Load trained health risk model and artifacts.
    
#     Args:
#         model_dir: Directory containing model artifacts
        
#     Returns:
#         Tuple of (model, scaler, encoders)
#     """
#     model_path = os.path.join(model_dir, "pet_health_risk_model.pkl")
#     scaler_path = os.path.join(model_dir, "pet_health_risk_scaler.pkl")
#     encoders_path = os.path.join(model_dir, "pet_health_risk_encoder.pkl")
    
#     with open(model_path, 'rb') as f:
#         model = pickle.load(f)
    
#     with open(scaler_path, 'rb') as f:
#         scaler = pickle.load(f)
    
#     with open(encoders_path, 'rb') as f:
#         encoders = pickle.load(f)
    
#     logger.info(f"Loaded model from {model_dir}")
#     return model, scaler, encoders


# def predict_health_risk(
#     model: Any,
#     scaler: StandardScaler,
#     encoders: Dict[str, LabelEncoder],
#     features: Dict[str, Any]
# ) -> float:
#     """
#     Make a prediction using the trained model.
    
#     Args:
#         model: Trained model
#         scaler: Fitted StandardScaler
#         encoders: Dictionary of LabelEncoders
#         features: Dictionary with feature values
        
#     Returns:
#         Predicted health risk score (0-1)
#     """
#     # Prepare feature vector
#     feature_values = []
    
#     # Encode categorical features
#     for col in CATEGORICAL_FEATURES:
#         value = features.get(col, 'unknown')
#         encoder = encoders[col]
        
#         try:
#             encoded = encoder.transform([value])[0]
#         except ValueError:
#             # Unseen label - use most common class
#             encoded = encoder.transform([encoder.classes_[0]])[0]
        
#         feature_values.append(encoded)
    
#     # Add numerical features
#     for col in NUMERICAL_FEATURES:
#         value = features.get(col, 0)
#         feature_values.append(float(value))
    
#     # Scale features
#     X = np.array([feature_values])
#     X_scaled = scaler.transform(X)
    
#     # Predict
#     prediction = model.predict(X_scaled)[0]
    
#     # Clip to valid range
#     return float(np.clip(prediction, 0.0, 1.0))


# # ==========================================
# # COMMAND LINE INTERFACE
# # ==========================================

# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Train pet health risk model")
#     parser.add_argument("training_file", help="Path to cleaned training CSV")
#     parser.add_argument("output_dir", help="Directory to save model artifacts")
    
#     args = parser.parse_args()
    
#     # Configure logging
#     logging.basicConfig(level=logging.INFO)
    
#     # Train model
#     results = train_health_risk_model(
#         training_file=args.training_file,
#         output_dir=args.output_dir
#     )
    
#     # Print final results
#     print("\n" + "=" * 60)
#     print("FINAL TRAINING RESULTS")
#     print("=" * 60)
    
#     if results["status"] == "success":
#         print(f"✅ Model training successful!")
#         print(f"Model type: {results['model_type']}")
#         print(f"Train R²: {results['train_r2']:.4f}")
#         print(f"Test R²:  {results['test_r2']:.4f}")
#         print(f"\nArtifacts saved to: {args.output_dir}")
#     else:
#         print(f"❌ Model training failed: {results.get('error', 'Unknown error')}")



# ml/train_model/train_health_risk_model.py
"""
Health Risk Model Training Module for PawCare+ ML pipeline.
Trains a regression model to predict pet health risk scores (0-1).
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed, falling back to RandomForest")

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature definitions - ORIGINAL 7 FEATURES
CATEGORICAL_FEATURES = [
    "pet_species",
    "weight_status",
    "living_situation",
    "exercise_level"
]

NUMERICAL_FEATURES = [
    "age_years",
    "conditions_count",
    "allergies_count"
]

TARGET_FEATURE = "health_risk_score"
EXPECTED_FEATURES = 7


def train_health_risk_model(training_file: str, output_dir: str) -> Dict[str, Any]:
    logger.info("=" * 60)
    logger.info("TRAINING PET HEALTH RISK MODEL")
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
        # STEP 1: LOAD DATA
        logger.info("\n📂 Loading training data...")
        df = pd.read_csv(training_file)
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        required_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TARGET_FEATURE]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            error_msg = f"Missing required columns: {missing_cols}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # STEP 2: PREPARE FEATURES AND TARGET
        logger.info("\n🔧 Preparing features and target...")
        
        X_categorical = df[CATEGORICAL_FEATURES].copy()
        X_numerical = df[NUMERICAL_FEATURES].copy()
        y = df[TARGET_FEATURE].copy()
        
        logger.info(f"Categorical features: {len(CATEGORICAL_FEATURES)}")
        logger.info(f"Numerical features: {len(NUMERICAL_FEATURES)}")
        
        # Handle missing values
        if X_categorical.isnull().any().any():
            logger.warning("Found NaN in categorical features, filling with 'unknown'")
            X_categorical = X_categorical.fillna('unknown')
        
        if X_numerical.isnull().any().any():
            logger.warning("Found NaN in numerical features, filling with median")
            X_numerical = X_numerical.fillna(X_numerical.median())
        
        # STEP 3: ENCODE CATEGORICAL FEATURES
        logger.info("\n🏷️ Encoding categorical features...")
        
        encoders = {}
        X_categorical_encoded = pd.DataFrame()
        
        for col in CATEGORICAL_FEATURES:
            logger.info(f"  Encoding {col}...")
            encoder = LabelEncoder()
            encoded_values = encoder.fit_transform(X_categorical[col])
            X_categorical_encoded[col] = encoded_values
            encoders[col] = encoder
            logger.info(f"    Classes: {list(encoder.classes_)}")
        
        # STEP 4: BUILD FEATURE MATRIX
        logger.info("\n🔢 Building feature matrix...")
        X = pd.concat([X_categorical_encoded, X_numerical], axis=1)
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Feature names: {list(X.columns)}")
        
        # STEP 5: SPLIT INTO TRAIN/TEST
        logger.info("\n✂️ Splitting into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
        )
        
        logger.info(f"Training samples: {len(X_train):,}")
        logger.info(f"Test samples: {len(X_test):,}")
        
        results["training_samples"] = len(X_train)
        results["test_samples"] = len(X_test)
        
        # STEP 6: SCALE FEATURES
        logger.info("\n📊 Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # STEP 7: CREATE AND TRAIN MODEL
        logger.info("\n🤖 Training regression model...")
        
        if XGBOOST_AVAILABLE:
            logger.info("Using XGBoost Regressor")
            model = XGBRegressor(
                n_estimators=200,
                max_depth=6,
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
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            model_type = "RandomForestRegressor"
        
        model.fit(X_train_scaled, y_train)
        logger.info(f"✓ Model training complete")
        results["model_type"] = model_type
        
        # STEP 8: EVALUATE MODEL
        logger.info("\n📈 Evaluating model...")
        
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        logger.info(f"Train R²: {train_r2:.4f}")
        logger.info(f"Test R²:  {test_r2:.4f}")
        logger.info(f"Train MAE: {train_mae:.4f}")
        logger.info(f"Test MAE:  {test_mae:.4f}")
        logger.info(f"Train RMSE: {train_rmse:.4f}")
        logger.info(f"Test RMSE:  {test_rmse:.4f}")
        
        results["train_r2"] = float(train_r2)
        results["test_r2"] = float(test_r2)
        results["metrics"] = {
            "train": {"r2": float(train_r2), "mae": float(train_mae), "rmse": float(train_rmse)},
            "test": {"r2": float(test_r2), "mae": float(test_mae), "rmse": float(test_rmse)}
        }
        
        # STEP 9: FEATURE IMPORTANCE
        logger.info("\n🔍 Calculating feature importance...")
        feature_names = list(X.columns)
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            feature_importance = {}
            for i, idx in enumerate(indices):
                feature_importance[feature_names[idx]] = float(importances[idx])
                logger.info(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
            results["feature_importance"] = feature_importance
        
        # STEP 10: SAVE ARTIFACTS
        logger.info("\n💾 Saving model artifacts...")
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = os.path.join(output_dir, "pet_health_risk_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        results["artifacts"]["model"] = model_path
        
        scaler_path = os.path.join(output_dir, "pet_health_risk_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        results["artifacts"]["scaler"] = scaler_path
        
        encoders_path = os.path.join(output_dir, "pet_health_risk_encoder.pkl")
        with open(encoders_path, 'wb') as f:
            pickle.dump(encoders, f)
        results["artifacts"]["encoders"] = encoders_path
        
        features_path = os.path.join(output_dir, "pet_health_risk_features.json")
        with open(features_path, 'w') as f:
            json.dump({
                "categorical_features": CATEGORICAL_FEATURES,
                "numerical_features": NUMERICAL_FEATURES,
                "target": TARGET_FEATURE,
                "all_features": feature_names
            }, f, indent=2)
        logger.info(f"✓ Feature config saved to: {features_path}")
        
        # STEP 11: TRAINING SUMMARY
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
        logger.info("\n✅ Health risk model training completed successfully")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error training health risk model: {str(e)}", exc_info=True)
        results["status"] = "failed"
        results["error"] = str(e)
        return results


def load_health_risk_model(model_dir: str) -> Tuple[Any, StandardScaler, Dict[str, LabelEncoder]]:
    model_path = os.path.join(model_dir, "pet_health_risk_model.pkl")
    scaler_path = os.path.join(model_dir, "pet_health_risk_scaler.pkl")
    encoders_path = os.path.join(model_dir, "pet_health_risk_encoder.pkl")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)
    
    return model, scaler, encoders


def predict_health_risk(
    model: Any,
    scaler: StandardScaler,
    encoders: Dict[str, LabelEncoder],
    features: Dict[str, Any]
) -> float:
    feature_values = []
    
    for col in CATEGORICAL_FEATURES:
        value = features.get(col, 'unknown')
        encoder = encoders[col]
        try:
            encoded = encoder.transform([value])[0]
        except ValueError:
            encoded = encoder.transform([encoder.classes_[0]])[0]
        feature_values.append(encoded)
    
    for col in NUMERICAL_FEATURES:
        value = features.get(col, 0)
        feature_values.append(float(value))
    
    X = np.array([feature_values])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    
    return float(np.clip(prediction, 0.0, 1.0))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train pet health risk model")
    parser.add_argument("training_file", help="Path to cleaned training CSV")
    parser.add_argument("output_dir", help="Directory to save model artifacts")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    results = train_health_risk_model(args.training_file, args.output_dir)
    
    print("\n" + "=" * 60)
    print("FINAL TRAINING RESULTS")
    print("=" * 60)
    
    if results["status"] == "success":
        print(f"✅ Model training successful!")
        print(f"Model type: {results['model_type']}")
        print(f"Train R²: {results['train_r2']:.4f}")
        print(f"Test R²:  {results['test_r2']:.4f}")
        if results.get("feature_importance"):
            print("\n📊 Feature Importance:")
            for feature, importance in results["feature_importance"].items():
                print(f"  • {feature}: {importance:.4f}")
        print(f"\nArtifacts saved to: {args.output_dir}")
    else:
        print(f"❌ Model training failed: {results.get('error', 'Unknown error')}")