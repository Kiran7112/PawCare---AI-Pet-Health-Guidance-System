# # agents/pet_health_risk_scorer_ml.py
# """
# Pet Health Risk Scorer ML Agent for PawCare+.
# Predicts health risk score (0-1) using trained ML model.
# """

# import os
# import sys
# import pickle
# import logging
# from pathlib import Path
# from typing import Dict, Any, List, Optional, Union

# import numpy as np

# # Configure logging
# logger = logging.getLogger(__name__)

# # Import sklearn conditionally to handle missing dependencies gracefully
# try:
#     from sklearn.base import BaseEstimator
#     from sklearn.preprocessing import StandardScaler, LabelEncoder
#     SKLEARN_AVAILABLE = True
# except ImportError:
#     SKLEARN_AVAILABLE = False
#     logger.warning("scikit-learn not installed. Install with: pip install scikit-learn")


# class PetHealthRiskScorerAgent:
#     """
#     ML agent for predicting pet health risk score.
    
#     Uses trained regression model to predict a health risk score between 0 and 1
#     based on extracted pet profile features.
    
#     Class Attributes:
#         model: Loaded regressor model from pickle
#         scaler: Loaded StandardScaler for feature scaling
#         encoder: Dictionary of LabelEncoders for categorical features
#     """
    
#     # Feature configuration
#     CATEGORICAL_FEATURES = [
#         'pet_species',
#         'weight_status',
#         'living_situation',
#         'exercise_level'
#     ]
    
#     NUMERICAL_FEATURES = [
#         'age_years',
#         'conditions_count',
#         'allergies_count'
#     ]
    
#     # Default values for missing features
#     DEFAULT_CATEGORICAL = 'unknown'
#     DEFAULT_NUMERICAL = 0
    
#     # Mapping of categorical values to ensure consistency
#     EXPECTED_CATEGORIES = {
#         'pet_species': ['dog', 'cat', 'rabbit', 'bird', 'reptile', 'other', 'unknown'],
#         'weight_status': ['underweight', 'normal', 'overweight', 'obese', 'unknown'],
#         'living_situation': ['apartment', 'house', 'farm', 'outdoor', 'mixed', 'unknown'],
#         'exercise_level': ['sedentary', 'light', 'moderate', 'very active', 'unknown']
#     }
    
#     def __init__(self):
#         """Initialize the health risk scorer agent with trained models."""
#         if not SKLEARN_AVAILABLE:
#             raise ImportError(
#                 "scikit-learn is required for ML agents. "
#                 "Install with: pip install scikit-learn"
#             )
        
#         # Configure scikit-learn for thread-safe execution
#         self._configure_sklearn()
        
#         # Load model artifacts
#         self.model = None
#         self.scaler = None
#         self.encoder = None
        
#         self._load_model_artifacts()
        
#         logger.info("PetHealthRiskScorerAgent initialized successfully")
    
#     def _configure_sklearn(self) -> None:
#         """Configure scikit-learn for thread-safe execution."""
#         try:
#             import sklearn
#             # Set n_jobs to 1 for thread safety
#             os.environ['SKLEARN_N_JOBS'] = '1'
#             # Configure for finite data assumption
#             sklearn.set_config(assume_finite=True)
#             logger.debug("Configured scikit-learn for thread-safe execution")
#         except Exception as e:
#             logger.warning(f"Could not configure scikit-learn: {str(e)}")
    
#     def _load_model_artifacts(self) -> None:
#         """
#         Load model, scaler, and encoder from pickle files.
        
#         Raises:
#             FileNotFoundError: If model artifacts not found
#             pickle.PickleError: If loading fails
#         """
#         # Construct path to ml/models directory
#         # Get the directory of the current file
#         current_dir = Path(__file__).parent.absolute()
#         # Go up to project root and then to ml/models
#         project_root = current_dir.parent
#         models_dir = project_root / 'ml' / 'models'
        
#         model_files = {
#             'model': 'pet_health_risk_model.pkl',
#             'scaler': 'pet_health_risk_scaler.pkl',
#             'encoder': 'pet_health_risk_encoder.pkl'
#         }
        
#         missing_files = []
        
#         for artifact_name, filename in model_files.items():
#             file_path = models_dir / filename
#             try:
#                 if not file_path.exists():
#                     missing_files.append(str(file_path))
#                     continue
                
#                 with open(file_path, 'rb') as f:
#                     artifact = pickle.load(f)
                
#                 setattr(self, artifact_name, artifact)
#                 logger.debug(f"Loaded {artifact_name} from {file_path}")
                
#             except Exception as e:
#                 logger.error(f"Failed to load {filename}: {str(e)}")
#                 raise
        
#         if missing_files:
#             error_msg = f"Missing model files: {missing_files}"
#             logger.error(error_msg)
#             raise FileNotFoundError(error_msg)
        
#         # Set model n_jobs to 1 to disable parallel execution
#         if hasattr(self.model, 'n_jobs'):
#             try:
#                 self.model.n_jobs = 1
#                 logger.debug("Set model n_jobs=1 for thread safety")
#             except Exception as e:
#                 logger.warning(f"Could not set model n_jobs: {str(e)}")
    
#     def predict_health_risk(self, profile: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Predict health risk score (0-1) from extracted pet profile.
        
#         Args:
#             profile: Dictionary containing extracted pet profile with fields:
#                 - pet_species: str
#                 - weight_status: str
#                 - living_situation: str
#                 - exercise_level: str
#                 - age_years: int
#                 - known_conditions: list
#                 - allergies_known: list
#                 (other fields are ignored)
        
#         Returns:
#             Dictionary containing:
#                 - health_risk_score: Float between 0.0 and 1.0
#                 - status: String "success" or "error"
#                 - message: Optional status message
#                 - feature_contributions: Optional dict of feature importance
#         """
#         try:
#             # Extract and prepare features
#             features = self._extract_features(profile)
            
#             # Build feature array
#             X = self._build_feature_array(features)
            
#             # Scale features
#             X_scaled = self.scaler.transform(X)
            
#             # Predict
#             raw_prediction = self.model.predict(X_scaled)[0]
            
#             # Clip to valid range
#             risk_score = float(np.clip(raw_prediction, 0.0, 1.0))
            
#             # Calculate feature contributions if model supports it
#             contributions = self._get_feature_contributions(X_scaled)
            
#             logger.info(f"Health risk prediction: {risk_score:.3f}")
            
#             return {
#                 "health_risk_score": risk_score,
#                 "status": "success",
#                 "message": "Prediction successful",
#                 "feature_contributions": contributions,
#                 "raw_prediction": float(raw_prediction)
#             }
            
#         except Exception as e:
#             logger.error(f"Health risk prediction failed: {str(e)}")
#             return {
#                 "health_risk_score": 0.5,  # Default medium risk on error
#                 "status": "error",
#                 "message": f"Prediction failed: {str(e)}",
#                 "feature_contributions": None
#             }
    
#     def _extract_features(self, profile: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Extract and prepare features from profile.
        
#         Args:
#             profile: Raw extracted profile
            
#         Returns:
#             Dictionary with processed features
#         """
#         features = {}
        
#         # Extract categorical features
#         for cat_feat in self.CATEGORICAL_FEATURES:
#             value = profile.get(cat_feat, self.DEFAULT_CATEGORICAL)
#             # Ensure value is string and lowercase for consistency
#             if not isinstance(value, str):
#                 value = str(value) if value is not None else self.DEFAULT_CATEGORICAL
#             features[cat_feat] = value.lower().strip()
        
#         # Extract numerical features
#         # Age
#         age = profile.get('age_years', self.DEFAULT_NUMERICAL)
#         try:
#             features['age_years'] = float(age) if age not in [None, ''] else 0.0
#         except (ValueError, TypeError):
#             features['age_years'] = 0.0
        
#         # Count of known conditions
#         conditions = profile.get('known_conditions', [])
#         if not isinstance(conditions, list):
#             conditions = []
#         features['conditions_count'] = len(conditions)
        
#         # Count of allergies
#         allergies = profile.get('allergies_known', [])
#         if not isinstance(allergies, list):
#             allergies = []
#         features['allergies_count'] = len(allergies)
        
#         return features
    
#     def _encode_categorical(self, feature_name: str, value: str) -> int:
#         """
#         Encode a categorical feature using the loaded encoder.
        
#         Args:
#             feature_name: Name of categorical feature
#             value: Raw categorical value
            
#         Returns:
#             Encoded integer value
#         """
#         try:
#             encoder_dict = self.encoder.get(feature_name, {})
#             if not encoder_dict:
#                 logger.warning(f"No encoder found for {feature_name}, using default")
#                 return 0
            
#             # Try to transform
#             encoded = encoder_dict.transform([value])[0]
#             return int(encoded)
            
#         except ValueError as e:
#             # Unseen label - use most common class
#             logger.debug(f"Unseen label '{value}' for {feature_name}, using default")
            
#             # Try to get most common class (first in classes_)
#             try:
#                 if hasattr(encoder_dict, 'classes_') and len(encoder_dict.classes_) > 0:
#                     # Use the first class as default (most common)
#                     default_value = encoder_dict.transform([encoder_dict.classes_[0]])[0]
#                     return int(default_value)
#             except Exception:
#                 pass
            
#             # Fallback to 0
#             return 0
            
#         except Exception as e:
#             logger.warning(f"Encoding failed for {feature_name}: {str(e)}")
#             return 0
    
#     def _build_feature_array(self, features: Dict[str, Any]) -> np.ndarray:
#         """
#         Build feature array for prediction.
        
#         Feature order: [encoded_species, encoded_weight, encoded_living, 
#                        encoded_exercise, age, conditions_count, allergies_count]
        
#         Args:
#             features: Extracted features dictionary
            
#         Returns:
#             2D numpy array with shape (1, 7)
#         """
#         # Encode categorical features
#         encoded_species = self._encode_categorical('pet_species', features['pet_species'])
#         encoded_weight = self._encode_categorical('weight_status', features['weight_status'])
#         encoded_living = self._encode_categorical('living_situation', features['living_situation'])
#         encoded_exercise = self._encode_categorical('exercise_level', features['exercise_level'])
        
#         # Build array
#         X = np.array([[
#             encoded_species,
#             encoded_weight,
#             encoded_living,
#             encoded_exercise,
#             features['age_years'],
#             features['conditions_count'],
#             features['allergies_count']
#         ]])
        
#         logger.debug(f"Feature array: {X}")
#         return X
    
#     def _get_feature_contributions(self, X_scaled: np.ndarray) -> Optional[Dict[str, float]]:
#         """
#         Calculate feature contributions to prediction if model supports it.
        
#         Args:
#             X_scaled: Scaled feature array
            
#         Returns:
#             Dictionary mapping feature names to contribution values, or None
#         """
#         try:
#             contributions = {}
            
#             # For tree-based models with feature_importances_
#             if hasattr(self.model, 'feature_importances_'):
#                 feature_names = [
#                     'species', 'weight', 'living', 'exercise',
#                     'age', 'conditions_count', 'allergies_count'
#                 ]
                
#                 importances = self.model.feature_importances_
#                 for name, imp in zip(feature_names, importances):
#                     contributions[name] = float(imp)
            
#             # For linear models with coefficients
#             elif hasattr(self.model, 'coef_'):
#                 feature_names = [
#                     'species', 'weight', 'living', 'exercise',
#                     'age', 'conditions_count', 'allergies_count'
#                 ]
                
#                 coefs = self.model.coef_[0] if self.model.coef_.ndim > 1 else self.model.coef_
#                 for name, coef in zip(feature_names, coefs):
#                     contributions[name] = float(coef)
            
#             return contributions if contributions else None
            
#         except Exception as e:
#             logger.debug(f"Could not calculate feature contributions: {str(e)}")
#             return None
    
#     def get_feature_names(self) -> List[str]:
#         """
#         Get list of feature names used by the model.
        
#         Returns:
#             List of feature names
#         """
#         return self.CATEGORICAL_FEATURES + self.NUMERICAL_FEATURES
    
#     def validate_profile_completeness(self, profile: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Validate that profile contains all required fields for prediction.
        
#         Args:
#             profile: Extracted profile
            
#         Returns:
#             Dictionary with validation results
#         """
#         required = self.CATEGORICAL_FEATURES + ['age_years', 'known_conditions', 'allergies_known']
#         missing = [field for field in required if field not in profile]
        
#         return {
#             "complete": len(missing) == 0,
#             "missing_fields": missing,
#             "available_fields": [f for f in required if f in profile]
#         }


# # ==========================================
# # CONVENIENCE FUNCTION
# # ==========================================

# def predict_health_risk(profile: Dict[str, Any]) -> float:
#     """
#     Convenience function to predict health risk score.
    
#     Args:
#         profile: Extracted pet profile
        
#     Returns:
#         Health risk score between 0 and 1
#     """
#     try:
#         agent = PetHealthRiskScorerAgent()
#         result = agent.predict_health_risk(profile)
#         return result.get("health_risk_score", 0.5)
#     except Exception as e:
#         logger.error(f"Prediction failed: {str(e)}")
#         return 0.5  # Default medium risk on error


# # ==========================================
# # MODULE TESTING
# # ==========================================

# if __name__ == "__main__":
#     print("=" * 60)
#     print("PET HEALTH RISK SCORER ML AGENT TEST")
#     print("=" * 60)
    
#     try:
#         # Initialize agent
#         agent = PetHealthRiskScorerAgent()
#         print("✅ Agent initialized successfully")
        
#         # Test with sample profiles
#         test_profiles = [
#             {
#                 "pet_species": "dog",
#                 "weight_status": "normal",
#                 "living_situation": "house",
#                 "exercise_level": "very active",
#                 "age_years": 3,
#                 "known_conditions": [],
#                 "allergies_known": []
#             },
#             {
#                 "pet_species": "cat",
#                 "weight_status": "overweight",
#                 "living_situation": "apartment",
#                 "exercise_level": "sedentary",
#                 "age_years": 12,
#                 "known_conditions": ["arthritis", "kidney disease"],
#                 "allergies_known": ["pollen"]
#             },
#             {
#                 "pet_species": "unknown",
#                 "weight_status": "unknown",
#                 "living_situation": "unknown",
#                 "exercise_level": "unknown",
#                 "age_years": 0,
#                 "known_conditions": [],
#                 "allergies_known": []
#             }
#         ]
        
#         for i, profile in enumerate(test_profiles, 1):
#             print(f"\n📝 Test Profile {i}:")
#             result = agent.predict_health_risk(profile)
#             print(f"  Risk Score: {result['health_risk_score']:.3f}")
#             print(f"  Status: {result['status']}")
#             if result.get('feature_contributions'):
#                 print(f"  Feature Contributions: {result['feature_contributions']}")
        
#         print("\n✅ Pet Health Risk Scorer Agent ready for use")
        
#     except Exception as e:
#         print(f"❌ Error: {str(e)}")



# agents/pet_health_risk_scorer_ml.py
"""
Pet Health Risk Scorer ML Agent for PawCare+.
Predicts health risk score (0-1) using trained ML model with symptom-based adjustment.
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed")


class PetHealthRiskScorerAgent:
    """
    ML agent for predicting pet health risk score with symptom-based adjustment.
    """
    
    CATEGORICAL_FEATURES = [
        'pet_species',
        'weight_status',
        'living_situation',
        'exercise_level'
    ]
    
    NUMERICAL_FEATURES = [
        'age_years',
        'conditions_count',
        'allergies_count'
    ]
    
    DEFAULT_CATEGORICAL = 'unknown'
    DEFAULT_NUMERICAL = 0
    
    def __init__(self):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required")
        
        self._configure_sklearn()
        self.model = None
        self.scaler = None
        self.encoder = None
        self._load_model_artifacts()
        logger.info("PetHealthRiskScorerAgent initialized")
    
    def _configure_sklearn(self) -> None:
        try:
            import sklearn
            os.environ['SKLEARN_N_JOBS'] = '1'
            sklearn.set_config(assume_finite=True)
        except Exception as e:
            logger.warning(f"Could not configure scikit-learn: {str(e)}")
    
    def _load_model_artifacts(self) -> None:
        current_dir = Path(__file__).parent.absolute()
        project_root = current_dir.parent
        models_dir = project_root / 'ml' / 'models'
        
        model_files = {
            'model': 'pet_health_risk_model.pkl',
            'scaler': 'pet_health_risk_scaler.pkl',
            'encoder': 'pet_health_risk_encoder.pkl'
        }
        
        for artifact_name, filename in model_files.items():
            file_path = models_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Missing model file: {file_path}")
            
            with open(file_path, 'rb') as f:
                setattr(self, artifact_name, pickle.load(f))
            logger.debug(f"Loaded {artifact_name}")
        
        if hasattr(self.model, 'n_jobs'):
            self.model.n_jobs = 1
    
    def predict_health_risk(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        try:
            features = self._extract_features(profile)
            X = self._build_feature_array(features)
            X_scaled = self.scaler.transform(X)
            
            # Base prediction - CONVERT TO PYTHON FLOAT
            base_risk = float(self.model.predict(X_scaled)[0])
            
            # Calculate symptom-based adjustment
            symptom_boost = self._calculate_symptom_boost(profile)
            
            # Final risk score - CONVERT TO PYTHON FLOAT
            final_risk = float(min(base_risk + symptom_boost, 1.0))
            
            logger.info(f"Base risk: {base_risk:.3f}, Symptom boost: {symptom_boost:.3f}, Final: {final_risk:.3f}")
            
            return {
                "health_risk_score": final_risk,
                "status": "success",
                "message": "Prediction successful",
                "base_risk": float(base_risk),
                "symptom_boost": float(symptom_boost)
            }
            
        except Exception as e:
            logger.error(f"Health risk prediction failed: {str(e)}")
            return {
                "health_risk_score": 0.5,
                "status": "error",
                "message": f"Prediction failed: {str(e)}"
            }
    
    def _calculate_symptom_boost(self, profile: Dict[str, Any]) -> float:
        """Calculate risk boost based on symptoms."""
        symptoms = profile.get('recent_symptoms', [])
        if not isinstance(symptoms, list):
            symptoms = []
        
        symptom_text = " ".join(symptoms).lower()
        symptom_boost = 0.0
        
        # Symptom combination boosts
        if 'thirst' in symptom_text and ('weight loss' in symptom_text or 'weight' in symptom_text):
            symptom_boost += 0.25
        
        if 'cough' in symptom_text and 'lethargy' in symptom_text:
            symptom_boost += 0.20
        
        if 'vomiting' in symptom_text and 'diarrhea' in symptom_text:
            symptom_boost += 0.20
        
        if 'pain' in symptom_text or 'crying' in symptom_text:
            symptom_boost += 0.15
        
        # Symptom count boost
        symptom_count = len(symptoms)
        if symptom_count >= 4:
            symptom_boost += 0.15
        elif symptom_count >= 3:
            symptom_boost += 0.10
        elif symptom_count >= 2:
            symptom_boost += 0.05
        
        # Duration boost
        duration = profile.get('symptom_duration_days', 0)
        if isinstance(duration, (int, float)):
            if duration > 14:
                symptom_boost += 0.10
            elif duration > 7:
                symptom_boost += 0.05
        
        # Severity boost
        severity = profile.get('symptom_severity', 'unknown')
        if severity == 'severe':
            symptom_boost += 0.15
        elif severity == 'moderate':
            symptom_boost += 0.08
        
        # Senior pet multiplier
        age = profile.get('age_years', 0)
        if isinstance(age, (int, float)) and age > 10 and symptom_count > 0:
            symptom_boost *= 1.2
        
        return float(min(symptom_boost, 0.4))
    
    def _extract_features(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        features = {}
        
        for cat_feat in self.CATEGORICAL_FEATURES:
            value = profile.get(cat_feat, self.DEFAULT_CATEGORICAL)
            if not isinstance(value, str):
                value = str(value) if value is not None else self.DEFAULT_CATEGORICAL
            features[cat_feat] = value.lower().strip()
        
        age = profile.get('age_years', self.DEFAULT_NUMERICAL)
        try:
            features['age_years'] = float(age) if age not in [None, ''] else 0.0
        except (ValueError, TypeError):
            features['age_years'] = 0.0
        
        conditions = profile.get('known_conditions', [])
        features['conditions_count'] = float(len(conditions)) if isinstance(conditions, list) else 0.0
        
        allergies = profile.get('allergies_known', [])
        features['allergies_count'] = float(len(allergies)) if isinstance(allergies, list) else 0.0
        
        return features
    
    def _encode_categorical(self, feature_name: str, value: str) -> int:
        try:
            encoder_dict = self.encoder.get(feature_name, {})
            if not encoder_dict:
                return 0
            return int(encoder_dict.transform([value])[0])
        except ValueError:
            if hasattr(encoder_dict, 'classes_') and len(encoder_dict.classes_) > 0:
                return int(encoder_dict.transform([encoder_dict.classes_[0]])[0])
            return 0
        except Exception:
            return 0
    
    def _build_feature_array(self, features: Dict[str, Any]) -> np.ndarray:
        encoded_species = self._encode_categorical('pet_species', features['pet_species'])
        encoded_weight = self._encode_categorical('weight_status', features['weight_status'])
        encoded_living = self._encode_categorical('living_situation', features['living_situation'])
        encoded_exercise = self._encode_categorical('exercise_level', features['exercise_level'])
        
        X = np.array([[
            float(encoded_species),
            float(encoded_weight),
            float(encoded_living),
            float(encoded_exercise),
            features['age_years'],
            features['conditions_count'],
            features['allergies_count']
        ]])
        
        return X
    
    def get_feature_names(self) -> List[str]:
        return self.CATEGORICAL_FEATURES + self.NUMERICAL_FEATURES


def predict_health_risk(profile: Dict[str, Any]) -> float:
    try:
        agent = PetHealthRiskScorerAgent()
        result = agent.predict_health_risk(profile)
        return float(result.get("health_risk_score", 0.5))
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return 0.5