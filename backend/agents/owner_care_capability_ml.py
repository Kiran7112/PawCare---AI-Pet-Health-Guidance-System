# # agents/owner_care_capability_ml.py
# """
# Owner Care Capability ML Agent for PawCare+.
# Predicts owner care capability score (0-100) using trained ML model.
# """

# import os
# import pickle
# import logging
# from pathlib import Path
# from typing import Dict, Any, List, Optional

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


# class OwnerCareCapabilityAgent:
#     """
#     ML agent for predicting owner care capability score.
    
#     Uses trained regression model to predict an owner's capability to provide care
#     (score between 0 and 100) based on three extracted profile features:
#     - owner_experience: Owner's experience level with pets
#     - vet_access: Availability of veterinary care
#     - owner_commitment: Owner's dedication level
    
#     Class Attributes:
#         model: Loaded regressor model from pickle
#         scaler: Loaded StandardScaler for feature scaling
#         encoder: Dictionary of LabelEncoders for categorical features
#     """
    
#     # Feature configuration
#     CATEGORICAL_FEATURES = [
#         'owner_experience',
#         'vet_access',
#         'owner_commitment'
#     ]
    
#     # Expected categories for validation
#     EXPECTED_CATEGORIES = {
#         'owner_experience': ['novice', 'experienced', 'expert', 'unknown'],
#         'vet_access': ['regular', 'emergency only', 'limited', 'none', 'unknown'],
#         'owner_commitment': ['casual', 'dedicated', 'obsessive', 'unknown']
#     }
    
#     # Default values for missing features
#     DEFAULT_CATEGORICAL = 'unknown'
    
#     # Score bounds
#     MIN_SCORE = 0.0
#     MAX_SCORE = 100.0
    
#     def __init__(self):
#         """Initialize the owner care capability agent with trained models."""
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
        
#         logger.info("OwnerCareCapabilityAgent initialized successfully")
    
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
#             'model': 'owner_care_capability_model.pkl',
#             'scaler': 'owner_care_capability_scaler.pkl',
#             'encoder': 'owner_care_capability_encoder.pkl'
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
    
#     def predict_capability(self, profile: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Predict owner care capability score (0-100) from extracted profile.
        
#         Args:
#             profile: Dictionary containing extracted profile with fields:
#                 - owner_experience: str (novice, experienced, expert, unknown)
#                 - vet_access: str (regular, emergency only, limited, none, unknown)
#                 - owner_commitment: str (casual, dedicated, obsessive, unknown)
        
#         Returns:
#             Dictionary containing:
#                 - care_capability_score: Float between 0.0 and 100.0
#                 - status: String "success" or "error"
#                 - message: Optional status message
#                 - feature_contributions: Optional dict of feature importance
#                 - confidence: Optional confidence score (0-1)
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
#             capability_score = float(np.clip(raw_prediction, self.MIN_SCORE, self.MAX_SCORE))
            
#             # Calculate feature contributions if model supports it
#             contributions = self._get_feature_contributions(X_scaled)
            
#             # Calculate confidence based on feature completeness
#             confidence = self._calculate_confidence(features)
            
#             logger.info(f"Care capability prediction: {capability_score:.1f} (confidence: {confidence:.2f})")
            
#             return {
#                 "care_capability_score": capability_score,
#                 "status": "success",
#                 "message": "Prediction successful",
#                 "feature_contributions": contributions,
#                 "confidence": confidence,
#                 "raw_prediction": float(raw_prediction),
#                 "features_used": features
#             }
            
#         except Exception as e:
#             logger.error(f"Care capability prediction failed: {str(e)}")
#             return {
#                 "care_capability_score": 50.0,  # Default medium capability on error
#                 "status": "error",
#                 "message": f"Prediction failed: {str(e)}",
#                 "feature_contributions": None,
#                 "confidence": 0.0
#             }
    
#     def _extract_features(self, profile: Dict[str, Any]) -> Dict[str, str]:
#         """
#         Extract and prepare features from profile.
        
#         Args:
#             profile: Raw extracted profile
            
#         Returns:
#             Dictionary with processed feature values
#         """
#         features = {}
        
#         for feature in self.CATEGORICAL_FEATURES:
#             value = profile.get(feature, self.DEFAULT_CATEGORICAL)
            
#             # Ensure value is string and lowercase for consistency
#             if not isinstance(value, str):
#                 value = str(value) if value is not None else self.DEFAULT_CATEGORICAL
            
#             # Clean and normalize
#             cleaned_value = value.lower().strip()
            
#             # Map common variations to expected categories
#             cleaned_value = self._normalize_category(feature, cleaned_value)
            
#             features[feature] = cleaned_value
#             logger.debug(f"Extracted {feature}: '{cleaned_value}' (from '{value}')")
        
#         return features
    
#     def _normalize_category(self, feature: str, value: str) -> str:
#         """
#         Normalize category value to expected format.
        
#         Args:
#             feature: Feature name
#             value: Raw category value
            
#         Returns:
#             Normalized category value
#         """
#         # Handle empty or None
#         if not value or value in ['none', 'null', 'nil', '']:
#             return 'unknown'
        
#         # Feature-specific mappings
#         mappings = {
#             'owner_experience': {
#                 'beginner': 'novice',
#                 'new': 'novice',
#                 'first time': 'novice',
#                 'intermediate': 'experienced',
#                 'advanced': 'expert',
#                 'professional': 'expert'
#             },
#             'vet_access': {
#                 '24/7': 'regular',
#                 '24-7': 'regular',
#                 'always': 'regular',
#                 'sometimes': 'emergency only',
#                 'rarely': 'limited',
#                 'never': 'none'
#             },
#             'owner_commitment': {
#                 'low': 'casual',
#                 'medium': 'dedicated',
#                 'high': 'obsessive',
#                 'very high': 'obsessive',
#                 'extreme': 'obsessive'
#             }
#         }
        
#         # Apply mappings if available
#         if feature in mappings and value in mappings[feature]:
#             return mappings[feature][value]
        
#         # Check if value is in expected categories
#         expected = self.EXPECTED_CATEGORIES.get(feature, [])
#         if value in expected:
#             return value
        
#         # If not found, try partial matching
#         for expected_value in expected:
#             if expected_value in value or value in expected_value:
#                 return expected_value
        
#         # Default to unknown
#         return 'unknown'
    
#     def _encode_categorical(self, feature_name: str, value: str) -> int:
#         """
#         Encode a categorical feature using the loaded encoder.
        
#         Args:
#             feature_name: Name of categorical feature
#             value: Raw categorical value (already normalized)
            
#         Returns:
#             Encoded integer value
#         """
#         try:
#             # Get encoder for this feature
#             encoder_dict = self.encoder.get(feature_name, {})
#             if not encoder_dict:
#                 logger.warning(f"No encoder found for {feature_name}, using default")
#                 return 0
            
#             # Try to transform
#             encoded = encoder_dict.transform([value])[0]
#             return int(encoded)
            
#         except ValueError as e:
#             # Unseen label - use most common class
#             logger.debug(f"Unseen label '{value}' for {feature_name}, using most common class")
            
#             # Try to get most common class (first in classes_)
#             try:
#                 if hasattr(encoder_dict, 'classes_') and len(encoder_dict.classes_) > 0:
#                     # Use the first class as default (most common in training)
#                     default_value = encoder_dict.transform([encoder_dict.classes_[0]])[0]
#                     return int(default_value)
#             except Exception:
#                 pass
            
#             # Fallback to 0
#             return 0
            
#         except Exception as e:
#             logger.warning(f"Encoding failed for {feature_name}: {str(e)}")
#             return 0
    
#     def _build_feature_array(self, features: Dict[str, str]) -> np.ndarray:
#         """
#         Build feature array for prediction.
        
#         Feature order: [encoded_experience, encoded_vet_access, encoded_commitment]
        
#         Args:
#             features: Extracted features dictionary
            
#         Returns:
#             2D numpy array with shape (1, 3)
#         """
#         # Encode categorical features
#         encoded_experience = self._encode_categorical('owner_experience', features['owner_experience'])
#         encoded_vet = self._encode_categorical('vet_access', features['vet_access'])
#         encoded_commitment = self._encode_categorical('owner_commitment', features['owner_commitment'])
        
#         # Build array
#         X = np.array([[
#             encoded_experience,
#             encoded_vet,
#             encoded_commitment
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
#             feature_names = ['experience', 'vet_access', 'commitment']
            
#             # For tree-based models with feature_importances_
#             if hasattr(self.model, 'feature_importances_'):
#                 importances = self.model.feature_importances_
#                 for name, imp in zip(feature_names, importances):
#                     contributions[name] = float(imp)
            
#             # For linear models with coefficients
#             elif hasattr(self.model, 'coef_'):
#                 coefs = self.model.coef_[0] if self.model.coef_.ndim > 1 else self.model.coef_
#                 for name, coef in zip(feature_names, coefs):
#                     contributions[name] = float(coef)
            
#             return contributions if contributions else None
            
#         except Exception as e:
#             logger.debug(f"Could not calculate feature contributions: {str(e)}")
#             return None
    
#     def _calculate_confidence(self, features: Dict[str, str]) -> float:
#         """
#         Calculate confidence score based on feature completeness.
        
#         Args:
#             features: Extracted features
            
#         Returns:
#             Confidence score between 0 and 1
#         """
#         if not features:
#             return 0.0
        
#         # Count features that are not 'unknown'
#         known_features = sum(1 for v in features.values() if v != 'unknown')
        
#         # Base confidence on known features (3 total)
#         confidence = known_features / len(self.CATEGORICAL_FEATURES)
        
#         # Bonus if all features are known and in expected categories
#         if known_features == len(self.CATEGORICAL_FEATURES):
#             # Check if all values are in expected categories
#             all_valid = True
#             for feature, value in features.items():
#                 if value not in self.EXPECTED_CATEGORIES.get(feature, []):
#                     all_valid = False
#                     break
            
#             if all_valid:
#                 confidence = min(1.0, confidence + 0.2)  # Bonus for valid categories
        
#         return confidence
    
#     def get_feature_names(self) -> List[str]:
#         """
#         Get list of feature names used by the model.
        
#         Returns:
#             List of feature names
#         """
#         return self.CATEGORICAL_FEATURES.copy()
    
#     def get_expected_categories(self) -> Dict[str, List[str]]:
#         """
#         Get expected categories for each feature.
        
#         Returns:
#             Dictionary mapping features to lists of expected categories
#         """
#         return self.EXPECTED_CATEGORIES.copy()
    
#     def validate_profile_completeness(self, profile: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Validate that profile contains all required fields for prediction.
        
#         Args:
#             profile: Extracted profile
            
#         Returns:
#             Dictionary with validation results
#         """
#         required = self.CATEGORICAL_FEATURES
#         missing = [field for field in required if field not in profile]
        
#         available = {}
#         for field in required:
#             if field in profile:
#                 value = profile[field]
#                 available[field] = {
#                     "value": value,
#                     "valid": value in self.EXPECTED_CATEGORIES.get(field, []),
#                     "normalized": self._normalize_category(field, str(value) if value else '')
#                 }
        
#         return {
#             "complete": len(missing) == 0,
#             "missing_fields": missing,
#             "available_fields": available,
#             "feature_count": len(required) - len(missing)
#         }


# # ==========================================
# # CONVENIENCE FUNCTIONS
# # ==========================================

# def predict_care_capability(profile: Dict[str, Any]) -> float:
#     """
#     Convenience function to predict care capability score.
    
#     Args:
#         profile: Extracted pet profile
        
#     Returns:
#         Care capability score between 0 and 100
#     """
#     try:
#         agent = OwnerCareCapabilityAgent()
#         result = agent.predict_capability(profile)
#         return result.get("care_capability_score", 50.0)
#     except Exception as e:
#         logger.error(f"Prediction failed: {str(e)}")
#         return 50.0  # Default medium capability on error


# def batch_predict_capability(profiles: List[Dict[str, Any]]) -> List[float]:
#     """
#     Predict care capability for multiple profiles.
    
#     Args:
#         profiles: List of profile dictionaries
        
#     Returns:
#         List of capability scores
#     """
#     try:
#         agent = OwnerCareCapabilityAgent()
#         return [agent.predict_capability(p)["care_capability_score"] for p in profiles]
#     except Exception as e:
#         logger.error(f"Batch prediction failed: {str(e)}")
#         return [50.0] * len(profiles)


# # ==========================================
# # MODULE TESTING
# # ==========================================

# if __name__ == "__main__":
#     print("=" * 60)
#     print("OWNER CARE CAPABILITY ML AGENT TEST")
#     print("=" * 60)
    
#     try:
#         # Initialize agent
#         agent = OwnerCareCapabilityAgent()
#         print("✅ Agent initialized successfully")
        
#         # Test with sample profiles
#         test_profiles = [
#             {
#                 "owner_experience": "expert",
#                 "vet_access": "regular",
#                 "owner_commitment": "obsessive"
#             },
#             {
#                 "owner_experience": "novice",
#                 "vet_access": "emergency only",
#                 "owner_commitment": "dedicated"
#             },
#             {
#                 "owner_experience": "experienced",
#                 "vet_access": "limited",
#                 "owner_commitment": "casual"
#             },
#             {
#                 # Profile with unknown values
#                 "owner_experience": "unknown",
#                 "vet_access": "unknown",
#                 "owner_commitment": "unknown"
#             },
#             {
#                 # Profile with variations that need normalization
#                 "owner_experience": "beginner",
#                 "vet_access": "24/7",
#                 "owner_commitment": "very high"
#             }
#         ]
        
#         print("\n📊 Feature Information:")
#         print(f"Features: {agent.get_feature_names()}")
#         print(f"Expected categories: {agent.get_expected_categories()}")
        
#         for i, profile in enumerate(test_profiles, 1):
#             print(f"\n📝 Test Profile {i}:")
#             print(f"  Input: {profile}")
            
#             # Validate completeness
#             validation = agent.validate_profile_completeness(profile)
#             print(f"  Complete: {validation['complete']}")
            
#             # Predict
#             result = agent.predict_capability(profile)
#             print(f"  Capability Score: {result['care_capability_score']:.1f}/100")
#             print(f"  Confidence: {result.get('confidence', 0):.2f}")
#             print(f"  Status: {result['status']}")
            
#             if result.get('feature_contributions'):
#                 print(f"  Feature Contributions: {result['feature_contributions']}")
        
#         print("\n✅ Owner Care Capability Agent ready for use")
        
#     except Exception as e:
#         print(f"❌ Error: {str(e)}")



# agents/owner_care_capability_ml.py
"""
Owner Care Capability ML Agent for PawCare+.
Predicts owner care capability score (0-100) using trained ML model.
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Import sklearn conditionally
try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed. Install with: pip install scikit-learn")


class OwnerCareCapabilityAgent:
    """
    ML agent for predicting owner care capability score.
    """
    
    # Feature configuration
    CATEGORICAL_FEATURES = [
        'owner_experience',
        'vet_access',
        'owner_commitment'
    ]
    
    # Expected categories for validation
    EXPECTED_CATEGORIES = {
        'owner_experience': ['novice', 'experienced', 'expert', 'unknown'],
        'vet_access': ['regular', 'emergency only', 'limited', 'none', 'unknown'],
        'owner_commitment': ['casual', 'dedicated', 'obsessive', 'unknown']
    }
    
    # Default values for missing features
    DEFAULT_CATEGORICAL = 'unknown'
    
    # Score bounds
    MIN_SCORE = 0.0
    MAX_SCORE = 100.0
    
    def __init__(self):
        """Initialize the owner care capability agent with trained models."""
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for ML agents. "
                "Install with: pip install scikit-learn"
            )
        
        # Configure scikit-learn for thread-safe execution
        self._configure_sklearn()
        
        # Load model artifacts
        self.model = None
        self.scaler = None
        self.encoder = None
        
        self._load_model_artifacts()
        
        logger.info("OwnerCareCapabilityAgent initialized successfully")
    
    def _configure_sklearn(self) -> None:
        """Configure scikit-learn for thread-safe execution."""
        try:
            import sklearn
            os.environ['SKLEARN_N_JOBS'] = '1'
            sklearn.set_config(assume_finite=True)
            logger.debug("Configured scikit-learn for thread-safe execution")
        except Exception as e:
            logger.warning(f"Could not configure scikit-learn: {str(e)}")
    
    def _load_model_artifacts(self) -> None:
        """Load model, scaler, and encoder from pickle files."""
        current_dir = Path(__file__).parent.absolute()
        project_root = current_dir.parent
        models_dir = project_root / 'ml' / 'models'
        
        model_files = {
            'model': 'owner_care_capability_model.pkl',
            'scaler': 'owner_care_capability_scaler.pkl',
            'encoder': 'owner_care_capability_encoder.pkl'
        }
        
        missing_files = []
        
        for artifact_name, filename in model_files.items():
            file_path = models_dir / filename
            try:
                if not file_path.exists():
                    missing_files.append(str(file_path))
                    continue
                
                with open(file_path, 'rb') as f:
                    artifact = pickle.load(f)
                
                setattr(self, artifact_name, artifact)
                logger.debug(f"Loaded {artifact_name} from {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to load {filename}: {str(e)}")
                raise
        
        if missing_files:
            error_msg = f"Missing model files: {missing_files}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Set model n_jobs to 1 for thread safety
        if hasattr(self.model, 'n_jobs'):
            try:
                self.model.n_jobs = 1
                logger.debug("Set model n_jobs=1 for thread safety")
            except Exception as e:
                logger.warning(f"Could not set model n_jobs: {str(e)}")
    
    def predict_capability(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict owner care capability score (0-100) from extracted profile.
        
        Returns:
            Dictionary with care_capability_score (Python float), status, confidence
        """
        try:
            # Extract and prepare features
            features = self._extract_features(profile)
            
            # Build feature array
            X = self._build_feature_array(features)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Predict - CONVERT TO PYTHON FLOAT
            raw_prediction = float(self.model.predict(X_scaled)[0])
            
            # Clip to valid range - CONVERT TO PYTHON FLOAT
            capability_score = float(np.clip(raw_prediction, self.MIN_SCORE, self.MAX_SCORE))
            
            # Calculate feature contributions if model supports it
            contributions = self._get_feature_contributions()
            
            # Calculate confidence based on feature completeness
            confidence = self._calculate_confidence(features)
            
            logger.info(f"Care capability prediction: {capability_score:.1f} (confidence: {confidence:.2f})")
            
            return {
                "care_capability_score": capability_score,
                "status": "success",
                "message": "Prediction successful",
                "feature_contributions": contributions,
                "confidence": float(confidence),
                "raw_prediction": float(raw_prediction),
                "features_used": features
            }
            
        except Exception as e:
            logger.error(f"Care capability prediction failed: {str(e)}")
            return {
                "care_capability_score": 50.0,
                "status": "error",
                "message": f"Prediction failed: {str(e)}",
                "feature_contributions": None,
                "confidence": 0.0
            }
    
    def _extract_features(self, profile: Dict[str, Any]) -> Dict[str, str]:
        """Extract and prepare features from profile."""
        features = {}
        
        for feature in self.CATEGORICAL_FEATURES:
            value = profile.get(feature, self.DEFAULT_CATEGORICAL)
            
            if not isinstance(value, str):
                value = str(value) if value is not None else self.DEFAULT_CATEGORICAL
            
            cleaned_value = value.lower().strip()
            cleaned_value = self._normalize_category(feature, cleaned_value)
            
            features[feature] = cleaned_value
            logger.debug(f"Extracted {feature}: '{cleaned_value}'")
        
        return features
    
    def _normalize_category(self, feature: str, value: str) -> str:
        """Normalize category value to expected format."""
        if not value or value in ['none', 'null', 'nil', '']:
            return 'unknown'
        
        mappings = {
            'owner_experience': {
                'beginner': 'novice',
                'new': 'novice',
                'first time': 'novice',
                'intermediate': 'experienced',
                'advanced': 'expert',
                'professional': 'expert'
            },
            'vet_access': {
                '24/7': 'regular',
                '24-7': 'regular',
                'always': 'regular',
                'sometimes': 'emergency only',
                'rarely': 'limited',
                'never': 'none'
            },
            'owner_commitment': {
                'low': 'casual',
                'medium': 'dedicated',
                'high': 'obsessive',
                'very high': 'obsessive',
                'extreme': 'obsessive'
            }
        }
        
        if feature in mappings and value in mappings[feature]:
            return mappings[feature][value]
        
        expected = self.EXPECTED_CATEGORIES.get(feature, [])
        if value in expected:
            return value
        
        for expected_value in expected:
            if expected_value in value or value in expected_value:
                return expected_value
        
        return 'unknown'
    
    def _encode_categorical(self, feature_name: str, value: str) -> int:
        """Encode a categorical feature using the loaded encoder."""
        try:
            encoder_dict = self.encoder.get(feature_name, {})
            if not encoder_dict:
                return 0
            
            encoded = int(encoder_dict.transform([value])[0])
            return encoded
            
        except ValueError:
            try:
                if hasattr(encoder_dict, 'classes_') and len(encoder_dict.classes_) > 0:
                    return int(encoder_dict.transform([encoder_dict.classes_[0]])[0])
            except Exception:
                pass
            return 0
        except Exception:
            return 0
    
    def _build_feature_array(self, features: Dict[str, str]) -> np.ndarray:
        """Build feature array for prediction."""
        encoded_experience = self._encode_categorical('owner_experience', features['owner_experience'])
        encoded_vet = self._encode_categorical('vet_access', features['vet_access'])
        encoded_commitment = self._encode_categorical('owner_commitment', features['owner_commitment'])
        
        X = np.array([[
            float(encoded_experience),
            float(encoded_vet),
            float(encoded_commitment)
        ]])
        
        logger.debug(f"Feature array: {X}")
        return X
    
    def _get_feature_contributions(self) -> Optional[Dict[str, float]]:
        """Calculate feature contributions if model supports it."""
        try:
            contributions = {}
            feature_names = ['experience', 'vet_access', 'commitment']
            
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                for name, imp in zip(feature_names, importances):
                    contributions[name] = float(imp)
            
            return contributions if contributions else None
            
        except Exception as e:
            logger.debug(f"Could not calculate feature contributions: {str(e)}")
            return None
    
    def _calculate_confidence(self, features: Dict[str, str]) -> float:
        """Calculate confidence score based on feature completeness."""
        if not features:
            return 0.0
        
        known_features = sum(1 for v in features.values() if v != 'unknown')
        confidence = float(known_features / len(self.CATEGORICAL_FEATURES))
        
        if known_features == len(self.CATEGORICAL_FEATURES):
            all_valid = True
            for feature, value in features.items():
                if value not in self.EXPECTED_CATEGORIES.get(feature, []):
                    all_valid = False
                    break
            if all_valid:
                confidence = min(1.0, confidence + 0.2)
        
        return float(confidence)
    
    def get_feature_names(self) -> List[str]:
        return self.CATEGORICAL_FEATURES.copy()
    
    def get_expected_categories(self) -> Dict[str, List[str]]:
        return self.EXPECTED_CATEGORIES.copy()
    
    def validate_profile_completeness(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        required = self.CATEGORICAL_FEATURES
        missing = [field for field in required if field not in profile]
        
        available = {}
        for field in required:
            if field in profile:
                value = profile[field]
                available[field] = {
                    "value": value,
                    "valid": value in self.EXPECTED_CATEGORIES.get(field, []),
                    "normalized": self._normalize_category(field, str(value) if value else '')
                }
        
        return {
            "complete": len(missing) == 0,
            "missing_fields": missing,
            "available_fields": available,
            "feature_count": len(required) - len(missing)
        }


# ==========================================
# CONVENIENCE FUNCTIONS
# ==========================================

def predict_care_capability(profile: Dict[str, Any]) -> float:
    """Convenience function to predict care capability score."""
    try:
        agent = OwnerCareCapabilityAgent()
        result = agent.predict_capability(profile)
        return float(result.get("care_capability_score", 50.0))
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return 50.0


def batch_predict_capability(profiles: List[Dict[str, Any]]) -> List[float]:
    """Predict care capability for multiple profiles."""
    try:
        agent = OwnerCareCapabilityAgent()
        return [float(agent.predict_capability(p)["care_capability_score"]) for p in profiles]
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        return [50.0] * len(profiles)


# ==========================================
# MODULE TESTING
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("OWNER CARE CAPABILITY ML AGENT TEST")
    print("=" * 60)
    
    try:
        agent = OwnerCareCapabilityAgent()
        print("✅ Agent initialized successfully")
        
        test_profiles = [
            {
                "owner_experience": "expert",
                "vet_access": "regular",
                "owner_commitment": "obsessive"
            },
            {
                "owner_experience": "novice",
                "vet_access": "emergency only",
                "owner_commitment": "dedicated"
            },
            {
                "owner_experience": "experienced",
                "vet_access": "limited",
                "owner_commitment": "casual"
            },
            {
                "owner_experience": "unknown",
                "vet_access": "unknown",
                "owner_commitment": "unknown"
            },
            {
                "owner_experience": "beginner",
                "vet_access": "24/7",
                "owner_commitment": "very high"
            }
        ]
        
        print("\n📊 Feature Information:")
        print(f"Features: {agent.get_feature_names()}")
        
        for i, profile in enumerate(test_profiles, 1):
            print(f"\n📝 Test Profile {i}:")
            validation = agent.validate_profile_completeness(profile)
            print(f"  Complete: {validation['complete']}")
            
            result = agent.predict_capability(profile)
            print(f"  Capability Score: {result['care_capability_score']:.1f}/100")
            print(f"  Confidence: {result.get('confidence', 0):.2f}")
            print(f"  Status: {result['status']}")
            
            if result.get('feature_contributions'):
                print(f"  Feature Contributions: {result['feature_contributions']}")
        
        print("\n✅ Owner Care Capability Agent ready for use")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")