# # nodes/pet_health_risk_scorer_node.py
# """
# Pet Health Risk Scorer Node for PawCare+ LangGraph workflow.
# Executes ML-based health risk prediction as a workflow step.
# """

# import logging
# from typing import Dict, Any

# from state import PetCareState
# from agents.pet_health_risk_scorer_ml import PetHealthRiskScorerAgent

# logger = logging.getLogger(__name__)


# def pet_health_risk_scorer_node(state: PetCareState) -> PetCareState:
#     """
#     Execute ML-based health risk prediction as a workflow step.
    
#     This node uses a trained ML model to predict a health risk score (0-1)
#     based on the extracted pet profile features.
    
#     Args:
#         state: Current PetCareState containing extracted profile fields
        
#     Returns:
#         Updated PetCareState with health_risk_score:
#         - health_risk_score: Float between 0.0 and 1.0
#         - health_risk_factors: Optional feature importance dictionary
#         - error_occurred: Set to True if prediction fails
#         - error_messages: Appended with any errors
#         - processing_stage: Updated to "risk_scored" or "risk_scoring_failed"
#     """
#     logger.info("=" * 50)
#     logger.info("EXECUTING PET HEALTH RISK SCORER NODE")
#     logger.info("=" * 50)
    
#     try:
#         # Log incoming state info
#         logger.debug(f"State request_id: {state.get('request_id', 'N/A')}")
#         logger.debug(f"Current processing stage: {state.get('processing_stage', 'unknown')}")
        
#         # Check if profile extraction was successful
#         if not state.get('profile_extraction_complete', False):
#             error_msg = "Cannot calculate health risk: profile extraction not complete"
#             logger.error(error_msg)
#             state["error_occurred"] = True
#             state["error_messages"].append(error_msg)
#             state["health_risk_score"] = 0.5  # Default medium risk
#             state["processing_stage"] = "risk_scoring_failed"
#             return state
        
#         # Verify extracted_profile exists
#         extracted_profile = state.get('extracted_profile', {})
#         if not extracted_profile:
#             logger.warning("extracted_profile is empty, building from flattened fields")
#             # Build profile from flattened fields as fallback
#             extracted_profile = {
#                 "pet_species": state.get("pet_species", "unknown"),
#                 "weight_status": state.get("weight_status", "unknown"),
#                 "living_situation": state.get("living_situation", "unknown"),
#                 "exercise_level": state.get("exercise_level", "unknown"),
#                 "age_years": state.get("age_years", 0),
#                 "known_conditions": state.get("known_conditions", []),
#                 "allergies_known": state.get("allergies_known", [])
#             }
        
#         # Log profile summary for debugging
#         logger.info(f"Calculating health risk for: "
#                    f"Species={extracted_profile.get('pet_species', 'unknown')}, "
#                    f"Age={extracted_profile.get('age_years', 0)}, "
#                    f"Conditions={len(extracted_profile.get('known_conditions', []))}")
        
#         # Create ML agent
#         scorer = PetHealthRiskScorerAgent()
#         logger.debug("PetHealthRiskScorerAgent created successfully")
        
#         # Call agent to predict health risk
#         result = scorer.predict_health_risk(extracted_profile)
        
#         # Process prediction result
#         if result.get("status") == "success":
#             # Extract risk score
#             risk_score = result.get("health_risk_score", 0.5)
            
#             logger.info(f"✅ Health risk prediction successful: {risk_score:.3f}")
            
#             # Store in state
#             state["health_risk_score"] = risk_score
            
#             # Store feature contributions if available
#             if "feature_contributions" in result:
#                 state["health_risk_factors"] = result["feature_contributions"]
#                 logger.debug(f"Feature contributions: {result['feature_contributions']}")
            
#             # Store raw prediction for debugging
#             if "raw_prediction" in result:
#                 state["health_risk_raw"] = result["raw_prediction"]
            
#             # Update processing stage
#             state["processing_stage"] = "risk_scored"
            
#         else:
#             # Prediction failed
#             error_msg = result.get("message", "Unknown ML prediction error")
#             logger.error(f"❌ Health risk prediction failed: {error_msg}")
            
#             state["error_occurred"] = True
#             state["error_messages"].append(f"Health risk prediction failed: {error_msg}")
#             state["health_risk_score"] = 0.5  # Default medium risk
#             state["processing_stage"] = "risk_scoring_failed"
        
#         return state
        
#     except Exception as e:
#         # Handle unexpected errors
#         error_msg = f"Unexpected error in pet health risk scorer node: {str(e)}"
#         logger.error(error_msg, exc_info=True)
        
#         # Update state with error information
#         state["error_occurred"] = True
#         state["error_messages"].append(error_msg)
#         state["health_risk_score"] = 0.5  # Default medium risk on error
#         state["processing_stage"] = "risk_scoring_error"
        
#         return state


# # ==========================================
# # OPTIONAL: Helper function for testing
# # ==========================================

# def test_pet_health_risk_scorer_node():
#     """
#     Test function to verify node behavior.
#     """
#     from state import get_initial_state
    
#     print("=" * 60)
#     print("TESTING PET HEALTH RISK SCORER NODE")
#     print("=" * 60)
    
#     try:
#         # Test Case 1: Complete profile with extracted_profile
#         print("\n📝 Test Case 1: Complete Profile")
#         state1 = get_initial_state({
#             "about_pet": "Test",
#             "daily_routine": "Test",
#             "health_concerns": "Test"
#         })
        
#         # Manually set extracted profile and completion flag
#         state1["profile_extraction_complete"] = True
#         state1["extracted_profile"] = {
#             "pet_species": "dog",
#             "weight_status": "overweight",
#             "living_situation": "house",
#             "exercise_level": "sedentary",
#             "age_years": 12,
#             "known_conditions": ["arthritis", "heart disease"],
#             "allergies_known": ["pollen"]
#         }
        
#         result1 = pet_health_risk_scorer_node(state1)
#         print(f"Risk score: {result1.get('health_risk_score', 'N/A'):.3f}")
#         print(f"Risk factors: {result1.get('health_risk_factors', {})}")
#         print(f"Stage: {result1['processing_stage']}")
        
#         # Test Case 2: Using flattened fields
#         print("\n📝 Test Case 2: Flattened Fields")
#         state2 = get_initial_state({
#             "about_pet": "Test",
#             "daily_routine": "Test",
#             "health_concerns": "Test"
#         })
        
#         # Set completion flag but no extracted_profile (will use flattened)
#         state2["profile_extraction_complete"] = True
#         state2["pet_species"] = "cat"
#         state2["weight_status"] = "normal"
#         state2["living_situation"] = "apartment"
#         state2["exercise_level"] = "moderate"
#         state2["age_years"] = 5
#         state2["known_conditions"] = []
#         state2["allergies_known"] = []
        
#         result2 = pet_health_risk_scorer_node(state2)
#         print(f"Risk score: {result2.get('health_risk_score', 'N/A'):.3f}")
#         print(f"Stage: {result2['processing_stage']}")
        
#         # Test Case 3: Missing profile
#         print("\n📝 Test Case 3: Missing Profile")
#         state3 = get_initial_state({
#             "about_pet": "Test",
#             "daily_routine": "Test",
#             "health_concerns": "Test"
#         })
        
#         # No completion flag set
#         result3 = pet_health_risk_scorer_node(state3)
#         print(f"Risk score (default): {result3.get('health_risk_score', 'N/A')}")
#         print(f"Error occurred: {result3['error_occurred']}")
#         print(f"Error messages: {result3['error_messages']}")
        
#         return result1, result2, result3
        
#     except Exception as e:
#         print(f"❌ Test error: {str(e)}")
#         return None, None, None


# if __name__ == "__main__":
#     # Run test if executed directly
#     test_pet_health_risk_scorer_node()

# nodes/pet_health_risk_scorer_node.py
"""
Pet Health Risk Scorer Node for PawCare+ LangGraph workflow.
Executes ML-based health risk prediction as a workflow step.
"""

import logging
from typing import Dict, Any

from state import PetCareState
from agents.pet_health_risk_scorer_ml import PetHealthRiskScorerAgent

logger = logging.getLogger(__name__)


def pet_health_risk_scorer_node(state: PetCareState) -> PetCareState:
    """
    Execute ML-based health risk prediction as a workflow step.
    """
    logger.info("=" * 50)
    logger.info("EXECUTING PET HEALTH RISK SCORER NODE")
    logger.info("=" * 50)
    
    try:
        logger.debug(f"State request_id: {state.get('request_id', 'N/A')}")
        logger.debug(f"Current processing stage: {state.get('processing_stage', 'unknown')}")
        
        # Check if profile extraction was successful
        if not state.get('profile_extraction_complete', False):
            error_msg = "Cannot calculate health risk: profile extraction not complete"
            logger.error(error_msg)
            state["error_occurred"] = True
            state["error_messages"].append(error_msg)
            state["health_risk_score"] = 0.5
            state["processing_stage"] = "risk_scoring_failed"
            return state
        
        # Build profile from state
        extracted_profile = state.get('extracted_profile', {})
        if not extracted_profile:
            extracted_profile = {
                "pet_species": state.get("pet_species", "unknown"),
                "weight_status": state.get("weight_status", "unknown"),
                "living_situation": state.get("living_situation", "unknown"),
                "exercise_level": state.get("exercise_level", "unknown"),
                "age_years": state.get("age_years", 0),
                "known_conditions": state.get("known_conditions", []),
                "allergies_known": state.get("allergies_known", []),
                "recent_symptoms": state.get("recent_symptoms", []),
                "symptom_duration_days": state.get("symptom_duration_days", 0),
                "symptom_severity": state.get("symptom_severity", "unknown")
            }
        
        logger.info(f"Calculating health risk for: "
                   f"Species={extracted_profile.get('pet_species', 'unknown')}, "
                   f"Age={extracted_profile.get('age_years', 0)}, "
                   f"Conditions={len(extracted_profile.get('known_conditions', []))}, "
                   f"Symptoms={len(extracted_profile.get('recent_symptoms', []))}")
        
        # Create ML agent
        scorer = PetHealthRiskScorerAgent()
        logger.debug("PetHealthRiskScorerAgent created successfully")
        
        # Call agent to predict health risk
        result = scorer.predict_health_risk(extracted_profile)
        
        # Process prediction result
        if result.get("status") == "success":
            # Extract and convert to Python float
            risk_score = float(result.get("health_risk_score", 0.5))
            
            logger.info(f"✅ Health risk prediction successful: {risk_score:.3f}")
            
            state["health_risk_score"] = risk_score
            
            if "base_risk" in result:
                state["health_risk_base"] = float(result["base_risk"])
            
            if "symptom_boost" in result:
                state["health_risk_boost"] = float(result["symptom_boost"])
            
            state["processing_stage"] = "risk_scored"
            
        else:
            error_msg = result.get("message", "Unknown ML prediction error")
            logger.error(f"❌ Health risk prediction failed: {error_msg}")
            
            state["error_occurred"] = True
            state["error_messages"].append(f"Health risk prediction failed: {error_msg}")
            state["health_risk_score"] = 0.5
            state["processing_stage"] = "risk_scoring_failed"
        
        return state
        
    except Exception as e:
        error_msg = f"Unexpected error in pet health risk scorer node: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        state["error_occurred"] = True
        state["error_messages"].append(error_msg)
        state["health_risk_score"] = 0.5
        state["processing_stage"] = "risk_scoring_error"
        
        return state


def test_pet_health_risk_scorer_node():
    from state import get_initial_state
    
    print("=" * 60)
    print("TESTING PET HEALTH RISK SCORER NODE")
    print("=" * 60)
    
    try:
        print("\n📝 Test Case 1: Senior Dog with Symptoms")
        state1 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        state1["profile_extraction_complete"] = True
        state1["extracted_profile"] = {
            "pet_species": "dog",
            "weight_status": "normal",
            "living_situation": "house",
            "exercise_level": "light",
            "age_years": 12,
            "known_conditions": [],
            "allergies_known": [],
            "recent_symptoms": ["increased thirst", "weight loss", "coughing", "lethargy"],
            "symptom_duration_days": 14,
            "symptom_severity": "moderate"
        }
        
        result1 = pet_health_risk_scorer_node(state1)
        print(f"Risk score: {result1.get('health_risk_score', 'N/A')}")
        print(f"Stage: {result1['processing_stage']}")
        
        return result1
        
    except Exception as e:
        print(f"❌ Test error: {str(e)}")
        return None


if __name__ == "__main__":
    test_pet_health_risk_scorer_node()