# # nodes/owner_care_capability_node.py
# """
# Owner Care Capability Node for PawCare+ LangGraph workflow.
# Executes ML-based owner care capability prediction as a workflow step.
# """

# import logging
# from typing import Dict, Any

# from state import PetCareState
# from agents.owner_care_capability_ml import OwnerCareCapabilityAgent

# logger = logging.getLogger(__name__)


# def owner_care_capability_node(state: PetCareState) -> PetCareState:
#     """
#     Execute ML-based owner care capability prediction as a workflow step.
    
#     This node uses a trained ML model to predict an owner's capability score (0-100)
#     based on three extracted profile features:
#     - owner_experience: Owner's experience level
#     - vet_access: Availability of veterinary care
#     - owner_commitment: Owner's dedication level
    
#     Args:
#         state: Current PetCareState containing extracted profile fields
        
#     Returns:
#         Updated PetCareState with care_capability_score:
#         - care_capability_score: Float between 0.0 and 100.0
#         - care_capability_factors: Optional feature importance dictionary
#         - care_capability_confidence: Confidence score (0-1)
#         - error_occurred: Set to True if prediction fails
#         - error_messages: Appended with any errors
#         - processing_stage: Updated appropriately
#     """
#     logger.info("=" * 50)
#     logger.info("EXECUTING OWNER CARE CAPABILITY NODE")
#     logger.info("=" * 50)
    
#     try:
#         # Log incoming state info
#         logger.debug(f"State request_id: {state.get('request_id', 'N/A')}")
#         logger.debug(f"Current processing stage: {state.get('processing_stage', 'unknown')}")
        
#         # Check if profile extraction was successful
#         if not state.get('profile_extraction_complete', False):
#             error_msg = "Cannot calculate care capability: profile extraction not complete"
#             logger.error(error_msg)
#             state["error_occurred"] = True
#             state["error_messages"].append(error_msg)
#             state["care_capability_score"] = 50.0  # Default medium capability
#             state["care_capability_confidence"] = 0.0
#             state["processing_stage"] = "capability_scoring_failed"
#             return state
        
#         # Extract the three required fields from state
#         # Try to get from extracted_profile first, then fall back to flattened fields
#         extracted_profile = state.get('extracted_profile', {})
        
#         owner_experience = (
#             extracted_profile.get('owner_experience') or 
#             state.get('owner_experience', 'unknown')
#         )
        
#         vet_access = (
#             extracted_profile.get('vet_access') or 
#             state.get('vet_access', 'unknown')
#         )
        
#         owner_commitment = (
#             extracted_profile.get('owner_commitment') or 
#             state.get('owner_commitment', 'unknown')
#         )
        
#         # Log extracted values for debugging
#         logger.info(f"Calculating care capability for: "
#                    f"Experience={owner_experience}, "
#                    f"Vet Access={vet_access}, "
#                    f"Commitment={owner_commitment}")
        
#         # Build profile dictionary for the agent
#         profile = {
#             "owner_experience": owner_experience,
#             "vet_access": vet_access,
#             "owner_commitment": owner_commitment
#         }
        
#         # Validate that we have at least some data
#         if all(v == 'unknown' for v in profile.values()):
#             logger.warning("All owner capability features are 'unknown', prediction will have low confidence")
        
#         # Create ML agent
#         capability_agent = OwnerCareCapabilityAgent()
#         logger.debug("OwnerCareCapabilityAgent created successfully")
        
#         # Call agent to predict capability
#         result = capability_agent.predict_capability(profile)
        
#         # Process prediction result
#         if result.get("status") == "success":
#             # Extract capability score
#             capability_score = result.get("care_capability_score", 50.0)
#             confidence = result.get("confidence", 0.5)
            
#             logger.info(f"✅ Care capability prediction successful: {capability_score:.1f}/100 "
#                        f"(confidence: {confidence:.2f})")
            
#             # Store in state
#             state["care_capability_score"] = capability_score
#             state["care_capability_confidence"] = confidence
            
#             # Store feature contributions if available
#             if "feature_contributions" in result:
#                 state["care_capability_factors"] = result["feature_contributions"]
#                 logger.debug(f"Feature contributions: {result['feature_contributions']}")
            
#             # Store raw prediction and features used
#             if "raw_prediction" in result:
#                 state["care_capability_raw"] = result["raw_prediction"]
            
#             if "features_used" in result:
#                 state["care_capability_features"] = result["features_used"]
            
#             # Update processing stage (but don't override main path)
#             # This node runs in parallel, so we use a separate stage field
#             state["capability_scoring_stage"] = "capability_scored"
            
#         else:
#             # Prediction failed
#             error_msg = result.get("message", "Unknown ML prediction error")
#             logger.error(f"❌ Care capability prediction failed: {error_msg}")
            
#             state["error_occurred"] = True
#             state["error_messages"].append(f"Care capability prediction failed: {error_msg}")
#             state["care_capability_score"] = 50.0  # Default medium capability
#             state["care_capability_confidence"] = 0.0
#             state["capability_scoring_stage"] = "capability_scoring_failed"
        
#         return state
        
#     except Exception as e:
#         # Handle unexpected errors
#         error_msg = f"Unexpected error in owner care capability node: {str(e)}"
#         logger.error(error_msg, exc_info=True)
        
#         # Update state with error information
#         state["error_occurred"] = True
#         state["error_messages"].append(error_msg)
#         state["care_capability_score"] = 50.0  # Default medium capability on error
#         state["care_capability_confidence"] = 0.0
#         state["capability_scoring_stage"] = "capability_scoring_error"
        
#         return state


# # ==========================================
# # OPTIONAL: Helper function for testing
# # ==========================================

# def test_owner_care_capability_node():
#     """
#     Test function to verify node behavior.
#     """
#     from state import get_initial_state
    
#     print("=" * 60)
#     print("TESTING OWNER CARE CAPABILITY NODE")
#     print("=" * 60)
    
#     try:
#         # Test Case 1: Complete profile with good values
#         print("\n📝 Test Case 1: High Capability Profile")
#         state1 = get_initial_state({
#             "about_pet": "Test",
#             "daily_routine": "Test",
#             "health_concerns": "Test"
#         })
        
#         state1["profile_extraction_complete"] = True
#         state1["owner_experience"] = "expert"
#         state1["vet_access"] = "regular"
#         state1["owner_commitment"] = "obsessive"
        
#         result1 = owner_care_capability_node(state1)
#         print(f"Capability score: {result1.get('care_capability_score', 'N/A'):.1f}/100")
#         print(f"Confidence: {result1.get('care_capability_confidence', 0):.2f}")
#         print(f"Factors: {result1.get('care_capability_factors', {})}")
        
#         # Test Case 2: Medium capability profile
#         print("\n📝 Test Case 2: Medium Capability Profile")
#         state2 = get_initial_state({
#             "about_pet": "Test",
#             "daily_routine": "Test",
#             "health_concerns": "Test"
#         })
        
#         state2["profile_extraction_complete"] = True
#         state2["owner_experience"] = "experienced"
#         state2["vet_access"] = "emergency only"
#         state2["owner_commitment"] = "dedicated"
        
#         result2 = owner_care_capability_node(state2)
#         print(f"Capability score: {result2.get('care_capability_score', 'N/A'):.1f}/100")
#         print(f"Confidence: {result2.get('care_capability_confidence', 0):.2f}")
        
#         # Test Case 3: Low capability profile
#         print("\n📝 Test Case 3: Low Capability Profile")
#         state3 = get_initial_state({
#             "about_pet": "Test",
#             "daily_routine": "Test",
#             "health_concerns": "Test"
#         })
        
#         state3["profile_extraction_complete"] = True
#         state3["owner_experience"] = "novice"
#         state3["vet_access"] = "limited"
#         state3["owner_commitment"] = "casual"
        
#         result3 = owner_care_capability_node(state3)
#         print(f"Capability score: {result3.get('care_capability_score', 'N/A'):.1f}/100")
        
#         # Test Case 4: Unknown values (low confidence)
#         print("\n📝 Test Case 4: Unknown Values")
#         state4 = get_initial_state({
#             "about_pet": "Test",
#             "daily_routine": "Test",
#             "health_concerns": "Test"
#         })
        
#         state4["profile_extraction_complete"] = True
#         state4["owner_experience"] = "unknown"
#         state4["vet_access"] = "unknown"
#         state4["owner_commitment"] = "unknown"
        
#         result4 = owner_care_capability_node(state4)
#         print(f"Capability score: {result4.get('care_capability_score', 'N/A'):.1f}/100")
#         print(f"Confidence: {result4.get('care_capability_confidence', 0):.2f}")
        
#         # Test Case 5: Missing extraction
#         print("\n📝 Test Case 5: Missing Extraction")
#         state5 = get_initial_state({
#             "about_pet": "Test",
#             "daily_routine": "Test",
#             "health_concerns": "Test"
#         })
        
#         # No profile_extraction_complete flag
#         result5 = owner_care_capability_node(state5)
#         print(f"Capability score (default): {result5.get('care_capability_score', 'N/A')}")
#         print(f"Error occurred: {result5['error_occurred']}")
        
#         return result1, result2, result3, result4, result5
        
#     except Exception as e:
#         print(f"❌ Test error: {str(e)}")
#         return None, None, None, None, None


# if __name__ == "__main__":
#     # Run test if executed directly
#     test_owner_care_capability_node()

# nodes/owner_care_capability_node.py
"""
Owner Care Capability Node for PawCare+ LangGraph workflow.
Executes ML-based owner care capability prediction as a workflow step.
"""

import logging
from typing import Dict, Any

from state import PetCareState
from agents.owner_care_capability_ml import OwnerCareCapabilityAgent

logger = logging.getLogger(__name__)


def owner_care_capability_node(state: PetCareState) -> PetCareState:
    """
    Execute ML-based owner care capability prediction as a workflow step.
    """
    logger.info("=" * 50)
    logger.info("EXECUTING OWNER CARE CAPABILITY NODE")
    logger.info("=" * 50)
    
    try:
        logger.debug(f"State request_id: {state.get('request_id', 'N/A')}")
        logger.debug(f"Current processing stage: {state.get('processing_stage', 'unknown')}")
        
        # Check if profile extraction was successful
        if not state.get('profile_extraction_complete', False):
            error_msg = "Cannot calculate care capability: profile extraction not complete"
            logger.error(error_msg)
            state["error_occurred"] = True
            state["error_messages"].append(error_msg)
            state["care_capability_score"] = 50.0
            state["care_capability_confidence"] = 0.0
            state["processing_stage"] = "capability_scoring_failed"
            return state
        
        # Extract the three required fields from state
        extracted_profile = state.get('extracted_profile', {})
        
        owner_experience = (
            extracted_profile.get('owner_experience') or 
            state.get('owner_experience', 'unknown')
        )
        
        vet_access = (
            extracted_profile.get('vet_access') or 
            state.get('vet_access', 'unknown')
        )
        
        owner_commitment = (
            extracted_profile.get('owner_commitment') or 
            state.get('owner_commitment', 'unknown')
        )
        
        logger.info(f"Calculating care capability for: "
                   f"Experience={owner_experience}, "
                   f"Vet Access={vet_access}, "
                   f"Commitment={owner_commitment}")
        
        profile = {
            "owner_experience": owner_experience,
            "vet_access": vet_access,
            "owner_commitment": owner_commitment
        }
        
        if all(v == 'unknown' for v in profile.values()):
            logger.warning("All owner capability features are 'unknown', prediction will have low confidence")
        
        # Create ML agent
        capability_agent = OwnerCareCapabilityAgent()
        logger.debug("OwnerCareCapabilityAgent created successfully")
        
        # Call agent to predict capability
        result = capability_agent.predict_capability(profile)
        
        # Process prediction result
        if result.get("status") == "success":
            # Extract and convert to Python float
            capability_score = float(result.get("care_capability_score", 50.0))
            confidence = float(result.get("confidence", 0.5))
            
            logger.info(f"✅ Care capability prediction successful: {capability_score:.1f}/100 "
                       f"(confidence: {confidence:.2f})")
            
            # Store in state as Python floats
            state["care_capability_score"] = capability_score
            state["care_capability_confidence"] = confidence
            
            # Store feature contributions (convert to Python floats)
            if "feature_contributions" in result and result["feature_contributions"]:
                contributions = {}
                for k, v in result["feature_contributions"].items():
                    contributions[k] = float(v)
                state["care_capability_factors"] = contributions
                logger.debug(f"Feature contributions: {contributions}")
            
            if "raw_prediction" in result:
                state["care_capability_raw"] = float(result["raw_prediction"])
            
            if "features_used" in result:
                state["care_capability_features"] = result["features_used"]
            
            state["capability_scoring_stage"] = "capability_scored"
            
        else:
            error_msg = result.get("message", "Unknown ML prediction error")
            logger.error(f"❌ Care capability prediction failed: {error_msg}")
            
            state["error_occurred"] = True
            state["error_messages"].append(f"Care capability prediction failed: {error_msg}")
            state["care_capability_score"] = 50.0
            state["care_capability_confidence"] = 0.0
            state["capability_scoring_stage"] = "capability_scoring_failed"
        
        return state
        
    except Exception as e:
        error_msg = f"Unexpected error in owner care capability node: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        state["error_occurred"] = True
        state["error_messages"].append(error_msg)
        state["care_capability_score"] = 50.0
        state["care_capability_confidence"] = 0.0
        state["capability_scoring_stage"] = "capability_scoring_error"
        
        return state


def test_owner_care_capability_node():
    from state import get_initial_state
    
    print("=" * 60)
    print("TESTING OWNER CARE CAPABILITY NODE")
    print("=" * 60)
    
    try:
        print("\n📝 Test Case 1: High Capability Profile")
        state1 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        state1["profile_extraction_complete"] = True
        state1["owner_experience"] = "expert"
        state1["vet_access"] = "regular"
        state1["owner_commitment"] = "obsessive"
        
        result1 = owner_care_capability_node(state1)
        print(f"Capability score: {result1.get('care_capability_score', 'N/A')}")
        print(f"Confidence: {result1.get('care_capability_confidence', 0):.2f}")
        
        return result1
        
    except Exception as e:
        print(f"❌ Test error: {str(e)}")
        return None


if __name__ == "__main__":
    test_owner_care_capability_node()