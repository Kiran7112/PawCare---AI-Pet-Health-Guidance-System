# # nodes/behavioral_coaching_node.py
# """
# Behavioral Coaching Node for PawCare+ LangGraph workflow (Critical Path).
# Executes behavioral coaching planning using LLM as a workflow step.
# """

# import logging
# from typing import Dict, Any

# from state import PetCareState
# from agents.behavioral_coaching_llm import BehavioralCoachingAgent
# from utils.openai_client import OpenAIClient

# logger = logging.getLogger(__name__)


# def behavioral_coaching_node(state: PetCareState, client: OpenAIClient) -> PetCareState:
#     """
#     Execute behavioral coaching planning as a workflow step in the critical path.
    
#     This node uses an LLM agent to generate a comprehensive behavioral coaching plan
#     for high-risk pets, including behavior assessment, training strategies, anxiety
#     management, and trigger identification.
    
#     Args:
#         state: Current PetCareState containing:
#             - Extracted profile fields with behavioral issues
#             - Must be on CRITICAL_CARE_PATH
#         client: OpenAIClient instance for LLM calls
        
#     Returns:
#         Updated PetCareState with behavioral_coaching_output:
#         - behavioral_coaching_output: Dictionary with coaching plan fields
#         - critical_path_outputs: Updated with behavioral
#         - error_occurred: Set to True if coaching fails
#         - error_messages: Appended with any errors
#         - processing_stage: Updated to "behavioral_coached" or "behavioral_coaching_failed"
#     """
#     logger.info("=" * 50)
#     logger.info("EXECUTING BEHAVIORAL COACHING NODE (CRITICAL PATH)")
#     logger.info("=" * 50)
    
#     try:
#         # Log incoming state info
#         logger.debug(f"State request_id: {state.get('request_id', 'N/A')}")
#         logger.debug(f"Current path: {state.get('path_taken', 'unknown')}")
#         logger.debug(f"Current processing stage: {state.get('processing_stage', 'unknown')}")
        
#         # Verify we're on the critical path
#         if state.get('path_taken') != "CRITICAL_CARE_PATH":
#             logger.warning(f"Behavioral coaching node executed on {state.get('path_taken')} path - this may be unexpected")
        
#         # Check if profile extraction was successful
#         if not state.get('profile_extraction_complete', False):
#             error_msg = "Cannot generate behavioral coaching: profile extraction not complete"
#             logger.error(error_msg)
#             state["error_occurred"] = True
#             state["error_messages"].append(error_msg)
#             state["processing_stage"] = "behavioral_coaching_failed"
#             return state
        
#         # Extract profile from state
#         profile = _build_profile_from_state(state)
        
#         # Log behavioral coaching context
#         logger.info(f"Generating behavioral coaching for: "
#                    f"Species={profile.get('pet_species', 'unknown')}, "
#                    f"Name={profile.get('pet_name', 'Unknown')}, "
#                    f"Age={profile.get('age_years', 0)}")
        
#         behavioral_issues = profile.get('behavioral_issues', [])
#         if behavioral_issues:
#             logger.info(f"Behavioral issues: {', '.join(behavioral_issues[:3])}")
#             if len(behavioral_issues) > 3:
#                 logger.info(f"  ... and {len(behavioral_issues) - 3} more")
#         else:
#             logger.info("No specific behavioral issues reported - providing general enrichment guidance")
        
#         conditions = profile.get('known_conditions', [])
#         if conditions:
#             logger.info(f"Medical context: {', '.join(conditions[:3])} may influence behavior")
        
#         # Create LLM agent
#         coaching_agent = BehavioralCoachingAgent(client)
#         logger.debug("BehavioralCoachingAgent created successfully")
        
#         # Call agent to generate behavioral coaching plan
#         result = coaching_agent.generate_behavior_coaching(profile)
        
#         # Process coaching result
#         if result.get("status") == "success":
#             # Extract coaching dictionary
#             coaching_plan = result.get("behavioral_coaching", {})
            
#             logger.info(f"✅ Behavioral coaching plan generated successfully")
#             logger.debug(f"Plan contains fields: {list(coaching_plan.keys())}")
            
#             # Store in state
#             state["behavioral_coaching_output"] = coaching_plan
            
#             # Also store in critical_path_outputs for aggregation
#             if "critical_path_outputs" not in state:
#                 state["critical_path_outputs"] = {}
#             state["critical_path_outputs"]["behavioral"] = coaching_plan
            
#             # Update processing stage
#             state["processing_stage"] = "behavioral_coached"
            
#             # Log summary of plan
#             if "behavior_assessment" in coaching_plan:
#                 logger.info(f"Assessment: {coaching_plan['behavior_assessment'][:100]}...")
            
#             if "training_strategies" in coaching_plan:
#                 strategies = coaching_plan["training_strategies"]
#                 if isinstance(strategies, list):
#                     logger.info(f"Training strategies: {len(strategies)}")
#                     for i, strategy in enumerate(strategies[:2]):  # Log first 2
#                         logger.debug(f"  Strategy {i+1}: {strategy[:100]}...")
            
#             if "anxiety_management" in coaching_plan:
#                 logger.info(f"Anxiety management: {coaching_plan['anxiety_management'][:100]}...")
            
#             if "common_triggers" in coaching_plan:
#                 triggers = coaching_plan["common_triggers"]
#                 if isinstance(triggers, list):
#                     logger.info(f"Identified {len(triggers)} common triggers")
            
#             if "training_timeline" in coaching_plan:
#                 logger.info(f"Training timeline: {coaching_plan['training_timeline'][:100]}...")
            
#         else:
#             # Coaching failed
#             error_msg = result.get("message", "Unknown behavioral coaching error")
#             logger.error(f"❌ Behavioral coaching plan generation failed: {error_msg}")
            
#             state["error_occurred"] = True
#             state["error_messages"].append(f"Behavioral coaching failed: {error_msg}")
#             state["processing_stage"] = "behavioral_coaching_failed"
            
#             # Store fallback plan if available
#             if "behavioral_coaching" in result:
#                 state["behavioral_coaching_output"] = result["behavioral_coaching"]
#                 state["critical_path_outputs"] = state.get("critical_path_outputs", {})
#                 state["critical_path_outputs"]["behavioral"] = result["behavioral_coaching"]
#                 logger.info("Stored fallback behavioral coaching plan")
        
#         return state
        
#     except Exception as e:
#         # Handle unexpected errors
#         error_msg = f"Unexpected error in behavioral coaching node: {str(e)}"
#         logger.error(error_msg, exc_info=True)
        
#         # Update state with error information
#         state["error_occurred"] = True
#         state["error_messages"].append(error_msg)
#         state["processing_stage"] = "behavioral_coaching_error"
        
#         return state


# # ==========================================
# # HELPER FUNCTIONS
# # ==========================================

# def _build_profile_from_state(state: PetCareState) -> Dict[str, Any]:
#     """
#     Build a complete profile dictionary from state fields for behavioral coaching.
    
#     Args:
#         state: Current PetCareState
        
#     Returns:
#         Dictionary with all profile fields needed for behavioral coaching
#     """
#     # Try to use extracted_profile first if available
#     profile = state.get('extracted_profile', {}).copy()
    
#     # If extracted_profile is empty, build from flattened fields
#     if not profile:
#         logger.debug("Building profile from flattened state fields")
#         profile = {
#             # Basic Information
#             "pet_name": state.get("pet_name", "Your Pet"),
#             "pet_species": state.get("pet_species", "unknown"),
#             "breed": state.get("breed", "unknown"),
#             "age_years": state.get("age_years", 0),
            
#             # Behavioral Information
#             "behavioral_issues": state.get("behavioral_issues", []),
            
#             # Medical Context (can affect behavior)
#             "known_conditions": state.get("known_conditions", []),
            
#             # Lifestyle Context
#             "living_situation": state.get("living_situation", "unknown"),
#             "exercise_level": state.get("exercise_level", "unknown"),
#             "daily_routine": state.get("daily_routine", "Not specified"),
            
#             # Owner Context
#             "owner_experience": state.get("owner_experience", "unknown"),
#         }
#     else:
#         # Ensure profile has all needed fields
#         if "pet_name" not in profile:
#             profile["pet_name"] = state.get("pet_name", "Your Pet")
#         if "behavioral_issues" not in profile:
#             profile["behavioral_issues"] = state.get("behavioral_issues", [])
#         if "living_situation" not in profile:
#             profile["living_situation"] = state.get("living_situation", "unknown")
#         if "exercise_level" not in profile:
#             profile["exercise_level"] = state.get("exercise_level", "unknown")
#         if "daily_routine" not in profile:
#             profile["daily_routine"] = state.get("daily_routine", "Not specified")
#         if "owner_experience" not in profile:
#             profile["owner_experience"] = state.get("owner_experience", "unknown")
    
#     # Ensure all required fields exist with defaults
#     required_fields = {
#         "pet_name": "Your Pet",
#         "pet_species": "unknown",
#         "breed": "unknown",
#         "age_years": 0,
#         "behavioral_issues": [],
#         "known_conditions": [],
#         "living_situation": "unknown",
#         "exercise_level": "unknown",
#         "daily_routine": "Not specified",
#         "owner_experience": "unknown"
#     }
    
#     for field, default in required_fields.items():
#         if field not in profile or profile[field] is None:
#             profile[field] = default
    
#     return profile


# # ==========================================
# # OPTIONAL: Helper function for testing
# # ==========================================

# def test_behavioral_coaching_node():
#     """
#     Test function to verify node behavior.
#     """
#     from state import get_initial_state
#     from utils.openai_client import build_openai_client
    
#     print("=" * 60)
#     print("TESTING BEHAVIORAL COACHING NODE")
#     print("=" * 60)
    
#     try:
#         # Initialize client
#         client = build_openai_client()
#         print("✅ OpenAI client created")
        
#         # Test Case 1: Dog with behavioral issues
#         print("\n📝 Test Case 1: Dog with Behavioral Issues")
#         state1 = get_initial_state({
#             "about_pet": "Test",
#             "daily_routine": "Test",
#             "health_concerns": "Test"
#         })
        
#         # Set up state
#         state1["path_taken"] = "CRITICAL_CARE_PATH"
#         state1["profile_extraction_complete"] = True
        
#         # Set profile fields
#         state1["pet_name"] = "Rocky"
#         state1["pet_species"] = "dog"
#         state1["breed"] = "german shepherd"
#         state1["age_years"] = 4
#         state1["living_situation"] = "house"
#         state1["exercise_level"] = "moderate"
#         state1["owner_experience"] = "experienced"
#         state1["behavioral_issues"] = [
#             "fear aggression towards strangers",
#             "separation anxiety",
#             "resource guarding of food bowl"
#         ]
#         state1["known_conditions"] = ["hip dysplasia", "anxiety"]
#         state1["daily_routine"] = "Walks morning and evening, fed twice daily, left alone during work hours"
        
#         result1 = behavioral_coaching_node(state1, client)
#         print(f"Coaching plan generated: {'behavioral_coaching_output' in result1}")
#         print(f"Stage: {result1.get('processing_stage')}")
        
#         if 'behavioral_coaching_output' in result1:
#             plan = result1['behavioral_coaching_output']
#             print(f"Plan fields: {list(plan.keys())}")
#             if 'behavior_assessment' in plan:
#                 print(f"Assessment: {plan['behavior_assessment'][:100]}...")
#             if 'training_strategies' in plan:
#                 strategies = plan['training_strategies']
#                 print(f"Training strategies: {len(strategies) if isinstance(strategies, list) else 'Provided'}")
        
#         # Test Case 2: Cat with anxiety
#         print("\n📝 Test Case 2: Cat with Anxiety")
#         state2 = get_initial_state({
#             "about_pet": "Test",
#             "daily_routine": "Test",
#             "health_concerns": "Test"
#         })
        
#         state2["path_taken"] = "CRITICAL_CARE_PATH"
#         state2["profile_extraction_complete"] = True
#         state2["pet_name"] = "Luna"
#         state2["pet_species"] = "cat"
#         state2["breed"] = "siamese"
#         state2["age_years"] = 6
#         state2["living_situation"] = "apartment"
#         state2["behavioral_issues"] = ["hiding from visitors", "excessive grooming", "vocalization at night"]
#         state2["known_conditions"] = ["hyperthyroidism"]
        
#         result2 = behavioral_coaching_node(state2, client)
#         print(f"Coaching plan generated: {'behavioral_coaching_output' in result2}")
        
#         # Test Case 3: No behavioral issues (should get general enrichment)
#         print("\n📝 Test Case 3: No Behavioral Issues")
#         state3 = get_initial_state({
#             "about_pet": "Test",
#             "daily_routine": "Test",
#             "health_concerns": "Test"
#         })
        
#         state3["path_taken"] = "CRITICAL_CARE_PATH"
#         state3["profile_extraction_complete"] = True
#         state3["pet_name"] = "Charlie"
#         state3["pet_species"] = "dog"
#         state3["behavioral_issues"] = []  # No issues
#         state3["known_conditions"] = ["diabetes"]  # Medical but no behavioral
        
#         result3 = behavioral_coaching_node(state3, client)
#         print(f"Coaching plan generated: {'behavioral_coaching_output' in result3}")
        
#         # Test Case 4: Missing profile extraction
#         print("\n📝 Test Case 4: Missing Profile Extraction")
#         state4 = get_initial_state({
#             "about_pet": "Test",
#             "daily_routine": "Test",
#             "health_concerns": "Test"
#         })
        
#         state4["path_taken"] = "CRITICAL_CARE_PATH"
#         state4["profile_extraction_complete"] = False  # Not complete
        
#         result4 = behavioral_coaching_node(state4, client)
#         print(f"Error occurred: {result4.get('error_occurred')}")
#         print(f"Error messages: {result4.get('error_messages')}")
        
#         # Test Case 5: Error handling
#         print("\n📝 Test Case 5: Error Handling")
#         state5 = get_initial_state({
#             "about_pet": "Test",
#             "daily_routine": "Test",
#             "health_concerns": "Test"
#         })
        
#         # Cause an error by not having required data
#         result5 = behavioral_coaching_node(state5, client)
#         print(f"Error occurred: {result5.get('error_occurred')}")
#         print(f"Stage: {result5.get('processing_stage')}")
        
#         return result1, result2, result3, result4, result5
        
#     except Exception as e:
#         print(f"❌ Test error: {str(e)}")
#         return None, None, None, None, None


# if __name__ == "__main__":
#     # Run test if executed directly
#     test_behavioral_coaching_node()



# nodes/behavioral_coaching_node.py
"""
Behavioral Coaching Node for PawCare+ LangGraph workflow (Critical Path).
Executes behavioral coaching planning using LLM as a workflow step.
"""

import logging
from typing import Dict, Any

from state import PetCareState
from agents.behavioral_coaching_llm import BehavioralCoachingAgent
from utils.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


def behavioral_coaching_node(state: PetCareState, client: OpenAIClient) -> PetCareState:
    """
    Execute behavioral coaching planning as a workflow step in the critical path.
    """
    logger.info("=" * 50)
    logger.info("EXECUTING BEHAVIORAL COACHING NODE (CRITICAL PATH)")
    logger.info("=" * 50)
    
    try:
        # Log incoming state info
        logger.debug(f"State request_id: {state.get('request_id', 'N/A')}")
        logger.debug(f"Current path: {state.get('path_taken', 'unknown')}")
        logger.debug(f"Current processing stage: {state.get('processing_stage', 'unknown')}")
        
        # Verify we're on the critical path
        if state.get('path_taken') != "CRITICAL_CARE_PATH":
            logger.warning(f"Behavioral coaching node executed on {state.get('path_taken')} path - this may be unexpected")
        
        # Check if profile extraction was successful
        if not state.get('profile_extraction_complete', False):
            error_msg = "Cannot generate behavioral coaching: profile extraction not complete"
            logger.error(error_msg)
            state["error_occurred"] = True
            state["error_messages"].append(error_msg)
            state["processing_stage"] = "behavioral_coaching_failed"
            return state
        
        # Extract profile from state
        profile = _build_profile_from_state(state)
        
        # Log behavioral coaching context
        logger.info(f"Generating behavioral coaching for: "
                   f"Species={profile.get('pet_species', 'unknown')}, "
                   f"Name={profile.get('pet_name', 'Unknown')}, "
                   f"Age={profile.get('age_years', 0)}")
        
        behavioral_issues = profile.get('behavioral_issues', [])
        if behavioral_issues:
            logger.info(f"Behavioral issues: {', '.join(behavioral_issues[:3])}")
            if len(behavioral_issues) > 3:
                logger.info(f"  ... and {len(behavioral_issues) - 3} more")
        else:
            logger.info("No specific behavioral issues reported - providing general enrichment guidance")
        
        conditions = profile.get('known_conditions', [])
        if conditions:
            logger.info(f"Medical context: {', '.join(conditions[:3])} may influence behavior")
        
        # Create LLM agent
        coaching_agent = BehavioralCoachingAgent(client)
        logger.debug("BehavioralCoachingAgent created successfully")
        
        # Call agent to generate behavioral coaching plan
        result = coaching_agent.generate_behavior_coaching(profile)
        
        # Process coaching result
        if result.get("status") == "success":
            # Extract coaching dictionary
            coaching_plan = result.get("behavioral_coaching", {})
            
            logger.info(f"✅ Behavioral coaching plan generated successfully")
            logger.debug(f"Plan contains fields: {list(coaching_plan.keys())}")
            
            # Store in state
            state["behavioral_coaching_output"] = coaching_plan
            
            # Also store in critical_path_outputs for aggregation
            if "critical_path_outputs" not in state:
                state["critical_path_outputs"] = {}
            state["critical_path_outputs"]["behavioral"] = coaching_plan
            
            # Update processing stage
            state["processing_stage"] = "behavioral_coached"
            
            # Log summary of plan - FIXED: Handle different data types
            if "behavior_assessment" in coaching_plan:
                assessment = coaching_plan["behavior_assessment"]
                if isinstance(assessment, str):
                    logger.info(f"Assessment: {assessment[:100]}...")
                elif isinstance(assessment, dict):
                    logger.info(f"Assessment: {str(assessment)[:100]}...")
                else:
                    logger.info(f"Assessment: {assessment}")
            
            if "training_strategies" in coaching_plan:
                strategies = coaching_plan["training_strategies"]
                if isinstance(strategies, list):
                    logger.info(f"Training strategies: {len(strategies)}")
                    for i, strategy in enumerate(strategies[:2]):  # Log first 2
                        if isinstance(strategy, str):
                            logger.debug(f"  Strategy {i+1}: {strategy[:100]}...")
                        else:
                            logger.debug(f"  Strategy {i+1}: {strategy}")
            
            if "anxiety_management" in coaching_plan:
                anxiety = coaching_plan["anxiety_management"]
                if isinstance(anxiety, str):
                    logger.info(f"Anxiety management: {anxiety[:100]}...")
                elif isinstance(anxiety, dict):
                    logger.info(f"Anxiety management: {str(anxiety)[:100]}...")
                else:
                    logger.info(f"Anxiety management: {anxiety}")
            
            if "common_triggers" in coaching_plan:
                triggers = coaching_plan["common_triggers"]
                if isinstance(triggers, list):
                    logger.info(f"Identified {len(triggers)} common triggers")
                else:
                    logger.info(f"Common triggers: Provided")
            
            if "training_timeline" in coaching_plan:
                timeline = coaching_plan["training_timeline"]
                if isinstance(timeline, str):
                    logger.info(f"Training timeline: {timeline[:100]}...")
                elif isinstance(timeline, dict):
                    logger.info(f"Training timeline: {str(timeline)[:100]}...")
                else:
                    logger.info(f"Training timeline: {timeline}")
            
        else:
            # Coaching failed
            error_msg = result.get("message", "Unknown behavioral coaching error")
            logger.error(f"❌ Behavioral coaching plan generation failed: {error_msg}")
            
            state["error_occurred"] = True
            state["error_messages"].append(f"Behavioral coaching failed: {error_msg}")
            state["processing_stage"] = "behavioral_coaching_failed"
            
            # Store fallback plan if available
            if "behavioral_coaching" in result:
                state["behavioral_coaching_output"] = result["behavioral_coaching"]
                state["critical_path_outputs"] = state.get("critical_path_outputs", {})
                state["critical_path_outputs"]["behavioral"] = result["behavioral_coaching"]
                logger.info("Stored fallback behavioral coaching plan")
        
        return state
        
    except Exception as e:
        # Handle unexpected errors
        error_msg = f"Unexpected error in behavioral coaching node: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Update state with error information
        state["error_occurred"] = True
        state["error_messages"].append(error_msg)
        state["processing_stage"] = "behavioral_coaching_error"
        
        return state


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def _build_profile_from_state(state: PetCareState) -> Dict[str, Any]:
    """
    Build a complete profile dictionary from state fields for behavioral coaching.
    """
    # Try to use extracted_profile first if available
    profile = state.get('extracted_profile', {}).copy()
    
    # If extracted_profile is empty, build from flattened fields
    if not profile:
        logger.debug("Building profile from flattened state fields")
        profile = {
            "pet_name": state.get("pet_name", "Your Pet"),
            "pet_species": state.get("pet_species", "unknown"),
            "breed": state.get("breed", "unknown"),
            "age_years": state.get("age_years", 0),
            "behavioral_issues": state.get("behavioral_issues", []),
            "known_conditions": state.get("known_conditions", []),
            "living_situation": state.get("living_situation", "unknown"),
            "exercise_level": state.get("exercise_level", "unknown"),
            "daily_routine": state.get("daily_routine", "Not specified"),
            "owner_experience": state.get("owner_experience", "unknown"),
        }
    else:
        if "pet_name" not in profile:
            profile["pet_name"] = state.get("pet_name", "Your Pet")
        if "behavioral_issues" not in profile:
            profile["behavioral_issues"] = state.get("behavioral_issues", [])
        if "living_situation" not in profile:
            profile["living_situation"] = state.get("living_situation", "unknown")
        if "exercise_level" not in profile:
            profile["exercise_level"] = state.get("exercise_level", "unknown")
        if "daily_routine" not in profile:
            profile["daily_routine"] = state.get("daily_routine", "Not specified")
        if "owner_experience" not in profile:
            profile["owner_experience"] = state.get("owner_experience", "unknown")
    
    required_fields = {
        "pet_name": "Your Pet",
        "pet_species": "unknown",
        "breed": "unknown",
        "age_years": 0,
        "behavioral_issues": [],
        "known_conditions": [],
        "living_situation": "unknown",
        "exercise_level": "unknown",
        "daily_routine": "Not specified",
        "owner_experience": "unknown"
    }
    
    for field, default in required_fields.items():
        if field not in profile or profile[field] is None:
            profile[field] = default
    
    return profile


# ==========================================
# OPTIONAL: Helper function for testing
# ==========================================

def test_behavioral_coaching_node():
    """
    Test function to verify node behavior.
    """
    from state import get_initial_state
    from utils.openai_client import build_openai_client
    
    print("=" * 60)
    print("TESTING BEHAVIORAL COACHING NODE")
    print("=" * 60)
    
    try:
        client = build_openai_client()
        print("✅ OpenAI client created")
        
        print("\n📝 Test Case 1: Dog with Behavioral Issues")
        state1 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        state1["path_taken"] = "CRITICAL_CARE_PATH"
        state1["profile_extraction_complete"] = True
        state1["pet_name"] = "Rocky"
        state1["pet_species"] = "dog"
        state1["breed"] = "german shepherd"
        state1["age_years"] = 4
        state1["living_situation"] = "house"
        state1["exercise_level"] = "moderate"
        state1["owner_experience"] = "experienced"
        state1["behavioral_issues"] = [
            "fear aggression towards strangers",
            "separation anxiety",
            "resource guarding of food bowl"
        ]
        state1["known_conditions"] = ["hip dysplasia", "anxiety"]
        state1["daily_routine"] = "Walks morning and evening, fed twice daily, left alone during work hours"
        
        result1 = behavioral_coaching_node(state1, client)
        print(f"Coaching plan generated: {'behavioral_coaching_output' in result1}")
        print(f"Stage: {result1.get('processing_stage')}")
        
        return result1
        
    except Exception as e:
        print(f"❌ Test error: {str(e)}")
        return None


if __name__ == "__main__":
    test_behavioral_coaching_node()