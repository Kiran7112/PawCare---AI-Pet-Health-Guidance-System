# nodes/wellness_tracking_preventive_node.py
"""
Wellness Tracking Preventive Node for PawCare+ LangGraph workflow (Preventive Path).
Executes preventive wellness tracking planning using LLM as a workflow step.
"""

import logging
from typing import Dict, Any

from state import PetCareState
from agents.wellness_tracking_preventive_llm import WellnessTrackingPreventiveAgent
from utils.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


def wellness_tracking_preventive_node(state: PetCareState, client: OpenAIClient) -> PetCareState:
    """
    Execute preventive wellness tracking planning as a workflow step in the preventive path.
    
    This node uses an LLM agent to generate a focused wellness tracking plan
    for moderate-risk pets, including tracking overview, monthly checklist,
    wellness goals, and early warning signs.
    
    Args:
        state: Current PetCareState containing:
            - Extracted profile fields
            - ML prediction results (health_risk_score)
            - Must be on PREVENTIVE_CARE_PATH
        client: OpenAIClient instance for LLM calls
        
    Returns:
        Updated PetCareState with wellness_tracking_output:
        - wellness_tracking_output: Dictionary with tracking plan fields
        - preventive_path_outputs: Updated with tracking
        - error_occurred: Set to True if planning fails
        - error_messages: Appended with any errors
        - processing_stage: Updated to "tracking_planned" or "tracking_planning_failed"
    """
    logger.info("=" * 50)
    logger.info("EXECUTING WELLNESS TRACKING PREVENTIVE NODE (PREVENTIVE PATH)")
    logger.info("=" * 50)
    
    try:
        # Log incoming state info
        logger.debug(f"State request_id: {state.get('request_id', 'N/A')}")
        logger.debug(f"Current path: {state.get('path_taken', 'unknown')}")
        logger.debug(f"Current processing stage: {state.get('processing_stage', 'unknown')}")
        
        # Verify we're on the preventive path
        if state.get('path_taken') != "PREVENTIVE_CARE_PATH":
            logger.warning(f"Wellness tracking preventive node executed on {state.get('path_taken')} path - this may be unexpected")
        
        # Check if profile extraction was successful
        if not state.get('profile_extraction_complete', False):
            error_msg = "Cannot generate tracking plan: profile extraction not complete"
            logger.error(error_msg)
            state["error_occurred"] = True
            state["error_messages"].append(error_msg)
            state["processing_stage"] = "tracking_planning_failed"
            return state
        
        # Extract profile from state
        profile = _build_profile_from_state(state)
        
        # Extract ML results from state
        ml_results = _build_ml_results_from_state(state)
        
        # Log tracking planning context
        logger.info(f"Generating wellness tracking plan for: "
                   f"Species={profile.get('pet_species', 'unknown')}, "
                   f"Name={profile.get('pet_name', 'Unknown')}, "
                   f"Age={profile.get('age_years', 0)}")
        
        risk_score = ml_results.get('health_risk_score', 0.45)
        logger.info(f"Health risk score: {risk_score:.3f} (Moderate risk range)")
        
        weight_status = profile.get('weight_status', 'unknown')
        logger.info(f"Weight status: {weight_status}")
        
        exercise_level = profile.get('exercise_level', 'unknown')
        logger.info(f"Exercise level: {exercise_level}")
        
        conditions = profile.get('known_conditions', [])
        if conditions:
            logger.info(f"Conditions to track: {', '.join(conditions[:3])}")
            if len(conditions) > 3:
                logger.info(f"  ... and {len(conditions) - 3} more")
        
        # Create LLM agent
        tracking_agent = WellnessTrackingPreventiveAgent(client)
        logger.debug("WellnessTrackingPreventiveAgent created successfully")
        
        # Call agent to generate tracking plan
        result = tracking_agent.generate_tracking_plan(profile, ml_results)
        
        # Process planning result
        if result.get("status") == "success":
            # Extract tracking plan dictionary
            tracking_plan = result.get("wellness_tracking", {})
            
            logger.info(f"✅ Wellness tracking plan generated successfully")
            logger.debug(f"Plan contains fields: {list(tracking_plan.keys())}")
            
            # Store in state
            state["wellness_tracking_output"] = tracking_plan
            
            # Also store in preventive_path_outputs for aggregation
            if "preventive_path_outputs" not in state:
                state["preventive_path_outputs"] = {}
            state["preventive_path_outputs"]["tracking"] = tracking_plan
            
            # Update processing stage
            state["processing_stage"] = "tracking_planned"
            
            # Log summary of plan
            if "tracking_overview" in tracking_plan:
                logger.info(f"Overview: {tracking_plan['tracking_overview'][:100]}...")
            
            if "monthly_checklist" in tracking_plan:
                checklist = tracking_plan["monthly_checklist"]
                if isinstance(checklist, list):
                    logger.info(f"Monthly checklist items: {len(checklist)}")
                    for i, item in enumerate(checklist[:3]):  # Log first 3
                        logger.debug(f"  Checklist {i+1}: {item}")
            
            if "wellness_goals" in tracking_plan:
                goals = tracking_plan["wellness_goals"]
                if isinstance(goals, list):
                    logger.info(f"Wellness goals: {len(goals)}")
                    for i, goal in enumerate(goals[:2]):  # Log first 2
                        logger.debug(f"  Goal {i+1}: {goal}")
            
            if "early_warning_signs" in tracking_plan:
                signs = tracking_plan["early_warning_signs"]
                if isinstance(signs, list):
                    logger.info(f"Early warning signs: {len(signs)}")
                    for i, sign in enumerate(signs[:2]):  # Log first 2
                        logger.debug(f"  Sign {i+1}: {sign}")
            
        else:
            # Planning failed
            error_msg = result.get("message", "Unknown tracking planning error")
            logger.error(f"❌ Wellness tracking plan generation failed: {error_msg}")
            
            state["error_occurred"] = True
            state["error_messages"].append(f"Tracking planning failed: {error_msg}")
            state["processing_stage"] = "tracking_planning_failed"
            
            # Store fallback plan if available
            if "wellness_tracking" in result:
                state["wellness_tracking_output"] = result["wellness_tracking"]
                state["preventive_path_outputs"] = state.get("preventive_path_outputs", {})
                state["preventive_path_outputs"]["tracking"] = result["wellness_tracking"]
                logger.info("Stored fallback tracking plan")
        
        return state
        
    except Exception as e:
        # Handle unexpected errors
        error_msg = f"Unexpected error in wellness tracking preventive node: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Update state with error information
        state["error_occurred"] = True
        state["error_messages"].append(error_msg)
        state["processing_stage"] = "tracking_planning_error"
        
        return state


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def _build_profile_from_state(state: PetCareState) -> Dict[str, Any]:
    """
    Build a complete profile dictionary from state fields for wellness tracking.
    
    Args:
        state: Current PetCareState
        
    Returns:
        Dictionary with all profile fields needed for wellness tracking
    """
    # Try to use extracted_profile first if available
    profile = state.get('extracted_profile', {}).copy()
    
    # If extracted_profile is empty, build from flattened fields
    if not profile:
        logger.debug("Building profile from flattened state fields")
        profile = {
            # Basic Information
            "pet_name": state.get("pet_name", "Your Pet"),
            "pet_species": state.get("pet_species", "unknown"),
            "breed": state.get("breed", "unknown"),
            "age_years": state.get("age_years", 0),
            "weight_status": state.get("weight_status", "unknown"),
            
            # Lifestyle Information
            "exercise_level": state.get("exercise_level", "unknown"),
            "living_situation": state.get("living_situation", "unknown"),
            
            # Medical Information
            "known_conditions": state.get("known_conditions", []),
        }
    else:
        # Ensure profile has all needed fields
        if "pet_name" not in profile:
            profile["pet_name"] = state.get("pet_name", "Your Pet")
        if "exercise_level" not in profile:
            profile["exercise_level"] = state.get("exercise_level", "unknown")
        if "living_situation" not in profile:
            profile["living_situation"] = state.get("living_situation", "unknown")
    
    # Ensure all required fields exist with defaults
    required_fields = {
        "pet_name": "Your Pet",
        "pet_species": "unknown",
        "breed": "unknown",
        "age_years": 0,
        "weight_status": "unknown",
        "exercise_level": "unknown",
        "living_situation": "unknown",
        "known_conditions": []
    }
    
    for field, default in required_fields.items():
        if field not in profile or profile[field] is None:
            profile[field] = default
    
    return profile


def _build_ml_results_from_state(state: PetCareState) -> Dict[str, Any]:
    """
    Build ML results dictionary from state fields for wellness tracking.
    
    Args:
        state: Current PetCareState
        
    Returns:
        Dictionary with ML prediction results
    """
    return {
        "health_risk_score": state.get("health_risk_score", 0.45),
        "care_capability_score": state.get("care_capability_score", 50.0)
    }


# ==========================================
# OPTIONAL: Helper function for testing
# ==========================================

def test_wellness_tracking_preventive_node():
    """
    Test function to verify node behavior.
    """
    from state import get_initial_state
    from utils.openai_client import build_openai_client
    
    print("=" * 60)
    print("TESTING WELLNESS TRACKING PREVENTIVE NODE")
    print("=" * 60)
    
    try:
        # Initialize client
        client = build_openai_client()
        print("✅ OpenAI client created")
        
        # Test Case 1: Adult dog with conditions
        print("\n📝 Test Case 1: Adult Dog with Conditions")
        state1 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        # Set up state
        state1["path_taken"] = "PREVENTIVE_CARE_PATH"
        state1["profile_extraction_complete"] = True
        
        # Set profile fields
        state1["pet_name"] = "Buddy"
        state1["pet_species"] = "dog"
        state1["breed"] = "beagle"
        state1["age_years"] = 6
        state1["weight_status"] = "overweight"
        state1["exercise_level"] = "moderate"
        state1["living_situation"] = "house"
        state1["known_conditions"] = ["mild arthritis", "seasonal allergies"]
        
        # Set ML results
        state1["health_risk_score"] = 0.45
        
        result1 = wellness_tracking_preventive_node(state1, client)
        print(f"Tracking plan generated: {'wellness_tracking_output' in result1}")
        print(f"Stage: {result1.get('processing_stage')}")
        
        if 'wellness_tracking_output' in result1:
            plan = result1['wellness_tracking_output']
            print(f"Plan fields: {list(plan.keys())}")
            if 'tracking_overview' in plan:
                print(f"Overview: {plan['tracking_overview'][:100]}...")
            if 'monthly_checklist' in plan:
                checklist = plan['monthly_checklist']
                print(f"Checklist items: {len(checklist) if isinstance(checklist, list) else 'Provided'}")
        
        # Test Case 2: Adult cat
        print("\n📝 Test Case 2: Adult Cat")
        state2 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        state2["path_taken"] = "PREVENTIVE_CARE_PATH"
        state2["profile_extraction_complete"] = True
        state2["pet_name"] = "Luna"
        state2["pet_species"] = "cat"
        state2["breed"] = "domestic shorthair"
        state2["age_years"] = 4
        state2["weight_status"] = "normal"
        state2["exercise_level"] = "moderate"
        state2["living_situation"] = "apartment"
        state2["known_conditions"] = []
        state2["health_risk_score"] = 0.32
        
        result2 = wellness_tracking_preventive_node(state2, client)
        print(f"Tracking plan generated: {'wellness_tracking_output' in result2}")
        
        # Test Case 3: Senior dog
        print("\n📝 Test Case 3: Senior Dog")
        state3 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        state3["path_taken"] = "PREVENTIVE_CARE_PATH"
        state3["profile_extraction_complete"] = True
        state3["pet_name"] = "Charlie"
        state3["pet_species"] = "dog"
        state3["breed"] = "golden retriever"
        state3["age_years"] = 10
        state3["weight_status"] = "normal"
        state3["exercise_level"] = "light"
        state3["living_situation"] = "house"
        state3["known_conditions"] = ["arthritis", "dental disease"]
        state3["health_risk_score"] = 0.58
        
        result3 = wellness_tracking_preventive_node(state3, client)
        print(f"Tracking plan generated: {'wellness_tracking_output' in result3}")
        
        # Test Case 4: Missing profile extraction
        print("\n📝 Test Case 4: Missing Profile Extraction")
        state4 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        state4["path_taken"] = "PREVENTIVE_CARE_PATH"
        state4["profile_extraction_complete"] = False  # Not complete
        
        result4 = wellness_tracking_preventive_node(state4, client)
        print(f"Error occurred: {result4.get('error_occurred')}")
        print(f"Error messages: {result4.get('error_messages')}")
        
        # Test Case 5: Error handling
        print("\n📝 Test Case 5: Error Handling")
        state5 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        # Cause an error by not having required data
        result5 = wellness_tracking_preventive_node(state5, client)
        print(f"Error occurred: {result5.get('error_occurred')}")
        print(f"Stage: {result5.get('processing_stage')}")
        
        return result1, result2, result3, result4, result5
        
    except Exception as e:
        print(f"❌ Test error: {str(e)}")
        return None, None, None, None, None


if __name__ == "__main__":
    # Run test if executed directly
    test_wellness_tracking_preventive_node()