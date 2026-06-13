# nodes/nutrition_critical_node.py
"""
Nutrition Critical Node for PawCare+ LangGraph workflow (Critical Path).
Executes critical nutrition planning using LLM as a workflow step.
"""

import logging
from typing import Dict, Any

from state import PetCareState
from agents.nutrition_critical_llm import NutritionCriticalAgent
from utils.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


def nutrition_critical_node(state: PetCareState, client: OpenAIClient) -> PetCareState:
    """
    Execute critical nutrition planning as a workflow step in the critical path.
    
    This node uses an LLM agent to generate a specialized nutrition plan for high-risk pets,
    including therapeutic diet recommendations, feeding schedules, supplements, and
    condition-specific dietary guidance.
    
    Args:
        state: Current PetCareState containing:
            - Extracted profile fields
            - Must be on CRITICAL_CARE_PATH
        client: OpenAIClient instance for LLM calls
        
    Returns:
        Updated PetCareState with nutrition_critical_output:
        - nutrition_critical_output: Dictionary with nutrition plan fields
        - critical_path_outputs: Updated with nutrition
        - error_occurred: Set to True if planning fails
        - error_messages: Appended with any errors
        - processing_stage: Updated to "nutrition_planned" or "nutrition_planning_failed"
    """
    logger.info("=" * 50)
    logger.info("EXECUTING NUTRITION CRITICAL NODE (CRITICAL PATH)")
    logger.info("=" * 50)
    
    try:
        # Log incoming state info
        logger.debug(f"State request_id: {state.get('request_id', 'N/A')}")
        logger.debug(f"Current path: {state.get('path_taken', 'unknown')}")
        logger.debug(f"Current processing stage: {state.get('processing_stage', 'unknown')}")
        
        # Verify we're on the critical path
        if state.get('path_taken') != "CRITICAL_CARE_PATH":
            logger.warning(f"Nutrition critical node executed on {state.get('path_taken')} path - this may be unexpected")
        
        # Check if profile extraction was successful
        if not state.get('profile_extraction_complete', False):
            error_msg = "Cannot generate nutrition plan: profile extraction not complete"
            logger.error(error_msg)
            state["error_occurred"] = True
            state["error_messages"].append(error_msg)
            state["processing_stage"] = "nutrition_planning_failed"
            return state
        
        # Extract profile from state
        profile = _build_profile_from_state(state)
        
        # Log nutrition planning context
        logger.info(f"Generating critical nutrition plan for: "
                   f"Species={profile.get('pet_species', 'unknown')}, "
                   f"Name={profile.get('pet_name', 'Unknown')}, "
                   f"Age={profile.get('age_years', 0)}")
        
        conditions = profile.get('known_conditions', [])
        if conditions:
            logger.info(f"Medical conditions: {', '.join(conditions[:3])}")
            if len(conditions) > 3:
                logger.info(f"  ... and {len(conditions) - 3} more")
        
        allergies = profile.get('allergies_known', [])
        if allergies:
            logger.info(f"Allergies: {', '.join(allergies)}")
        
        # Create LLM agent
        nutrition_agent = NutritionCriticalAgent(client)
        logger.debug("NutritionCriticalAgent created successfully")
        
        # Call agent to generate nutrition plan
        result = nutrition_agent.generate_nutrition_plan(profile)
        
        # Process planning result
        if result.get("status") == "success":
            # Extract nutrition plan dictionary
            nutrition_plan = result.get("nutrition_critical", {})
            
            logger.info(f"✅ Critical nutrition plan generated successfully")
            logger.debug(f"Plan contains fields: {list(nutrition_plan.keys())}")
            
            # Store in state
            state["nutrition_critical_output"] = nutrition_plan
            
            # Also store in critical_path_outputs for aggregation
            if "critical_path_outputs" not in state:
                state["critical_path_outputs"] = {}
            state["critical_path_outputs"]["nutrition"] = nutrition_plan
            
            # Update processing stage
            state["processing_stage"] = "nutrition_planned"
            
            # Log summary of plan
            if "nutrition_overview" in nutrition_plan:
                logger.info(f"Overview: {nutrition_plan['nutrition_overview'][:100]}...")
            
            if "recommended_diet" in nutrition_plan:
                logger.info(f"Diet recommendation: {nutrition_plan['recommended_diet'][:100]}...")
            
            if "feeding_schedule" in nutrition_plan:
                logger.info(f"Feeding schedule: {nutrition_plan['feeding_schedule'][:100]}...")
            
            # Extract and log supplements if available
            if "supplements" in nutrition_plan:
                supplements = nutrition_plan["supplements"]
                if isinstance(supplements, list):
                    logger.info(f"Recommended supplements: {len(supplements)}")
                    for i, supp in enumerate(supplements[:2]):  # Log first 2
                        logger.debug(f"  Supplement {i+1}: {supp[:100]}...")
            
            # Log foods to avoid if available
            if "foods_to_avoid" in nutrition_plan:
                foods = nutrition_plan["foods_to_avoid"]
                if isinstance(foods, list):
                    logger.info(f"Foods to avoid: {len(foods)} items")
            
        else:
            # Planning failed
            error_msg = result.get("message", "Unknown nutrition planning error")
            logger.error(f"❌ Critical nutrition plan generation failed: {error_msg}")
            
            state["error_occurred"] = True
            state["error_messages"].append(f"Nutrition planning failed: {error_msg}")
            state["processing_stage"] = "nutrition_planning_failed"
            
            # Store fallback plan if available
            if "nutrition_critical" in result:
                state["nutrition_critical_output"] = result["nutrition_critical"]
                state["critical_path_outputs"] = state.get("critical_path_outputs", {})
                state["critical_path_outputs"]["nutrition"] = result["nutrition_critical"]
                logger.info("Stored fallback nutrition plan")
        
        return state
        
    except Exception as e:
        # Handle unexpected errors
        error_msg = f"Unexpected error in nutrition critical node: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Update state with error information
        state["error_occurred"] = True
        state["error_messages"].append(error_msg)
        state["processing_stage"] = "nutrition_planning_error"
        
        return state


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def _build_profile_from_state(state: PetCareState) -> Dict[str, Any]:
    """
    Build a complete profile dictionary from state fields for nutrition planning.
    
    Args:
        state: Current PetCareState
        
    Returns:
        Dictionary with all profile fields needed for nutrition planning
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
            
            # Medical Information
            "known_conditions": state.get("known_conditions", []),
            "allergies_known": state.get("allergies_known", []),
            
            # Dietary Information
            "diet_type": state.get("diet_type", "unknown"),
            "diet_quality": state.get("diet_quality", "unknown"),
            "feeding_frequency": state.get("feeding_frequency", "Not specified"),
            "current_diet": state.get("current_diet", "Not specified"),
        }
    else:
        # Ensure profile has all needed fields
        if "pet_name" not in profile:
            profile["pet_name"] = state.get("pet_name", "Your Pet")
        if "diet_type" not in profile:
            profile["diet_type"] = state.get("diet_type", "unknown")
        if "diet_quality" not in profile:
            profile["diet_quality"] = state.get("diet_quality", "unknown")
        if "feeding_frequency" not in profile:
            profile["feeding_frequency"] = state.get("feeding_frequency", "Not specified")
        if "current_diet" not in profile:
            profile["current_diet"] = state.get("current_diet", "Not specified")
    
    # Ensure all required fields exist with defaults
    required_fields = {
        "pet_name": "Your Pet",
        "pet_species": "unknown",
        "breed": "unknown",
        "age_years": 0,
        "weight_status": "unknown",
        "known_conditions": [],
        "allergies_known": [],
        "diet_type": "unknown",
        "diet_quality": "unknown",
        "feeding_frequency": "Not specified",
        "current_diet": "Not specified"
    }
    
    for field, default in required_fields.items():
        if field not in profile or profile[field] is None:
            profile[field] = default
    
    return profile


# ==========================================
# OPTIONAL: Helper function for testing
# ==========================================

def test_nutrition_critical_node():
    """
    Test function to verify node behavior.
    """
    from state import get_initial_state
    from utils.openai_client import build_openai_client
    
    print("=" * 60)
    print("TESTING NUTRITION CRITICAL NODE")
    print("=" * 60)
    
    try:
        # Initialize client
        client = build_openai_client()
        print("✅ OpenAI client created")
        
        # Test Case 1: Complete profile on critical path
        print("\n📝 Test Case 1: Complete Profile (Critical Path)")
        state1 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        # Set up state
        state1["path_taken"] = "CRITICAL_CARE_PATH"
        state1["profile_extraction_complete"] = True
        
        # Set profile fields
        state1["pet_name"] = "Max"
        state1["pet_species"] = "dog"
        state1["breed"] = "labrador"
        state1["age_years"] = 12
        state1["weight_status"] = "overweight"
        state1["known_conditions"] = ["diabetes", "arthritis", "heart murmur"]
        state1["allergies_known"] = ["chicken"]
        state1["diet_type"] = "kibble"
        state1["diet_quality"] = "good"
        state1["feeding_frequency"] = "2 times daily"
        state1["current_diet"] = "Adult maintenance kibble"
        
        result1 = nutrition_critical_node(state1, client)
        print(f"Nutrition plan generated: {'nutrition_critical_output' in result1}")
        print(f"Stage: {result1.get('processing_stage')}")
        
        if 'nutrition_critical_output' in result1:
            plan = result1['nutrition_critical_output']
            print(f"Plan fields: {list(plan.keys())}")
            if 'nutrition_overview' in plan:
                print(f"Overview: {plan['nutrition_overview'][:100]}...")
            if 'recommended_diet' in plan:
                print(f"Diet: {plan['recommended_diet'][:100]}...")
        
        # Test Case 2: Cat with kidney disease
        print("\n📝 Test Case 2: Cat with Kidney Disease")
        state2 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        state2["path_taken"] = "CRITICAL_CARE_PATH"
        state2["profile_extraction_complete"] = True
        state2["pet_name"] = "Whiskers"
        state2["pet_species"] = "cat"
        state2["breed"] = "domestic shorthair"
        state2["age_years"] = 14
        state2["weight_status"] = "underweight"
        state2["known_conditions"] = ["chronic kidney disease", "hyperthyroidism"]
        state2["diet_type"] = "wet food"
        state2["diet_quality"] = "premium"
        
        result2 = nutrition_critical_node(state2, client)
        print(f"Nutrition plan generated: {'nutrition_critical_output' in result2}")
        
        # Test Case 3: Missing profile extraction
        print("\n📝 Test Case 3: Missing Profile Extraction")
        state3 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        state3["path_taken"] = "CRITICAL_CARE_PATH"
        state3["profile_extraction_complete"] = False  # Not complete
        
        result3 = nutrition_critical_node(state3, client)
        print(f"Error occurred: {result3.get('error_occurred')}")
        print(f"Error messages: {result3.get('error_messages')}")
        
        # Test Case 4: Error handling
        print("\n📝 Test Case 4: Error Handling")
        state4 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        # Cause an error by not having required data
        result4 = nutrition_critical_node(state4, client)
        print(f"Error occurred: {result4.get('error_occurred')}")
        print(f"Stage: {result4.get('processing_stage')}")
        
        return result1, result2, result3, result4
        
    except Exception as e:
        print(f"❌ Test error: {str(e)}")
        return None, None, None, None


if __name__ == "__main__":
    # Run test if executed directly
    test_nutrition_critical_node()