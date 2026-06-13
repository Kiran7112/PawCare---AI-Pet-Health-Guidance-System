# nodes/wellness_optimization_node.py
"""
Wellness Optimization Node for PawCare+ LangGraph workflow (Wellness Path).
Executes wellness optimization planning using LLM as a workflow step.
"""

import logging
from typing import Dict, Any

from state import PetCareState
from agents.wellness_optimization_llm import WellnessOptimizationAgent
from utils.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


def wellness_optimization_node(state: PetCareState, client: OpenAIClient) -> PetCareState:
    """
    Execute wellness optimization planning as a workflow step in the wellness path.
    
    This node uses an LLM agent to generate an optimization plan for healthy,
    low-risk pets, including wellness overview, enhancements, activity suggestions,
    and bonding activities.
    
    Args:
        state: Current PetCareState containing:
            - Extracted profile fields
            - ML prediction results (health_risk_score)
            - Must be on WELLNESS_PATH
        client: OpenAIClient instance for LLM calls
        
    Returns:
        Updated PetCareState with wellness_optimization_output:
        - wellness_optimization_output: Dictionary with optimization plan fields
        - wellness_path_outputs: Updated with optimization
        - error_occurred: Set to True if planning fails
        - error_messages: Appended with any errors
        - processing_stage: Updated to "wellness_optimized" or "wellness_optimization_failed"
    """
    logger.info("=" * 50)
    logger.info("EXECUTING WELLNESS OPTIMIZATION NODE (WELLNESS PATH)")
    logger.info("=" * 50)
    
    try:
        # Log incoming state info
        logger.debug(f"State request_id: {state.get('request_id', 'N/A')}")
        logger.debug(f"Current path: {state.get('path_taken', 'unknown')}")
        logger.debug(f"Current processing stage: {state.get('processing_stage', 'unknown')}")
        
        # Verify we're on the wellness path
        if state.get('path_taken') != "WELLNESS_PATH":
            logger.warning(f"Wellness optimization node executed on {state.get('path_taken')} path - this may be unexpected")
        
        # Check if profile extraction was successful
        if not state.get('profile_extraction_complete', False):
            error_msg = "Cannot generate wellness plan: profile extraction not complete"
            logger.error(error_msg)
            state["error_occurred"] = True
            state["error_messages"].append(error_msg)
            state["processing_stage"] = "wellness_optimization_failed"
            return state
        
        # Extract profile from state
        profile = _build_profile_from_state(state)
        
        # Extract ML results from state
        ml_results = _build_ml_results_from_state(state)
        
        # Log optimization context
        logger.info(f"Generating wellness optimization plan for: "
                   f"Species={profile.get('pet_species', 'unknown')}, "
                   f"Name={profile.get('pet_name', 'Unknown')}, "
                   f"Age={profile.get('age_years', 0)}")
        
        risk_score = ml_results.get('health_risk_score', 0.2)
        logger.info(f"Health risk score: {risk_score:.3f} (Low risk - Wellness Path)")
        
        exercise_level = profile.get('exercise_level', 'moderate')
        logger.info(f"Current exercise level: {exercise_level}")
        
        living_situation = profile.get('living_situation', 'unknown')
        logger.info(f"Living situation: {living_situation}")
        
        behavioral_issues = profile.get('behavioral_issues', [])
        if behavioral_issues:
            logger.info(f"Behavioral notes: {', '.join(behavioral_issues[:3])}")
        
        # Create LLM agent
        optimization_agent = WellnessOptimizationAgent(client)
        logger.debug("WellnessOptimizationAgent created successfully")
        
        # Call agent to generate optimization plan
        result = optimization_agent.generate_optimization_plan(profile, ml_results)
        
        # Process planning result
        if result.get("status") == "success":
            # Extract optimization plan dictionary
            optimization_plan = result.get("wellness_optimization", {})
            
            logger.info(f"✅ Wellness optimization plan generated successfully")
            logger.debug(f"Plan contains fields: {list(optimization_plan.keys())}")
            
            # Store in state
            state["wellness_optimization_output"] = optimization_plan
            
            # Also store in wellness_path_outputs for aggregation
            if "wellness_path_outputs" not in state:
                state["wellness_path_outputs"] = {}
            state["wellness_path_outputs"]["optimization"] = optimization_plan
            
            # Update processing stage
            state["processing_stage"] = "wellness_optimized"
            
            # Log summary of plan
            if "optimization_overview" in optimization_plan:
                logger.info(f"Overview: {optimization_plan['optimization_overview'][:100]}...")
            
            if "wellness_enhancements" in optimization_plan:
                enhancements = optimization_plan["wellness_enhancements"]
                if isinstance(enhancements, list):
                    logger.info(f"Wellness enhancements: {len(enhancements)}")
                    for i, enhancement in enumerate(enhancements[:3]):  # Log first 3
                        logger.debug(f"  Enhancement {i+1}: {enhancement}")
            
            if "activity_suggestions" in optimization_plan:
                activities = optimization_plan["activity_suggestions"]
                if isinstance(activities, list):
                    logger.info(f"Activity suggestions: {len(activities)}")
                    for i, activity in enumerate(activities[:2]):  # Log first 2
                        logger.debug(f"  Activity {i+1}: {activity}")
            
            if "bonding_activities" in optimization_plan:
                bonding = optimization_plan["bonding_activities"]
                if isinstance(bonding, list):
                    logger.info(f"Bonding activities: {len(bonding)}")
                    for i, activity in enumerate(bonding[:2]):  # Log first 2
                        logger.debug(f"  Bonding {i+1}: {activity}")
            
        else:
            # Planning failed
            error_msg = result.get("message", "Unknown optimization error")
            logger.error(f"❌ Wellness optimization plan generation failed: {error_msg}")
            
            state["error_occurred"] = True
            state["error_messages"].append(f"Wellness optimization failed: {error_msg}")
            state["processing_stage"] = "wellness_optimization_failed"
            
            # Store fallback plan if available
            if "wellness_optimization" in result:
                state["wellness_optimization_output"] = result["wellness_optimization"]
                state["wellness_path_outputs"] = state.get("wellness_path_outputs", {})
                state["wellness_path_outputs"]["optimization"] = result["wellness_optimization"]
                logger.info("Stored fallback optimization plan")
        
        return state
        
    except Exception as e:
        # Handle unexpected errors
        error_msg = f"Unexpected error in wellness optimization node: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Update state with error information
        state["error_occurred"] = True
        state["error_messages"].append(error_msg)
        state["processing_stage"] = "wellness_optimization_error"
        
        return state


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def _build_profile_from_state(state: PetCareState) -> Dict[str, Any]:
    """
    Build a complete profile dictionary from state fields for wellness optimization.
    
    Args:
        state: Current PetCareState
        
    Returns:
        Dictionary with all profile fields needed for wellness optimization
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
            
            # Lifestyle Information
            "exercise_level": state.get("exercise_level", "moderate"),
            "living_situation": state.get("living_situation", "unknown"),
            
            # Behavioral Information
            "behavioral_issues": state.get("behavioral_issues", []),
        }
    else:
        # Ensure profile has all needed fields
        if "pet_name" not in profile:
            profile["pet_name"] = state.get("pet_name", "Your Pet")
        if "exercise_level" not in profile:
            profile["exercise_level"] = state.get("exercise_level", "moderate")
        if "living_situation" not in profile:
            profile["living_situation"] = state.get("living_situation", "unknown")
        if "behavioral_issues" not in profile:
            profile["behavioral_issues"] = state.get("behavioral_issues", [])
    
    # Ensure all required fields exist with defaults
    required_fields = {
        "pet_name": "Your Pet",
        "pet_species": "unknown",
        "breed": "unknown",
        "age_years": 0,
        "exercise_level": "moderate",
        "living_situation": "unknown",
        "behavioral_issues": []
    }
    
    for field, default in required_fields.items():
        if field not in profile or profile[field] is None:
            profile[field] = default
    
    return profile


def _build_ml_results_from_state(state: PetCareState) -> Dict[str, Any]:
    """
    Build ML results dictionary from state fields for wellness optimization.
    
    Args:
        state: Current PetCareState
        
    Returns:
        Dictionary with ML prediction results
    """
    return {
        "health_risk_score": state.get("health_risk_score", 0.2),
        "care_capability_score": state.get("care_capability_score", 70.0)
    }


# ==========================================
# OPTIONAL: Helper function for testing
# ==========================================

def test_wellness_optimization_node():
    """
    Test function to verify node behavior.
    """
    from state import get_initial_state
    from utils.openai_client import build_openai_client
    
    print("=" * 60)
    print("TESTING WELLNESS OPTIMIZATION NODE")
    print("=" * 60)
    
    try:
        # Initialize client
        client = build_openai_client()
        print("✅ OpenAI client created")
        
        # Test Case 1: Active young dog
        print("\n📝 Test Case 1: Active Young Dog")
        state1 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        # Set up state
        state1["path_taken"] = "WELLNESS_PATH"
        state1["profile_extraction_complete"] = True
        
        # Set profile fields
        state1["pet_name"] = "Luna"
        state1["pet_species"] = "dog"
        state1["breed"] = "australian shepherd"
        state1["age_years"] = 2
        state1["exercise_level"] = "active"
        state1["living_situation"] = "house with yard"
        state1["behavioral_issues"] = []
        
        # Set ML results
        state1["health_risk_score"] = 0.12
        
        result1 = wellness_optimization_node(state1, client)
        print(f"Optimization plan generated: {'wellness_optimization_output' in result1}")
        print(f"Stage: {result1.get('processing_stage')}")
        
        if 'wellness_optimization_output' in result1:
            plan = result1['wellness_optimization_output']
            print(f"Plan fields: {list(plan.keys())}")
            if 'optimization_overview' in plan:
                print(f"Overview: {plan['optimization_overview'][:100]}...")
            if 'wellness_enhancements' in plan:
                enhancements = plan['wellness_enhancements']
                print(f"Enhancements: {len(enhancements) if isinstance(enhancements, list) else 'Provided'}")
        
        # Test Case 2: Adult cat
        print("\n📝 Test Case 2: Adult Cat")
        state2 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        state2["path_taken"] = "WELLNESS_PATH"
        state2["profile_extraction_complete"] = True
        state2["pet_name"] = "Whiskers"
        state2["pet_species"] = "cat"
        state2["breed"] = "siamese"
        state2["age_years"] = 4
        state2["exercise_level"] = "moderate"
        state2["living_situation"] = "apartment"
        state2["behavioral_issues"] = []
        state2["health_risk_score"] = 0.15
        
        result2 = wellness_optimization_node(state2, client)
        print(f"Optimization plan generated: {'wellness_optimization_output' in result2}")
        
        # Test Case 3: Senior dog with mild issues
        print("\n📝 Test Case 3: Senior Dog (still wellness path due to low risk)")
        state3 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        state3["path_taken"] = "WELLNESS_PATH"
        state3["profile_extraction_complete"] = True
        state3["pet_name"] = "Charlie"
        state3["pet_species"] = "dog"
        state3["breed"] = "golden retriever"
        state3["age_years"] = 10
        state3["exercise_level"] = "light"
        state3["living_situation"] = "house"
        state3["behavioral_issues"] = ["mild anxiety during storms"]
        state3["health_risk_score"] = 0.28  # Still ≤ 0.3 for wellness path
        
        result3 = wellness_optimization_node(state3, client)
        print(f"Optimization plan generated: {'wellness_optimization_output' in result3}")
        
        # Test Case 4: Missing profile extraction
        print("\n📝 Test Case 4: Missing Profile Extraction")
        state4 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        state4["path_taken"] = "WELLNESS_PATH"
        state4["profile_extraction_complete"] = False  # Not complete
        
        result4 = wellness_optimization_node(state4, client)
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
        result5 = wellness_optimization_node(state5, client)
        print(f"Error occurred: {result5.get('error_occurred')}")
        print(f"Stage: {result5.get('processing_stage')}")
        
        return result1, result2, result3, result4, result5
        
    except Exception as e:
        print(f"❌ Test error: {str(e)}")
        return None, None, None, None, None


if __name__ == "__main__":
    # Run test if executed directly
    test_wellness_optimization_node()