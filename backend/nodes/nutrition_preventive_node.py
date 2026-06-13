# nodes/nutrition_preventive_node.py
"""
Nutrition Preventive Node for PawCare+ LangGraph workflow (Preventive Path).
Executes preventive nutrition planning using LLM as a workflow step.
"""

import logging
from typing import Dict, Any

from state import PetCareState
from agents.nutrition_preventive_llm import NutritionPreventiveAgent
from utils.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


def nutrition_preventive_node(state: PetCareState, client: OpenAIClient) -> PetCareState:
    """
    Execute preventive nutrition planning as a workflow step in the preventive path.
    
    This node uses an LLM agent to generate a focused preventive nutrition guide
    for moderate-risk pets, including nutrition overview, diet recommendations,
    portion guidance, and healthy treat options.
    
    Args:
        state: Current PetCareState containing:
            - Extracted profile fields
            - ML prediction results (health_risk_score)
            - Must be on PREVENTIVE_CARE_PATH
        client: OpenAIClient instance for LLM calls
        
    Returns:
        Updated PetCareState with nutrition_preventive_output:
        - nutrition_preventive_output: Dictionary with nutrition guide fields
        - preventive_path_outputs: Updated with nutrition
        - error_occurred: Set to True if planning fails
        - error_messages: Appended with any errors
        - processing_stage: Updated to "nutrition_guided" or "nutrition_guidance_failed"
    """
    logger.info("=" * 50)
    logger.info("EXECUTING NUTRITION PREVENTIVE NODE (PREVENTIVE PATH)")
    logger.info("=" * 50)
    
    try:
        # Log incoming state info
        logger.debug(f"State request_id: {state.get('request_id', 'N/A')}")
        logger.debug(f"Current path: {state.get('path_taken', 'unknown')}")
        logger.debug(f"Current processing stage: {state.get('processing_stage', 'unknown')}")
        
        # Verify we're on the preventive path
        if state.get('path_taken') != "PREVENTIVE_CARE_PATH":
            logger.warning(f"Nutrition preventive node executed on {state.get('path_taken')} path - this may be unexpected")
        
        # Check if profile extraction was successful
        if not state.get('profile_extraction_complete', False):
            error_msg = "Cannot generate nutrition guide: profile extraction not complete"
            logger.error(error_msg)
            state["error_occurred"] = True
            state["error_messages"].append(error_msg)
            state["processing_stage"] = "nutrition_guidance_failed"
            return state
        
        # Extract profile from state
        profile = _build_profile_from_state(state)
        
        # Extract ML results from state
        ml_results = _build_ml_results_from_state(state)
        
        # Log nutrition planning context
        logger.info(f"Generating preventive nutrition guide for: "
                   f"Species={profile.get('pet_species', 'unknown')}, "
                   f"Name={profile.get('pet_name', 'Unknown')}, "
                   f"Age={profile.get('age_years', 0)}")
        
        risk_score = ml_results.get('health_risk_score', 0.45)
        logger.info(f"Health risk score: {risk_score:.3f}")
        
        weight_status = profile.get('weight_status', 'unknown')
        logger.info(f"Weight status: {weight_status}")
        
        diet_type = profile.get('diet_type', 'unknown')
        diet_quality = profile.get('diet_quality', 'unknown')
        logger.info(f"Current diet: {diet_type} ({diet_quality})")
        
        conditions = profile.get('known_conditions', [])
        if conditions:
            logger.info(f"Conditions with dietary implications: {', '.join(conditions[:3])}")
            if len(conditions) > 3:
                logger.info(f"  ... and {len(conditions) - 3} more")
        
        allergies = profile.get('allergies_known', [])
        if allergies:
            logger.info(f"Allergies affecting diet: {', '.join(allergies)}")
        
        # Create LLM agent
        nutrition_agent = NutritionPreventiveAgent(client)
        logger.debug("NutritionPreventiveAgent created successfully")
        
        # Call agent to generate nutrition guide
        result = nutrition_agent.generate_nutrition_guide(profile, ml_results)
        
        # Process planning result
        if result.get("status") == "success":
            # Extract nutrition guide dictionary
            nutrition_guide = result.get("nutrition_preventive", {})
            
            logger.info(f"✅ Preventive nutrition guide generated successfully")
            logger.debug(f"Guide contains fields: {list(nutrition_guide.keys())}")
            
            # Store in state
            state["nutrition_preventive_output"] = nutrition_guide
            
            # Also store in preventive_path_outputs for aggregation
            if "preventive_path_outputs" not in state:
                state["preventive_path_outputs"] = {}
            state["preventive_path_outputs"]["nutrition"] = nutrition_guide
            
            # Update processing stage
            state["processing_stage"] = "nutrition_guided"
            
            # Log summary of guide
            if "nutrition_overview" in nutrition_guide:
                logger.info(f"Overview: {nutrition_guide['nutrition_overview'][:100]}...")
            
            if "recommended_diet" in nutrition_guide:
                logger.info(f"Diet recommendation: {nutrition_guide['recommended_diet'][:100]}...")
            
            if "portion_guidance" in nutrition_guide:
                logger.info(f"Portion guidance: {nutrition_guide['portion_guidance'][:100]}...")
            
            if "healthy_treats" in nutrition_guide:
                treats = nutrition_guide["healthy_treats"]
                if isinstance(treats, list):
                    logger.info(f"Healthy treat options: {len(treats)}")
                    for i, treat in enumerate(treats[:3]):  # Log first 3
                        logger.debug(f"  Treat {i+1}: {treat}")
            
        else:
            # Planning failed
            error_msg = result.get("message", "Unknown nutrition guidance error")
            logger.error(f"❌ Preventive nutrition guide generation failed: {error_msg}")
            
            state["error_occurred"] = True
            state["error_messages"].append(f"Nutrition guidance failed: {error_msg}")
            state["processing_stage"] = "nutrition_guidance_failed"
            
            # Store fallback guide if available
            if "nutrition_preventive" in result:
                state["nutrition_preventive_output"] = result["nutrition_preventive"]
                state["preventive_path_outputs"] = state.get("preventive_path_outputs", {})
                state["preventive_path_outputs"]["nutrition"] = result["nutrition_preventive"]
                logger.info("Stored fallback nutrition guide")
        
        return state
        
    except Exception as e:
        # Handle unexpected errors
        error_msg = f"Unexpected error in nutrition preventive node: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Update state with error information
        state["error_occurred"] = True
        state["error_messages"].append(error_msg)
        state["processing_stage"] = "nutrition_guidance_error"
        
        return state


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def _build_profile_from_state(state: PetCareState) -> Dict[str, Any]:
    """
    Build a complete profile dictionary from state fields for preventive nutrition.
    
    Args:
        state: Current PetCareState
        
    Returns:
        Dictionary with all profile fields needed for preventive nutrition guidance
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
            
            # Dietary Information
            "diet_type": state.get("diet_type", "unknown"),
            "diet_quality": state.get("diet_quality", "unknown"),
            
            # Medical Information
            "known_conditions": state.get("known_conditions", []),
            "allergies_known": state.get("allergies_known", []),
        }
    else:
        # Ensure profile has all needed fields
        if "pet_name" not in profile:
            profile["pet_name"] = state.get("pet_name", "Your Pet")
        if "diet_type" not in profile:
            profile["diet_type"] = state.get("diet_type", "unknown")
        if "diet_quality" not in profile:
            profile["diet_quality"] = state.get("diet_quality", "unknown")
        if "allergies_known" not in profile:
            profile["allergies_known"] = state.get("allergies_known", [])
    
    # Ensure all required fields exist with defaults
    required_fields = {
        "pet_name": "Your Pet",
        "pet_species": "unknown",
        "breed": "unknown",
        "age_years": 0,
        "weight_status": "unknown",
        "diet_type": "unknown",
        "diet_quality": "unknown",
        "known_conditions": [],
        "allergies_known": []
    }
    
    for field, default in required_fields.items():
        if field not in profile or profile[field] is None:
            profile[field] = default
    
    return profile


def _build_ml_results_from_state(state: PetCareState) -> Dict[str, Any]:
    """
    Build ML results dictionary from state fields for preventive nutrition.
    
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

def test_nutrition_preventive_node():
    """
    Test function to verify node behavior.
    """
    from state import get_initial_state
    from utils.openai_client import build_openai_client
    
    print("=" * 60)
    print("TESTING NUTRITION PREVENTIVE NODE")
    print("=" * 60)
    
    try:
        # Initialize client
        client = build_openai_client()
        print("✅ OpenAI client created")
        
        # Test Case 1: Overweight dog needing weight management
        print("\n📝 Test Case 1: Overweight Dog")
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
        state1["diet_type"] = "kibble"
        state1["diet_quality"] = "average"
        state1["known_conditions"] = ["mild arthritis"]
        state1["allergies_known"] = []
        
        # Set ML results
        state1["health_risk_score"] = 0.45
        
        result1 = nutrition_preventive_node(state1, client)
        print(f"Nutrition guide generated: {'nutrition_preventive_output' in result1}")
        print(f"Stage: {result1.get('processing_stage')}")
        
        if 'nutrition_preventive_output' in result1:
            guide = result1['nutrition_preventive_output']
            print(f"Guide fields: {list(guide.keys())}")
            if 'nutrition_overview' in guide:
                print(f"Overview: {guide['nutrition_overview'][:100]}...")
            if 'recommended_diet' in guide:
                print(f"Diet: {guide['recommended_diet'][:100]}...")
            if 'healthy_treats' in guide:
                treats = guide['healthy_treats']
                print(f"Treat options: {len(treats) if isinstance(treats, list) else 'Provided'}")
        
        # Test Case 2: Overweight cat
        print("\n📝 Test Case 2: Overweight Cat")
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
        state2["weight_status"] = "overweight"
        state2["diet_type"] = "kibble"
        state2["diet_quality"] = "average"
        state2["known_conditions"] = []
        state2["health_risk_score"] = 0.38
        
        result2 = nutrition_preventive_node(state2, client)
        print(f"Nutrition guide generated: {'nutrition_preventive_output' in result2}")
        
        # Test Case 3: Dog with allergies
        print("\n📝 Test Case 3: Dog with Allergies")
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
        state3["age_years"] = 5
        state3["weight_status"] = "normal"
        state3["diet_type"] = "kibble"
        state3["diet_quality"] = "good"
        state3["known_conditions"] = []
        state3["allergies_known"] = ["chicken", "beef"]
        state3["health_risk_score"] = 0.32
        
        result3 = nutrition_preventive_node(state3, client)
        print(f"Nutrition guide generated: {'nutrition_preventive_output' in result3}")
        
        # Test Case 4: Healthy adult dog
        print("\n📝 Test Case 4: Healthy Adult Dog")
        state4 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        state4["path_taken"] = "PREVENTIVE_CARE_PATH"
        state4["profile_extraction_complete"] = True
        state4["pet_name"] = "Max"
        state4["pet_species"] = "dog"
        state4["breed"] = "labrador"
        state4["age_years"] = 3
        state4["weight_status"] = "normal"
        state4["diet_type"] = "kibble"
        state4["diet_quality"] = "premium"
        state4["known_conditions"] = []
        state4["health_risk_score"] = 0.28  # Actually wellness path, but testing preventive node
        
        result4 = nutrition_preventive_node(state4, client)
        print(f"Nutrition guide generated: {'nutrition_preventive_output' in result4}")
        
        # Test Case 5: Missing profile extraction
        print("\n📝 Test Case 5: Missing Profile Extraction")
        state5 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        state5["path_taken"] = "PREVENTIVE_CARE_PATH"
        state5["profile_extraction_complete"] = False  # Not complete
        
        result5 = nutrition_preventive_node(state5, client)
        print(f"Error occurred: {result5.get('error_occurred')}")
        print(f"Error messages: {result5.get('error_messages')}")
        
        # Test Case 6: Error handling
        print("\n📝 Test Case 6: Error Handling")
        state6 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        # Cause an error by not having required data
        result6 = nutrition_preventive_node(state6, client)
        print(f"Error occurred: {result6.get('error_occurred')}")
        print(f"Stage: {result6.get('processing_stage')}")
        
        return result1, result2, result3, result4, result5, result6
        
    except Exception as e:
        print(f"❌ Test error: {str(e)}")
        return None, None, None, None, None, None


if __name__ == "__main__":
    # Run test if executed directly
    test_nutrition_preventive_node()