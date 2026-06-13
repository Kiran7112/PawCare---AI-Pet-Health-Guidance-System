# nodes/nutrition_wellness_node.py
"""
Nutrition Wellness Node for PawCare+ LangGraph workflow (Wellness Path).
Executes wellness nutrition enhancement using LLM as a workflow step.
"""

import logging
from typing import Dict, Any

from state import PetCareState
from agents.nutrition_wellness_llm import NutritionWellnessAgent
from utils.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


def nutrition_wellness_node(state: PetCareState, client: OpenAIClient) -> PetCareState:
    """
    Execute wellness nutrition enhancement as a workflow step in the wellness path.
    
    This node uses an LLM agent to generate nutrition enhancement suggestions
    for healthy, low-risk pets, including nutrition overview, enhancement tips,
    variety suggestions, and supplement options.
    
    Args:
        state: Current PetCareState containing:
            - Extracted profile fields
            - ML prediction results (health_risk_score)
            - Must be on WELLNESS_PATH
        client: OpenAIClient instance for LLM calls
        
    Returns:
        Updated PetCareState with nutrition_wellness_output:
        - nutrition_wellness_output: Dictionary with nutrition enhancement fields
        - wellness_path_outputs: Updated with nutrition
        - error_occurred: Set to True if generation fails
        - error_messages: Appended with any errors
        - processing_stage: Updated to "nutrition_enhanced" or "nutrition_enhancement_failed"
    """
    logger.info("=" * 50)
    logger.info("EXECUTING NUTRITION WELLNESS NODE (WELLNESS PATH)")
    logger.info("=" * 50)
    
    try:
        # Log incoming state info
        logger.debug(f"State request_id: {state.get('request_id', 'N/A')}")
        logger.debug(f"Current path: {state.get('path_taken', 'unknown')}")
        logger.debug(f"Current processing stage: {state.get('processing_stage', 'unknown')}")
        
        # Verify we're on the wellness path
        if state.get('path_taken') != "WELLNESS_PATH":
            logger.warning(f"Nutrition wellness node executed on {state.get('path_taken')} path - this may be unexpected")
        
        # Check if profile extraction was successful
        if not state.get('profile_extraction_complete', False):
            error_msg = "Cannot generate nutrition enhancement: profile extraction not complete"
            logger.error(error_msg)
            state["error_occurred"] = True
            state["error_messages"].append(error_msg)
            state["processing_stage"] = "nutrition_enhancement_failed"
            return state
        
        # Extract profile from state
        profile = _build_profile_from_state(state)
        
        # Extract ML results from state
        ml_results = _build_ml_results_from_state(state)
        
        # Log nutrition context
        logger.info(f"Generating nutrition wellness enhancement for: "
                   f"Species={profile.get('pet_species', 'unknown')}, "
                   f"Name={profile.get('pet_name', 'Unknown')}, "
                   f"Age={profile.get('age_years', 0)}")
        
        risk_score = ml_results.get('health_risk_score', 0.2)
        logger.info(f"Health risk score: {risk_score:.3f} (Low risk - Wellness Path)")
        
        weight_status = profile.get('weight_status', 'normal')
        logger.info(f"Weight status: {weight_status}")
        
        diet_type = profile.get('diet_type', 'unknown')
        diet_quality = profile.get('diet_quality', 'unknown')
        logger.info(f"Current diet: {diet_type} ({diet_quality})")
        
        allergies = profile.get('allergies_known', [])
        if allergies:
            logger.info(f"Allergies to consider: {', '.join(allergies)}")
        
        # Create LLM agent
        nutrition_agent = NutritionWellnessAgent(client)
        logger.debug("NutritionWellnessAgent created successfully")
        
        # Call agent to generate nutrition enhancement
        result = nutrition_agent.generate_nutrition_enhancement(profile, ml_results)
        
        # Process enhancement result
        if result.get("status") == "success":
            # Extract nutrition enhancement dictionary
            nutrition_enhancement = result.get("nutrition_wellness", {})
            
            logger.info(f"✅ Nutrition wellness enhancement generated successfully")
            logger.debug(f"Enhancement contains fields: {list(nutrition_enhancement.keys())}")
            
            # Store in state
            state["nutrition_wellness_output"] = nutrition_enhancement
            
            # Also store in wellness_path_outputs for aggregation
            if "wellness_path_outputs" not in state:
                state["wellness_path_outputs"] = {}
            state["wellness_path_outputs"]["nutrition"] = nutrition_enhancement
            
            # Update processing stage
            state["processing_stage"] = "nutrition_enhanced"
            
            # Log summary of enhancement
            if "nutrition_overview" in nutrition_enhancement:
                logger.info(f"Overview: {nutrition_enhancement['nutrition_overview'][:100]}...")
            
            if "enhancement_tips" in nutrition_enhancement:
                tips = nutrition_enhancement["enhancement_tips"]
                if isinstance(tips, list):
                    logger.info(f"Enhancement tips: {len(tips)}")
                    for i, tip in enumerate(tips[:3]):  # Log first 3
                        logger.debug(f"  Tip {i+1}: {tip}")
            
            if "variety_suggestions" in nutrition_enhancement:
                variety = nutrition_enhancement["variety_suggestions"]
                if isinstance(variety, list):
                    logger.info(f"Variety suggestions: {len(variety)}")
                    for i, suggestion in enumerate(variety[:2]):  # Log first 2
                        logger.debug(f"  Suggestion {i+1}: {suggestion}")
            
            if "supplement_options" in nutrition_enhancement:
                supplements = nutrition_enhancement["supplement_options"]
                if isinstance(supplements, list):
                    logger.info(f"Supplement options: {len(supplements)}")
                    for i, supplement in enumerate(supplements[:2]):  # Log first 2
                        logger.debug(f"  Supplement {i+1}: {supplement[:100]}...")
            
        else:
            # Enhancement failed
            error_msg = result.get("message", "Unknown nutrition enhancement error")
            logger.error(f"❌ Nutrition wellness enhancement generation failed: {error_msg}")
            
            state["error_occurred"] = True
            state["error_messages"].append(f"Nutrition enhancement failed: {error_msg}")
            state["processing_stage"] = "nutrition_enhancement_failed"
            
            # Store fallback enhancement if available
            if "nutrition_wellness" in result:
                state["nutrition_wellness_output"] = result["nutrition_wellness"]
                state["wellness_path_outputs"] = state.get("wellness_path_outputs", {})
                state["wellness_path_outputs"]["nutrition"] = result["nutrition_wellness"]
                logger.info("Stored fallback nutrition enhancement")
        
        return state
        
    except Exception as e:
        # Handle unexpected errors
        error_msg = f"Unexpected error in nutrition wellness node: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Update state with error information
        state["error_occurred"] = True
        state["error_messages"].append(error_msg)
        state["processing_stage"] = "nutrition_enhancement_error"
        
        return state


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def _build_profile_from_state(state: PetCareState) -> Dict[str, Any]:
    """
    Build a complete profile dictionary from state fields for nutrition wellness.
    
    Args:
        state: Current PetCareState
        
    Returns:
        Dictionary with all profile fields needed for nutrition enhancement
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
            "weight_status": state.get("weight_status", "normal"),
            
            # Dietary Information
            "diet_type": state.get("diet_type", "unknown"),
            "diet_quality": state.get("diet_quality", "unknown"),
            
            # Medical Information
            "allergies_known": state.get("allergies_known", []),
        }
    else:
        # Ensure profile has all needed fields
        if "pet_name" not in profile:
            profile["pet_name"] = state.get("pet_name", "Your Pet")
        if "weight_status" not in profile:
            profile["weight_status"] = state.get("weight_status", "normal")
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
        "weight_status": "normal",
        "diet_type": "unknown",
        "diet_quality": "unknown",
        "allergies_known": []
    }
    
    for field, default in required_fields.items():
        if field not in profile or profile[field] is None:
            profile[field] = default
    
    return profile


def _build_ml_results_from_state(state: PetCareState) -> Dict[str, Any]:
    """
    Build ML results dictionary from state fields for nutrition wellness.
    
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

def test_nutrition_wellness_node():
    """
    Test function to verify node behavior.
    """
    from state import get_initial_state
    from utils.openai_client import build_openai_client
    
    print("=" * 60)
    print("TESTING NUTRITION WELLNESS NODE")
    print("=" * 60)
    
    try:
        # Initialize client
        client = build_openai_client()
        print("✅ OpenAI client created")
        
        # Test Case 1: Healthy adult dog on premium diet
        print("\n📝 Test Case 1: Healthy Adult Dog")
        state1 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        # Set up state
        state1["path_taken"] = "WELLNESS_PATH"
        state1["profile_extraction_complete"] = True
        
        # Set profile fields
        state1["pet_name"] = "Bailey"
        state1["pet_species"] = "dog"
        state1["breed"] = "golden retriever"
        state1["age_years"] = 3
        state1["weight_status"] = "normal"
        state1["diet_type"] = "kibble"
        state1["diet_quality"] = "premium"
        state1["allergies_known"] = []
        
        # Set ML results
        state1["health_risk_score"] = 0.15
        
        result1 = nutrition_wellness_node(state1, client)
        print(f"Nutrition enhancement generated: {'nutrition_wellness_output' in result1}")
        print(f"Stage: {result1.get('processing_stage')}")
        
        if 'nutrition_wellness_output' in result1:
            enhancement = result1['nutrition_wellness_output']
            print(f"Enhancement fields: {list(enhancement.keys())}")
            if 'nutrition_overview' in enhancement:
                print(f"Overview: {enhancement['nutrition_overview'][:100]}...")
            if 'enhancement_tips' in enhancement:
                tips = enhancement['enhancement_tips']
                print(f"Tips: {len(tips) if isinstance(tips, list) else 'Provided'}")
        
        # Test Case 2: Healthy cat
        print("\n📝 Test Case 2: Healthy Cat")
        state2 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        state2["path_taken"] = "WELLNESS_PATH"
        state2["profile_extraction_complete"] = True
        state2["pet_name"] = "Luna"
        state2["pet_species"] = "cat"
        state2["breed"] = "domestic shorthair"
        state2["age_years"] = 4
        state2["weight_status"] = "normal"
        state2["diet_type"] = "mixed"
        state2["diet_quality"] = "good"
        state2["allergies_known"] = []
        state2["health_risk_score"] = 0.12
        
        result2 = nutrition_wellness_node(state2, client)
        print(f"Nutrition enhancement generated: {'nutrition_wellness_output' in result2}")
        
        # Test Case 3: Dog with allergies (but still wellness path)
        print("\n📝 Test Case 3: Dog with Allergies")
        state3 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        state3["path_taken"] = "WELLNESS_PATH"
        state3["profile_extraction_complete"] = True
        state3["pet_name"] = "Charlie"
        state3["pet_species"] = "dog"
        state3["breed"] = "beagle"
        state3["age_years"] = 5
        state3["weight_status"] = "normal"
        state3["diet_type"] = "kibble"
        state3["diet_quality"] = "good"
        state3["allergies_known"] = ["chicken", "beef"]
        state3["health_risk_score"] = 0.22
        
        result3 = nutrition_wellness_node(state3, client)
        print(f"Nutrition enhancement generated: {'nutrition_wellness_output' in result3}")
        
        # Test Case 4: Healthy senior dog
        print("\n📝 Test Case 4: Healthy Senior Dog")
        state4 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        state4["path_taken"] = "WELLNESS_PATH"
        state4["profile_extraction_complete"] = True
        state4["pet_name"] = "Max"
        state4["pet_species"] = "dog"
        state4["breed"] = "labrador"
        state4["age_years"] = 9
        state4["weight_status"] = "normal"
        state4["diet_type"] = "kibble"
        state4["diet_quality"] = "good"
        state4["allergies_known"] = []
        state4["health_risk_score"] = 0.28  # Still ≤ 0.3 for wellness path
        
        result4 = nutrition_wellness_node(state4, client)
        print(f"Nutrition enhancement generated: {'nutrition_wellness_output' in result4}")
        
        # Test Case 5: Missing profile extraction
        print("\n📝 Test Case 5: Missing Profile Extraction")
        state5 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        state5["path_taken"] = "WELLNESS_PATH"
        state5["profile_extraction_complete"] = False  # Not complete
        
        result5 = nutrition_wellness_node(state5, client)
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
        result6 = nutrition_wellness_node(state6, client)
        print(f"Error occurred: {result6.get('error_occurred')}")
        print(f"Stage: {result6.get('processing_stage')}")
        
        return result1, result2, result3, result4, result5, result6
        
    except Exception as e:
        print(f"❌ Test error: {str(e)}")
        return None, None, None, None, None, None


if __name__ == "__main__":
    # Run test if executed directly
    test_nutrition_wellness_node()