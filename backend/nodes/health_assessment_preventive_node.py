# nodes/health_assessment_preventive_node.py
"""
Health Assessment Preventive Node for PawCare+ LangGraph workflow (Preventive Path).
Executes preventive health assessment using LLM as a workflow step.
"""

import logging
from typing import Dict, Any

from state import PetCareState
from agents.health_assessment_preventive_llm import HealthAssessmentPreventiveAgent
from utils.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


def health_assessment_preventive_node(state: PetCareState, client: OpenAIClient) -> PetCareState:
    """
    Execute preventive health assessment as a workflow step in the preventive path.
    
    This node uses an LLM agent to generate a focused preventive health assessment
    for moderate-risk pets, including preventive assessment, key health areas,
    recommended checkups, and prevention strategies.
    
    Args:
        state: Current PetCareState containing:
            - Extracted profile fields
            - ML prediction results (health_risk_score, care_capability_score)
            - Must be on PREVENTIVE_CARE_PATH
        client: OpenAIClient instance for LLM calls
        
    Returns:
        Updated PetCareState with health_assessment_output:
        - health_assessment_output: Dictionary with assessment fields
        - preventive_path_outputs: Updated with health_assessment
        - error_occurred: Set to True if assessment fails
        - error_messages: Appended with any errors
        - processing_stage: Updated to "health_assessed" or "health_assessment_failed"
    """
    logger.info("=" * 50)
    logger.info("EXECUTING HEALTH ASSESSMENT PREVENTIVE NODE (PREVENTIVE PATH)")
    logger.info("=" * 50)
    
    try:
        # Log incoming state info
        logger.debug(f"State request_id: {state.get('request_id', 'N/A')}")
        logger.debug(f"Current path: {state.get('path_taken', 'unknown')}")
        logger.debug(f"Current processing stage: {state.get('processing_stage', 'unknown')}")
        
        # Verify we're on the preventive path
        if state.get('path_taken') != "PREVENTIVE_CARE_PATH":
            logger.warning(f"Health assessment preventive node executed on {state.get('path_taken')} path - this may be unexpected")
        
        # Check if profile extraction was successful
        if not state.get('profile_extraction_complete', False):
            error_msg = "Cannot generate health assessment: profile extraction not complete"
            logger.error(error_msg)
            state["error_occurred"] = True
            state["error_messages"].append(error_msg)
            state["processing_stage"] = "health_assessment_failed"
            return state
        
        # Extract profile from state
        profile = _build_profile_from_state(state)
        
        # Extract ML results from state
        ml_results = _build_ml_results_from_state(state)
        
        # Log assessment context
        logger.info(f"Generating preventive health assessment for: "
                   f"Species={profile.get('pet_species', 'unknown')}, "
                   f"Name={profile.get('pet_name', 'Unknown')}, "
                   f"Age={profile.get('age_years', 0)}")
        
        risk_score = ml_results.get('health_risk_score', 0.45)
        logger.info(f"Health risk score: {risk_score:.3f} (Moderate risk range: 0.3-0.6)")
        
        conditions = profile.get('known_conditions', [])
        if conditions:
            logger.info(f"Conditions to monitor: {', '.join(conditions[:3])}")
            if len(conditions) > 3:
                logger.info(f"  ... and {len(conditions) - 3} more")
        
        weight_status = profile.get('weight_status', 'unknown')
        if weight_status in ['overweight', 'obese']:
            logger.info(f"Weight status: {weight_status} - weight management key focus")
        
        # Create LLM agent
        assessment_agent = HealthAssessmentPreventiveAgent(client)
        logger.debug("HealthAssessmentPreventiveAgent created successfully")
        
        # Call agent to generate health assessment
        result = assessment_agent.generate_health_assessment(profile, ml_results)
        
        # Process assessment result
        if result.get("status") == "success":
            # Extract assessment dictionary
            assessment = result.get("health_assessment", {})
            
            logger.info(f"✅ Preventive health assessment generated successfully")
            logger.debug(f"Assessment contains fields: {list(assessment.keys())}")
            
            # Store in state
            state["health_assessment_output"] = assessment
            
            # Also store in preventive_path_outputs for aggregation
            if "preventive_path_outputs" not in state:
                state["preventive_path_outputs"] = {}
            state["preventive_path_outputs"]["health_assessment"] = assessment
            
            # Update processing stage
            state["processing_stage"] = "health_assessed"
            
            # Log summary of assessment
            if "preventive_assessment" in assessment:
                logger.info(f"Assessment: {assessment['preventive_assessment'][:100]}...")
            
            if "key_health_areas" in assessment:
                areas = assessment["key_health_areas"]
                if isinstance(areas, list):
                    logger.info(f"Key health areas to monitor: {len(areas)}")
                    for i, area in enumerate(areas[:3]):  # Log first 3
                        logger.debug(f"  Area {i+1}: {area}")
            
            if "recommended_checkups" in assessment:
                logger.info(f"Checkup schedule: {assessment['recommended_checkups'][:100]}...")
            
            if "prevention_strategies" in assessment:
                strategies = assessment["prevention_strategies"]
                if isinstance(strategies, list):
                    logger.info(f"Prevention strategies: {len(strategies)}")
                    for i, strategy in enumerate(strategies[:2]):  # Log first 2
                        logger.debug(f"  Strategy {i+1}: {strategy[:100]}...")
            
        else:
            # Assessment failed
            error_msg = result.get("message", "Unknown health assessment error")
            logger.error(f"❌ Preventive health assessment generation failed: {error_msg}")
            
            state["error_occurred"] = True
            state["error_messages"].append(f"Health assessment failed: {error_msg}")
            state["processing_stage"] = "health_assessment_failed"
            
            # Store fallback assessment if available
            if "health_assessment" in result:
                state["health_assessment_output"] = result["health_assessment"]
                state["preventive_path_outputs"] = state.get("preventive_path_outputs", {})
                state["preventive_path_outputs"]["health_assessment"] = result["health_assessment"]
                logger.info("Stored fallback health assessment")
        
        return state
        
    except Exception as e:
        # Handle unexpected errors
        error_msg = f"Unexpected error in health assessment preventive node: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Update state with error information
        state["error_occurred"] = True
        state["error_messages"].append(error_msg)
        state["processing_stage"] = "health_assessment_error"
        
        return state


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def _build_profile_from_state(state: PetCareState) -> Dict[str, Any]:
    """
    Build a complete profile dictionary from state fields for preventive assessment.
    
    Args:
        state: Current PetCareState
        
    Returns:
        Dictionary with all profile fields needed for preventive assessment
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
            
            # Lifestyle Information
            "exercise_level": state.get("exercise_level", "unknown"),
            "diet_type": state.get("diet_type", "unknown"),
            "diet_quality": state.get("diet_quality", "unknown"),
        }
    else:
        # Ensure profile has all needed fields
        if "pet_name" not in profile:
            profile["pet_name"] = state.get("pet_name", "Your Pet")
        if "exercise_level" not in profile:
            profile["exercise_level"] = state.get("exercise_level", "unknown")
        if "diet_type" not in profile:
            profile["diet_type"] = state.get("diet_type", "unknown")
        if "diet_quality" not in profile:
            profile["diet_quality"] = state.get("diet_quality", "unknown")
    
    # Ensure all required fields exist with defaults
    required_fields = {
        "pet_name": "Your Pet",
        "pet_species": "unknown",
        "breed": "unknown",
        "age_years": 0,
        "weight_status": "unknown",
        "known_conditions": [],
        "exercise_level": "unknown",
        "diet_type": "unknown",
        "diet_quality": "unknown"
    }
    
    for field, default in required_fields.items():
        if field not in profile or profile[field] is None:
            profile[field] = default
    
    return profile


def _build_ml_results_from_state(state: PetCareState) -> Dict[str, Any]:
    """
    Build ML results dictionary from state fields for preventive assessment.
    
    Args:
        state: Current PetCareState
        
    Returns:
        Dictionary with ML prediction results
    """
    return {
        "health_risk_score": state.get("health_risk_score", 0.45),
        "care_capability_score": state.get("care_capability_score", 50.0),
        "health_risk_factors": state.get("health_risk_factors", {})
    }


# ==========================================
# OPTIONAL: Helper function for testing
# ==========================================

def test_health_assessment_preventive_node():
    """
    Test function to verify node behavior.
    """
    from state import get_initial_state
    from utils.openai_client import build_openai_client
    
    print("=" * 60)
    print("TESTING HEALTH ASSESSMENT PREVENTIVE NODE")
    print("=" * 60)
    
    try:
        # Initialize client
        client = build_openai_client()
        print("✅ OpenAI client created")
        
        # Test Case 1: Moderate-risk overweight dog
        print("\n📝 Test Case 1: Moderate-Risk Overweight Dog")
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
        state1["diet_type"] = "kibble"
        state1["diet_quality"] = "average"
        state1["known_conditions"] = ["mild arthritis", "seasonal allergies"]
        
        # Set ML results
        state1["health_risk_score"] = 0.45  # Moderate risk
        state1["care_capability_score"] = 70.0
        state1["health_risk_factors"] = {
            "weight": 0.4,
            "age": 0.3,
            "conditions": 0.2
        }
        
        result1 = health_assessment_preventive_node(state1, client)
        print(f"Health assessment generated: {'health_assessment_output' in result1}")
        print(f"Stage: {result1.get('processing_stage')}")
        
        if 'health_assessment_output' in result1:
            assessment = result1['health_assessment_output']
            print(f"Assessment fields: {list(assessment.keys())}")
            if 'preventive_assessment' in assessment:
                print(f"Assessment: {assessment['preventive_assessment'][:100]}...")
            if 'key_health_areas' in assessment:
                areas = assessment['key_health_areas']
                print(f"Key areas: {', '.join(areas[:3]) if isinstance(areas, list) else areas}")
        
        # Test Case 2: Healthy adult cat
        print("\n📝 Test Case 2: Healthy Adult Cat (Low-Moderate Risk)")
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
        state2["known_conditions"] = []
        state2["health_risk_score"] = 0.32  # Low-moderate
        
        result2 = health_assessment_preventive_node(state2, client)
        print(f"Health assessment generated: {'health_assessment_output' in result2}")
        
        # Test Case 3: Senior dog with multiple conditions
        print("\n📝 Test Case 3: Senior Dog (Upper Moderate Risk)")
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
        state3["known_conditions"] = ["arthritis", "dental disease"]
        state3["health_risk_score"] = 0.58  # Upper moderate
        
        result3 = health_assessment_preventive_node(state3, client)
        print(f"Health assessment generated: {'health_assessment_output' in result3}")
        
        # Test Case 4: Missing profile extraction
        print("\n📝 Test Case 4: Missing Profile Extraction")
        state4 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        state4["path_taken"] = "PREVENTIVE_CARE_PATH"
        state4["profile_extraction_complete"] = False  # Not complete
        
        result4 = health_assessment_preventive_node(state4, client)
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
        result5 = health_assessment_preventive_node(state5, client)
        print(f"Error occurred: {result5.get('error_occurred')}")
        print(f"Stage: {result5.get('processing_stage')}")
        
        return result1, result2, result3, result4, result5
        
    except Exception as e:
        print(f"❌ Test error: {str(e)}")
        return None, None, None, None, None


if __name__ == "__main__":
    # Run test if executed directly
    test_health_assessment_preventive_node()