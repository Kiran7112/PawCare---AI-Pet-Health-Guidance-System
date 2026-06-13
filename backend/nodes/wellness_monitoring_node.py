# nodes/wellness_monitoring_node.py
"""
Wellness Monitoring Node for PawCare+ LangGraph workflow (Critical Path).
Executes wellness monitoring planning using LLM as a workflow step.
"""

import logging
from typing import Dict, Any

from state import PetCareState
from agents.wellness_monitoring_llm import WellnessMonitoringAgent
from utils.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


def wellness_monitoring_node(state: PetCareState, client: OpenAIClient) -> PetCareState:
    """
    Execute wellness monitoring planning as a workflow step in the critical path.
    
    This node uses an LLM agent to generate a comprehensive wellness monitoring plan
    for high-risk pets, including daily monitoring checklists, weekly assessments,
    monthly detailed checks, vet visit schedules, and red flag identification.
    
    Args:
        state: Current PetCareState containing:
            - Extracted profile fields
            - ML prediction results (health_risk_score, care_capability_score)
            - Must be on CRITICAL_CARE_PATH
        client: OpenAIClient instance for LLM calls
        
    Returns:
        Updated PetCareState with wellness_monitoring_output:
        - wellness_monitoring_output: Dictionary with monitoring plan fields
        - critical_path_outputs: Updated with monitoring
        - error_occurred: Set to True if planning fails
        - error_messages: Appended with any errors
        - processing_stage: Updated to "monitoring_planned" or "monitoring_planning_failed"
    """
    logger.info("=" * 50)
    logger.info("EXECUTING WELLNESS MONITORING NODE (CRITICAL PATH)")
    logger.info("=" * 50)
    
    try:
        # Log incoming state info
        logger.debug(f"State request_id: {state.get('request_id', 'N/A')}")
        logger.debug(f"Current path: {state.get('path_taken', 'unknown')}")
        logger.debug(f"Current processing stage: {state.get('processing_stage', 'unknown')}")
        
        # Verify we're on the critical path
        if state.get('path_taken') != "CRITICAL_CARE_PATH":
            logger.warning(f"Wellness monitoring node executed on {state.get('path_taken')} path - this may be unexpected")
        
        # Check if profile extraction was successful
        if not state.get('profile_extraction_complete', False):
            error_msg = "Cannot generate monitoring plan: profile extraction not complete"
            logger.error(error_msg)
            state["error_occurred"] = True
            state["error_messages"].append(error_msg)
            state["processing_stage"] = "monitoring_planning_failed"
            return state
        
        # Extract profile from state
        profile = _build_profile_from_state(state)
        
        # Extract ML results from state
        ml_results = _build_ml_results_from_state(state)
        
        # Log monitoring planning context
        logger.info(f"Generating wellness monitoring plan for: "
                   f"Species={profile.get('pet_species', 'unknown')}, "
                   f"Name={profile.get('pet_name', 'Unknown')}, "
                   f"Age={profile.get('age_years', 0)}")
        
        conditions = profile.get('known_conditions', [])
        if conditions:
            logger.info(f"Medical conditions: {', '.join(conditions[:3])}")
            if len(conditions) > 3:
                logger.info(f"  ... and {len(conditions) - 3} more")
        
        risk_score = ml_results.get('health_risk_score', 0.5)
        logger.info(f"Health risk score: {risk_score:.3f}")
        
        # Get monitoring intensity based on risk score
        if risk_score > 0.8:
            logger.info("Monitoring intensity: INTENSIVE (multiple times daily)")
        elif risk_score > 0.6:
            logger.info("Monitoring intensity: ELEVATED (daily)")
        else:
            logger.info("Monitoring intensity: STANDARD (daily with weekly assessments)")
        
        # Create LLM agent
        monitoring_agent = WellnessMonitoringAgent(client)
        logger.debug("WellnessMonitoringAgent created successfully")
        
        # Call agent to generate monitoring plan
        result = monitoring_agent.generate_monitoring_plan(profile, ml_results)
        
        # Process planning result
        if result.get("status") == "success":
            # Extract monitoring plan dictionary
            monitoring_plan = result.get("wellness_monitoring", {})
            
            logger.info(f"✅ Wellness monitoring plan generated successfully")
            logger.debug(f"Plan contains fields: {list(monitoring_plan.keys())}")
            
            # Store in state
            state["wellness_monitoring_output"] = monitoring_plan
            
            # Also store in critical_path_outputs for aggregation
            if "critical_path_outputs" not in state:
                state["critical_path_outputs"] = {}
            state["critical_path_outputs"]["monitoring"] = monitoring_plan
            
            # Update processing stage
            state["processing_stage"] = "monitoring_planned"
            
            # Log summary of plan
            if "monitoring_overview" in monitoring_plan:
                logger.info(f"Overview: {monitoring_plan['monitoring_overview'][:100]}...")
            
            if "daily_monitoring" in monitoring_plan:
                daily = monitoring_plan["daily_monitoring"]
                if isinstance(daily, list):
                    logger.info(f"Daily monitoring tasks: {len(daily)}")
                    for i, task in enumerate(daily[:3]):  # Log first 3
                        logger.debug(f"  Daily task {i+1}: {task}")
            
            if "weekly_assessment" in monitoring_plan:
                weekly = monitoring_plan["weekly_assessment"]
                if isinstance(weekly, list):
                    logger.info(f"Weekly assessment tasks: {len(weekly)}")
            
            if "red_flags" in monitoring_plan:
                red_flags = monitoring_plan["red_flags"]
                if isinstance(red_flags, list):
                    logger.info(f"Red flags to watch: {len(red_flags)}")
                    for i, flag in enumerate(red_flags[:2]):  # Log first 2
                        logger.debug(f"  Red flag {i+1}: {flag}")
            
            if "vet_visit_schedule" in monitoring_plan:
                logger.info(f"Vet visit schedule: {monitoring_plan['vet_visit_schedule'][:100]}...")
            
        else:
            # Planning failed
            error_msg = result.get("message", "Unknown monitoring planning error")
            logger.error(f"❌ Wellness monitoring plan generation failed: {error_msg}")
            
            state["error_occurred"] = True
            state["error_messages"].append(f"Monitoring planning failed: {error_msg}")
            state["processing_stage"] = "monitoring_planning_failed"
            
            # Store fallback plan if available
            if "wellness_monitoring" in result:
                state["wellness_monitoring_output"] = result["wellness_monitoring"]
                state["critical_path_outputs"] = state.get("critical_path_outputs", {})
                state["critical_path_outputs"]["monitoring"] = result["wellness_monitoring"]
                logger.info("Stored fallback monitoring plan")
        
        return state
        
    except Exception as e:
        # Handle unexpected errors
        error_msg = f"Unexpected error in wellness monitoring node: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Update state with error information
        state["error_occurred"] = True
        state["error_messages"].append(error_msg)
        state["processing_stage"] = "monitoring_planning_error"
        
        return state


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def _build_profile_from_state(state: PetCareState) -> Dict[str, Any]:
    """
    Build a complete profile dictionary from state fields for monitoring planning.
    
    Args:
        state: Current PetCareState
        
    Returns:
        Dictionary with all profile fields needed for monitoring planning
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
            "medications_current": state.get("medications_current", []),
            "recent_symptoms": state.get("recent_symptoms", []),
            
            # Symptom duration (if available)
            "symptom_duration_days": state.get("symptom_duration_days", None),
        }
    else:
        # Ensure profile has all needed fields
        if "pet_name" not in profile:
            profile["pet_name"] = state.get("pet_name", "Your Pet")
        if "medications_current" not in profile:
            profile["medications_current"] = state.get("medications_current", [])
        if "recent_symptoms" not in profile:
            profile["recent_symptoms"] = state.get("recent_symptoms", [])
        if "symptom_duration_days" not in profile:
            profile["symptom_duration_days"] = state.get("symptom_duration_days", None)
    
    # Ensure all required fields exist with defaults
    required_fields = {
        "pet_name": "Your Pet",
        "pet_species": "unknown",
        "breed": "unknown",
        "age_years": 0,
        "weight_status": "unknown",
        "known_conditions": [],
        "medications_current": [],
        "recent_symptoms": [],
        "symptom_duration_days": None
    }
    
    for field, default in required_fields.items():
        if field not in profile or profile[field] is None:
            profile[field] = default
    
    return profile


def _build_ml_results_from_state(state: PetCareState) -> Dict[str, Any]:
    """
    Build ML results dictionary from state fields for monitoring planning.
    
    Args:
        state: Current PetCareState
        
    Returns:
        Dictionary with ML prediction results
    """
    return {
        "health_risk_score": state.get("health_risk_score", 0.5),
        "care_capability_score": state.get("care_capability_score", 50.0),
        "health_risk_factors": state.get("health_risk_factors", {})
    }


# ==========================================
# OPTIONAL: Helper function for testing
# ==========================================

def test_wellness_monitoring_node():
    """
    Test function to verify node behavior.
    """
    from state import get_initial_state
    from utils.openai_client import build_openai_client
    
    print("=" * 60)
    print("TESTING WELLNESS MONITORING NODE")
    print("=" * 60)
    
    try:
        # Initialize client
        client = build_openai_client()
        print("✅ OpenAI client created")
        
        # Test Case 1: High-risk diabetic dog
        print("\n📝 Test Case 1: High-Risk Diabetic Dog")
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
        state1["medications_current"] = ["insulin", "carprofen"]
        state1["recent_symptoms"] = ["increased thirst", "lethargy", "occasional cough"]
        state1["symptom_duration_days"] = 5
        
        # Set ML results
        state1["health_risk_score"] = 0.85
        state1["care_capability_score"] = 75.0
        state1["health_risk_factors"] = {
            "age": 0.4,
            "conditions_count": 0.3,
            "symptoms": 0.2
        }
        
        result1 = wellness_monitoring_node(state1, client)
        print(f"Monitoring plan generated: {'wellness_monitoring_output' in result1}")
        print(f"Stage: {result1.get('processing_stage')}")
        
        if 'wellness_monitoring_output' in result1:
            plan = result1['wellness_monitoring_output']
            print(f"Plan fields: {list(plan.keys())}")
            if 'monitoring_overview' in plan:
                print(f"Overview: {plan['monitoring_overview'][:100]}...")
            if 'daily_monitoring' in plan:
                daily = plan['daily_monitoring']
                print(f"Daily tasks: {len(daily) if isinstance(daily, list) else 'Provided'}")
            if 'red_flags' in plan:
                flags = plan['red_flags']
                print(f"Red flags: {len(flags) if isinstance(flags, list) else 'Listed'}")
        
        # Test Case 2: High-risk cat with kidney disease
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
        state2["age_years"] = 14
        state2["known_conditions"] = ["chronic kidney disease", "hyperthyroidism"]
        state2["medications_current"] = ["methimazole"]
        state2["recent_symptoms"] = ["increased thirst", "weight loss"]
        state2["health_risk_score"] = 0.78
        
        result2 = wellness_monitoring_node(state2, client)
        print(f"Monitoring plan generated: {'wellness_monitoring_output' in result2}")
        
        # Test Case 3: Missing profile extraction
        print("\n📝 Test Case 3: Missing Profile Extraction")
        state3 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        state3["path_taken"] = "CRITICAL_CARE_PATH"
        state3["profile_extraction_complete"] = False  # Not complete
        
        result3 = wellness_monitoring_node(state3, client)
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
        result4 = wellness_monitoring_node(state4, client)
        print(f"Error occurred: {result4.get('error_occurred')}")
        print(f"Stage: {result4.get('processing_stage')}")
        
        return result1, result2, result3, result4
        
    except Exception as e:
        print(f"❌ Test error: {str(e)}")
        return None, None, None, None


if __name__ == "__main__":
    # Run test if executed directly
    test_wellness_monitoring_node()