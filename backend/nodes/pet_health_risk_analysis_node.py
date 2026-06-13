# nodes/pet_health_risk_analysis_node.py
"""
Pet Health Risk Analysis Node for PawCare+ LangGraph workflow (Critical Path).
Executes detailed health risk analysis using LLM as a workflow step.
"""

import logging
from typing import Dict, Any

from state import PetCareState
from agents.pet_health_risk_analysis_llm import PetHealthRiskAnalysisAgent
from utils.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


def pet_health_risk_analysis_node(state: PetCareState, client: OpenAIClient) -> PetCareState:
    """
    Execute health risk analysis as a workflow step in the critical path.
    
    This node uses an LLM agent to generate a comprehensive health risk analysis
    for high-risk pets, including risk assessment, critical factors, warning signs,
    and urgency timeline.
    
    Args:
        state: Current PetCareState containing:
            - Extracted profile fields
            - ML prediction results (health_risk_score, care_capability_score)
            - Must be on CRITICAL_CARE_PATH
        client: OpenAIClient instance for LLM calls
        
    Returns:
        Updated PetCareState with health_risk_analysis_output:
        - health_risk_analysis_output: Dictionary with analysis fields
        - critical_path_outputs: Updated with risk_analysis
        - error_occurred: Set to True if analysis fails
        - error_messages: Appended with any errors
        - processing_stage: Updated to "risk_analyzed" or "risk_analysis_failed"
    """
    logger.info("=" * 50)
    logger.info("EXECUTING PET HEALTH RISK ANALYSIS NODE (CRITICAL PATH)")
    logger.info("=" * 50)
    
    try:
        # Log incoming state info
        logger.debug(f"State request_id: {state.get('request_id', 'N/A')}")
        logger.debug(f"Current path: {state.get('path_taken', 'unknown')}")
        logger.debug(f"Current processing stage: {state.get('processing_stage', 'unknown')}")
        
        # Verify we're on the critical path
        if state.get('path_taken') != "CRITICAL_CARE_PATH":
            logger.warning(f"Risk analysis node executed on {state.get('path_taken')} path - this may be unexpected")
        
        # Check if profile extraction was successful
        if not state.get('profile_extraction_complete', False):
            error_msg = "Cannot generate risk analysis: profile extraction not complete"
            logger.error(error_msg)
            state["error_occurred"] = True
            state["error_messages"].append(error_msg)
            state["processing_stage"] = "risk_analysis_failed"
            return state
        
        # Extract profile from state
        profile = _build_profile_from_state(state)
        
        # Extract ML results from state
        ml_results = _build_ml_results_from_state(state)
        
        # Log analysis context
        logger.info(f"Generating risk analysis for: "
                   f"Species={profile.get('pet_species', 'unknown')}, "
                   f"Age={profile.get('age_years', 0)}, "
                   f"Conditions={len(profile.get('known_conditions', []))}")
        logger.info(f"Risk score: {ml_results.get('health_risk_score', 0):.3f}, "
                   f"Care capability: {ml_results.get('care_capability_score', 0):.1f}")
        
        # Create LLM agent
        analyzer = PetHealthRiskAnalysisAgent(client)
        logger.debug("PetHealthRiskAnalysisAgent created successfully")
        
        # Call agent to generate risk analysis
        result = analyzer.generate_risk_analysis(profile, ml_results)
        
        # Process analysis result
        if result.get("status") == "success":
            # Extract analysis dictionary
            analysis = result.get("health_risk_analysis", {})
            
            logger.info(f"✅ Risk analysis generated successfully")
            logger.debug(f"Analysis contains fields: {list(analysis.keys())}")
            
            # Store in state
            state["health_risk_analysis_output"] = analysis
            
            # Also store in critical_path_outputs for aggregation
            if "critical_path_outputs" not in state:
                state["critical_path_outputs"] = {}
            state["critical_path_outputs"]["risk_analysis"] = analysis
            
            # Update processing stage
            state["processing_stage"] = "risk_analyzed"
            
            # Log summary of analysis
            risk_level = "CRITICAL" if ml_results.get('health_risk_score', 0) > 0.8 else "HIGH"
            logger.info(f"Risk level: {risk_level}")
            
            # Extract and log key information from analysis if available
            if "critical_risk_factors" in analysis:
                factors = analysis["critical_risk_factors"]
                logger.info(f"Identified {len(factors)} critical risk factors")
                for i, factor in enumerate(factors[:2]):  # Log first 2
                    logger.debug(f"  Risk factor {i+1}: {factor[:100]}...")
            
            if "urgency_timeline" in analysis:
                logger.info(f"Urgency timeline: {analysis['urgency_timeline'][:100]}...")
            
        else:
            # Analysis failed
            error_msg = result.get("message", "Unknown analysis error")
            logger.error(f"❌ Risk analysis failed: {error_msg}")
            
            state["error_occurred"] = True
            state["error_messages"].append(f"Risk analysis failed: {error_msg}")
            state["processing_stage"] = "risk_analysis_failed"
            
            # Store fallback analysis if available
            if "health_risk_analysis" in result:
                state["health_risk_analysis_output"] = result["health_risk_analysis"]
                state["critical_path_outputs"] = state.get("critical_path_outputs", {})
                state["critical_path_outputs"]["risk_analysis"] = result["health_risk_analysis"]
        
        return state
        
    except Exception as e:
        # Handle unexpected errors
        error_msg = f"Unexpected error in pet health risk analysis node: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Update state with error information
        state["error_occurred"] = True
        state["error_messages"].append(error_msg)
        state["processing_stage"] = "risk_analysis_error"
        
        return state


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def _build_profile_from_state(state: PetCareState) -> Dict[str, Any]:
    """
    Build a complete profile dictionary from state fields.
    
    Args:
        state: Current PetCareState
        
    Returns:
        Dictionary with all profile fields needed for analysis
    """
    # Try to use extracted_profile first if available
    profile = state.get('extracted_profile', {}).copy()
    
    # If extracted_profile is empty, build from flattened fields
    if not profile:
        logger.debug("Building profile from flattened state fields")
        profile = {
            # Basic Information
            "pet_species": state.get("pet_species", "unknown"),
            "breed": state.get("breed", "unknown"),
            "age_years": state.get("age_years", 0),
            "weight_status": state.get("weight_status", "unknown"),
            "sex": state.get("sex", "unknown"),
            
            # Medical Information
            "known_conditions": state.get("known_conditions", []),
            "past_surgeries": state.get("past_surgeries", []),
            "allergies_known": state.get("allergies_known", []),
            "medications_current": state.get("medications_current", []),
            
            # Lifestyle Information
            "living_situation": state.get("living_situation", "unknown"),
            "exercise_level": state.get("exercise_level", "unknown"),
            "diet_type": state.get("diet_type", "unknown"),
            "diet_quality": state.get("diet_quality", "unknown"),
            "behavioral_issues": state.get("behavioral_issues", []),
            
            # Owner Information
            "owner_experience": state.get("owner_experience", "unknown"),
            "vet_access": state.get("vet_access", "unknown"),
            "owner_commitment": state.get("owner_commitment", "unknown"),
        }
    
    # Ensure all required fields exist
    required_fields = [
        "pet_species", "breed", "age_years", "weight_status",
        "known_conditions", "recent_symptoms", "symptom_duration_days",
        "medications_current", "behavioral_issues"
    ]
    
    for field in required_fields:
        if field not in profile:
            if field in ["known_conditions", "recent_symptoms", "medications_current", "behavioral_issues"]:
                profile[field] = []
            elif field == "symptom_duration_days":
                profile[field] = None
            else:
                profile[field] = "unknown"
    
    # Add recent symptoms if available (might be in state separately)
    if "recent_symptoms" not in profile and "recent_symptoms" in state:
        profile["recent_symptoms"] = state["recent_symptoms"]
    
    # Add symptom duration if available
    if "symptom_duration_days" not in profile and "symptom_duration_days" in state:
        profile["symptom_duration_days"] = state["symptom_duration_days"]
    
    return profile


def _build_ml_results_from_state(state: PetCareState) -> Dict[str, Any]:
    """
    Build ML results dictionary from state fields.
    
    Args:
        state: Current PetCareState
        
    Returns:
        Dictionary with ML prediction results
    """
    return {
        "health_risk_score": state.get("health_risk_score", 0.5),
        "care_capability_score": state.get("care_capability_score", 50.0),
        "health_risk_factors": state.get("health_risk_factors", {}),
        "feature_contributions": state.get("health_risk_factors", {})
    }


# ==========================================
# OPTIONAL: Helper function for testing
# ==========================================

def test_pet_health_risk_analysis_node():
    """
    Test function to verify node behavior.
    """
    from state import get_initial_state
    from utils.openai_client import build_openai_client
    
    print("=" * 60)
    print("TESTING PET HEALTH RISK ANALYSIS NODE")
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
        state1["health_risk_score"] = 0.85
        state1["care_capability_score"] = 75.0
        
        # Set profile fields
        state1["pet_species"] = "dog"
        state1["breed"] = "labrador"
        state1["age_years"] = 12
        state1["weight_status"] = "overweight"
        state1["known_conditions"] = ["arthritis", "heart murmur", "diabetes"]
        state1["medications_current"] = ["insulin", "carprofen"]
        state1["recent_symptoms"] = ["lethargy", "increased thirst", "coughing"]
        state1["symptom_duration_days"] = 5
        
        result1 = pet_health_risk_analysis_node(state1, client)
        print(f"Analysis generated: {'health_risk_analysis_output' in result1}")
        print(f"Stage: {result1.get('processing_stage')}")
        
        if 'health_risk_analysis_output' in result1:
            analysis = result1['health_risk_analysis_output']
            print(f"Analysis fields: {list(analysis.keys())}")
            if 'honest_risk_assessment' in analysis:
                print(f"Risk assessment: {analysis['honest_risk_assessment'][:100]}...")
        
        # Test Case 2: Missing profile extraction
        print("\n📝 Test Case 2: Missing Profile Extraction")
        state2 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        state2["path_taken"] = "CRITICAL_CARE_PATH"
        state2["profile_extraction_complete"] = False  # Not complete
        
        result2 = pet_health_risk_analysis_node(state2, client)
        print(f"Error occurred: {result2.get('error_occurred')}")
        print(f"Error messages: {result2.get('error_messages')}")
        
        # Test Case 3: Error handling
        print("\n📝 Test Case 3: Error Handling")
        state3 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        # Cause an error by not having required data
        # This should be caught by the exception handler
        
        result3 = pet_health_risk_analysis_node(state3, client)
        print(f"Error occurred: {result3.get('error_occurred')}")
        print(f"Stage: {result3.get('processing_stage')}")
        
        return result1, result2, result3
        
    except Exception as e:
        print(f"❌ Test error: {str(e)}")
        return None, None, None


if __name__ == "__main__":
    # Run test if executed directly
    test_pet_health_risk_analysis_node()