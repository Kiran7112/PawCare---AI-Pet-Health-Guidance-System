# nodes/lifestyle_enrichment_node.py
"""
Lifestyle Enrichment Node for PawCare+ LangGraph workflow (Wellness Path).
Executes lifestyle enrichment planning using LLM as a workflow step.
"""

import logging
from typing import Dict, Any

from state import PetCareState
from agents.lifestyle_enrichment_llm import LifestyleEnrichmentAgent
from utils.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


def lifestyle_enrichment_node(state: PetCareState, client: OpenAIClient) -> PetCareState:
    """
    Execute lifestyle enrichment planning as a workflow step in the wellness path.
    
    This node uses an LLM agent to generate a comprehensive lifestyle enrichment plan
    for healthy, low-risk pets, including mental stimulation, social opportunities,
    and environmental enrichment suggestions.
    
    Args:
        state: Current PetCareState containing:
            - Extracted profile fields
            - ML prediction results (health_risk_score)
            - Must be on WELLNESS_PATH
        client: OpenAIClient instance for LLM calls
        
    Returns:
        Updated PetCareState with lifestyle_enrichment_output:
        - lifestyle_enrichment_output: Dictionary with enrichment plan fields
        - wellness_path_outputs: Updated with lifestyle
        - error_occurred: Set to True if planning fails
        - error_messages: Appended with any errors
        - processing_stage: Updated to "enrichment_planned" or "enrichment_planning_failed"
    """
    logger.info("=" * 50)
    logger.info("EXECUTING LIFESTYLE ENRICHMENT NODE (WELLNESS PATH)")
    logger.info("=" * 50)
    
    try:
        # Log incoming state info
        logger.debug(f"State request_id: {state.get('request_id', 'N/A')}")
        logger.debug(f"Current path: {state.get('path_taken', 'unknown')}")
        logger.debug(f"Current processing stage: {state.get('processing_stage', 'unknown')}")
        
        # Verify we're on the wellness path
        if state.get('path_taken') != "WELLNESS_PATH":
            logger.warning(f"Lifestyle enrichment node executed on {state.get('path_taken')} path - this may be unexpected")
        
        # Check if profile extraction was successful
        if not state.get('profile_extraction_complete', False):
            error_msg = "Cannot generate enrichment plan: profile extraction not complete"
            logger.error(error_msg)
            state["error_occurred"] = True
            state["error_messages"].append(error_msg)
            state["processing_stage"] = "enrichment_planning_failed"
            return state
        
        # Extract profile from state
        profile = _build_profile_from_state(state)
        
        # Extract ML results from state
        ml_results = _build_ml_results_from_state(state)
        
        # Log enrichment context
        logger.info(f"Generating lifestyle enrichment plan for: "
                   f"Species={profile.get('pet_species', 'unknown')}, "
                   f"Name={profile.get('pet_name', 'Unknown')}, "
                   f"Age={profile.get('age_years', 0)}")
        
        risk_score = ml_results.get('health_risk_score', 0.2)
        logger.info(f"Health risk score: {risk_score:.3f} (Low risk - Wellness Path)")
        
        living_situation = profile.get('living_situation', 'unknown')
        logger.info(f"Living situation: {living_situation}")
        
        exercise_level = profile.get('exercise_level', 'moderate')
        logger.info(f"Exercise level: {exercise_level}")
        
        behavioral_issues = profile.get('behavioral_issues', [])
        if behavioral_issues:
            logger.info(f"Behavioral notes: {', '.join(behavioral_issues[:3])}")
        
        # Create LLM agent
        enrichment_agent = LifestyleEnrichmentAgent(client)
        logger.debug("LifestyleEnrichmentAgent created successfully")
        
        # Call agent to generate enrichment plan
        result = enrichment_agent.generate_enrichment_plan(profile, ml_results)
        
        # Process planning result
        if result.get("status") == "success":
            # Extract enrichment plan dictionary
            enrichment_plan = result.get("lifestyle_enrichment", {})
            
            logger.info(f"✅ Lifestyle enrichment plan generated successfully")
            logger.debug(f"Plan contains fields: {list(enrichment_plan.keys())}")
            
            # Store in state
            state["lifestyle_enrichment_output"] = enrichment_plan
            
            # Also store in wellness_path_outputs for aggregation
            if "wellness_path_outputs" not in state:
                state["wellness_path_outputs"] = {}
            state["wellness_path_outputs"]["lifestyle"] = enrichment_plan
            
            # Update processing stage
            state["processing_stage"] = "enrichment_planned"
            
            # Log summary of plan
            if "enrichment_overview" in enrichment_plan:
                logger.info(f"Overview: {enrichment_plan['enrichment_overview'][:100]}...")
            
            if "mental_stimulation" in enrichment_plan:
                mental = enrichment_plan["mental_stimulation"]
                if isinstance(mental, list):
                    logger.info(f"Mental stimulation activities: {len(mental)}")
                    for i, activity in enumerate(mental[:3]):  # Log first 3
                        logger.debug(f"  Mental {i+1}: {activity}")
            
            if "social_opportunities" in enrichment_plan:
                social = enrichment_plan["social_opportunities"]
                if isinstance(social, list):
                    logger.info(f"Social opportunities: {len(social)}")
                    for i, opportunity in enumerate(social[:2]):  # Log first 2
                        logger.debug(f"  Social {i+1}: {opportunity}")
            
            if "environmental_enrichment" in enrichment_plan:
                env = enrichment_plan["environmental_enrichment"]
                if isinstance(env, list):
                    logger.info(f"Environmental enrichment ideas: {len(env)}")
                    for i, idea in enumerate(env[:2]):  # Log first 2
                        logger.debug(f"  Environmental {i+1}: {idea}")
            
        else:
            # Planning failed
            error_msg = result.get("message", "Unknown enrichment planning error")
            logger.error(f"❌ Lifestyle enrichment plan generation failed: {error_msg}")
            
            state["error_occurred"] = True
            state["error_messages"].append(f"Enrichment planning failed: {error_msg}")
            state["processing_stage"] = "enrichment_planning_failed"
            
            # Store fallback plan if available
            if "lifestyle_enrichment" in result:
                state["lifestyle_enrichment_output"] = result["lifestyle_enrichment"]
                state["wellness_path_outputs"] = state.get("wellness_path_outputs", {})
                state["wellness_path_outputs"]["lifestyle"] = result["lifestyle_enrichment"]
                logger.info("Stored fallback enrichment plan")
        
        return state
        
    except Exception as e:
        # Handle unexpected errors
        error_msg = f"Unexpected error in lifestyle enrichment node: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Update state with error information
        state["error_occurred"] = True
        state["error_messages"].append(error_msg)
        state["processing_stage"] = "enrichment_planning_error"
        
        return state


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def _build_profile_from_state(state: PetCareState) -> Dict[str, Any]:
    """
    Build a complete profile dictionary from state fields for lifestyle enrichment.
    
    Args:
        state: Current PetCareState
        
    Returns:
        Dictionary with all profile fields needed for lifestyle enrichment
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
            "living_situation": state.get("living_situation", "unknown"),
            "exercise_level": state.get("exercise_level", "moderate"),
            
            # Behavioral Information
            "behavioral_issues": state.get("behavioral_issues", []),
        }
    else:
        # Ensure profile has all needed fields
        if "pet_name" not in profile:
            profile["pet_name"] = state.get("pet_name", "Your Pet")
        if "living_situation" not in profile:
            profile["living_situation"] = state.get("living_situation", "unknown")
        if "exercise_level" not in profile:
            profile["exercise_level"] = state.get("exercise_level", "moderate")
        if "behavioral_issues" not in profile:
            profile["behavioral_issues"] = state.get("behavioral_issues", [])
    
    # Ensure all required fields exist with defaults
    required_fields = {
        "pet_name": "Your Pet",
        "pet_species": "unknown",
        "breed": "unknown",
        "age_years": 0,
        "living_situation": "unknown",
        "exercise_level": "moderate",
        "behavioral_issues": []
    }
    
    for field, default in required_fields.items():
        if field not in profile or profile[field] is None:
            profile[field] = default
    
    return profile


def _build_ml_results_from_state(state: PetCareState) -> Dict[str, Any]:
    """
    Build ML results dictionary from state fields for lifestyle enrichment.
    
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

def test_lifestyle_enrichment_node():
    """
    Test function to verify node behavior.
    """
    from state import get_initial_state
    from utils.openai_client import build_openai_client
    
    print("=" * 60)
    print("TESTING LIFESTYLE ENRICHMENT NODE")
    print("=" * 60)
    
    try:
        # Initialize client
        client = build_openai_client()
        print("✅ OpenAI client created")
        
        # Test Case 1: Active dog in house with yard
        print("\n📝 Test Case 1: Active Dog in House with Yard")
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
        state1["living_situation"] = "house with yard"
        state1["exercise_level"] = "active"
        state1["behavioral_issues"] = []
        
        # Set ML results
        state1["health_risk_score"] = 0.12
        
        result1 = lifestyle_enrichment_node(state1, client)
        print(f"Enrichment plan generated: {'lifestyle_enrichment_output' in result1}")
        print(f"Stage: {result1.get('processing_stage')}")
        
        if 'lifestyle_enrichment_output' in result1:
            plan = result1['lifestyle_enrichment_output']
            print(f"Plan fields: {list(plan.keys())}")
            if 'enrichment_overview' in plan:
                print(f"Overview: {plan['enrichment_overview'][:100]}...")
            if 'mental_stimulation' in plan:
                mental = plan['mental_stimulation']
                print(f"Mental activities: {len(mental) if isinstance(mental, list) else 'Provided'}")
        
        # Test Case 2: Indoor cat in apartment
        print("\n📝 Test Case 2: Indoor Cat in Apartment")
        state2 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        state2["path_taken"] = "WELLNESS_PATH"
        state2["profile_extraction_complete"] = True
        state2["pet_name"] = "Whiskers"
        state2["pet_species"] = "cat"
        state2["breed"] = "domestic shorthair"
        state2["age_years"] = 4
        state2["living_situation"] = "apartment"
        state2["exercise_level"] = "moderate"
        state2["behavioral_issues"] = []
        state2["health_risk_score"] = 0.15
        
        result2 = lifestyle_enrichment_node(state2, client)
        print(f"Enrichment plan generated: {'lifestyle_enrichment_output' in result2}")
        
        # Test Case 3: Senior dog with mild anxiety
        print("\n📝 Test Case 3: Senior Dog with Mild Anxiety")
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
        state3["living_situation"] = "house"
        state3["exercise_level"] = "light"
        state3["behavioral_issues"] = ["mild anxiety during storms"]
        state3["health_risk_score"] = 0.28
        
        result3 = lifestyle_enrichment_node(state3, client)
        print(f"Enrichment plan generated: {'lifestyle_enrichment_output' in result3}")
        
        # Test Case 4: Rabbit
        print("\n📝 Test Case 4: Rabbit")
        state4 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        state4["path_taken"] = "WELLNESS_PATH"
        state4["profile_extraction_complete"] = True
        state4["pet_name"] = "Fluffy"
        state4["pet_species"] = "rabbit"
        state4["breed"] = "holland lop"
        state4["age_years"] = 2
        state4["living_situation"] = "indoor enclosure"
        state4["exercise_level"] = "moderate"
        state4["behavioral_issues"] = []
        state4["health_risk_score"] = 0.10
        
        result4 = lifestyle_enrichment_node(state4, client)
        print(f"Enrichment plan generated: {'lifestyle_enrichment_output' in result4}")
        
        # Test Case 5: Missing profile extraction
        print("\n📝 Test Case 5: Missing Profile Extraction")
        state5 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        state5["path_taken"] = "WELLNESS_PATH"
        state5["profile_extraction_complete"] = False  # Not complete
        
        result5 = lifestyle_enrichment_node(state5, client)
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
        result6 = lifestyle_enrichment_node(state6, client)
        print(f"Error occurred: {result6.get('error_occurred')}")
        print(f"Stage: {result6.get('processing_stage')}")
        
        return result1, result2, result3, result4, result5, result6
        
    except Exception as e:
        print(f"❌ Test error: {str(e)}")
        return None, None, None, None, None, None


if __name__ == "__main__":
    # Run test if executed directly
    test_lifestyle_enrichment_node()