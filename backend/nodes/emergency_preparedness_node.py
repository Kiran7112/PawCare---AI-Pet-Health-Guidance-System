# # nodes/emergency_preparedness_node.py
# """
# Emergency Preparedness Node for PawCare+ LangGraph workflow (Critical Path).
# Executes emergency preparedness planning using LLM as a workflow step.
# """

# import logging
# from typing import Dict, Any

# from state import PetCareState
# from agents.emergency_preparedness_llm import EmergencyPreparednessAgent
# from utils.openai_client import OpenAIClient

# logger = logging.getLogger(__name__)


# def emergency_preparedness_node(state: PetCareState, client: OpenAIClient) -> PetCareState:
#     """
#     Execute emergency preparedness planning as a workflow step in the critical path.
    
#     This node uses an LLM agent to generate a comprehensive emergency preparedness plan
#     for high-risk pets, including emergency contacts, first aid supplies, crisis procedures,
#     evacuation plans, and financial preparation.
    
#     Args:
#         state: Current PetCareState containing:
#             - Extracted profile fields
#             - Health risk analysis from previous node
#             - Must be on CRITICAL_CARE_PATH
#         client: OpenAIClient instance for LLM calls
        
#     Returns:
#         Updated PetCareState with emergency_prep_output:
#         - emergency_prep_output: Dictionary with emergency plan fields
#         - critical_path_outputs: Updated with emergency_prep
#         - error_occurred: Set to True if planning fails
#         - error_messages: Appended with any errors
#         - processing_stage: Updated to "emergency_planned" or "emergency_planning_failed"
#     """
#     logger.info("=" * 50)
#     logger.info("EXECUTING EMERGENCY PREPAREDNESS NODE (CRITICAL PATH)")
#     logger.info("=" * 50)
    
#     try:
#         # Log incoming state info
#         logger.debug(f"State request_id: {state.get('request_id', 'N/A')}")
#         logger.debug(f"Current path: {state.get('path_taken', 'unknown')}")
#         logger.debug(f"Current processing stage: {state.get('processing_stage', 'unknown')}")
        
#         # Verify we're on the critical path
#         if state.get('path_taken') != "CRITICAL_CARE_PATH":
#             logger.warning(f"Emergency preparedness node executed on {state.get('path_taken')} path - this may be unexpected")
        
#         # Check if profile extraction was successful
#         if not state.get('profile_extraction_complete', False):
#             error_msg = "Cannot generate emergency plan: profile extraction not complete"
#             logger.error(error_msg)
#             state["error_occurred"] = True
#             state["error_messages"].append(error_msg)
#             state["processing_stage"] = "emergency_planning_failed"
#             return state
        
#         # Check if risk analysis is available
#         health_analysis = state.get('health_risk_analysis_output', {})
#         if not health_analysis:
#             logger.warning("Health risk analysis not found, emergency plan may be less specific")
        
#         # Extract profile from state
#         profile = _build_profile_from_state(state)
        
#         # Log planning context
#         logger.info(f"Generating emergency plan for: "
#                    f"Species={profile.get('pet_species', 'unknown')}, "
#                    f"Name={profile.get('pet_name', 'Unknown')}, "
#                    f"Conditions={len(profile.get('known_conditions', []))}")
        
#         if health_analysis:
#             risk_factors = health_analysis.get('critical_risk_factors', [])
#             logger.info(f"Based on {len(risk_factors)} identified risk factors")
        
#         # Create LLM agent
#         planner = EmergencyPreparednessAgent(client)
#         logger.debug("EmergencyPreparednessAgent created successfully")
        
#         # Call agent to generate emergency plan
#         result = planner.generate_emergency_plan(profile, health_analysis)
        
#         # Process planning result
#         if result.get("status") == "success":
#             # Extract emergency plan dictionary
#             emergency_plan = result.get("emergency_preparedness", {})
            
#             logger.info(f"✅ Emergency plan generated successfully")
#             logger.debug(f"Plan contains fields: {list(emergency_plan.keys())}")
            
#             # Store in state
#             state["emergency_prep_output"] = emergency_plan
            
#             # Also store in critical_path_outputs for aggregation
#             if "critical_path_outputs" not in state:
#                 state["critical_path_outputs"] = {}
#             state["critical_path_outputs"]["emergency_prep"] = emergency_plan
            
#             # Update processing stage
#             state["processing_stage"] = "emergency_planned"
            
#             # Log summary of plan
#             if "emergency_overview" in emergency_plan:
#                 logger.info(f"Overview: {emergency_plan['emergency_overview'][:100]}...")
            
#             if "emergency_contacts" in emergency_plan:
#                 contacts = emergency_plan["emergency_contacts"]
#                 logger.info(f"Emergency contacts: {len(contacts) if isinstance(contacts, list) else 'Provided'}")
            
#             if "first_aid_supplies" in emergency_plan:
#                 supplies = emergency_plan["first_aid_supplies"]
#                 logger.info(f"First aid supplies: {len(supplies) if isinstance(supplies, list) else 'Listed'}")
            
#             # Extract and log crisis procedures if available
#             if "crisis_procedures" in emergency_plan:
#                 procedures = emergency_plan["crisis_procedures"]
#                 if isinstance(procedures, list):
#                     logger.info(f"Crisis procedures: {len(procedures)} specific protocols")
#                     for i, proc in enumerate(procedures[:2]):  # Log first 2
#                         logger.debug(f"  Protocol {i+1}: {proc[:100]}...")
            
#         else:
#             # Planning failed
#             error_msg = result.get("message", "Unknown emergency planning error")
#             logger.error(f"❌ Emergency plan generation failed: {error_msg}")
            
#             state["error_occurred"] = True
#             state["error_messages"].append(f"Emergency planning failed: {error_msg}")
#             state["processing_stage"] = "emergency_planning_failed"
            
#             # Store fallback plan if available
#             if "emergency_preparedness" in result:
#                 state["emergency_prep_output"] = result["emergency_preparedness"]
#                 state["critical_path_outputs"] = state.get("critical_path_outputs", {})
#                 state["critical_path_outputs"]["emergency_prep"] = result["emergency_preparedness"]
#                 logger.info("Stored fallback emergency plan")
        
#         return state
        
#     except Exception as e:
#         # Handle unexpected errors
#         error_msg = f"Unexpected error in emergency preparedness node: {str(e)}"
#         logger.error(error_msg, exc_info=True)
        
#         # Update state with error information
#         state["error_occurred"] = True
#         state["error_messages"].append(error_msg)
#         state["processing_stage"] = "emergency_planning_error"
        
#         return state


# # ==========================================
# # HELPER FUNCTIONS
# # ==========================================

# def _build_profile_from_state(state: PetCareState) -> Dict[str, Any]:
#     """
#     Build a complete profile dictionary from state fields for emergency planning.
    
#     Args:
#         state: Current PetCareState
        
#     Returns:
#         Dictionary with all profile fields needed for emergency planning
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
#             "weight_status": state.get("weight_status", "unknown"),
            
#             # Medical Information
#             "known_conditions": state.get("known_conditions", []),
#             "medications_current": state.get("medications_current", []),
#             "allergies_known": state.get("allergies_known", []),
            
#             # Lifestyle Information
#             "living_situation": state.get("living_situation", "unknown"),
#         }
#     else:
#         # Ensure profile has all needed fields
#         if "pet_name" not in profile:
#             profile["pet_name"] = state.get("pet_name", "Your Pet")
#         if "medications_current" not in profile:
#             profile["medications_current"] = state.get("medications_current", [])
#         if "allergies_known" not in profile:
#             profile["allergies_known"] = state.get("allergies_known", [])
#         if "living_situation" not in profile:
#             profile["living_situation"] = state.get("living_situation", "unknown")
    
#     # Ensure all required fields exist with defaults
#     required_fields = {
#         "pet_name": "Your Pet",
#         "pet_species": "unknown",
#         "breed": "unknown",
#         "age_years": 0,
#         "weight_status": "unknown",
#         "known_conditions": [],
#         "medications_current": [],
#         "allergies_known": [],
#         "living_situation": "unknown"
#     }
    
#     for field, default in required_fields.items():
#         if field not in profile or profile[field] is None:
#             profile[field] = default
    
#     return profile


# # ==========================================
# # OPTIONAL: Helper function for testing
# # ==========================================

# def test_emergency_preparedness_node():
#     """
#     Test function to verify node behavior.
#     """
#     from state import get_initial_state
#     from utils.openai_client import build_openai_client
    
#     print("=" * 60)
#     print("TESTING EMERGENCY PREPAREDNESS NODE")
#     print("=" * 60)
    
#     try:
#         # Initialize client
#         client = build_openai_client()
#         print("✅ OpenAI client created")
        
#         # Test Case 1: Complete profile with health analysis
#         print("\n📝 Test Case 1: Complete Profile with Health Analysis")
#         state1 = get_initial_state({
#             "about_pet": "Test",
#             "daily_routine": "Test",
#             "health_concerns": "Test"
#         })
        
#         # Set up state
#         state1["path_taken"] = "CRITICAL_CARE_PATH"
#         state1["profile_extraction_complete"] = True
        
#         # Set profile fields
#         state1["pet_name"] = "Max"
#         state1["pet_species"] = "dog"
#         state1["breed"] = "labrador"
#         state1["age_years"] = 12
#         state1["weight_status"] = "overweight"
#         state1["living_situation"] = "house"
#         state1["known_conditions"] = ["diabetes", "arthritis", "heart murmur"]
#         state1["medications_current"] = ["insulin", "carprofen"]
#         state1["allergies_known"] = ["pollen"]
        
#         # Set health analysis
#         state1["health_risk_analysis_output"] = {
#             "critical_risk_factors": [
#                 "Advanced age with multiple chronic conditions",
#                 "Diabetes requiring insulin management",
#                 "Heart murmur indicating possible cardiac disease"
#             ],
#             "warning_signs": [
#                 "Difficulty breathing",
#                 "Collapse",
#                 "Severe lethargy"
#             ],
#             "urgency_timeline": "Seek emergency care within 24 hours"
#         }
        
#         result1 = emergency_preparedness_node(state1, client)
#         print(f"Emergency plan generated: {'emergency_prep_output' in result1}")
#         print(f"Stage: {result1.get('processing_stage')}")
        
#         if 'emergency_prep_output' in result1:
#             plan = result1['emergency_prep_output']
#             print(f"Plan fields: {list(plan.keys())}")
#             if 'emergency_overview' in plan:
#                 print(f"Overview: {plan['emergency_overview'][:100]}...")
#             if 'emergency_contacts' in plan:
#                 contacts = plan['emergency_contacts']
#                 print(f"Contacts: {len(contacts) if isinstance(contacts, list) else 'Provided'}")
        
#         # Test Case 2: Missing health analysis
#         print("\n📝 Test Case 2: Missing Health Analysis")
#         state2 = get_initial_state({
#             "about_pet": "Test",
#             "daily_routine": "Test",
#             "health_concerns": "Test"
#         })
        
#         state2["path_taken"] = "CRITICAL_CARE_PATH"
#         state2["profile_extraction_complete"] = True
#         state2["pet_species"] = "cat"
#         state2["pet_name"] = "Whiskers"
#         state2["known_conditions"] = ["kidney disease"]
#         # No health_risk_analysis_output
        
#         result2 = emergency_preparedness_node(state2, client)
#         print(f"Emergency plan generated: {'emergency_prep_output' in result2}")
#         print(f"Stage: {result2.get('processing_stage')}")
        
#         # Test Case 3: Missing profile extraction
#         print("\n📝 Test Case 3: Missing Profile Extraction")
#         state3 = get_initial_state({
#             "about_pet": "Test",
#             "daily_routine": "Test",
#             "health_concerns": "Test"
#         })
        
#         state3["path_taken"] = "CRITICAL_CARE_PATH"
#         state3["profile_extraction_complete"] = False  # Not complete
        
#         result3 = emergency_preparedness_node(state3, client)
#         print(f"Error occurred: {result3.get('error_occurred')}")
#         print(f"Error messages: {result3.get('error_messages')}")
        
#         # Test Case 4: Error handling
#         print("\n📝 Test Case 4: Error Handling")
#         state4 = get_initial_state({
#             "about_pet": "Test",
#             "daily_routine": "Test",
#             "health_concerns": "Test"
#         })
        
#         # Cause an error by not having required data
#         result4 = emergency_preparedness_node(state4, client)
#         print(f"Error occurred: {result4.get('error_occurred')}")
#         print(f"Stage: {result4.get('processing_stage')}")
        
#         return result1, result2, result3, result4
        
#     except Exception as e:
#         print(f"❌ Test error: {str(e)}")
#         return None, None, None, None


# if __name__ == "__main__":
#     # Run test if executed directly
#     test_emergency_preparedness_node()

# nodes/emergency_preparedness_node.py
"""
Emergency Preparedness Node for PawCare+ LangGraph workflow (Critical Path).
Executes emergency preparedness planning using LLM as a workflow step.
"""

import logging
from typing import Dict, Any

from state import PetCareState
from agents.emergency_preparedness_llm import EmergencyPreparednessAgent
from utils.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


def emergency_preparedness_node(state: PetCareState, client: OpenAIClient) -> PetCareState:
    """
    Execute emergency preparedness planning as a workflow step in the critical path.
    """
    logger.info("=" * 50)
    logger.info("EXECUTING EMERGENCY PREPAREDNESS NODE (CRITICAL PATH)")
    logger.info("=" * 50)
    
    try:
        # Log incoming state info
        logger.debug(f"State request_id: {state.get('request_id', 'N/A')}")
        logger.debug(f"Current path: {state.get('path_taken', 'unknown')}")
        logger.debug(f"Current processing stage: {state.get('processing_stage', 'unknown')}")
        
        # Verify we're on the critical path
        if state.get('path_taken') != "CRITICAL_CARE_PATH":
            logger.warning(f"Emergency preparedness node executed on {state.get('path_taken')} path - this may be unexpected")
        
        # Check if profile extraction was successful
        if not state.get('profile_extraction_complete', False):
            error_msg = "Cannot generate emergency plan: profile extraction not complete"
            logger.error(error_msg)
            state["error_occurred"] = True
            state["error_messages"].append(error_msg)
            state["processing_stage"] = "emergency_planning_failed"
            return state
        
        # Check if risk analysis is available
        health_analysis = state.get('health_risk_analysis_output', {})
        if not health_analysis:
            logger.warning("Health risk analysis not found, emergency plan may be less specific")
        
        # Extract profile from state
        profile = _build_profile_from_state(state)
        
        # Log planning context
        logger.info(f"Generating emergency plan for: "
                   f"Species={profile.get('pet_species', 'unknown')}, "
                   f"Name={profile.get('pet_name', 'Unknown')}, "
                   f"Conditions={len(profile.get('known_conditions', []))}")
        
        if health_analysis:
            risk_factors = health_analysis.get('critical_risk_factors', [])
            logger.info(f"Based on {len(risk_factors)} identified risk factors")
        
        # Create LLM agent
        planner = EmergencyPreparednessAgent(client)
        logger.debug("EmergencyPreparednessAgent created successfully")
        
        # Call agent to generate emergency plan
        result = planner.generate_emergency_plan(profile, health_analysis)
        
        # Process planning result
        if result.get("status") == "success":
            # Extract emergency plan dictionary
            emergency_plan = result.get("emergency_preparedness", {})
            
            logger.info(f"✅ Emergency plan generated successfully")
            logger.debug(f"Plan contains fields: {list(emergency_plan.keys())}")
            
            # Store in state
            state["emergency_prep_output"] = emergency_plan
            
            # Also store in critical_path_outputs for aggregation
            if "critical_path_outputs" not in state:
                state["critical_path_outputs"] = {}
            state["critical_path_outputs"]["emergency_prep"] = emergency_plan
            
            # Update processing stage
            state["processing_stage"] = "emergency_planned"
            
            # Log summary of plan - FIXED: Handle different data types
            if "emergency_overview" in emergency_plan:
                overview = emergency_plan["emergency_overview"]
                if isinstance(overview, str):
                    logger.info(f"Overview: {overview[:100]}...")
                elif isinstance(overview, dict):
                    logger.info(f"Overview: {str(overview)[:100]}...")
                else:
                    logger.info(f"Overview: {overview}")
            
            if "emergency_contacts" in emergency_plan:
                contacts = emergency_plan["emergency_contacts"]
                if isinstance(contacts, list):
                    logger.info(f"Emergency contacts: {len(contacts)}")
                else:
                    logger.info(f"Emergency contacts: Provided")
            
            if "first_aid_supplies" in emergency_plan:
                supplies = emergency_plan["first_aid_supplies"]
                if isinstance(supplies, list):
                    logger.info(f"First aid supplies: {len(supplies)} items")
                else:
                    logger.info(f"First aid supplies: Listed")
            
            # Extract and log crisis procedures if available
            if "crisis_procedures" in emergency_plan:
                procedures = emergency_plan["crisis_procedures"]
                if isinstance(procedures, list):
                    logger.info(f"Crisis procedures: {len(procedures)} specific protocols")
                    for i, proc in enumerate(procedures[:2]):  # Log first 2
                        if isinstance(proc, str):
                            logger.debug(f"  Protocol {i+1}: {proc[:100]}...")
                        else:
                            logger.debug(f"  Protocol {i+1}: {proc}")
            
        else:
            # Planning failed
            error_msg = result.get("message", "Unknown emergency planning error")
            logger.error(f"❌ Emergency plan generation failed: {error_msg}")
            
            state["error_occurred"] = True
            state["error_messages"].append(f"Emergency planning failed: {error_msg}")
            state["processing_stage"] = "emergency_planning_failed"
            
            # Store fallback plan if available
            if "emergency_preparedness" in result:
                state["emergency_prep_output"] = result["emergency_preparedness"]
                state["critical_path_outputs"] = state.get("critical_path_outputs", {})
                state["critical_path_outputs"]["emergency_prep"] = result["emergency_preparedness"]
                logger.info("Stored fallback emergency plan")
        
        return state
        
    except Exception as e:
        # Handle unexpected errors
        error_msg = f"Unexpected error in emergency preparedness node: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Update state with error information
        state["error_occurred"] = True
        state["error_messages"].append(error_msg)
        state["processing_stage"] = "emergency_planning_error"
        
        return state


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def _build_profile_from_state(state: PetCareState) -> Dict[str, Any]:
    """
    Build a complete profile dictionary from state fields for emergency planning.
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
            "weight_status": state.get("weight_status", "unknown"),
            "known_conditions": state.get("known_conditions", []),
            "medications_current": state.get("medications_current", []),
            "allergies_known": state.get("allergies_known", []),
            "living_situation": state.get("living_situation", "unknown"),
        }
    else:
        # Ensure profile has all needed fields
        if "pet_name" not in profile:
            profile["pet_name"] = state.get("pet_name", "Your Pet")
        if "medications_current" not in profile:
            profile["medications_current"] = state.get("medications_current", [])
        if "allergies_known" not in profile:
            profile["allergies_known"] = state.get("allergies_known", [])
        if "living_situation" not in profile:
            profile["living_situation"] = state.get("living_situation", "unknown")
    
    # Ensure all required fields exist with defaults
    required_fields = {
        "pet_name": "Your Pet",
        "pet_species": "unknown",
        "breed": "unknown",
        "age_years": 0,
        "weight_status": "unknown",
        "known_conditions": [],
        "medications_current": [],
        "allergies_known": [],
        "living_situation": "unknown"
    }
    
    for field, default in required_fields.items():
        if field not in profile or profile[field] is None:
            profile[field] = default
    
    return profile


# ==========================================
# OPTIONAL: Helper function for testing
# ==========================================

def test_emergency_preparedness_node():
    """
    Test function to verify node behavior.
    """
    from state import get_initial_state
    from utils.openai_client import build_openai_client
    
    print("=" * 60)
    print("TESTING EMERGENCY PREPAREDNESS NODE")
    print("=" * 60)
    
    try:
        # Initialize client
        client = build_openai_client()
        print("✅ OpenAI client created")
        
        # Test Case 1: Complete profile with health analysis
        print("\n📝 Test Case 1: Complete Profile with Health Analysis")
        state1 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        state1["path_taken"] = "CRITICAL_CARE_PATH"
        state1["profile_extraction_complete"] = True
        state1["pet_name"] = "Max"
        state1["pet_species"] = "dog"
        state1["breed"] = "labrador"
        state1["age_years"] = 12
        state1["weight_status"] = "overweight"
        state1["living_situation"] = "house"
        state1["known_conditions"] = ["diabetes", "arthritis", "heart murmur"]
        state1["medications_current"] = ["insulin", "carprofen"]
        state1["allergies_known"] = ["pollen"]
        
        state1["health_risk_analysis_output"] = {
            "critical_risk_factors": [
                "Advanced age with multiple chronic conditions",
                "Diabetes requiring insulin management",
                "Heart murmur indicating possible cardiac disease"
            ],
            "warning_signs": [
                "Difficulty breathing",
                "Collapse",
                "Severe lethargy"
            ],
            "urgency_timeline": "Seek emergency care within 24 hours"
        }
        
        result1 = emergency_preparedness_node(state1, client)
        print(f"Emergency plan generated: {'emergency_prep_output' in result1}")
        print(f"Stage: {result1.get('processing_stage')}")
        
        if 'emergency_prep_output' in result1:
            plan = result1['emergency_prep_output']
            print(f"Plan fields: {list(plan.keys())}")
            if 'emergency_overview' in plan:
                overview = plan['emergency_overview']
                if isinstance(overview, str):
                    print(f"Overview: {overview[:100]}...")
                else:
                    print(f"Overview: {overview}")
            if 'emergency_contacts' in plan:
                contacts = plan['emergency_contacts']
                print(f"Contacts: {len(contacts) if isinstance(contacts, list) else 'Provided'}")
        
        # Test Case 2: Missing health analysis
        print("\n📝 Test Case 2: Missing Health Analysis")
        state2 = get_initial_state({
            "about_pet": "Test",
            "daily_routine": "Test",
            "health_concerns": "Test"
        })
        
        state2["path_taken"] = "CRITICAL_CARE_PATH"
        state2["profile_extraction_complete"] = True
        state2["pet_species"] = "cat"
        state2["pet_name"] = "Whiskers"
        state2["known_conditions"] = ["kidney disease"]
        
        result2 = emergency_preparedness_node(state2, client)
        print(f"Emergency plan generated: {'emergency_prep_output' in result2}")
        print(f"Stage: {result2.get('processing_stage')}")
        
        return result1, result2
        
    except Exception as e:
        print(f"❌ Test error: {str(e)}")
        return None, None


if __name__ == "__main__":
    test_emergency_preparedness_node()