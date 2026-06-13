# # nodes/pet_profile_extractor_node.py
# """
# Pet Profile Extractor Node for PawCare+ LangGraph workflow.
# Executes profile extraction using LLM as the second step in the workflow.
# """

# import logging
# import sys
# from typing import Dict, Any

# from state import PetCareState
# from utils.openai_client import OpenAIClient

# # Import with error handling
# try:
#     from agents.pet_profile_extractor_llm import PetProfileExtractorAgent
#     AGENT_AVAILABLE = True
# except ImportError as e:
#     logging.error(f"Failed to import PetProfileExtractorAgent: {e}")
#     AGENT_AVAILABLE = False
#     PetProfileExtractorAgent = None

# logger = logging.getLogger(__name__)


# def pet_profile_extractor_node(state: PetCareState, client: OpenAIClient) -> PetCareState:
#     """
#     Execute profile extraction as the second workflow step.
    
#     This node uses an LLM agent to extract 17 structured profile fields from
#     the three user input fields (about_pet, daily_routine, health_concerns).
    
#     Args:
#         state: Current PetCareState containing validated user inputs
#         client: OpenAIClient instance for LLM calls
        
#     Returns:
#         Updated PetCareState with extracted profile:
#         - extracted_profile: Complete profile dictionary
#         - profile_extraction_complete: Boolean indicating success
#         - 17 individual profile fields flattened into state
#         - error_occurred: Set to True if extraction fails
#         - error_messages: Appended with any errors
#         - processing_stage: Updated to "extraction_complete" or "extraction_failed"
#     """
#     logger.info("=" * 50)
#     logger.info("EXECUTING PET PROFILE EXTRACTOR NODE")
#     logger.info("=" * 50)
    
#     # Default profile to use as fallback
#     default_profile = {
#         "pet_species": "unknown",
#         "breed": "unknown",
#         "age_years": 0,
#         "weight_status": "unknown",
#         "sex": "unknown",
#         "known_conditions": [],
#         "past_surgeries": [],
#         "allergies_known": [],
#         "medications_current": [],
#         "living_situation": "unknown",
#         "exercise_level": "unknown",
#         "diet_type": "unknown",
#         "diet_quality": "unknown",
#         "behavioral_issues": [],
#         "owner_experience": "unknown",
#         "vet_access": "unknown",
#         "owner_commitment": "unknown"
#     }
    
#     try:
#         # Check if agent is available
#         if not AGENT_AVAILABLE or PetProfileExtractorAgent is None:
#             error_msg = "PetProfileExtractorAgent could not be imported. Check your imports and ensure no circular dependencies."
#             logger.error(error_msg)
#             state["error_occurred"] = True
#             state["error_messages"].append(error_msg)
#             state["profile_extraction_complete"] = False
#             state["extraction_failed"] = True
#             state["extracted_profile"] = default_profile
#             state["processing_stage"] = "extraction_failed"
#             return state
        
#         # Log incoming state info
#         logger.debug(f"State request_id: {state.get('request_id', 'N/A')}")
#         logger.debug(f"Current processing stage: {state.get('processing_stage', 'unknown')}")
        
#         # Validate that we have inputs to extract
#         about_pet = state.get('about_pet', '')
#         daily_routine = state.get('daily_routine', '')
#         health_concerns = state.get('health_concerns', '')
        
#         if not about_pet or not daily_routine or not health_concerns:
#             error_msg = "Missing required input fields for profile extraction"
#             logger.error(error_msg)
#             state["error_occurred"] = True
#             state["error_messages"].append(error_msg)
#             state["profile_extraction_complete"] = False
#             state["extraction_failed"] = True
#             state["extracted_profile"] = default_profile
#             state["processing_stage"] = "extraction_failed"
#             return state
        
#         logger.info(f"Extracting profile from - about_pet: {len(about_pet)} chars, "
#                    f"daily_routine: {len(daily_routine)} chars, "
#                    f"health_concerns: {len(health_concerns)} chars")
        
#         # Create profile extractor agent
#         try:
#             extractor = PetProfileExtractorAgent(client)
#             logger.debug("PetProfileExtractorAgent created successfully")
#         except Exception as e:
#             error_msg = f"Failed to create PetProfileExtractorAgent: {str(e)}"
#             logger.error(error_msg)
#             state["error_occurred"] = True
#             state["error_messages"].append(error_msg)
#             state["profile_extraction_complete"] = False
#             state["extraction_failed"] = True
#             state["extracted_profile"] = default_profile
#             state["processing_stage"] = "extraction_failed"
#             return state
        
#         # Prepare input dictionary
#         raw_inputs = {
#             "about_pet": about_pet,
#             "daily_routine": daily_routine,
#             "health_concerns": health_concerns
#         }
        
#         # Call agent to extract profile
#         try:
#             result = extractor.extract_pet_profile(raw_inputs)
#         except Exception as e:
#             error_msg = f"Error during profile extraction: {str(e)}"
#             logger.error(error_msg)
#             state["error_occurred"] = True
#             state["error_messages"].append(error_msg)
#             state["profile_extraction_complete"] = False
#             state["extraction_failed"] = True
#             state["extracted_profile"] = default_profile
#             state["processing_stage"] = "extraction_failed"
#             return state
        
#         # Process extraction result
#         if result and result.get("status") == "success":
#             # Extract the profile dictionary
#             profile = result.get("extracted_profile", default_profile)
            
#             logger.info(f"✅ Profile extraction successful with confidence "
#                        f"{result.get('extraction_confidence', 0):.2f}")
#             logger.debug(f"Extracted {len(profile)} fields")
            
#             # Store the complete profile dictionary
#             state["extracted_profile"] = profile
            
#             # Flatten all 17 profile fields into state for easy access
#             # Basic Information
#             state["pet_species"] = profile.get("pet_species", "unknown")
#             state["breed"] = profile.get("breed", "unknown")
#             state["age_years"] = profile.get("age_years", 0)
#             state["weight_status"] = profile.get("weight_status", "unknown")
#             state["sex"] = profile.get("sex", "unknown")
            
#             # Medical Information
#             state["known_conditions"] = profile.get("known_conditions", [])
#             state["past_surgeries"] = profile.get("past_surgeries", [])
#             state["allergies_known"] = profile.get("allergies_known", [])
#             state["medications_current"] = profile.get("medications_current", [])
            
#             # Lifestyle Information
#             state["living_situation"] = profile.get("living_situation", "unknown")
#             state["exercise_level"] = profile.get("exercise_level", "unknown")
#             state["diet_type"] = profile.get("diet_type", "unknown")
#             state["diet_quality"] = profile.get("diet_quality", "unknown")
#             state["behavioral_issues"] = profile.get("behavioral_issues", [])
            
#             # Owner Information
#             state["owner_experience"] = profile.get("owner_experience", "unknown")
#             state["vet_access"] = profile.get("vet_access", "unknown")
#             state["owner_commitment"] = profile.get("owner_commitment", "unknown")
            
#             # Set completion flags
#             state["profile_extraction_complete"] = True
#             state["extraction_confidence"] = result.get("extraction_confidence", 0.0)
#             state["extraction_failed"] = False
#             state["processing_stage"] = "extraction_complete"
            
#             # Log extracted fields (summary)
#             logger.info(f"Extracted - Species: {state['pet_species']}, "
#                        f"Breed: {state['breed']}, Age: {state['age_years']}")
#             logger.info(f"Conditions: {len(state['known_conditions'])} found")
            
#         else:
#             # Extraction failed
#             error_msg = result.get("message", "Unknown extraction error") if result else "No result returned"
#             logger.error(f"❌ Profile extraction failed: {error_msg}")
            
#             state["error_occurred"] = True
#             state["error_messages"].append(f"Profile extraction failed: {error_msg}")
#             state["profile_extraction_complete"] = False
#             state["extraction_failed"] = True
#             state["extracted_profile"] = default_profile
#             state["processing_stage"] = "extraction_failed"
            
#         return state
        
#     except Exception as e:
#         # Handle unexpected errors
#         error_msg = f"Unexpected error in pet profile extractor node: {str(e)}"
#         logger.error(error_msg, exc_info=True)
        
#         # Update state with error information
#         state["error_occurred"] = True
#         state["error_messages"].append(error_msg)
#         state["profile_extraction_complete"] = False
#         state["extraction_failed"] = True
#         state["extracted_profile"] = default_profile
#         state["processing_stage"] = "extraction_error"
        
#         return state


# # ==========================================
# # OPTIONAL: Helper function for testing
# # ==========================================

# def test_pet_profile_extractor_node():
#     """
#     Test function to verify node behavior.
#     """
#     from state import get_initial_state
#     from utils.openai_client import build_openai_client
    
#     print("=" * 60)
#     print("TESTING PET PROFILE EXTRACTOR NODE")
#     print("=" * 60)
    
#     try:
#         # Initialize client
#         client = build_openai_client()
#         print("✅ OpenAI client created")
        
#         # Test Case 1: Valid inputs
#         print("\n📝 Test Case 1: Valid Inputs")
#         state1 = get_initial_state({
#             "about_pet": "My dog is a 5-year-old Labrador Retriever named Max. He's male, not neutered, weighs about 30kg.",
#             "daily_routine": "He eats twice daily (kibble), walks for 30 minutes twice a day, and sleeps indoors.",
#             "health_concerns": "Recently noticed increased thirst and lethargy. No other symptoms."
#         })
        
#         result1 = pet_profile_extractor_node(state1, client)
#         print(f"Extraction complete: {result1['profile_extraction_complete']}")
#         print(f"Species: {result1.get('pet_species')}")
#         print(f"Breed: {result1.get('breed')}")
#         print(f"Age: {result1.get('age_years')}")
#         print(f"Conditions: {result1.get('known_conditions')}")
#         print(f"Stage: {result1['processing_stage']}")
        
#         # Test Case 2: Minimal inputs
#         print("\n📝 Test Case 2: Minimal Inputs")
#         state2 = get_initial_state({
#             "about_pet": "My cat is 3 years old.",
#             "daily_routine": "Indoor cat, eats dry food.",
#             "health_concerns": "None."
#         })
        
#         result2 = pet_profile_extractor_node(state2, client)
#         print(f"Extraction complete: {result2['profile_extraction_complete']}")
#         print(f"Species: {result2.get('pet_species')}")
#         print(f"Age: {result2.get('age_years')}")
#         print(f"Confidence: {result2.get('extraction_confidence', 0):.2f}")
        
#         # Test Case 3: Missing inputs (should fail gracefully)
#         print("\n📝 Test Case 3: Missing Inputs")
#         state3 = get_initial_state({
#             "about_pet": "",
#             "daily_routine": "Walks daily",
#             "health_concerns": ""
#         })
        
#         result3 = pet_profile_extractor_node(state3, client)
#         print(f"Extraction complete: {result3['profile_extraction_complete']}")
#         print(f"Error occurred: {result3['error_occurred']}")
#         print(f"Error messages: {result3['error_messages']}")
        
#         return result1, result2, result3
        
#     except Exception as e:
#         print(f"❌ Test error: {str(e)}")
#         return None, None, None


# if __name__ == "__main__":
#     # Run test if executed directly
#     test_pet_profile_extractor_node()


# nodes/pet_profile_extractor_node.py
"""
Pet Profile Extractor Node for PawCare+ LangGraph workflow.
Executes profile extraction using LLM as the second step in the workflow.
"""

import logging
from typing import Dict, Any

from state import PetCareState
from utils.openai_client import OpenAIClient

# Import with error handling
try:
    from agents.pet_profile_extractor_llm import PetProfileExtractorAgent
    AGENT_AVAILABLE = True
except ImportError as e:
    logging.error(f"Failed to import PetProfileExtractorAgent: {e}")
    AGENT_AVAILABLE = False
    PetProfileExtractorAgent = None

logger = logging.getLogger(__name__)


# Default profile with 20 fields
DEFAULT_PROFILE = {
    "pet_species": "unknown",
    "breed": "unknown",
    "age_years": 0,
    "weight_status": "unknown",
    "sex": "unknown",
    "known_conditions": [],
    "past_surgeries": [],
    "allergies_known": [],
    "medications_current": [],
    "living_situation": "unknown",
    "exercise_level": "unknown",
    "diet_type": "unknown",
    "diet_quality": "unknown",
    "behavioral_issues": [],
    "owner_experience": "unknown",
    "vet_access": "unknown",
    "owner_commitment": "unknown",
    "recent_symptoms": [],
    "symptom_duration_days": 0,
    "symptom_severity": "unknown"
}


def pet_profile_extractor_node(state: PetCareState, client: OpenAIClient) -> PetCareState:
    """
    Execute profile extraction as the second workflow step.
    
    This node uses an LLM agent to extract 20 structured profile fields from
    the three user input fields (about_pet, daily_routine, health_concerns).
    """
    logger.info("=" * 50)
    logger.info("EXECUTING PET PROFILE EXTRACTOR NODE")
    logger.info("=" * 50)
    
    try:
        # Check if agent is available
        if not AGENT_AVAILABLE or PetProfileExtractorAgent is None:
            error_msg = "PetProfileExtractorAgent could not be imported."
            logger.error(error_msg)
            state["error_occurred"] = True
            state["error_messages"].append(error_msg)
            state["profile_extraction_complete"] = False
            state["extraction_failed"] = True
            state["extracted_profile"] = DEFAULT_PROFILE
            state["processing_stage"] = "extraction_failed"
            return state
        
        # Log incoming state info
        logger.debug(f"State request_id: {state.get('request_id', 'N/A')}")
        logger.debug(f"Current processing stage: {state.get('processing_stage', 'unknown')}")
        
        # Validate that we have inputs to extract
        about_pet = state.get('about_pet', '')
        daily_routine = state.get('daily_routine', '')
        health_concerns = state.get('health_concerns', '')
        
        if not about_pet or not daily_routine or not health_concerns:
            error_msg = "Missing required input fields for profile extraction"
            logger.error(error_msg)
            state["error_occurred"] = True
            state["error_messages"].append(error_msg)
            state["profile_extraction_complete"] = False
            state["extraction_failed"] = True
            state["extracted_profile"] = DEFAULT_PROFILE
            state["processing_stage"] = "extraction_failed"
            return state
        
        logger.info(f"Extracting profile from inputs")
        
        # Create profile extractor agent
        try:
            extractor = PetProfileExtractorAgent(client)
            logger.debug("PetProfileExtractorAgent created successfully")
        except Exception as e:
            error_msg = f"Failed to create PetProfileExtractorAgent: {str(e)}"
            logger.error(error_msg)
            state["error_occurred"] = True
            state["error_messages"].append(error_msg)
            state["profile_extraction_complete"] = False
            state["extraction_failed"] = True
            state["extracted_profile"] = DEFAULT_PROFILE
            state["processing_stage"] = "extraction_failed"
            return state
        
        # Prepare input dictionary
        raw_inputs = {
            "about_pet": about_pet,
            "daily_routine": daily_routine,
            "health_concerns": health_concerns
        }
        
        # Call agent to extract profile
        try:
            result = extractor.extract_pet_profile(raw_inputs)
        except Exception as e:
            error_msg = f"Error during profile extraction: {str(e)}"
            logger.error(error_msg)
            state["error_occurred"] = True
            state["error_messages"].append(error_msg)
            state["profile_extraction_complete"] = False
            state["extraction_failed"] = True
            state["extracted_profile"] = DEFAULT_PROFILE
            state["processing_stage"] = "extraction_failed"
            return state
        
        # Process extraction result
        if result and result.get("status") == "success":
            profile = result.get("extracted_profile", DEFAULT_PROFILE)
            
            logger.info(f"✅ Profile extraction successful with confidence "
                       f"{result.get('extraction_confidence', 0):.2f}")
            logger.debug(f"Extracted {len(profile)} fields")
            
            # Store the complete profile dictionary
            state["extracted_profile"] = profile
            
            # Flatten all 20 profile fields into state for easy access
            # Basic Information
            state["pet_species"] = profile.get("pet_species", "unknown")
            state["breed"] = profile.get("breed", "unknown")
            state["age_years"] = profile.get("age_years", 0)
            state["weight_status"] = profile.get("weight_status", "unknown")
            state["sex"] = profile.get("sex", "unknown")
            
            # Medical Information
            state["known_conditions"] = profile.get("known_conditions", [])
            state["past_surgeries"] = profile.get("past_surgeries", [])
            state["allergies_known"] = profile.get("allergies_known", [])
            state["medications_current"] = profile.get("medications_current", [])
            
            # Lifestyle Information
            state["living_situation"] = profile.get("living_situation", "unknown")
            state["exercise_level"] = profile.get("exercise_level", "unknown")
            state["diet_type"] = profile.get("diet_type", "unknown")
            state["diet_quality"] = profile.get("diet_quality", "unknown")
            state["behavioral_issues"] = profile.get("behavioral_issues", [])
            
            # Owner Information
            state["owner_experience"] = profile.get("owner_experience", "unknown")
            state["vet_access"] = profile.get("vet_access", "unknown")
            state["owner_commitment"] = profile.get("owner_commitment", "unknown")
            
            # NEW: Symptom Information
            state["recent_symptoms"] = profile.get("recent_symptoms", [])
            state["symptom_duration_days"] = profile.get("symptom_duration_days", 0)
            state["symptom_severity"] = profile.get("symptom_severity", "unknown")
            
            # Set completion flags
            state["profile_extraction_complete"] = True
            state["extraction_confidence"] = result.get("extraction_confidence", 0.0)
            state["extraction_failed"] = False
            state["processing_stage"] = "extraction_complete"
            
            # Log extracted fields summary
            logger.info(f"Extracted - Species: {state['pet_species']}, "
                       f"Breed: {state['breed']}, Age: {state['age_years']}")
            logger.info(f"Conditions: {len(state['known_conditions'])}, "
                       f"Symptoms: {len(state['recent_symptoms'])}")
            
        else:
            # Extraction failed
            error_msg = result.get("message", "Unknown extraction error") if result else "No result returned"
            logger.error(f"❌ Profile extraction failed: {error_msg}")
            
            state["error_occurred"] = True
            state["error_messages"].append(f"Profile extraction failed: {error_msg}")
            state["profile_extraction_complete"] = False
            state["extraction_failed"] = True
            state["extracted_profile"] = DEFAULT_PROFILE
            state["processing_stage"] = "extraction_failed"
            
        return state
        
    except Exception as e:
        error_msg = f"Unexpected error in pet profile extractor node: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        state["error_occurred"] = True
        state["error_messages"].append(error_msg)
        state["profile_extraction_complete"] = False
        state["extraction_failed"] = True
        state["extracted_profile"] = DEFAULT_PROFILE
        state["processing_stage"] = "extraction_error"
        
        return state


# ==========================================
# TESTING FUNCTION
# ==========================================

def test_pet_profile_extractor_node():
    from state import get_initial_state
    from utils.openai_client import build_openai_client
    
    print("=" * 60)
    print("TESTING PET PROFILE EXTRACTOR NODE")
    print("=" * 60)
    
    try:
        client = build_openai_client()
        print("✅ OpenAI client created")
        
        # Test Case 1: Senior dog with symptoms
        print("\n📝 Test Case 1: Senior Dog with Symptoms")
        state1 = get_initial_state({
            "about_pet": "My dog is a 12-year-old Labrador Retriever named Max.",
            "daily_routine": "He eats twice daily, walks for 15 minutes, sleeps indoors.",
            "health_concerns": "Recently increased thirst, weight loss, coughing, lethargy."
        })
        
        result1 = pet_profile_extractor_node(state1, client)
        print(f"Extraction complete: {result1['profile_extraction_complete']}")
        print(f"Species: {result1.get('pet_species')}")
        print(f"Age: {result1.get('age_years')}")
        print(f"Symptoms: {result1.get('recent_symptoms')}")
        print(f"Symptom severity: {result1.get('symptom_severity')}")
        print(f"Stage: {result1['processing_stage']}")
        
        return result1
        
    except Exception as e:
        print(f"❌ Test error: {str(e)}")
        return None


if __name__ == "__main__":
    test_pet_profile_extractor_node()