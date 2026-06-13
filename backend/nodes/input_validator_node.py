# nodes/input_validator_node.py
"""
Input Validator Node for PawCare+ LangGraph workflow.
Executes input validation as the first step in the workflow.
"""

import logging
from typing import Dict, Any

from state import PetCareState
from agents.input_validator_agent import InputValidatorAgent

logger = logging.getLogger(__name__)


def input_validator_node(state: PetCareState) -> PetCareState:
    """
    Execute input validation as the first workflow step.
    
    This node validates the three user input fields (about_pet, daily_routine, health_concerns)
    for basic requirements like non-emptiness and reasonable length.
    
    Args:
        state: Current PetCareState containing user inputs
        
    Returns:
        Updated PetCareState with validation results:
        - validation_errors: List of error messages
        - parsing_complete: Boolean indicating validation completion
        - error_occurred: Set to True if validation fails
        - error_messages: Appended with any errors
        - processing_stage: Updated to "validation_complete" or "validation_failed"
    """
    logger.info("=" * 50)
    logger.info("EXECUTING INPUT VALIDATOR NODE")
    logger.info("=" * 50)
    
    try:
        # Log incoming state info
        logger.debug(f"State request_id: {state.get('request_id', 'N/A')}")
        logger.debug(f"Current processing stage: {state.get('processing_stage', 'unknown')}")
        
        # Create validator agent
        validator = InputValidatorAgent()
        logger.debug("InputValidatorAgent created successfully")
        
        # Extract user inputs from state
        about_pet = state.get('about_pet', '')
        daily_routine = state.get('daily_routine', '')
        health_concerns = state.get('health_concerns', '')
        
        # Log input lengths for debugging
        logger.info(f"Validating inputs - about_pet: {len(about_pet)} chars, "
                   f"daily_routine: {len(daily_routine)} chars, "
                   f"health_concerns: {len(health_concerns)} chars")
        
        # Validate inputs
        result = validator.validate_inputs({
            "about_pet": about_pet,
            "daily_routine": daily_routine,
            "health_concerns": health_concerns
        })
        
        # Update state with validation results
        state["validation_errors"] = result["validation_errors"]
        state["parsing_complete"] = result["is_valid"]
        
        # Also store validated inputs (cleaned versions)
        if "validated_inputs" in result:
            state["validated_inputs"] = result["validated_inputs"]
        
        # Handle validation failure
        if not result["is_valid"]:
            logger.warning(f"Validation failed with {len(result['validation_errors'])} errors")
            for error in result["validation_errors"]:
                logger.warning(f"  - {error}")
            
            state["error_occurred"] = True
            state["error_messages"].append(
                f"Input validation failed: {', '.join(result['validation_errors'])}"
            )
            state["processing_stage"] = "validation_failed"
            
            # Store field status for debugging
            if "field_status" in result:
                state["field_status"] = result["field_status"]
        else:
            logger.info("✅ Input validation passed successfully")
            state["processing_stage"] = "validation_complete"
            
            # Store validation stats if available
            if "stats" in result:
                state["validation_stats"] = result["stats"]
        
        # Log completion
        logger.info(f"Input validator node completed. Valid: {result['is_valid']}")
        
        return state
        
    except Exception as e:
        # Handle unexpected errors
        error_msg = f"Unexpected error in input validator node: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Update state with error information
        state["error_occurred"] = True
        state["error_messages"].append(error_msg)
        state["validation_errors"] = state.get("validation_errors", [])
        state["validation_errors"].append("System error during validation")
        state["parsing_complete"] = False
        state["processing_stage"] = "validation_error"
        
        return state


# ==========================================
# OPTIONAL: Helper function for testing
# ==========================================

def test_input_validator_node():
    """
    Test function to verify node behavior.
    """
    from state import get_initial_state
    
    print("=" * 60)
    print("TESTING INPUT VALIDATOR NODE")
    print("=" * 60)
    
    # Test Case 1: Valid inputs
    print("\n📝 Test Case 1: Valid Inputs")
    state1 = get_initial_state({
        "about_pet": "My dog is a 5-year-old Labrador Retriever named Max.",
        "daily_routine": "He eats twice daily, walks for 30 minutes, and sleeps indoors.",
        "health_concerns": "Recently noticed increased thirst and lethargy."
    })
    
    result1 = input_validator_node(state1)
    print(f"Valid: {result1['parsing_complete']}")
    print(f"Errors: {result1['validation_errors']}")
    print(f"Stage: {result1['processing_stage']}")
    
    # Test Case 2: Invalid inputs (empty)
    print("\n📝 Test Case 2: Empty Inputs")
    state2 = get_initial_state({
        "about_pet": "",
        "daily_routine": "  ",
        "health_concerns": ""
    })
    
    result2 = input_validator_node(state2)
    print(f"Valid: {result2['parsing_complete']}")
    print(f"Errors: {result2['validation_errors']}")
    print(f"Error occurred: {result2['error_occurred']}")
    
    # Test Case 3: Mixed inputs
    print("\n📝 Test Case 3: Mixed Inputs")
    state3 = get_initial_state({
        "about_pet": "My cat",
        "daily_routine": "Indoor cat, eats dry food.",
        "health_concerns": ""
    })
    
    result3 = input_validator_node(state3)
    print(f"Valid: {result3['parsing_complete']}")
    print(f"Errors: {result3['validation_errors']}")
    
    return result1, result2, result3


if __name__ == "__main__":
    # Run test if executed directly
    test_input_validator_node()