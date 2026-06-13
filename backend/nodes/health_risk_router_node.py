# nodes/health_risk_router_node.py
"""
Health Risk Router Node for PawCare+ LangGraph workflow.
Routes to appropriate care path based on health risk score thresholds.
"""

import logging
from typing import Dict, Any, Literal

from state import PetCareState

# Import thresholds from workflow module
try:
    from workflow.workflow import CRITICAL_THRESHOLD, WELLNESS_THRESHOLD
except ImportError:
    # Fallback thresholds if workflow module not available
    CRITICAL_THRESHOLD = 0.6
    WELLNESS_THRESHOLD = 0.3
    logging.warning(f"Using fallback thresholds: Critical>{CRITICAL_THRESHOLD}, Wellness≤{WELLNESS_THRESHOLD}")

logger = logging.getLogger(__name__)

# Path constants
CRITICAL_CARE_PATH = "CRITICAL_CARE_PATH"
PREVENTIVE_CARE_PATH = "PREVENTIVE_CARE_PATH"
WELLNESS_PATH = "WELLNESS_PATH"


def health_risk_router_node(state: PetCareState) -> PetCareState:
    """
    Route to appropriate care path based on health risk score.
    
    This node evaluates the health_risk_score (0-1) and sets the path_taken
    field according to predefined thresholds:
    - Score > 0.6 → CRITICAL_CARE_PATH
    - 0.3 < Score ≤ 0.6 → PREVENTIVE_CARE_PATH
    - Score ≤ 0.3 → WELLNESS_PATH
    
    Args:
        state: Current PetCareState containing health_risk_score
        
    Returns:
        Updated PetCareState with path_taken set and routing metadata:
        - path_taken: One of the three path constants
        - path_decision_rationale: Explanation of routing decision
        - path_thresholds_used: Dictionary of thresholds applied
        - processing_stage: Updated to "routed"
        - error_occurred: Set to True if routing fails
        - error_messages: Appended with any errors
    """
    logger.info("=" * 50)
    logger.info("EXECUTING HEALTH RISK ROUTER NODE")
    logger.info("=" * 50)
    
    try:
        # Log incoming state info
        logger.debug(f"State request_id: {state.get('request_id', 'N/A')}")
        logger.debug(f"Current processing stage: {state.get('processing_stage', 'unknown')}")
        
        # Extract health_risk_score with default
        risk_score = state.get('health_risk_score', 0.5)
        
        # Validate score is within expected range
        if not isinstance(risk_score, (int, float)):
            logger.warning(f"health_risk_score is not numeric: {risk_score}, defaulting to 0.5")
            risk_score = 0.5
        else:
            # Clamp to valid range just in case
            risk_score = max(0.0, min(1.0, float(risk_score)))
        
        logger.info(f"Health risk score: {risk_score:.3f}")
        logger.info(f"Thresholds: Critical > {CRITICAL_THRESHOLD}, Wellness ≤ {WELLNESS_THRESHOLD}")
        
        # Store thresholds used for transparency
        state["path_thresholds_used"] = {
            "critical": CRITICAL_THRESHOLD,
            "wellness": WELLNESS_THRESHOLD,
            "score": risk_score
        }
        
        # Determine path based on risk score
        if risk_score > CRITICAL_THRESHOLD:
            path = CRITICAL_CARE_PATH
            rationale = f"High risk score {risk_score:.3f} > {CRITICAL_THRESHOLD} indicates critical care needed"
            logger.info(f"🔴 ROUTING TO: {path}")
            
        elif risk_score > WELLNESS_THRESHOLD:
            path = PREVENTIVE_CARE_PATH
            rationale = f"Moderate risk score {risk_score:.3f} between {WELLNESS_THRESHOLD} and {CRITICAL_THRESHOLD} indicates preventive care"
            logger.info(f"🟡 ROUTING TO: {path}")
            
        else:
            path = WELLNESS_PATH
            rationale = f"Low risk score {risk_score:.3f} ≤ {WELLNESS_THRESHOLD} indicates wellness optimization"
            logger.info(f"🟢 ROUTING TO: {path}")
        
        # Update state with routing decision
        state["path_taken"] = path
        state["path_decision_rationale"] = rationale
        state["processing_stage"] = "routed"
        
        # Also store care capability score for context if available
        care_score = state.get('care_capability_score')
        if care_score is not None:
            logger.info(f"Owner care capability score: {care_score:.1f}/100")
            state["path_decision_context"] = {
                "health_risk_score": risk_score,
                "care_capability_score": care_score
            }
        
        # Log summary
        logger.info(f"✅ Routing complete: {path}")
        
        return state
        
    except Exception as e:
        # Handle unexpected errors
        error_msg = f"Unexpected error in health risk router node: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Update state with error information
        state["error_occurred"] = True
        state["error_messages"].append(error_msg)
        
        # Set a safe default path (preventive) to allow workflow to continue
        state["path_taken"] = PREVENTIVE_CARE_PATH
        state["path_decision_rationale"] = f"Default path due to routing error: {str(e)}"
        state["processing_stage"] = "routing_error"
        
        logger.warning(f"Using default path: {PREVENTIVE_CARE_PATH} due to error")
        
        return state


# ==========================================
# CONDITIONAL EDGE FUNCTION FOR GRAPH
# ==========================================

def get_next_node_after_routing(state: PetCareState) -> Literal[
    "risk_analysis", 
    "health_assessment_preventive", 
    "wellness_optimization",
    "output_aggregator"  # Fallback
]:
    """
    Determine the next node to execute based on the routed path.
    
    This function is used by LangGraph for conditional edges.
    
    Args:
        state: Current PetCareState with path_taken set
        
    Returns:
        Name of the next node to execute
    """
    path = state.get("path_taken", PREVENTIVE_CARE_PATH)
    
    logger.debug(f"Getting next node for path: {path}")
    
    if path == CRITICAL_CARE_PATH:
        return "risk_analysis"
    elif path == PREVENTIVE_CARE_PATH:
        return "health_assessment_preventive"
    elif path == WELLNESS_PATH:
        return "wellness_optimization"
    else:
        # Fallback for unknown path
        logger.warning(f"Unknown path: {path}, defaulting to preventive")
        return "health_assessment_preventive"


# ==========================================
# OPTIONAL: Helper function for testing
# ==========================================

def test_health_risk_router_node():
    """
    Test function to verify node behavior.
    """
    from state import get_initial_state
    
    print("=" * 60)
    print("TESTING HEALTH RISK ROUTER NODE")
    print("=" * 60)
    print(f"Thresholds: Critical > {CRITICAL_THRESHOLD}, Wellness ≤ {WELLNESS_THRESHOLD}")
    
    # Test Case 1: Critical path (score > 0.6)
    print("\n📝 Test Case 1: Critical Path")
    state1 = get_initial_state({})
    state1["health_risk_score"] = 0.85
    state1["care_capability_score"] = 75.0
    
    result1 = health_risk_router_node(state1)
    print(f"Risk score: {result1['health_risk_score']}")
    print(f"Path taken: {result1['path_taken']}")
    print(f"Rationale: {result1['path_decision_rationale']}")
    print(f"Next node: {get_next_node_after_routing(result1)}")
    
    # Test Case 2: Preventive path (0.3 < score ≤ 0.6)
    print("\n📝 Test Case 2: Preventive Path")
    state2 = get_initial_state({})
    state2["health_risk_score"] = 0.45
    
    result2 = health_risk_router_node(state2)
    print(f"Risk score: {result2['health_risk_score']}")
    print(f"Path taken: {result2['path_taken']}")
    print(f"Next node: {get_next_node_after_routing(result2)}")
    
    # Test Case 3: Wellness path (score ≤ 0.3)
    print("\n📝 Test Case 3: Wellness Path")
    state3 = get_initial_state({})
    state3["health_risk_score"] = 0.15
    
    result3 = health_risk_router_node(state3)
    print(f"Risk score: {result3['health_risk_score']}")
    print(f"Path taken: {result3['path_taken']}")
    print(f"Next node: {get_next_node_after_routing(result3)}")
    
    # Test Case 4: Boundary values
    print("\n📝 Test Case 4: Boundary Values")
    boundaries = [0.601, 0.6, 0.301, 0.3, 0.299]
    
    for score in boundaries:
        state = get_initial_state({})
        state["health_risk_score"] = score
        result = health_risk_router_node(state.copy())
        print(f"  Score {score:.3f} → {result['path_taken']}")
    
    # Test Case 5: Missing score (default)
    print("\n📝 Test Case 5: Missing Score (defaults to 0.5)")
    state5 = get_initial_state({})
    # No health_risk_score set
    
    result5 = health_risk_router_node(state5)
    print(f"Path taken: {result5['path_taken']}")
    print(f"Rationale: {result5['path_decision_rationale']}")
    
    return result1, result2, result3, result5


if __name__ == "__main__":
    # Run test if executed directly
    test_health_risk_router_node()