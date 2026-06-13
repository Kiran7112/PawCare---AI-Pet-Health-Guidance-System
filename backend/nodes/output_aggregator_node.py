# nodes/output_aggregator_node.py
"""
Output Aggregator Node for PawCare+ LangGraph workflow.
Aggregates all path-specific outputs into a single comprehensive result.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from state import PetCareState

logger = logging.getLogger(__name__)

# Path constants (should match those in health_risk_router_node)
CRITICAL_CARE_PATH = "CRITICAL_CARE_PATH"
PREVENTIVE_CARE_PATH = "PREVENTIVE_CARE_PATH"
WELLNESS_PATH = "WELLNESS_PATH"


def output_aggregator_node(state: PetCareState) -> PetCareState:
    """
    Aggregate all outputs based on care path into a single comprehensive result.
    
    This node consolidates all generated content from the specific care path
    into a unified aggregated_output dictionary for final display.
    
    Args:
        state: Current PetCareState containing:
            - path_taken: One of the three path constants
            - Path-specific output fields based on the path
            
    Returns:
        Updated PetCareState with aggregated_output and processing_complete flag:
        - aggregated_output: Dictionary containing all path-specific outputs
        - processing_complete: Set to True
        - processing_stage: Updated to "aggregated" or "aggregation_failed"
        - error_occurred: Set to True if aggregation fails
        - error_messages: Appended with any errors
    """
    logger.info("=" * 50)
    logger.info("EXECUTING OUTPUT AGGREGATOR NODE")
    logger.info("=" * 50)
    
    try:
        # Log incoming state info
        logger.debug(f"State request_id: {state.get('request_id', 'N/A')}")
        logger.debug(f"Current path: {state.get('path_taken', 'unknown')}")
        logger.debug(f"Current processing stage: {state.get('processing_stage', 'unknown')}")
        
        # Extract path_taken
        path = state.get('path_taken')
        
        if not path:
            logger.warning("No path_taken found in state, using default aggregation")
            path = "UNKNOWN_PATH"
        
        logger.info(f"Aggregating outputs for path: {path}")
        
        # Create base aggregated output with metadata
        aggregated = {
            "request_id": state.get('request_id', 'N/A'),
            "analysis_timestamp": state.get('analysis_timestamp', datetime.now().isoformat()),
            "path_taken": path,
            "health_risk_score": state.get('health_risk_score'),
            "care_capability_score": state.get('care_capability_score'),
            "pet_profile_summary": _build_profile_summary(state),
            "outputs": {},
            "summary": None,
            "recommendations": []
        }
        
        # Aggregate based on path
        if path == CRITICAL_CARE_PATH:
            aggregated = _aggregate_critical_path(state, aggregated)
        elif path == PREVENTIVE_CARE_PATH:
            aggregated = _aggregate_preventive_path(state, aggregated)
        elif path == WELLNESS_PATH:
            aggregated = _aggregate_wellness_path(state, aggregated)
        else:
            # Unknown path - try to aggregate anything available
            logger.warning(f"Unknown path: {path}, aggregating available outputs")
            aggregated = _aggregate_available_outputs(state, aggregated)
        
        # Add summary and recommendations
        aggregated["summary"] = _generate_summary(aggregated)
        aggregated["recommendations"] = _extract_recommendations(aggregated)
        
        # Add error summary if any errors occurred
        if state.get('error_occurred', False):
            aggregated["errors"] = {
                "has_errors": True,
                "error_count": len(state.get('error_messages', [])),
                "error_messages": state.get('error_messages', [])[:5]  # Limit to 5 messages
            }
        else:
            aggregated["errors"] = {"has_errors": False}
        
        # Store in state
        state["aggregated_output"] = aggregated
        state["processing_complete"] = True
        state["processing_stage"] = "aggregated"
        
        # Log success
        logger.info(f"✅ Output aggregation completed successfully")
        logger.info(f"Aggregated {len(aggregated['outputs'])} output sections")
        
        return state
        
    except Exception as e:
        # Handle unexpected errors
        error_msg = f"Unexpected error in output aggregator node: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Update state with error information
        state["error_occurred"] = True
        state["error_messages"].append(error_msg)
        state["processing_complete"] = True  # Still mark as complete to end workflow
        state["processing_stage"] = "aggregation_error"
        
        # Create minimal aggregated output
        state["aggregated_output"] = {
            "request_id": state.get('request_id', 'N/A'),
            "analysis_timestamp": state.get('analysis_timestamp', datetime.now().isoformat()),
            "path_taken": state.get('path_taken', 'unknown'),
            "error": "Aggregation failed",
            "error_details": error_msg
        }
        
        return state


# ==========================================
# PATH-SPECIFIC AGGREGATION FUNCTIONS
# ==========================================

def _aggregate_critical_path(state: PetCareState, aggregated: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate outputs for critical care path.
    
    Includes: health_risk_analysis, emergency_preparedness, nutrition_plan,
    behavioral_coaching, wellness_monitoring
    
    Args:
        state: Current PetCareState
        aggregated: Base aggregated dictionary
        
    Returns:
        Updated aggregated dictionary
    """
    logger.info("Aggregating critical path outputs (5 sections)")
    
    outputs = {}
    
    # 1. Health Risk Analysis
    risk_analysis = state.get('health_risk_analysis_output') or state.get('critical_path_outputs', {}).get('risk_analysis')
    if risk_analysis:
        outputs["health_risk_analysis"] = {
            "title": "Health Risk Analysis",
            "icon": "🔴",
            "content": risk_analysis
        }
        logger.debug("Added health risk analysis")
    
    # 2. Emergency Preparedness
    emergency = state.get('emergency_prep_output') or state.get('critical_path_outputs', {}).get('emergency_prep')
    if emergency:
        outputs["emergency_preparedness"] = {
            "title": "Emergency Preparedness",
            "icon": "🚨",
            "content": emergency
        }
        logger.debug("Added emergency preparedness")
    
    # 3. Critical Nutrition
    nutrition = state.get('nutrition_critical_output') or state.get('critical_path_outputs', {}).get('nutrition')
    if nutrition:
        outputs["nutrition_plan"] = {
            "title": "Critical Nutrition Plan",
            "icon": "🥗",
            "content": nutrition
        }
        logger.debug("Added critical nutrition")
    
    # 4. Behavioral Coaching
    behavioral = state.get('behavioral_coaching_output') or state.get('critical_path_outputs', {}).get('behavioral')
    if behavioral:
        outputs["behavioral_coaching"] = {
            "title": "Behavioral Coaching",
            "icon": "🐾",
            "content": behavioral
        }
        logger.debug("Added behavioral coaching")
    
    # 5. Wellness Monitoring
    monitoring = state.get('wellness_monitoring_output') or state.get('critical_path_outputs', {}).get('monitoring')
    if monitoring:
        outputs["wellness_monitoring"] = {
            "title": "Wellness Monitoring",
            "icon": "📊",
            "content": monitoring
        }
        logger.debug("Added wellness monitoring")
    
    # Add to aggregated
    aggregated["outputs"] = outputs
    aggregated["output_count"] = len(outputs)
    aggregated["path_description"] = "Critical Care Path - Immediate veterinary attention recommended"
    
    return aggregated


def _aggregate_preventive_path(state: PetCareState, aggregated: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate outputs for preventive care path.
    
    Includes: health_assessment, nutrition_guide, wellness_tracking
    
    Args:
        state: Current PetCareState
        aggregated: Base aggregated dictionary
        
    Returns:
        Updated aggregated dictionary
    """
    logger.info("Aggregating preventive path outputs (3 sections)")
    
    outputs = {}
    
    # 1. Health Assessment
    assessment = state.get('health_assessment_output') or state.get('preventive_path_outputs', {}).get('health_assessment')
    if assessment:
        outputs["health_assessment"] = {
            "title": "Preventive Health Assessment",
            "icon": "🩺",
            "content": assessment
        }
        logger.debug("Added health assessment")
    
    # 2. Preventive Nutrition
    nutrition = state.get('nutrition_preventive_output') or state.get('preventive_path_outputs', {}).get('nutrition')
    if nutrition:
        outputs["nutrition_guide"] = {
            "title": "Preventive Nutrition Guide",
            "icon": "🥗",
            "content": nutrition
        }
        logger.debug("Added preventive nutrition")
    
    # 3. Wellness Tracking
    tracking = state.get('wellness_tracking_output') or state.get('preventive_path_outputs', {}).get('tracking')
    if tracking:
        outputs["wellness_tracking"] = {
            "title": "Wellness Tracking Plan",
            "icon": "📋",
            "content": tracking
        }
        logger.debug("Added wellness tracking")
    
    # Add to aggregated
    aggregated["outputs"] = outputs
    aggregated["output_count"] = len(outputs)
    aggregated["path_description"] = "Preventive Care Path - Schedule vet visit and implement home care"
    
    return aggregated


def _aggregate_wellness_path(state: PetCareState, aggregated: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate outputs for wellness path.
    
    Includes: wellness_optimization, nutrition_enhancement, lifestyle_enrichment
    
    Args:
        state: Current PetCareState
        aggregated: Base aggregated dictionary
        
    Returns:
        Updated aggregated dictionary
    """
    logger.info("Aggregating wellness path outputs (3 sections)")
    
    outputs = {}
    
    # 1. Wellness Optimization
    optimization = state.get('wellness_optimization_output') or state.get('wellness_path_outputs', {}).get('optimization')
    if optimization:
        outputs["wellness_optimization"] = {
            "title": "Wellness Optimization",
            "icon": "🌟",
            "content": optimization
        }
        logger.debug("Added wellness optimization")
    
    # 2. Nutrition Enhancement
    nutrition = state.get('nutrition_wellness_output') or state.get('wellness_path_outputs', {}).get('nutrition')
    if nutrition:
        outputs["nutrition_enhancement"] = {
            "title": "Nutrition Enhancement",
            "icon": "🥕",
            "content": nutrition
        }
        logger.debug("Added nutrition enhancement")
    
    # 3. Lifestyle Enrichment
    lifestyle = state.get('lifestyle_enrichment_output') or state.get('wellness_path_outputs', {}).get('lifestyle')
    if lifestyle:
        outputs["lifestyle_enrichment"] = {
            "title": "Lifestyle Enrichment",
            "icon": "🎾",
            "content": lifestyle
        }
        logger.debug("Added lifestyle enrichment")
    
    # Add to aggregated
    aggregated["outputs"] = outputs
    aggregated["output_count"] = len(outputs)
    aggregated["path_description"] = "Wellness Path - Focus on optimization and enrichment"
    
    return aggregated


def _aggregate_available_outputs(state: PetCareState, aggregated: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fallback function to aggregate any available outputs regardless of path.
    
    Args:
        state: Current PetCareState
        aggregated: Base aggregated dictionary
        
    Returns:
        Updated aggregated dictionary
    """
    logger.info("Aggregating any available outputs")
    
    outputs = {}
    
    # Check all possible output fields
    possible_outputs = [
        ("health_risk_analysis_output", "Health Risk Analysis", "🔴"),
        ("emergency_prep_output", "Emergency Preparedness", "🚨"),
        ("nutrition_critical_output", "Critical Nutrition", "🥗"),
        ("behavioral_coaching_output", "Behavioral Coaching", "🐾"),
        ("wellness_monitoring_output", "Wellness Monitoring", "📊"),
        ("health_assessment_output", "Health Assessment", "🩺"),
        ("nutrition_preventive_output", "Preventive Nutrition", "🥗"),
        ("wellness_tracking_output", "Wellness Tracking", "📋"),
        ("wellness_optimization_output", "Wellness Optimization", "🌟"),
        ("nutrition_wellness_output", "Nutrition Enhancement", "🥕"),
        ("lifestyle_enrichment_output", "Lifestyle Enrichment", "🎾")
    ]
    
    for field, title, icon in possible_outputs:
        content = state.get(field)
        if content:
            outputs[field.replace("_output", "")] = {
                "title": title,
                "icon": icon,
                "content": content
            }
            logger.debug(f"Added {title}")
    
    # Also check path-specific output containers
    if "critical_path_outputs" in state:
        for key, content in state["critical_path_outputs"].items():
            if key not in outputs:
                outputs[f"critical_{key}"] = {
                    "title": f"Critical: {key.replace('_', ' ').title()}",
                    "icon": "🔴",
                    "content": content
                }
    
    if "preventive_path_outputs" in state:
        for key, content in state["preventive_path_outputs"].items():
            if key not in outputs:
                outputs[f"preventive_{key}"] = {
                    "title": f"Preventive: {key.replace('_', ' ').title()}",
                    "icon": "🟡",
                    "content": content
                }
    
    if "wellness_path_outputs" in state:
        for key, content in state["wellness_path_outputs"].items():
            if key not in outputs:
                outputs[f"wellness_{key}"] = {
                    "title": f"Wellness: {key.replace('_', ' ').title()}",
                    "icon": "🟢",
                    "content": content
                }
    
    aggregated["outputs"] = outputs
    aggregated["output_count"] = len(outputs)
    aggregated["path_description"] = "Mixed/Unknown Path - Review all available guidance"
    
    return aggregated


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def _build_profile_summary(state: PetCareState) -> Dict[str, Any]:
    """
    Build a summary of the pet profile from state.
    
    Args:
        state: Current PetCareState
        
    Returns:
        Dictionary with profile summary
    """
    return {
        "name": state.get('pet_name', 'Your Pet'),
        "species": state.get('pet_species', 'unknown'),
        "breed": state.get('breed', 'unknown'),
        "age_years": state.get('age_years', 0),
        "weight_status": state.get('weight_status', 'unknown'),
        "known_conditions": state.get('known_conditions', []),
        "medications": state.get('medications_current', []),
        "behavioral_issues": state.get('behavioral_issues', [])
    }


def _generate_summary(aggregated: Dict[str, Any]) -> str:
    """
    Generate a brief summary text from aggregated outputs.
    
    Args:
        aggregated: Aggregated outputs dictionary
        
    Returns:
        Summary string
    """
    path = aggregated.get('path_taken', 'unknown')
    count = aggregated.get('output_count', 0)
    
    if path == CRITICAL_CARE_PATH:
        return f"Critical care assessment complete. Review {count} sections for emergency guidance."
    elif path == PREVENTIVE_CARE_PATH:
        return f"Preventive care assessment complete. Review {count} sections for proactive health guidance."
    elif path == WELLNESS_PATH:
        return f"Wellness assessment complete. Review {count} sections for optimization tips."
    else:
        return f"Assessment complete. Review {count} guidance sections."


def _extract_recommendations(aggregated: Dict[str, Any]) -> list:
    """
    Extract key recommendations from outputs for quick reference.
    
    Args:
        aggregated: Aggregated outputs dictionary
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    # Try to extract from different output types
    for output_key, output_data in aggregated.get('outputs', {}).items():
        content = output_data.get('content', {})
        
        # Check for different field names that might contain recommendations
        if isinstance(content, dict):
            # Look for list-type fields that might contain recommendations
            for field in ['training_strategies', 'prevention_strategies', 'enhancement_tips', 
                         'critical_risk_factors', 'emergency_contacts', 'first_aid_supplies',
                         'crisis_procedures', 'warning_signs', 'healthy_treats']:
                if field in content and isinstance(content[field], list):
                    items = content[field]
                    if items and len(items) > 0:
                        # Take first item as sample recommendation
                        recommendations.append(f"{output_data.get('title')}: {items[0]}")
                        break
            
            # Check for text fields that might contain summary recommendations
            for field in ['preventive_assessment', 'honest_risk_assessment', 'nutrition_overview']:
                if field in content and isinstance(content[field], str):
                    text = content[field]
                    if text and len(text) > 30:
                        # Truncate long text
                        if len(text) > 100:
                            text = text[:97] + "..."
                        recommendations.append(f"{output_data.get('title')}: {text}")
                        break
    
    # Limit to top 3 recommendations
    return recommendations[:3]


# ==========================================
# OPTIONAL: Helper function for testing
# ==========================================

def test_output_aggregator_node():
    """
    Test function to verify node behavior.
    """
    from state import get_initial_state
    
    print("=" * 60)
    print("TESTING OUTPUT AGGREGATOR NODE")
    print("=" * 60)
    
    # Test Case 1: Critical path
    print("\n📝 Test Case 1: Critical Path")
    state1 = get_initial_state({})
    state1["path_taken"] = CRITICAL_CARE_PATH
    state1["health_risk_score"] = 0.85
    state1["pet_name"] = "Max"
    state1["pet_species"] = "dog"
    
    # Add mock outputs
    state1["health_risk_analysis_output"] = {"risk": "High risk", "factors": ["age", "conditions"]}
    state1["emergency_prep_output"] = {"contacts": ["vet"], "supplies": ["meds"]}
    state1["nutrition_critical_output"] = {"diet": "therapeutic", "supplements": []}
    
    result1 = output_aggregator_node(state1)
    print(f"Aggregated: {'aggregated_output' in result1}")
    print(f"Path: {result1['aggregated_output'].get('path_taken')}")
    print(f"Output count: {result1['aggregated_output'].get('output_count')}")
    print(f"Complete: {result1.get('processing_complete')}")
    
    # Test Case 2: Preventive path
    print("\n📝 Test Case 2: Preventive Path")
    state2 = get_initial_state({})
    state2["path_taken"] = PREVENTIVE_CARE_PATH
    state2["health_risk_score"] = 0.45
    state2["pet_name"] = "Buddy"
    
    state2["health_assessment_output"] = {"assessment": "moderate risk", "areas": ["weight"]}
    state2["nutrition_preventive_output"] = {"diet": "weight management", "treats": []}
    
    result2 = output_aggregator_node(state2)
    print(f"Aggregated: {'aggregated_output' in result2}")
    print(f"Output count: {result2['aggregated_output'].get('output_count')}")
    
    # Test Case 3: Wellness path
    print("\n📝 Test Case 3: Wellness Path")
    state3 = get_initial_state({})
    state3["path_taken"] = WELLNESS_PATH
    state3["health_risk_score"] = 0.15
    state3["pet_name"] = "Luna"
    
    state3["wellness_optimization_output"] = {"enhancements": ["agility", "puzzles"]}
    state3["nutrition_wellness_output"] = {"tips": ["toppers", "variety"]}
    state3["lifestyle_enrichment_output"] = {"mental": ["games"], "social": ["playdates"]}
    
    result3 = output_aggregator_node(state3)
    print(f"Aggregated: {'aggregated_output' in result3}")
    print(f"Output count: {result3['aggregated_output'].get('output_count')}")
    
    # Test Case 4: Unknown path (should aggregate available)
    print("\n📝 Test Case 4: Unknown Path")
    state4 = get_initial_state({})
    # No path_taken set
    state4["health_risk_score"] = 0.5
    state4["pet_name"] = "Charlie"
    
    state4["wellness_optimization_output"] = {"enhancements": ["agility"]}
    state4["nutrition_preventive_output"] = {"diet": "maintenance"}
    
    result4 = output_aggregator_node(state4)
    print(f"Aggregated: {'aggregated_output' in result4}")
    print(f"Output count: {result4['aggregated_output'].get('output_count')}")
    print(f"Summary: {result4['aggregated_output'].get('summary')}")
    
    return result1, result2, result3, result4


if __name__ == "__main__":
    # Run test if executed directly
    test_output_aggregator_node()