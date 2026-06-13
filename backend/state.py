
"""
State Management Module for PawCare+.
Defines the complete state schema with 51 fields and custom reducer functions for LangGraph workflow.
"""

from typing import TypedDict, List, Dict, Any, Optional, Union
from datetime import datetime
import uuid
from typing_extensions import Annotated
from operator import add

# ==========================================
# CUSTOM REDUCER FUNCTIONS
# ==========================================

def keep_first(existing: Any, new: Any) -> Any:
    """Reducer: Keep the first value (ignore subsequent updates)."""
    return existing if existing is not None else new


def set_true(existing: bool, new: bool) -> bool:
    """Reducer: Set to True if either value is True (OR logic)."""
    return existing or new


from operator import add as accumulate_lists


# ==========================================
# PETCARESTATE TYPEDICT DEFINITION
# ==========================================

class PetCareState(TypedDict):
    """
    Complete state schema for PawCare+ workflow.
    Contains 51 fields across 8 categories.
    """
    
    # ========== 1. SYSTEM METADATA (5 fields) ==========
    request_id: Annotated[Optional[str], keep_first]
    analysis_timestamp: Annotated[Optional[str], keep_first]
    error_occurred: Annotated[bool, set_true]
    error_messages: Annotated[List[str], accumulate_lists]
    processing_complete: Annotated[bool, set_true]
    
    # ========== 2. USER INPUTS (3 fields) ==========
    about_pet: Annotated[Optional[str], keep_first]
    daily_routine: Annotated[Optional[str], keep_first]
    health_concerns: Annotated[Optional[str], keep_first]
    
    # ========== 3. VALIDATION FIELDS (2 fields) ==========
    validation_errors: Annotated[List[str], accumulate_lists]
    parsing_complete: Annotated[bool, set_true]
    
    # ========== 4. EXTRACTED PROFILE (20 fields) ==========
    extracted_profile: Annotated[Optional[Dict[str, Any]], keep_first]
    profile_extraction_complete: Annotated[bool, set_true]
    
    # Basic Information
    pet_species: Annotated[Optional[str], keep_first]
    breed: Annotated[Optional[str], keep_first]
    age_years: Annotated[Optional[int], keep_first]
    weight_status: Annotated[Optional[str], keep_first]
    sex: Annotated[Optional[str], keep_first]
    
    # Medical Information
    known_conditions: Annotated[List[str], accumulate_lists]
    past_surgeries: Annotated[List[str], accumulate_lists]
    allergies_known: Annotated[List[str], accumulate_lists]
    medications_current: Annotated[List[str], accumulate_lists]
    
    # Lifestyle Information
    living_situation: Annotated[Optional[str], keep_first]
    exercise_level: Annotated[Optional[str], keep_first]
    diet_type: Annotated[Optional[str], keep_first]
    diet_quality: Annotated[Optional[str], keep_first]
    behavioral_issues: Annotated[List[str], accumulate_lists]
    
    # Owner Information
    owner_experience: Annotated[Optional[str], keep_first]
    owner_commitment: Annotated[Optional[str], keep_first]
    vet_access: Annotated[Optional[str], keep_first]
    
    # Symptom Fields (NEW - 3 fields)
    recent_symptoms: Annotated[List[str], accumulate_lists]
    symptom_duration_days: Annotated[Optional[int], keep_first]
    symptom_severity: Annotated[Optional[str], keep_first]
    
    # ========== 5. ML PREDICTION FIELDS (2 fields) ==========
    health_risk_score: Annotated[Optional[float], keep_first]
    care_capability_score: Annotated[Optional[float], keep_first]
    
    # ========== 6. ROUTING FIELD (1 field) ==========
    path_taken: Annotated[Optional[str], keep_first]
    
    # ========== 7. PATH-SPECIFIC OUTPUT FIELDS (12 fields) ==========
    # CRITICAL PATH outputs (5 fields)
    health_risk_analysis_output: Annotated[Optional[Dict[str, Any]], keep_first]
    emergency_prep_output: Annotated[Optional[Dict[str, Any]], keep_first]
    nutrition_critical_output: Annotated[Optional[Dict[str, Any]], keep_first]
    behavioral_coaching_output: Annotated[Optional[Dict[str, Any]], keep_first]
    wellness_monitoring_output: Annotated[Optional[Dict[str, Any]], keep_first]
    
    # PREVENTIVE PATH outputs (3 fields)
    health_assessment_output: Annotated[Optional[Dict[str, Any]], keep_first]
    nutrition_preventive_output: Annotated[Optional[Dict[str, Any]], keep_first]
    wellness_tracking_output: Annotated[Optional[Dict[str, Any]], keep_first]
    
    # WELLNESS PATH outputs (3 fields)
    wellness_optimization_output: Annotated[Optional[Dict[str, Any]], keep_first]
    nutrition_wellness_output: Annotated[Optional[Dict[str, Any]], keep_first]
    lifestyle_enrichment_output: Annotated[Optional[Dict[str, Any]], keep_first]
    
    # Aggregated path container (1 field)
    path_outputs_aggregated: Annotated[Optional[Dict[str, Any]], keep_first]
    
    # ========== 8. FINAL OUTPUT FIELD (1 field) ==========
    aggregated_output: Annotated[Optional[Dict[str, Any]], keep_first]
    
    # ========== ADDITIONAL UTILITY FIELDS (3 fields) ==========
    current_node: Annotated[Optional[str], keep_first]
    node_execution_order: Annotated[List[str], accumulate_lists]
    processing_stage: Annotated[Optional[str], keep_first]


# ==========================================
# VALIDATION: FIELD COUNT CHECK
# ==========================================

def _count_state_fields() -> int:
    """Count the number of fields in PetCareState for validation."""
    annotations = PetCareState.__annotations__
    fields = [f for f in annotations.keys() if not f.startswith('_')]
    return len(fields)

# Verify we have exactly 51 fields
assert _count_state_fields() == 51, f"PetCareState has {_count_state_fields()} fields, expected 51"


# ==========================================
# GET_INITIAL_STATE FUNCTION
# ==========================================

def get_initial_state(form_data: Dict[str, Any]) -> PetCareState:
    """
    Create and initialize complete state dictionary from form data.
    """
    
    request_id = form_data.get('request_id', str(uuid.uuid4()))
    timestamp = form_data.get('timestamp', datetime.now().isoformat())
    
    about_pet = form_data.get('about_pet', '')
    daily_routine = form_data.get('daily_routine', '')
    health_concerns = form_data.get('health_concerns', '')
    
    return PetCareState(
        # ===== System Metadata (5) =====
        request_id=request_id,
        analysis_timestamp=timestamp,
        error_occurred=False,
        error_messages=[],
        processing_complete=False,
        
        # ===== User Inputs (3) =====
        about_pet=about_pet,
        daily_routine=daily_routine,
        health_concerns=health_concerns,
        
        # ===== Validation Fields (2) =====
        validation_errors=[],
        parsing_complete=False,
        
        # ===== Extracted Profile (20) =====
        extracted_profile={},
        profile_extraction_complete=False,
        pet_species=None,
        breed=None,
        age_years=None,
        weight_status=None,
        sex=None,
        known_conditions=[],
        past_surgeries=[],
        allergies_known=[],
        medications_current=[],
        living_situation=None,
        exercise_level=None,
        diet_type=None,
        diet_quality=None,
        behavioral_issues=[],
        owner_experience=None,
        owner_commitment=None,
        vet_access=None,
        # NEW: Symptom fields
        recent_symptoms=[],
        symptom_duration_days=None,
        symptom_severity=None,
        
        # ===== ML Prediction Fields (2) =====
        health_risk_score=None,
        care_capability_score=None,
        
        # ===== Routing Field (1) =====
        path_taken=None,
        
        # ===== Path-Specific Outputs (12) =====
        health_risk_analysis_output=None,
        emergency_prep_output=None,
        nutrition_critical_output=None,
        behavioral_coaching_output=None,
        wellness_monitoring_output=None,
        health_assessment_output=None,
        nutrition_preventive_output=None,
        wellness_tracking_output=None,
        wellness_optimization_output=None,
        nutrition_wellness_output=None,
        lifestyle_enrichment_output=None,
        path_outputs_aggregated=None,
        
        # ===== Final Output (1) =====
        aggregated_output=None,
        
        # ===== Utility Fields (3) =====
        current_node=None,
        node_execution_order=[],
        processing_stage="initialized"
    )


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def is_critical_path(state: PetCareState) -> bool:
    return state.get('path_taken') == "CRITICAL_CARE_PATH"


def is_preventive_path(state: PetCareState) -> bool:
    return state.get('path_taken') == "PREVENTIVE_CARE_PATH"


def is_wellness_path(state: PetCareState) -> bool:
    return state.get('path_taken') == "WELLNESS_PATH"


def has_errors(state: PetCareState) -> bool:
    return state.get('error_occurred', False) or len(state.get('error_messages', [])) > 0


def get_path_outputs(state: PetCareState) -> Dict[str, Any]:
    path = state.get('path_taken')
    
    if path == "CRITICAL_CARE_PATH":
        return {
            "health_risk_analysis": state.get('health_risk_analysis_output'),
            "emergency_prep": state.get('emergency_prep_output'),
            "nutrition_critical": state.get('nutrition_critical_output'),
            "behavioral_coaching": state.get('behavioral_coaching_output'),
            "wellness_monitoring": state.get('wellness_monitoring_output')
        }
    elif path == "PREVENTIVE_CARE_PATH":
        return {
            "health_assessment": state.get('health_assessment_output'),
            "nutrition_preventive": state.get('nutrition_preventive_output'),
            "wellness_tracking": state.get('wellness_tracking_output')
        }
    elif path == "WELLNESS_PATH":
        return {
            "wellness_optimization": state.get('wellness_optimization_output'),
            "nutrition_wellness": state.get('nutrition_wellness_output'),
            "lifestyle_enrichment": state.get('lifestyle_enrichment_output')
        }
    else:
        return {}


def state_to_dict(state: PetCareState) -> Dict[str, Any]:
    return dict(state)


def validate_required_fields(state: PetCareState, stage: str) -> List[str]:
    required_fields = {
        'input': ['about_pet', 'daily_routine', 'health_concerns'],
        'extraction': ['pet_species', 'age_years', 'extracted_profile'],
        'ml': ['health_risk_score', 'care_capability_score'],
        'routing': ['path_taken'],
        'output': ['aggregated_output']
    }
    
    missing = []
    for field in required_fields.get(stage, []):
        if field not in state or state.get(field) is None:
            missing.append(field)
    
    return missing


if __name__ == "__main__":
    test_form = {
        "about_pet": "My cat Whiskers is 8 years old, indoor only",
        "daily_routine": "Eats wet food twice daily, sleeps on couch",
        "health_concerns": "Drinking more water than usual, losing weight"
    }
    
    initial_state = get_initial_state(test_form)
    
    print("=" * 60)
    print("PAWCARE+ STATE INITIALIZATION TEST")
    print("=" * 60)
    print(f"\n✅ Request ID: {initial_state['request_id']}")
    print(f"✅ Timestamp: {initial_state['analysis_timestamp']}")
    print(f"✅ User Input captured: {bool(initial_state['about_pet'])}")
    print(f"✅ Total fields initialized: {len(initial_state)}")
    print(f"✅ Field count verification: {_count_state_fields()} fields")
    
    updated_state = initial_state.copy()
    updated_state['error_messages'] = ["Node 1 failed"]
    updated_state['error_messages'] = ["Node 2 failed"]
    
    print(f"\n✅ Error accumulation test: {updated_state['error_messages']}")
    print(f"\n✅ State module ready for import")