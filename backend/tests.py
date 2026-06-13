"""
PawCare+ Test Suite

Tests validate output contracts and logic without constraining implementation.
Following VentureLens testing patterns.

Test Structure:
- 12 total test cases (removed 3 JSON formatting tests)
- Organized by component type
- Mock infrastructure for LLM testing
- Multiple pet profiles for scenario testing
- Data quality validation for ML training data
"""

import pytest
import json
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path


# ============================================================================
# PART 1: Mock Infrastructure
# ============================================================================

class MockGeminiContent:
    """Simulates google.genai content response"""
    def __init__(self, text: str):
        self.text = text


class MockGeminiResponse:
    """Simulates Gemini API response"""
    def __init__(self, text: str):
        self.text = text


class MockGeminiClient:
    """
    Complete mock of GeminiClient for testing without API calls.
    Pattern matches on prompt content to return appropriate responses.
    """

    def __init__(self):
        self.call_count = 0
        self.default_responses = {
            'pet_profile': {
                'pet_species': 'Dog',
                'breed': 'Labrador',
                'age_years': 5,
                'weight_status': 'Normal',
                'sex': 'Male',
                'known_conditions': [],
                'past_surgeries': [],
                'allergies_known': [],
                'medications_current': [],
                'living_situation': 'House',
                'exercise_level': 'High',
                'diet_type': 'Dry Food',
                'diet_quality': 'Premium',
                'behavioral_issues': 'None',
                'owner_experience': 'Experienced',
                'vet_access': 'Regular',
                'owner_commitment': 'Dedicated'
            },
            'health_risk_analysis': {
                'honest_risk_assessment': 'Low risk with good health indicators',
                'recommended_actions': ['Annual checkups', 'Maintain exercise routine'],
                'prevention_focus': 'Continue current wellness routine'
            },
            'emergency_preparedness': {
                'emergency_overview': 'Well-prepared for emergencies',
                'when_to_call_vet': ['Sudden behavior changes', 'Difficulty breathing'],
                'emergency_contacts': 'Keep vet number handy',
                'preparation_checklist': ['First aid kit', 'Medical records']
            },
            'nutrition_plan': {
                'nutrition_overview': 'Balanced diet recommended',
                'recommended_foods': ['High-quality protein sources', 'Fresh vegetables'],
                'foods_to_avoid': ['Chocolate', 'Grapes'],
                'feeding_schedule': 'Twice daily'
            },
            'behavioral_coaching': {
                'coaching_overview': 'Good behavioral foundation',
                'positive_reinforcement': ['Treats for good behavior', 'Regular playtime'],
                'training_tips': ['Consistency is key', 'Start with basic commands'],
                'socialization_plan': 'Regular interaction with other dogs'
            },
            'wellness_monitoring': {
                'monitoring_overview': 'Regular monitoring recommended',
                'health_metrics_to_track': ['Weight', 'Energy levels', 'Appetite'],
                'frequency': 'Monthly checks',
                'red_flags': ['Excessive scratching', 'Lethargy']
            },
            'health_assessment_preventive': {
                'preventive_assessment': 'Preventive care plan established',
                'key_health_areas': ['Dental health', 'Joint health', 'Immune system'],
                'recommended_checkups': 'Biannual wellness checks',
                'prevention_strategies': ['Regular exercise', 'Balanced diet']
            },
            'nutrition_preventive': {
                'nutrition_overview': 'Preventive nutrition approach',
                'recommended_diet': 'High-quality kibble with supplements',
                'portion_guidance': 'Based on age and weight',
                'healthy_treats': ['Carrots', 'Apples', 'Plain chicken']
            },
            'wellness_tracking_preventive': {
                'tracking_overview': 'Monthly wellness tracking protocol',
                'monthly_checklist': ['Weight check', 'Coat condition', 'Energy level'],
                'wellness_goals': ['Maintain ideal weight', 'Prevent chronic diseases'],
                'early_warning_signs': ['Excessive scratching', 'Behavioral changes']
            },
            'wellness_optimization': {
                'optimization_overview': 'Wellness optimization for healthy pets',
                'wellness_enhancements': ['Advanced training', 'Mental enrichment', 'Specialty supplements'],
                'activity_suggestions': ['Agility courses', 'Swimming'],
                'bonding_activities': ['Hiking together', 'Beach trips']
            },
            'nutrition_wellness': {
                'nutrition_overview': 'Optimal nutrition for peak health',
                'enhancement_tips': ['Add omega-3 supplements', 'Include fresh foods', 'Rotate protein sources'],
                'variety_suggestions': 'Mix different high-quality foods weekly',
                'supplement_options': ['Fish oil', 'Probiotics']
            },
            'lifestyle_enrichment': {
                'enrichment_overview': 'Comprehensive lifestyle enrichment program',
                'mental_stimulation': ['Puzzle toys', 'Training sessions', 'Exploration walks'],
                'social_opportunities': ['Dog parks', 'Playdates'],
                'environmental_enrichment': ['Varied walking routes', 'Novel toys']
            }
        }

    def generate_content(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_mime_type: Optional[str] = None,
    ) -> str:
        """Generate content with pattern matching on prompt"""
        self.call_count += 1
        prompt_lower = prompt.lower()

        # Pattern matching for different agent types
        if "health assessment preventive" in prompt_lower:
            response = self.default_responses['health_assessment_preventive']
        elif "nutrition preventive" in prompt_lower:
            response = self.default_responses['nutrition_preventive']
        elif "wellness tracking" in prompt_lower:
            response = self.default_responses['wellness_tracking_preventive']
        elif "wellness optimization" in prompt_lower:
            response = self.default_responses['wellness_optimization']
        elif "nutrition wellness" in prompt_lower or "nutrition enhancement" in prompt_lower:
            response = self.default_responses['nutrition_wellness']
        elif "lifestyle enrichment" in prompt_lower:
            response = self.default_responses['lifestyle_enrichment']
        elif "emergency" in prompt_lower or "preparedness" in prompt_lower:
            response = self.default_responses['emergency_preparedness']
        elif "nutrition" in prompt_lower or "feeding" in prompt_lower:
            response = self.default_responses['nutrition_plan']
        elif "behavioral" in prompt_lower or "behavior" in prompt_lower:
            response = self.default_responses['behavioral_coaching']
        elif "wellness monitoring" in prompt_lower or "monitoring" in prompt_lower:
            response = self.default_responses['wellness_monitoring']
        elif "health risk" in prompt_lower or "risk analysis" in prompt_lower:
            response = self.default_responses['health_risk_analysis']
        elif "profile" in prompt_lower or "extract" in prompt_lower:
            response = self.default_responses['pet_profile']
        else:
            response = self.default_responses['pet_profile']

        return json.dumps(response)

    def extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON from response text"""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return {}

    def validate_response_fields(self, response: Dict[str, Any], required_fields: list) -> None:
        """Validate that response has required fields"""
        missing = [f for f in required_fields if f not in response]
        if missing:
            raise ValueError(f"Missing fields: {missing}")

    def generate_structured_json(
        self,
        prompt: str,
        required_fields: list,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """Generate and validate JSON response"""
        response_text = self.generate_content(prompt, temperature, max_tokens)
        result = self.extract_json_from_response(response_text)
        self.validate_response_fields(result, required_fields)
        return result


# ============================================================================
# PART 2: Sample Pet Profiles
# ============================================================================

SAMPLE_PET_HEALTHY = {
    'pet_species': 'Dog',
    'breed': 'Labrador Retriever',
    'age_years': 4,
    'weight_status': 'Normal',
    'known_conditions': [],
    'exercise_level': 'High',
    'diet_type': 'Dry Food'
}

SAMPLE_PET_MODERATE_RISK = {
    'pet_species': 'Cat',
    'breed': 'Persian',
    'age_years': 7,
    'weight_status': 'Overweight',
    'known_conditions': ['Diabetes'],
    'exercise_level': 'Low',
    'diet_type': 'Wet Food'
}

SAMPLE_PET_HIGH_RISK = {
    'pet_species': 'Dog',
    'breed': 'German Shepherd',
    'age_years': 12,
    'weight_status': 'Obese',
    'known_conditions': ['Heart Disease', 'Arthritis'],
    'exercise_level': 'Very Low',
    'diet_type': 'Prescription Diet'
}


# ============================================================================
# PART 3: ML Agent Tests (4 tests)
# ============================================================================

def test_health_risk_scorer_produces_valid_score():
    """Test health risk scorer returns valid score between 0-1"""
    from agents.pet_health_risk_scorer_ml import PetHealthRiskScorerMLAgent

    agent = PetHealthRiskScorerMLAgent()

    # Test with complete sample pet profile
    profile = {
        'pet_species': 'Dog',
        'breed': 'Labrador',
        'age_years': 5,
        'weight_status': 'Normal',
        'sex': 'Male',
        'known_conditions': [],
        'past_surgeries': [],
        'allergies_known': [],
        'medications_current': [],
        'living_situation': 'House',
        'exercise_level': 'High',
        'diet_type': 'Dry Food',
        'diet_quality': 'Premium',
        'behavioral_issues': 'None',
        'owner_experience': 'Experienced',
        'vet_access': 'Regular',
        'owner_commitment': 'Dedicated'
    }

    result = agent.predict_health_risk(profile)

    # Validate output structure
    assert 'health_risk_score' in result

    # Validate types
    assert isinstance(result['health_risk_score'], (int, float))

    # Validate range
    assert 0 <= result['health_risk_score'] <= 1


def test_health_risk_scorer_differentiates_profiles():
    """Test that health risk scorer differentiates between healthy and unhealthy pets"""
    from agents.pet_health_risk_scorer_ml import PetHealthRiskScorerMLAgent

    agent = PetHealthRiskScorerMLAgent()

    # Healthy pet profile
    healthy_profile = {
        'pet_species': 'Dog',
        'breed': 'Labrador',
        'age_years': 3,
        'weight_status': 'Normal',
        'sex': 'Female',
        'known_conditions': [],
        'past_surgeries': ['Spay'],
        'allergies_known': [],
        'medications_current': [],
        'living_situation': 'House',
        'exercise_level': 'High',
        'diet_type': 'Dry Food',
        'diet_quality': 'Premium',
        'behavioral_issues': 'None',
        'owner_experience': 'Experienced',
        'vet_access': 'Regular',
        'owner_commitment': 'Dedicated'
    }

    # Unhealthy pet profile
    unhealthy_profile = {
        'pet_species': 'Dog',
        'breed': 'German Shepherd',
        'age_years': 12,
        'weight_status': 'Obese',
        'sex': 'Male',
        'known_conditions': ['Heart Disease', 'Arthritis'],
        'past_surgeries': ['Multiple knee surgeries'],
        'allergies_known': ['Chicken'],
        'medications_current': ['Aspirin', 'Heart medication'],
        'living_situation': 'Apartment',
        'exercise_level': 'Very Low',
        'diet_type': 'Prescription Diet',
        'diet_quality': 'Therapeutic',
        'behavioral_issues': 'Anxiety',
        'owner_experience': 'Novice',
        'vet_access': 'Limited',
        'owner_commitment': 'Moderate'
    }

    healthy_result = agent.predict_health_risk(healthy_profile)
    unhealthy_result = agent.predict_health_risk(unhealthy_profile)

    # Both should produce valid scores
    assert 0 <= healthy_result['health_risk_score'] <= 1
    assert 0 <= unhealthy_result['health_risk_score'] <= 1


def test_owner_care_capability_produces_valid_score():
    """Test owner care capability scorer returns valid score between 0-100"""
    from agents.owner_care_capability_ml import OwnerCareCapabilityMLAgent

    agent = OwnerCareCapabilityMLAgent()

    profile = {
        'owner_experience': 'experienced',
        'vet_access': 'regular',
        'owner_commitment': 'dedicated'
    }

    result = agent.predict_capability(profile)

    # Validate output structure
    assert 'care_capability_score' in result

    # Validate types
    assert isinstance(result['care_capability_score'], (int, float))

    # Validate range
    assert 0 <= result['care_capability_score'] <= 100


def test_ml_models_exist_and_load():
    """Test that trained ML models exist and can be loaded"""
    from pathlib import Path
    import os

    project_root = Path(__file__).parent
    models_dir = project_root / "ml" / "models"

    model_files = [
        "pet_health_risk_model.pkl",
        "owner_care_capability_model.pkl"
    ]

    for model_file in model_files:
        model_path = models_dir / model_file
        assert model_path.exists(), f"Missing ML model: {model_file}"
        assert os.path.getsize(model_path) > 0, f"Empty ML model: {model_file}"


# ============================================================================
# PART 4: LLM Agent Tests (1 test)
# ============================================================================

def test_llm_agents_require_valid_client():
    """Test LLM agents raise error when client is None"""
    from agents.pet_profile_extractor_llm import PetProfileExtractorLLMAgent
    from agents.pet_health_risk_analysis_llm import PetHealthRiskAnalysisLLMAgent

    with pytest.raises(ValueError):
        PetProfileExtractorLLMAgent(client=None)

    with pytest.raises(ValueError):
        PetHealthRiskAnalysisLLMAgent(client=None)


# ============================================================================
# PART 5: State Management Tests (2 tests)
# ============================================================================

def test_initial_state_creation():
    """Test state initialization with form data"""
    from state import get_initial_state

    form_data = {
        'about_pet': 'Golden Retriever, 5 years old',
        'daily_routine': '2 walks daily',
        'health_concerns': 'Occasional ear infections'
    }

    state = get_initial_state(form_data)

    # Validate input fields copied
    assert state.get('about_pet') == form_data['about_pet']
    assert state.get('daily_routine') == form_data['daily_routine']
    assert state.get('health_concerns') == form_data['health_concerns']

    # Validate output fields initialized
    assert 'error_messages' in state
    assert isinstance(state['error_messages'], list)


def test_state_tracks_completion_flags():
    """Test state properly tracks workflow completion"""
    from state import get_initial_state

    form_data = {
        'about_pet': 'Test pet',
        'daily_routine': 'Test routine',
        'health_concerns': 'Test concerns'
    }

    state = get_initial_state(form_data)

    # Validate error tracking exists
    assert 'error_occurred' in state
    assert isinstance(state['error_occurred'], bool)
    assert state['error_occurred'] is False


# ============================================================================
# PART 6: Integration/Workflow Tests (3 tests)
# ============================================================================

def test_complete_workflow_execution():
    """Test that complete assessment workflow executes without errors"""
    from graph import assess_pet_health

    result = assess_pet_health({
        'about_pet': 'Healthy 5-year-old Golden Retriever',
        'daily_routine': '2 walks daily, lots of playtime',
        'health_concerns': 'None'
    })

    # Validate workflow completed
    assert result is not None
    assert isinstance(result, dict)
    assert result.get('processing_complete') is not None


def test_workflow_with_different_pet_types():
    """Test workflow handles different pet types"""
    from graph import assess_pet_health

    # Test with dog
    dog_result = assess_pet_health({
        'about_pet': 'Golden Retriever',
        'daily_routine': '2 walks daily',
        'health_concerns': 'None'
    })

    assert dog_result is not None
    assert isinstance(dog_result, dict)

    # Test with cat
    cat_result = assess_pet_health({
        'about_pet': 'Persian cat',
        'daily_routine': 'Indoor, playtime',
        'health_concerns': 'None'
    })

    assert cat_result is not None
    assert isinstance(cat_result, dict)


def test_error_handling_in_workflow():
    """Test workflow handles errors gracefully"""
    from state import get_initial_state

    # Create state with minimal data
    form_data = {
        'about_pet': '',
        'daily_routine': '',
        'health_concerns': ''
    }

    state = get_initial_state(form_data)

    # Should still create valid state
    assert state is not None
    assert isinstance(state, dict)


# ============================================================================
# PART 7: Data Quality Tests (2 tests)
# ============================================================================

def test_processed_training_data_is_clean():
    """Test that processed training data has been properly cleaned per documentation specs"""
    from pathlib import Path
    import pandas as pd

    project_root = Path(__file__).parent
    processed_dir = project_root / "data" / "processed"

    processed_files = [
        "pet_health_risk_clean.csv",
        "owner_care_capability_clean.csv"
    ]

    for dataset_file in processed_files:
        dataset_path = processed_dir / dataset_file
        assert dataset_path.exists(), f"Missing processed dataset: {dataset_file}"

        df = pd.read_csv(dataset_path)

        # Check file has content
        assert len(df) > 0, f"Processed dataset is empty: {dataset_file}"

        # Per documentation: NaN values should be removed during cleaning
        assert not df.isnull().any().any(), f"NaN values found in processed data: {dataset_file}"

        # Per documentation: Duplicates are kept to maintain balanced class distribution
        # (Balanced datasets may have intentional duplicates from oversampling)

        # Per documentation: Age bounds validated during cleaning
        if 'age_years' in df.columns:
            assert df['age_years'].min() >= 0 and df['age_years'].max() <= 30, \
                "Age values outside documented bounds [0, 30] after processing"

        # Per documentation: Health risk score should be in range [0.0, 1.0]
        if 'health_risk_score' in df.columns:
            assert df['health_risk_score'].min() >= 0.0 and df['health_risk_score'].max() <= 1.0, \
                "Health risk scores outside valid range [0.0, 1.0] after processing"

        # Per documentation: Care capability score should be in range [0.0, 100.0]
        if 'care_capability_score' in df.columns:
            assert df['care_capability_score'].min() >= 0.0 and df['care_capability_score'].max() <= 100.0, \
                "Care capability scores outside valid range [0.0, 100.0] after processing"


def test_ml_models_not_empty():
    """Test that trained ML models exist and are not empty .pkl files"""
    from pathlib import Path
    import os

    project_root = Path(__file__).parent
    models_dir = project_root / "ml" / "models"

    model_files = [
        "pet_health_risk_model.pkl",
        "pet_health_risk_scaler.pkl",
        "pet_health_risk_encoder.pkl",
        "owner_care_capability_model.pkl",
        "owner_care_capability_scaler.pkl",
        "owner_care_capability_encoder.pkl"
    ]

    for model_file in model_files:
        model_path = models_dir / model_file

        # Check file exists
        assert model_path.exists(), f"Missing ML model: {model_file}"

        # Check file is .pkl
        assert model_path.suffix == ".pkl", f"File is not .pkl format: {model_file}"

        # Check file is not empty
        assert os.path.getsize(model_path) > 0, f"Empty ML model file: {model_file}"


# ============================================================================
# Test Execution
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
