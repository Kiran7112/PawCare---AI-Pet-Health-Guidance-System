
"""
Pet Profile Extractor LLM Agent for PawCare+.
Extracts structured profile fields from free-text pet owner descriptions using OpenAI.
"""

from typing import Dict, Any, List, Optional
import logging

from agents.base_llm_agent import BaseLLMAgent
from utils.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


class PetProfileExtractorAgent(BaseLLMAgent):
    """
    LLM agent for extracting structured pet profile from natural language inputs.
    """
    
    # Define fields to extract
    EXTRACTION_FIELDS = {
        "pet_species": "Species type (dog, cat, rabbit, bird, reptile, other)",
        "breed": "Specific breed or mix",
        "age_years": "Age as integer (approximate if not specified, default 0 if unknown)",
        "weight_status": "Weight category (underweight, normal, overweight, obese, unknown)",
        "sex": "Gender (male, female, unknown)",
        "known_conditions": "List of known health conditions (empty list if none)",
        "past_surgeries": "List of previous surgeries (empty list if none)",
        "allergies_known": "List of known allergies (empty list if none)",
        "medications_current": "List of current medications (empty list if none)",
        "living_situation": "Environment type (apartment, house, farm, outdoor, mixed)",
        "exercise_level": "Activity level (sedentary, light, moderate, very active, unknown)",
        "diet_type": "Diet category (kibble, raw, wet, mixed, homemade, prescription, unknown)",
        "diet_quality": "Diet quality rating (poor, average, good, premium, unknown)",
        "behavioral_issues": "List of behavioral concerns (empty list if none)",
        "owner_experience": "Owner expertise level (novice, experienced, expert, unknown)",
        "vet_access": "Veterinary access availability (regular, emergency only, limited, none, unknown)",
        "owner_commitment": "Owner dedication level (casual, dedicated, obsessive, unknown)",
        "recent_symptoms": "List of current symptoms (e.g., coughing, increased thirst, lethargy, weight loss)",
        "symptom_duration_days": "How many days symptoms have been present (0 if none)",
        "symptom_severity": "Severity of symptoms (mild, moderate, severe, unknown)"
    }
    
    DEFAULT_VALUES = {
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
    
    def __init__(self, client: OpenAIClient):
        super().__init__(
            client=client,
            agent_name="PetProfileExtractor",
            default_temperature=0.3,
            default_max_tokens=900
        )
        logger.info(f"PetProfileExtractorAgent initialized")
    
    def generate(self, *args, **kwargs) -> Dict[str, Any]:
        if args and len(args) > 0 and isinstance(args[0], dict):
            return self.extract_pet_profile(args[0])
        elif 'raw_inputs' in kwargs:
            return self.extract_pet_profile(kwargs['raw_inputs'])
        else:
            return {
                "status": "error",
                "message": "Invalid arguments to generate()",
                "extracted_profile": self.DEFAULT_VALUES.copy()
            }
    
    def extract_pet_profile(self, raw_inputs: Dict[str, str]) -> Dict[str, Any]:
        try:
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(raw_inputs)
            required_fields = list(self.EXTRACTION_FIELDS.keys())
            
            result = self._generate_with_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                required_fields=required_fields,
                temperature=self.default_temperature,
                max_tokens=self.default_max_tokens
            )
            
            if not result.get("_generation_success", False):
                return {
                    "extracted_profile": self.DEFAULT_VALUES.copy(),
                    "status": "error",
                    "message": result.get("_error", "Unknown error"),
                    "extraction_confidence": 0.0
                }
            
            extracted_profile = {k: v for k, v in result.items() if not k.startswith('_')}
            extracted_profile = self._validate_and_clean_profile(extracted_profile)
            confidence = self._calculate_extraction_confidence(extracted_profile)
            
            return {
                "extracted_profile": extracted_profile,
                "status": "success",
                "message": "Profile extracted successfully",
                "extraction_confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Unexpected error in profile extraction: {str(e)}")
            return {
                "extracted_profile": self.DEFAULT_VALUES.copy(),
                "status": "error",
                "message": f"Extraction failed: {str(e)}",
                "extraction_confidence": 0.0
            }
    
    def _build_system_prompt(self) -> str:
        field_descriptions = "\n".join([
            f"  - {field}: {desc}" 
            for field, desc in self.EXTRACTION_FIELDS.items()
        ])
        
        return f"""You are a veterinary intake specialist expert at extracting structured pet information from owner descriptions.

Extract exactly 20 specific fields. Mark as "unknown" when uncertain.

EXTRACTION FIELDS:
{field_descriptions}

EXTRACTION RULES:
1. List fields: return as JSON array, empty list [] if none
2. Categorical fields: use allowed values only
3. age_years: extract as integer, default 0
4. symptoms: extract all mentioned symptoms as list
5. Be conservative - only extract explicitly stated information
6. Return ONLY valid JSON matching the field names exactly"""
    
    def _build_user_prompt(self, raw_inputs: Dict[str, str]) -> str:
        about_pet = raw_inputs.get('about_pet', '').strip()
        daily_routine = raw_inputs.get('daily_routine', '').strip()
        health_concerns = raw_inputs.get('health_concerns', '').strip()
        
        return f"""Extract the complete pet profile:

=== ABOUT THE PET ===
{about_pet}

=== DAILY ROUTINE ===
{daily_routine}

=== HEALTH CONCERNS ===
{health_concerns}

Extract all 20 fields. Return a JSON object."""
    
    def _validate_and_clean_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = {}
        
        for field, default_value in self.DEFAULT_VALUES.items():
            value = profile.get(field, default_value)
            
            if field in ["known_conditions", "past_surgeries", "allergies_known", 
                        "medications_current", "behavioral_issues", "recent_symptoms"]:
                if not isinstance(value, list):
                    if isinstance(value, str) and value:
                        value = [item.strip() for item in value.split(',')]
                    else:
                        value = []
            
            elif field in ["age_years", "symptom_duration_days"]:
                try:
                    value = int(float(value)) if value not in [None, "", "unknown"] else 0
                except (ValueError, TypeError):
                    value = 0
            
            elif field in ["pet_species", "breed", "weight_status", "sex", "living_situation",
                          "exercise_level", "diet_type", "diet_quality", "owner_experience",
                          "vet_access", "owner_commitment", "symptom_severity"]:
                if not isinstance(value, str):
                    value = str(value) if value is not None else "unknown"
                value = value.strip()[:50]
                if not value or value.lower() in ["null", "none", "nil"]:
                    value = "unknown"
            
            cleaned[field] = value
        
        return cleaned
    
    def _calculate_extraction_confidence(self, profile: Dict[str, Any]) -> float:
        if not profile:
            return 0.0
        
        total_fields = len(self.DEFAULT_VALUES)
        meaningful_fields = 0
        
        for field, value in profile.items():
            if field in ["known_conditions", "past_surgeries", "allergies_known",
                        "medications_current", "behavioral_issues", "recent_symptoms"]:
                if isinstance(value, list) and len(value) > 0:
                    meaningful_fields += 1
            elif field in ["age_years", "symptom_duration_days"]:
                if isinstance(value, (int, float)) and value > 0:
                    meaningful_fields += 1
            else:
                if isinstance(value, str) and value.lower() not in ["unknown", "", "none"]:
                    meaningful_fields += 1
        
        return meaningful_fields / total_fields
    
    def get_default_profile(self) -> Dict[str, Any]:
        return self.DEFAULT_VALUES.copy()


def extract_pet_profile(client: OpenAIClient, about_pet: str, daily_routine: str, health_concerns: str) -> Dict[str, Any]:
    agent = PetProfileExtractorAgent(client)
    return agent.extract_pet_profile({
        "about_pet": about_pet,
        "daily_routine": daily_routine,
        "health_concerns": health_concerns
    })