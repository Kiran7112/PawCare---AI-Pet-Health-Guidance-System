# agents/health_assessment_preventive_llm.py

from typing import Dict, Any, List
import logging

from agents.base_llm_agent import BaseLLMAgent
from utils.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


class HealthAssessmentPreventiveAgent(BaseLLMAgent):

    REQUIRED_FIELDS = [
        "preventive_assessment",
        "key_health_areas",
        "recommended_checkups",
        "prevention_strategies"
    ]

    PREVENTIVE_CATEGORIES = {
        "dog": ["Vaccinations", "Parasite prevention", "Dental health", "Weight management", "Joint health"],
        "cat": ["Vaccinations", "Parasite prevention", "Dental health", "Weight management", "Kidney health"],
        "rabbit": ["Dental health", "GI motility", "Parasite prevention", "Weight monitoring"]
    }

    def __init__(self, client: OpenAIClient):
        super().__init__(
            client=client,
            agent_name="HealthAssessmentPreventive",
            default_temperature=0.4,
            default_max_tokens=400
        )

    # ================= CORE =================

    def generate(self, profile: Dict[str, Any], ml_results: Dict[str, Any]) -> Dict[str, Any]:
        return self.generate_health_assessment(profile, ml_results)

    def generate_health_assessment(
        self,
        profile: Dict[str, Any],
        ml_results: Dict[str, Any]
    ) -> Dict[str, Any]:

        try:
            pet = self._extract_pet(profile)
            risk = self._extract_risk(ml_results)

            result = self._generate_with_prompt(
                system_prompt=self._system_prompt(),
                user_prompt=self._user_prompt(pet, risk),
                required_fields=self.REQUIRED_FIELDS
            )

            if not result.get("_generation_success"):
                return self._error_response(pet, risk, result.get("_error"))

            assessment = {k: v for k, v in result.items() if not k.startswith("_")}
            return {
                "health_assessment": self._format(assessment, pet),
                "status": "success"
            }

        except Exception as e:
            logger.error(f"{self.agent_name}: {e}")
            return self._error_response(
                self._extract_pet(profile),
                self._extract_risk(ml_results),
                str(e)
            )

    # ================= EXTRACTION =================

    def _extract_pet(self, p: Dict[str, Any]) -> Dict[str, Any]:
        species = str(p.get("pet_species", "unknown")).capitalize()
        age = p.get("age_years", 0)

        return {
            "species": species,
            "breed": str(p.get("breed", "unknown")).capitalize(),
            "age": age,
            "age_category": "senior" if age > 7 else "adult" if age > 1 else "young",
            "conditions": p.get("known_conditions", []) or [],
            "conditions_text": self._format_list_for_prompt(p.get("known_conditions", []), "None"),
            "weight": p.get("weight_status", "unknown"),
            "exercise": p.get("exercise_level", "unknown"),
            "diet": f"{p.get('diet_type', 'unknown')} ({p.get('diet_quality', 'unknown')})",
            "categories": self.PREVENTIVE_CATEGORIES.get(species.lower(), ["General wellness", "Dental health"])
        }

    def _extract_risk(self, r: Dict[str, Any]) -> Dict[str, Any]:
        score = r.get("health_risk_score", 0.45)

        return {
            "score": score,
            "percent": score * 100,
            "category": (
                "ELEVATED" if score > 0.6 else
                "MODERATE" if score > 0.45 else
                "MILD" if score > 0.3 else "LOW"
            ),
            "care": r.get("care_capability_score", 50.0)
        }

    # ================= PROMPTS =================

    def _system_prompt(self) -> str:
        return """You are a preventive veterinary specialist.

Return JSON with:
- preventive_assessment (2-3 sentences)
- key_health_areas (3-5 items)
- recommended_checkups
- prevention_strategies (3-5 items)

Be concise and practical."""

    def _user_prompt(self, pet: Dict[str, Any], risk: Dict[str, Any]) -> str:
        return f"""
Species: {pet['species']}
Breed: {pet['breed']}
Age: {pet['age']} ({pet['age_category']})
Weight: {pet['weight']}
Exercise: {pet['exercise']}
Diet: {pet['diet']}
Conditions: {pet['conditions_text']}

Risk Score: {risk['percent']:.1f}% ({risk['category']})
Care Capability: {risk['care']}/100

Focus Areas: {", ".join(pet['categories'])}
"""

    # ================= FORMAT =================

    def _format(self, a: Dict[str, Any], pet: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "preventive_assessment": str(a.get(
                "preventive_assessment",
                f"This {pet['species'].lower()} requires preventive care monitoring."
            )),
            "key_health_areas": (a.get("key_health_areas") or ["General wellness", "Dental health", "Weight"])[:5],
            "recommended_checkups": str(a.get(
                "recommended_checkups",
                "Annual wellness exam recommended."
            )),
            "prevention_strategies": (a.get("prevention_strategies") or [
                "Regular vet visits",
                "Maintain healthy weight",
                "Routine dental care"
            ])[:5]
        }

    # ================= FALLBACK =================

    def _error_response(self, pet, risk, err):
        return {
            "health_assessment": {
                "preventive_assessment": f"{pet['species']} has moderate risk. Preventive care needed.",
                "key_health_areas": ["General wellness", "Dental", "Weight"],
                "recommended_checkups": "Annual vet visits",
                "prevention_strategies": [
                    "Routine checkups",
                    "Balanced diet",
                    "Exercise regularly"
                ]
            },
            "status": "error",
            "message": err or "Generation failed"
        }


# ================= CONVENIENCE =================
def generate(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Implementation of abstract generate method from BaseLLMAgent.
        Delegates to the specific generation method.
        """
        # Call the specific method based on what the agent does
        return self.generate_preventive_health_assessment(*args, **kwargs)
def generate_preventive_health_assessment(
    client: OpenAIClient,
    profile: Dict[str, Any],
    ml_results: Dict[str, Any]
) -> Dict[str, Any]:
    return HealthAssessmentPreventiveAgent(client).generate(profile, ml_results)