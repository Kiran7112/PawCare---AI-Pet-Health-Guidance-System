# agents/emergency_preparedness_llm.py

from typing import Dict, Any, List
import logging

from agents.base_llm_agent import BaseLLMAgent
from utils.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


class EmergencyPreparednessAgent(BaseLLMAgent):

    REQUIRED_FIELDS = [
        "emergency_overview",
        "emergency_contacts",
        "first_aid_supplies",
        "crisis_procedures",
        "when_to_call_vet",
        "evacuation_plan",
        "medical_history_prep",
        "financial_prep"
    ]

    def __init__(self, client: OpenAIClient):
        super().__init__(
            client=client,
            agent_name="EmergencyPreparedness",
            default_temperature=0.5,
            default_max_tokens=1200
        )

    # ================= MAIN =================
    def generate(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Implementation of abstract generate method from BaseLLMAgent.
        Delegates to the specific generation method.
        """
        # Call the specific method based on what the agent does
        return self.generate_emergency_plan(*args, **kwargs)
    def generate_emergency_plan(
        self,
        profile: Dict[str, Any],
        health: Dict[str, Any]
    ) -> Dict[str, Any]:

        pet = self._extract_pet(profile)
        risk = self._extract_risk(health)

        result = self._generate_with_prompt(
            system_prompt=self._system_prompt(),
            user_prompt=self._user_prompt(pet, risk),
            required_fields=self.REQUIRED_FIELDS
        )

        if not result.get("_generation_success"):
            return {
                "emergency_preparedness": self._fallback(pet, risk),
                "status": "error"
            }

        plan = {k: v for k, v in result.items() if not k.startswith("_")}

        return {
            "emergency_preparedness": plan,
            "status": "success"
        }

    # ================= HELPERS =================

    def _extract_pet(self, p: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": p.get("pet_name", "Pet"),
            "species": p.get("pet_species", "unknown").capitalize(),
            "breed": p.get("breed", "unknown").capitalize(),
            "age": p.get("age_years", 0),
            "weight": p.get("weight_status", ""),
            "conditions": self._format_list_for_prompt(p.get("known_conditions", [])),
            "meds": self._format_list_for_prompt(p.get("medications_current", [])),
            "allergies": self._format_list_for_prompt(p.get("allergies_known", [])),
            "living": p.get("living_situation", "")
        }

    def _extract_risk(self, h: Dict[str, Any]) -> Dict[str, Any]:
        risks = h.get("critical_risk_factors", [])
        warnings = h.get("warning_signs", [])

        if isinstance(risks, str):
            risks = [risks]
        if isinstance(warnings, str):
            warnings = [warnings]

        return {
            "risks": risks,
            "risks_text": "\n".join(risks),
            "warnings": warnings,
            "warnings_text": "\n".join(warnings),
            "urgency": h.get("urgency_timeline", "")
        }

    # ================= PROMPTS =================

    def _system_prompt(self) -> str:
        return """You are a veterinary emergency expert.
Return JSON with:
emergency_overview, emergency_contacts, first_aid_supplies,
crisis_procedures, when_to_call_vet, evacuation_plan,
medical_history_prep, financial_prep."""

    def _user_prompt(self, pet: Dict[str, Any], risk: Dict[str, Any]) -> str:
        return f"""
Pet: {pet['name']} ({pet['species']} - {pet['breed']})
Age: {pet['age']} | Weight: {pet['weight']}

Conditions: {pet['conditions']}
Medications: {pet['meds']}
Allergies: {pet['allergies']}

Risks:
{risk['risks_text']}

Warnings:
{risk['warnings_text']}

Urgency: {risk['urgency']}
Living: {pet['living']}
"""

    # ================= FALLBACK =================

    def _fallback(self, pet: Dict[str, Any], risk: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "emergency_overview": f"{pet['species']} needs emergency planning due to health risks.",
            "emergency_contacts": [
                "Emergency Vet",
                "Regular Vet",
                "Poison Control",
                "Emergency Contact"
            ],
            "first_aid_supplies": [
                "Medical records",
                "Medications",
                "First aid kit",
                "Food & water",
                "Carrier/leash"
            ],
            "crisis_procedures": [
                "Stay calm and assess",
                "Contact vet immediately",
                "Transport safely"
            ],
            "when_to_call_vet": "Call for breathing issues, collapse, or severe symptoms",
            "evacuation_plan": "Prepare kit, carrier, transport plan",
            "medical_history_prep": "Keep digital + physical records ready",
            "financial_prep": "Keep emergency fund and insurance"
        }

    # ================= UTILITIES =================

    def get_plan_summary(self, plan: Dict[str, Any]) -> str:
        return plan.get("emergency_overview", "")

    def extract_essential_contacts(self, plan: Dict[str, Any]) -> List[str]:
        return plan.get("emergency_contacts", [])[:3]

    def get_emergency_kit_checklist(self, plan: Dict[str, Any]) -> List[str]:
        return plan.get("first_aid_supplies", [])


# ================= CONVENIENCE =================

def generate_emergency_plan(
    client: OpenAIClient,
    profile: Dict[str, Any],
    health: Dict[str, Any]
):
    return EmergencyPreparednessAgent(client).generate_emergency_plan(profile, health)