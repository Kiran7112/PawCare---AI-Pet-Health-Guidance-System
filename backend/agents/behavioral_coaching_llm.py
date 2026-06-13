# agents/behavioral_coaching_llm.py

from typing import Dict, Any, List
import logging

from agents.base_llm_agent import BaseLLMAgent
from utils.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


class BehavioralCoachingAgent(BaseLLMAgent):

    REQUIRED_FIELDS = [
        "behavior_assessment",
        "training_strategies",
        "anxiety_management",
        "training_timeline",
        "when_professional",
        "positive_reinforcement_plan",
        "common_triggers"
    ]

    BEHAVIOR_CATEGORIES = {
        "aggression": ["aggression", "biting", "growling", "lunging"],
        "anxiety": ["anxiety", "fear", "stress", "separation anxiety"],
        "destructive": ["chewing", "scratching", "digging"],
        "vocalization": ["barking", "howling", "whining"],
        "house_soiling": ["accidents", "marking"],
        "compulsive": ["pacing", "circling", "obsessive"],
        "fearful": ["shy", "hiding", "cowering"],
        "resource_guarding": ["food guarding", "possessive"]
    }

    def __init__(self, client: OpenAIClient):
        super().__init__(
            client=client,
            agent_name="BehavioralCoaching",
            default_temperature=0.6,
            default_max_tokens=1000
        )

    # ================= MAIN =================
    def generate(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Implementation of abstract generate method from BaseLLMAgent.
        Delegates to the specific generation method.
        """
        # Call the specific method based on what the agent does
        return self.generate_behavior_coaching(*args, **kwargs)
    def generate_behavior_coaching(self, profile: Dict[str, Any]) -> Dict[str, Any]:

        pet = self._extract_pet_info(profile)

        result = self._generate_with_prompt(
            system_prompt=self._system_prompt(),
            user_prompt=self._user_prompt(pet),
            required_fields=self.REQUIRED_FIELDS
        )

        if not result.get("_generation_success"):
            return {
                "behavioral_coaching": self._fallback(pet),
                "status": "error"
            }

        coaching = {k: v for k, v in result.items() if not k.startswith("_")}

        return {
            "behavioral_coaching": coaching,
            "status": "success"
        }

    # ================= HELPERS =================

    def _extract_pet_info(self, profile: Dict[str, Any]) -> Dict[str, Any]:

        issues = profile.get("behavioral_issues", [])
        if isinstance(issues, str):
            issues = [i.strip() for i in issues.split(",")]

        categorized = self._categorize(issues)

        return {
            "name": profile.get("pet_name", "Pet"),
            "species": profile.get("pet_species", "unknown").capitalize(),
            "breed": profile.get("breed", "unknown").capitalize(),
            "age": profile.get("age_years", 0),
            "issues": issues,
            "issues_text": self._format_list_for_prompt(issues),
            "categorized": categorized,
            "categorized_text": self._format_categories(categorized),
            "conditions": self._format_list_for_prompt(profile.get("known_conditions", [])),
            "living": profile.get("living_situation", ""),
            "exercise": profile.get("exercise_level", ""),
            "owner": profile.get("owner_experience", ""),
            "routine": (profile.get("daily_routine", "") or "")[:200]
        }

    def _categorize(self, issues: List[str]) -> Dict[str, List[str]]:
        result = {}

        for issue in issues:
            text = issue.lower()
            matched = False

            for cat, keywords in self.BEHAVIOR_CATEGORIES.items():
                if any(k in text for k in keywords):
                    result.setdefault(cat, []).append(issue)
                    matched = True
                    break

            if not matched:
                result.setdefault("other", []).append(issue)

        return result

    def _format_categories(self, categorized: Dict[str, List[str]]) -> str:
        return "\n".join(
            f"{k}: {', '.join(v)}" for k, v in categorized.items()
        )

    # ================= PROMPTS =================

    def _system_prompt(self) -> str:
        return """You are an animal behavior expert. 
Return a JSON with:
behavior_assessment, training_strategies, anxiety_management,
training_timeline, when_professional,
positive_reinforcement_plan, common_triggers."""

    def _user_prompt(self, pet: Dict[str, Any]) -> str:
        return f"""
Pet: {pet['name']} ({pet['species']} - {pet['breed']})
Age: {pet['age']}
Issues: {pet['issues_text']}

Categories:
{pet['categorized_text']}

Conditions: {pet['conditions']}
Living: {pet['living']}
Exercise: {pet['exercise']}
Owner: {pet['owner']}
Routine: {pet['routine']}
"""

    # ================= FALLBACK =================

    def _fallback(self, pet: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "behavior_assessment": f"{pet['species']} shows {pet['issues_text']}.",
            "training_strategies": [
                "Use positive reinforcement",
                "Maintain routine",
                "Short training sessions"
            ],
            "anxiety_management": "Provide calm environment and safe space",
            "training_timeline": "2-12 weeks gradual improvement",
            "when_professional": "If behavior worsens or is dangerous",
            "positive_reinforcement_plan": "Reward good behavior consistently",
            "common_triggers": ["noise", "strangers", "environment change"]
        }

    # ================= UTILITIES =================

    def get_coaching_summary(self, coaching: Dict[str, Any]) -> str:
        return coaching.get("behavior_assessment", "")

    def get_training_strategies_list(self, coaching: Dict[str, Any]) -> List[str]:
        return coaching.get("training_strategies", [])

    def get_trigger_management_tips(self, coaching: Dict[str, Any]) -> List[str]:
        return coaching.get("common_triggers", [])

    def get_behavioral_issue_category(self, issue: str) -> str:
        text = issue.lower()
        for cat, keys in self.BEHAVIOR_CATEGORIES.items():
            if any(k in text for k in keys):
                return cat
        return "other"


# ================= CONVENIENCE =================

def generate_behavioral_coaching(client: OpenAIClient, profile: Dict[str, Any]):
    return BehavioralCoachingAgent(client).generate_behavior_coaching(profile)