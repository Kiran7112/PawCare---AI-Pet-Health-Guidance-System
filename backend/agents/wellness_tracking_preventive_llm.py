# agents/wellness_tracking_preventive_llm.py
"""
Wellness Tracking Preventive LLM Agent for PawCare+ (Preventive Path).
Generates preventive wellness tracking plans for moderate-risk pets.
"""

from typing import Dict, Any, List, Optional
import logging

from agents.base_llm_agent import BaseLLMAgent
from utils.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


class WellnessTrackingPreventiveAgent(BaseLLMAgent):
    """
    LLM agent for generating preventive wellness tracking plans for moderate-risk pets.
    
    This agent is part of the PREVENTIVE PATH and provides focused tracking guidance including:
    - Overview of tracking strategy
    - Monthly wellness checklist
    - Proactive wellness goals
    - Early warning signs to watch
    """
    
    # Required fields in the output JSON
    REQUIRED_FIELDS = [
        "tracking_overview",
        "monthly_checklist",
        "wellness_goals",
        "early_warning_signs"
    ]
    
    # Species-specific wellness indicators
    SPECIES_INDICATORS = {
        "dog": {
            "appetite": "Monitor for changes in eating habits",
            "energy": "Track activity levels and stamina",
            "coat": "Check coat shine and skin condition",
            "weight": "Monitor weight trends",
            "behavior": "Watch for behavioral changes",
            "mobility": "Observe for stiffness or difficulty moving"
        },
        "cat": {
            "appetite": "Monitor food intake (cats hide illness well)",
            "litter_box": "Track litter box habits",
            "grooming": "Check for changes in grooming behavior",
            "weight": "Monitor weight trends",
            "behavior": "Watch for hiding or personality changes",
            "vocalization": "Note changes in meowing patterns"
        },
        "rabbit": {
            "appetite": "Monitor eating and drinking",
            "fecal_output": "Check droppings for size and quantity",
            "activity": "Track energy and movement",
            "weight": "Monitor weight trends",
            "dental": "Watch for drooling or difficulty eating",
            "grooming": "Check for changes in grooming"
        }
    }
    
    def __init__(self, client: OpenAIClient):
        """
        Initialize the wellness tracking preventive agent.
        
        Args:
            client: OpenAIClient instance for making API calls
        """
        super().__init__(
            client=client,
            agent_name="WellnessTrackingPreventive",
            default_temperature=0.4,  # Focused/deterministic for tracking plans
            default_max_tokens=300     # As specified (very concise)
        )
        logger.info("WellnessTrackingPreventiveAgent initialized")
    
    # ==========================================
    # IMPLEMENTATION OF ABSTRACT GENERATE METHOD
    # ==========================================
    def generate(self, profile: Dict[str, Any], ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implementation of abstract generate method from BaseLLMAgent.
        Delegates to generate_tracking_plan.
        
        Args:
            profile: Extracted pet profile
            ml_results: ML prediction results
            
        Returns:
            Dictionary with tracking plan results
        """
        return self.generate_tracking_plan(profile, ml_results)
    
    def generate_tracking_plan(
        self,
        profile: Dict[str, Any],
        ml_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate preventive wellness tracking plan for moderate-risk pets.
        
        Args:
            profile: Extracted pet profile containing:
                - pet_species: str
                - breed: str
                - age_years: int
                - weight_status: str
                - known_conditions: List[str] (if any)
                - exercise_level: str
                
            ml_results: ML predictions containing:
                - health_risk_score: float (0-1)
                - care_capability_score: float (0-100)
        
        Returns:
            Dictionary containing:
                - wellness_tracking: Dictionary with 4 required fields
                - status: String "success" or "error"
                - message: Optional status message
        """
        try:
            # Extract key information
            pet_info = self._extract_pet_info(profile)
            risk_info = self._extract_risk_info(ml_results)
            
            # Build prompts
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(pet_info, risk_info)
            
            logger.info(f"Generating wellness tracking plan for {pet_info['species']} {pet_info['breed']}")
            
            # Generate structured JSON
            result = self._generate_with_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                required_fields=self.REQUIRED_FIELDS,
                temperature=self.default_temperature,
                max_tokens=self.default_max_tokens
            )
            
            # Check if generation was successful
            if not result.get("_generation_success", False):
                error_msg = result.get("_error", "Unknown error in tracking plan generation")
                logger.error(f"Tracking plan generation failed: {error_msg}")
                
                return {
                    "wellness_tracking": self._get_fallback_plan(pet_info),
                    "status": "error",
                    "message": f"Plan generation failed: {error_msg}"
                }
            
            # Remove internal metadata
            plan = {k: v for k, v in result.items() if not k.startswith('_')}
            
            # Ensure all required fields are present with correct types
            plan = self._validate_and_format_plan(plan, pet_info)
            
            logger.info("Wellness tracking plan generated successfully")
            
            return {
                "wellness_tracking": plan,
                "status": "success",
                "message": "Tracking plan generated successfully"
            }
            
        except Exception as e:
            logger.error(f"Unexpected error in tracking plan generation: {str(e)}")
            
            # Provide fallback plan
            pet_info = self._extract_pet_info(profile)
            
            return {
                "wellness_tracking": self._get_fallback_plan(pet_info),
                "status": "error",
                "message": f"Plan generation failed: {str(e)}"
            }
    
    def _extract_pet_info(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant pet information for wellness tracking.
        
        Args:
            profile: Raw extracted profile
            
        Returns:
            Dictionary with formatted pet information
        """
        # Basic info
        species = profile.get('pet_species', 'unknown')
        if isinstance(species, str):
            species = species.capitalize()
        
        breed = profile.get('breed', 'unknown')
        if isinstance(breed, str):
            breed = breed.capitalize()
        
        name = profile.get('pet_name', 'Your Pet')
        
        # Age and life stage
        age = profile.get('age_years', 0)
        if age < 1:
            life_stage = "puppy/kitten"
        elif age < 7:
            life_stage = "adult"
        else:
            life_stage = "senior"
        
        # Health status
        weight_status = profile.get('weight_status', 'unknown')
        exercise_level = profile.get('exercise_level', 'unknown')
        
        conditions = profile.get('known_conditions', [])
        if not isinstance(conditions, list):
            conditions = []
        
        # Get species-specific wellness indicators
        species_key = species.lower()
        indicators = self.SPECIES_INDICATORS.get(
            species_key, 
            self.SPECIES_INDICATORS.get('dog')  # Default to dog
        )
        
        return {
            "name": name,
            "species": species,
            "breed": breed,
            "age": age,
            "life_stage": life_stage,
            "weight_status": weight_status,
            "exercise_level": exercise_level,
            "conditions": conditions,
            "conditions_text": self._format_list_for_prompt(conditions, "No specific conditions"),
            "indicators": indicators,
            "indicators_list": list(indicators.values())
        }
    
    def _extract_risk_info(self, ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract risk information from ML results.
        
        Args:
            ml_results: ML prediction results
            
        Returns:
            Dictionary with formatted risk information
        """
        risk_score = ml_results.get('health_risk_score', 0.45)
        care_score = ml_results.get('care_capability_score', 50.0)
        
        return {
            "risk_score": risk_score,
            "risk_score_percent": risk_score * 100,
            "care_score": care_score
        }
    
    def _build_system_prompt(self) -> str:
        """
        Build the system prompt defining the agent's role and output format.
        
        Returns:
            System prompt string
        """
        return """You are a preventive wellness specialist. Your role is to provide concise, practical wellness tracking plans that help owners monitor their pet's health proactively and detect issues early.

For each case, you must provide a JSON response with exactly these four fields:

1. "tracking_overview": A brief overview (1-2 sentences) explaining the importance of proactive wellness tracking for this pet.

2. "monthly_checklist": A list of 3-5 specific items to check monthly, formatted as actionable tasks (e.g., "Check weight and body condition").

3. "wellness_goals": A list of 2-4 proactive wellness goals appropriate for this pet's age and health status (e.g., "Maintain healthy weight", "Improve dental health").

4. "early_warning_signs": A list of 3-5 specific early warning signs that warrant attention, tailored to this pet's species and risk factors.

Your response must be concise, practical, and focused on prevention. Use clear, actionable language."""
    
    def _build_user_prompt(
        self,
        pet_info: Dict[str, Any],
        risk_info: Dict[str, Any]
    ) -> str:
        """
        Build the user prompt with specific pet information.
        
        Args:
            pet_info: Extracted pet information
            risk_info: Extracted risk information
            
        Returns:
            User prompt string
        """
        return f"""Create a preventive wellness tracking plan for this pet:

=== PET PROFILE ===
Species: {pet_info['species']}
Breed: {pet_info['breed']}
Age: {pet_info['age']} years ({pet_info['life_stage']})
Weight Status: {pet_info['weight_status']}
Exercise Level: {pet_info['exercise_level']}
Current Conditions: {pet_info['conditions_text']}

=== RISK CONTEXT ===
Health Risk Score: {risk_info['risk_score_percent']:.1f}%

Generate a concise wellness tracking plan with the four required fields."""
    
    def _validate_and_format_plan(
        self,
        plan: Dict[str, Any],
        pet_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate and format the tracking plan to ensure correct structure.
        
        Args:
            plan: Raw plan dictionary
            pet_info: Pet information for context
            
        Returns:
            Formatted plan with proper types
        """
        formatted = {}
        species = pet_info['species'].lower()
        life_stage = pet_info['life_stage']
        
        # Format tracking_overview
        if "tracking_overview" in plan:
            formatted["tracking_overview"] = str(plan["tracking_overview"])
        else:
            formatted["tracking_overview"] = (
                f"Regular wellness tracking helps detect changes early in this {life_stage} {species}, "
                f"allowing for timely intervention and preventive care."
            )
        
        # Format monthly_checklist (should be list)
        if "monthly_checklist" in plan:
            checklist = plan["monthly_checklist"]
            if isinstance(checklist, list):
                formatted["monthly_checklist"] = [str(item) for item in checklist]
            elif isinstance(checklist, str):
                # Split by lines or bullets
                lines = [line.strip() for line in checklist.split('\n') if line.strip()]
                if lines:
                    formatted["monthly_checklist"] = lines
                elif ',' in checklist:
                    formatted["monthly_checklist"] = [item.strip() for item in checklist.split(',')]
                else:
                    formatted["monthly_checklist"] = [checklist]
            else:
                formatted["monthly_checklist"] = [str(checklist)] if checklist else []
        else:
            # Default checklist based on species
            formatted["monthly_checklist"] = list(pet_info['indicators'].values())[:4]
        
        # Ensure at least 3 checklist items
        while len(formatted["monthly_checklist"]) < 3:
            formatted["monthly_checklist"].append("Monitor overall behavior and activity")
        
        # Limit to 5 items
        formatted["monthly_checklist"] = formatted["monthly_checklist"][:5]
        
        # Format wellness_goals (should be list)
        if "wellness_goals" in plan:
            goals = plan["wellness_goals"]
            if isinstance(goals, list):
                formatted["wellness_goals"] = [str(goal) for goal in goals]
            elif isinstance(goals, str):
                # Split by lines or commas
                if '\n' in goals:
                    formatted["wellness_goals"] = [g.strip() for g in goals.split('\n') if g.strip()]
                elif ',' in goals:
                    formatted["wellness_goals"] = [g.strip() for g in goals.split(',')]
                else:
                    formatted["wellness_goals"] = [goals]
            else:
                formatted["wellness_goals"] = [str(goals)] if goals else []
        else:
            # Default goals based on life stage
            if life_stage == "senior":
                formatted["wellness_goals"] = [
                    "Maintain healthy weight",
                    "Support joint health",
                    "Monitor for age-related changes",
                    "Keep up with regular vet visits"
                ]
            elif life_stage == "puppy/kitten":
                formatted["wellness_goals"] = [
                    "Complete vaccination series",
                    "Establish healthy eating habits",
                    "Socialization and training",
                    "Regular growth monitoring"
                ]
            else:
                formatted["wellness_goals"] = [
                    "Maintain ideal body condition",
                    "Prevent dental disease",
                    "Regular parasite prevention",
                    "Annual wellness exams"
                ]
        
        # Limit to 4 goals
        formatted["wellness_goals"] = formatted["wellness_goals"][:4]
        
        # Format early_warning_signs (should be list)
        if "early_warning_signs" in plan:
            signs = plan["early_warning_signs"]
            if isinstance(signs, list):
                formatted["early_warning_signs"] = [str(sign) for sign in signs]
            elif isinstance(signs, str):
                # Split by lines or commas
                if '\n' in signs:
                    formatted["early_warning_signs"] = [s.strip() for s in signs.split('\n') if s.strip()]
                elif ',' in signs:
                    formatted["early_warning_signs"] = [s.strip() for s in signs.split(',')]
                else:
                    formatted["early_warning_signs"] = [signs]
            else:
                formatted["early_warning_signs"] = [str(signs)] if signs else []
        else:
            # Default early warning signs
            if species == 'dog':
                formatted["early_warning_signs"] = [
                    "Changes in appetite or water intake",
                    "Lethargy or decreased activity",
                    "Vomiting or diarrhea lasting >24 hours",
                    "Coughing or difficulty breathing",
                    "Limping or difficulty standing"
                ]
            elif species == 'cat':
                formatted["early_warning_signs"] = [
                    "Not eating for >24 hours",
                    "Changes in litter box habits",
                    "Hiding or behavior changes",
                    "Weight loss",
                    "Vomiting or diarrhea"
                ]
            else:
                formatted["early_warning_signs"] = [
                    "Changes in appetite or eating habits",
                    "Lethargy or decreased activity",
                    "Changes in droppings/urination",
                    "Difficulty moving",
                    "Any sudden behavior changes"
                ]
        
        # Limit to 5 signs
        formatted["early_warning_signs"] = formatted["early_warning_signs"][:5]
        
        return formatted
    
    def _get_fallback_plan(self, pet_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate fallback tracking plan when LLM fails.
        
        Args:
            pet_info: Pet information
            
        Returns:
            Dictionary with fallback tracking plan
        """
        species = pet_info['species'].lower()
        life_stage = pet_info['life_stage']
        
        return {
            "tracking_overview": (
                f"Regular wellness tracking helps maintain optimal health for this "
                f"{life_stage} {species} by detecting subtle changes early."
            ),
            "monthly_checklist": [
                "Check weight and body condition",
                "Monitor appetite and water intake",
                "Observe energy levels and activity",
                "Inspect coat and skin condition",
                "Note any behavioral changes"
            ],
            "wellness_goals": [
                "Maintain healthy weight",
                "Keep vaccinations current",
                "Annual veterinary checkup",
                "Regular parasite prevention"
            ],
            "early_warning_signs": [
                "Changes in eating or drinking habits",
                "Lethargy or decreased activity",
                "Vomiting or diarrhea",
                "Difficulty urinating or defecating",
                "Any sudden behavior changes"
            ]
        }
    
    def get_plan_summary(self, plan: Dict[str, Any]) -> str:
        """
        Get a brief summary of the tracking plan for display.
        
        Args:
            plan: Tracking plan dictionary
            
        Returns:
            Brief summary string
        """
        if not plan:
            return "No tracking plan available."
        
        overview = plan.get('tracking_overview', '')
        checklist = plan.get('monthly_checklist', [])
        
        summary = f"{overview}\n\nMonthly Tasks:\n"
        
        if checklist:
            for task in checklist[:3]:  # Show first 3 tasks
                summary += f"  • {task}\n"
        
        return summary
    
    def get_checklist_items(self, plan: Dict[str, Any]) -> List[str]:
        """
        Extract monthly checklist items.
        
        Args:
            plan: Tracking plan dictionary
            
        Returns:
            List of checklist items
        """
        checklist = plan.get('monthly_checklist', [])
        if isinstance(checklist, list):
            return checklist
        return []
    
    def get_warning_signs(self, plan: Dict[str, Any]) -> List[str]:
        """
        Extract early warning signs.
        
        Args:
            plan: Tracking plan dictionary
            
        Returns:
            List of warning signs
        """
        signs = plan.get('early_warning_signs', [])
        if isinstance(signs, list):
            return signs
        return []


# ==========================================
# CONVENIENCE FUNCTION
# ==========================================

def generate_wellness_tracking_plan(
    client: OpenAIClient,
    profile: Dict[str, Any],
    ml_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convenience function to generate wellness tracking plan.
    
    Args:
        client: OpenAIClient instance
        profile: Extracted pet profile
        ml_results: ML prediction results
        
    Returns:
        Dictionary with tracking plan and status
    """
    agent = WellnessTrackingPreventiveAgent(client)
    return agent.generate_tracking_plan(profile, ml_results)


# ==========================================
# MODULE TESTING
# ==========================================

if __name__ == "__main__":
    import json
    from utils.openai_client import build_openai_client
    
    print("=" * 60)
    print("WELLNESS TRACKING PREVENTIVE AGENT TEST")
    print("=" * 60)
    
    try:
        # Initialize client
        client = build_openai_client()
        print("✅ OpenAI client created")
        
        # Initialize agent
        agent = WellnessTrackingPreventiveAgent(client)
        print(f"✅ Agent initialized: {agent.agent_name}")
        
        # Test with sample data - Adult dog
        test_profile = {
            "pet_name": "Charlie",
            "pet_species": "dog",
            "breed": "golden retriever",
            "age_years": 5,
            "weight_status": "normal",
            "exercise_level": "active",
            "known_conditions": []
        }
        
        test_ml_results = {
            "health_risk_score": 0.35,
            "care_capability_score": 80.0
        }
        
        print("\n📤 Generating wellness tracking plan...")
        result = agent.generate_tracking_plan(test_profile, test_ml_results)
        
        print(f"\n📊 Tracking Plan Result:")
        print(f"Status: {result['status']}")
        
        if result['status'] == 'success':
            plan = result['wellness_tracking']
            
            print("\n📋 WELLNESS TRACKING PLAN")
            print("=" * 40)
            
            print(f"\n📝 Tracking Overview:")
            print(f"  {plan.get('tracking_overview', 'N/A')}")
            
            print(f"\n✅ Monthly Checklist:")
            for item in plan.get('monthly_checklist', []):
                print(f"  • {item}")
            
            print(f"\n🎯 Wellness Goals:")
            for goal in plan.get('wellness_goals', []):
                print(f"  • {goal}")
            
            print(f"\n⚠️ Early Warning Signs:")
            for sign in plan.get('early_warning_signs', []):
                print(f"  • {sign}")
            
            # Test utility methods
            print(f"\n📝 Plan Summary:")
            print(f"  {agent.get_plan_summary(plan)}")
            
            print(f"\n✅ Monthly Tasks:")
            for task in agent.get_checklist_items(plan)[:3]:
                print(f"  • {task}")
        
        print("\n✅ Wellness Tracking Preventive Agent ready for use")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")