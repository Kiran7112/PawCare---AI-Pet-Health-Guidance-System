# agents/wellness_optimization_llm.py
"""
Wellness Optimization LLM Agent for PawCare+ (Wellness Path).
Generates wellness optimization plans for healthy, low-risk pets.
"""

from typing import Dict, Any, List, Optional
import logging

from agents.base_llm_agent import BaseLLMAgent
from utils.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


class WellnessOptimizationAgent(BaseLLMAgent):
    """
    LLM agent for generating wellness optimization plans for healthy, low-risk pets.
    
    This agent is part of the WELLNESS PATH and provides enhancement guidance including:
    - Overview of wellness optimization approach
    - Specific wellness enhancements
    - Physical activity suggestions
    - Bonding activities for owner and pet
    """
    
    # Required fields in the output JSON
    REQUIRED_FIELDS = [
        "optimization_overview",
        "wellness_enhancements",
        "activity_suggestions",
        "bonding_activities"
    ]
    
    # Species-specific wellness enhancement areas
    SPECIES_ENHANCEMENTS = {
        "dog": {
            "physical": ["Agility training", "Swimming", "Hiking", "Fetch variations"],
            "mental": ["Puzzle toys", "Scent work", "Training new tricks", "Hide and seek"],
            "nutritional": ["Food toppers", "Seasonal vegetables", "Joint supplements"],
            "social": ["Dog park visits", "Play dates", "Group training classes"]
        },
        "cat": {
            "physical": ["Vertical climbing spaces", "Interactive toys", "Laser play", "Cat trees"],
            "mental": ["Puzzle feeders", "Window perches", "Bird watching", "Treat puzzles"],
            "nutritional": ["Wet food variety", "Cat grass", "Hairball remedies"],
            "social": ["Interactive play sessions", "Clicker training", "Catnip toys"]
        },
        "rabbit": {
            "physical": ["Tunnels and hideouts", "Obstacle courses", "Digging boxes"],
            "mental": ["Foraging toys", "Treat balls", "Maze challenges"],
            "nutritional": ["Fresh herb variety", "Seasonal vegetables", "Hay enrichment"],
            "social": ["Floor time exploration", "Gentle handling practice", "Clicker training"]
        }
    }
    
    def __init__(self, client: OpenAIClient):
        """
        Initialize the wellness optimization agent.
        
        Args:
            client: OpenAIClient instance for making API calls
        """
        super().__init__(
            client=client,
            agent_name="WellnessOptimization",
            default_temperature=0.5,  # Balanced for creative suggestions
            default_max_tokens=350     # As specified
        )
        logger.info("WellnessOptimizationAgent initialized")
    def generate(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Implementation of abstract generate method from BaseLLMAgent.
        Delegates to the specific generation method.
        """
        # Call the specific method based on what the agent does
        return self.generate_optimization_plan(*args, **kwargs)
    def generate_optimization_plan(
        self,
        profile: Dict[str, Any],
        ml_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate wellness optimization plan for healthy, low-risk pets.
        
        Args:
            profile: Extracted pet profile containing:
                - pet_species: str
                - breed: str
                - age_years: int
                - exercise_level: str
                - living_situation: str
                - behavioral_issues: List[str] (if any)
                
            ml_results: ML predictions containing:
                - health_risk_score: float (0-1) - should be ≤ 0.3 for wellness path
                - care_capability_score: float (0-100)
        
        Returns:
            Dictionary containing:
                - wellness_optimization: Dictionary with 4 required fields
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
            
            logger.info(f"Generating wellness optimization plan for {pet_info['species']} {pet_info['breed']}, "
                       f"risk score {risk_info['risk_score']:.2f}")
            
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
                error_msg = result.get("_error", "Unknown error in optimization plan generation")
                logger.error(f"Optimization plan generation failed: {error_msg}")
                
                return {
                    "wellness_optimization": self._get_fallback_plan(pet_info),
                    "status": "error",
                    "message": f"Plan generation failed: {error_msg}"
                }
            
            # Remove internal metadata
            plan = {k: v for k, v in result.items() if not k.startswith('_')}
            
            # Ensure all required fields are present with correct types
            plan = self._validate_and_format_plan(plan, pet_info)
            
            logger.info("Wellness optimization plan generated successfully")
            
            return {
                "wellness_optimization": plan,
                "status": "success",
                "message": "Optimization plan generated successfully"
            }
            
        except Exception as e:
            logger.error(f"Unexpected error in optimization plan generation: {str(e)}")
            
            # Provide fallback plan
            pet_info = self._extract_pet_info(profile)
            
            return {
                "wellness_optimization": self._get_fallback_plan(pet_info),
                "status": "error",
                "message": f"Plan generation failed: {str(e)}"
            }
    
    def _extract_pet_info(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant pet information for wellness optimization.
        
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
        
        # Lifestyle
        exercise_level = profile.get('exercise_level', 'moderate')
        living_situation = profile.get('living_situation', 'unknown')
        if isinstance(living_situation, str):
            living_situation = living_situation.capitalize()
        
        # Behavioral context
        behavioral_issues = profile.get('behavioral_issues', [])
        if not isinstance(behavioral_issues, list):
            behavioral_issues = []
        
        # Get species-specific enhancement options
        species_key = species.lower()
        enhancements = self.SPECIES_ENHANCEMENTS.get(
            species_key, 
            self.SPECIES_ENHANCEMENTS.get('dog')  # Default to dog
        )
        
        return {
            "name": name,
            "species": species,
            "breed": breed,
            "age": age,
            "life_stage": life_stage,
            "exercise_level": exercise_level,
            "living_situation": living_situation,
            "behavioral_issues": behavioral_issues,
            "behavioral_issues_text": self._format_list_for_prompt(behavioral_issues, "None reported"),
            "enhancements": enhancements
        }
    
    def _extract_risk_info(self, ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract risk information from ML results.
        
        Args:
            ml_results: ML prediction results
            
        Returns:
            Dictionary with formatted risk information
        """
        risk_score = ml_results.get('health_risk_score', 0.2)  # Default to low risk
        care_score = ml_results.get('care_capability_score', 70.0)
        
        # Determine wellness level
        if risk_score <= 0.15:
            wellness_level = "EXCELLENT"
        elif risk_score <= 0.3:
            wellness_level = "GOOD"
        else:
            wellness_level = "MAINTENANCE"
        
        return {
            "risk_score": risk_score,
            "risk_score_percent": risk_score * 100,
            "wellness_level": wellness_level,
            "care_score": care_score
        }
    
    def _build_system_prompt(self) -> str:
        """
        Build the system prompt defining the agent's role and output format.
        
        Returns:
            System prompt string
        """
        return """You are a pet wellness optimization specialist. Your role is to provide creative, engaging suggestions that enhance the wellbeing of healthy pets and strengthen the bond between pets and their owners.

For each case, you must provide a JSON response with exactly these four fields:

1. "optimization_overview": A brief overview (1-2 sentences) explaining the wellness optimization approach for this healthy pet.

2. "wellness_enhancements": A list of 3-5 specific enhancements to improve overall wellness, such as enrichment activities, nutritional boosts, or preventive measures.

3. "activity_suggestions": A list of 3-5 physical activity suggestions appropriate for this pet's age, breed, and living situation.

4. "bonding_activities": A list of 3-5 activities specifically designed to strengthen the bond between the pet and owner.

Your response should be positive, encouraging, and tailored to the pet's specific characteristics. Focus on enhancement rather than treatment."""
    
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
        return f"""Create a wellness optimization plan for this healthy pet:

=== PET PROFILE ===
Name: {pet_info['name']}
Species: {pet_info['species']}
Breed: {pet_info['breed']}
Age: {pet_info['age']} years ({pet_info['life_stage']})
Current Exercise Level: {pet_info['exercise_level']}
Living Situation: {pet_info['living_situation']}
Behavioral Notes: {pet_info['behavioral_issues_text']}

=== WELLNESS STATUS ===
Health Risk Score: {risk_info['risk_score_percent']:.1f}% (LOW)
Wellness Level: {risk_info['wellness_level']}

Generate a wellness optimization plan with the four required fields, focusing on enhancement and enrichment."""
    
    def _validate_and_format_plan(
        self,
        plan: Dict[str, Any],
        pet_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate and format the optimization plan to ensure correct structure.
        
        Args:
            plan: Raw plan dictionary
            pet_info: Pet information for context
            
        Returns:
            Formatted plan with proper types
        """
        formatted = {}
        species = pet_info['species'].lower()
        life_stage = pet_info['life_stage']
        
        # Format optimization_overview
        if "optimization_overview" in plan:
            formatted["optimization_overview"] = str(plan["optimization_overview"])
        else:
            formatted["optimization_overview"] = (
                f"This {life_stage} {species} is in excellent health! Focus on enhancing "
                f"wellbeing through enrichment, varied activities, and quality bonding time."
            )
        
        # Format wellness_enhancements (should be list)
        if "wellness_enhancements" in plan:
            enhancements = plan["wellness_enhancements"]
            if isinstance(enhancements, list):
                formatted["wellness_enhancements"] = [str(e) for e in enhancements]
            elif isinstance(enhancements, str):
                # Split by lines or commas
                if '\n' in enhancements:
                    formatted["wellness_enhancements"] = [e.strip() for e in enhancements.split('\n') if e.strip()]
                elif ',' in enhancements:
                    formatted["wellness_enhancements"] = [e.strip() for e in enhancements.split(',')]
                else:
                    formatted["wellness_enhancements"] = [enhancements]
            else:
                formatted["wellness_enhancements"] = [str(enhancements)] if enhancements else []
        else:
            # Default enhancements from species mapping
            enhancements_list = []
            for category, items in pet_info['enhancements'].items():
                if items and len(enhancements_list) < 3:
                    enhancements_list.append(items[0])  # Take first from each category
            formatted["wellness_enhancements"] = enhancements_list[:4]
        
        # Ensure at least 3 enhancements
        while len(formatted["wellness_enhancements"]) < 3:
            formatted["wellness_enhancements"].append(f"Try new {species}-appropriate enrichment toys")
        
        # Limit to 5 enhancements
        formatted["wellness_enhancements"] = formatted["wellness_enhancements"][:5]
        
        # Format activity_suggestions (should be list)
        if "activity_suggestions" in plan:
            activities = plan["activity_suggestions"]
            if isinstance(activities, list):
                formatted["activity_suggestions"] = [str(a) for a in activities]
            elif isinstance(activities, str):
                # Split by lines or commas
                if '\n' in activities:
                    formatted["activity_suggestions"] = [a.strip() for a in activities.split('\n') if a.strip()]
                elif ',' in activities:
                    formatted["activity_suggestions"] = [a.strip() for a in activities.split(',')]
                else:
                    formatted["activity_suggestions"] = [activities]
            else:
                formatted["activity_suggestions"] = [str(activities)] if activities else []
        else:
            # Default activities based on species
            if species == 'dog':
                formatted["activity_suggestions"] = [
                    "Vary walking routes for new scents and experiences",
                    "Try hiking on gentle trails",
                    "Incorporate fetch with different toys",
                    "Swimming (if breed-appropriate)",
                    "Agility or obedience classes"
                ]
            elif species == 'cat':
                formatted["activity_suggestions"] = [
                    "Interactive wand toys for hunting practice",
                    "Create vertical climbing spaces",
                    "Hide treats for foraging",
                    "Laser play sessions (end with a tangible reward)",
                    "Catnip or silvervine toys"
                ]
            else:
                formatted["activity_suggestions"] = [
                    "Create obstacle courses with tunnels",
                    "Foraging activities with hidden treats",
                    "Supervised exploration time",
                    "Gentle handling and petting sessions",
                    "Novel toys rotated regularly"
                ]
        
        # Limit to 5 activities
        formatted["activity_suggestions"] = formatted["activity_suggestions"][:5]
        
        # Format bonding_activities (should be list)
        if "bonding_activities" in plan:
            bonding = plan["bonding_activities"]
            if isinstance(bonding, list):
                formatted["bonding_activities"] = [str(b) for b in bonding]
            elif isinstance(bonding, str):
                # Split by lines or commas
                if '\n' in bonding:
                    formatted["bonding_activities"] = [b.strip() for b in bonding.split('\n') if b.strip()]
                elif ',' in bonding:
                    formatted["bonding_activities"] = [b.strip() for b in bonding.split(',')]
                else:
                    formatted["bonding_activities"] = [bonding]
            else:
                formatted["bonding_activities"] = [str(bonding)] if bonding else []
        else:
            # Default bonding activities
            if species == 'dog':
                formatted["bonding_activities"] = [
                    "Training sessions using positive reinforcement",
                    "Couch cuddle time with gentle massage",
                    "Hand-feeding treats during training",
                    "Quiet walks focusing on shared experience",
                    "Learning a new trick together"
                ]
            elif species == 'cat':
                formatted["bonding_activities"] = [
                    "Gentle grooming sessions",
                    "Interactive play with wand toys",
                    "Clicker training for tricks",
                    "Quiet time together with petting",
                    "Offering treats by hand"
                ]
            else:
                formatted["bonding_activities"] = [
                    "Daily gentle handling sessions",
                    "Hand-feeding favorite treats",
                    "Quiet time in same room",
                    "Gentle petting and massage",
                    "Talking softly during interactions"
                ]
        
        # Limit to 5 activities
        formatted["bonding_activities"] = formatted["bonding_activities"][:5]
        
        return formatted
    
    def _get_fallback_plan(self, pet_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate fallback optimization plan when LLM fails.
        
        Args:
            pet_info: Pet information
            
        Returns:
            Dictionary with fallback optimization plan
        """
        species = pet_info['species'].lower()
        life_stage = pet_info['life_stage']
        
        return {
            "optimization_overview": (
                f"Your {life_stage} {species} is healthy and thriving! These suggestions "
                f"will enhance wellbeing and strengthen your bond."
            ),
            "wellness_enhancements": [
                "Rotate toys weekly to maintain interest",
                "Try puzzle feeders for mental stimulation",
                "Incorporate new scents during walks",
                "Add variety to treats with healthy options"
            ],
            "activity_suggestions": [
                "Vary exercise routines to prevent boredom",
                "Try new environments for exploration",
                "Incorporate play that mimics natural behaviors",
                "Schedule regular play sessions",
                "Consider pet-friendly classes or activities"
            ],
            "bonding_activities": [
                "Daily dedicated one-on-one time",
                "Learn a new skill or trick together",
                "Practice gentle handling and massage",
                "Create a special routine just for you two",
                "Capture memories with photos and videos"
            ]
        }
    
    def get_plan_summary(self, plan: Dict[str, Any]) -> str:
        """
        Get a brief summary of the optimization plan for display.
        
        Args:
            plan: Optimization plan dictionary
            
        Returns:
            Brief summary string
        """
        if not plan:
            return "No optimization plan available."
        
        overview = plan.get('optimization_overview', '')
        enhancements = plan.get('wellness_enhancements', [])
        
        summary = f"{overview}\n\nKey Enhancements:\n"
        
        if enhancements:
            for enhancement in enhancements[:3]:  # Show first 3
                summary += f"  • {enhancement}\n"
        
        return summary
    
    def get_enhancements_list(self, plan: Dict[str, Any]) -> List[str]:
        """
        Extract wellness enhancements list.
        
        Args:
            plan: Optimization plan dictionary
            
        Returns:
            List of enhancement suggestions
        """
        enhancements = plan.get('wellness_enhancements', [])
        if isinstance(enhancements, list):
            return enhancements
        return []
    
    def get_activities_list(self, plan: Dict[str, Any]) -> List[str]:
        """
        Extract activity suggestions list.
        
        Args:
            plan: Optimization plan dictionary
            
        Returns:
            List of activity suggestions
        """
        activities = plan.get('activity_suggestions', [])
        if isinstance(activities, list):
            return activities
        return []
    
    def get_bonding_list(self, plan: Dict[str, Any]) -> List[str]:
        """
        Extract bonding activities list.
        
        Args:
            plan: Optimization plan dictionary
            
        Returns:
            List of bonding activities
        """
        bonding = plan.get('bonding_activities', [])
        if isinstance(bonding, list):
            return bonding
        return []


# ==========================================
# CONVENIENCE FUNCTION
# ==========================================

def generate_wellness_optimization_plan(
    client: OpenAIClient,
    profile: Dict[str, Any],
    ml_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convenience function to generate wellness optimization plan.
    
    Args:
        client: OpenAIClient instance
        profile: Extracted pet profile
        ml_results: ML prediction results
        
    Returns:
        Dictionary with optimization plan and status
    """
    agent = WellnessOptimizationAgent(client)
    return agent.generate_optimization_plan(profile, ml_results)


# ==========================================
# MODULE TESTING
# ==========================================

if __name__ == "__main__":
    import json
    from utils.openai_client import build_openai_client
    
    print("=" * 60)
    print("WELLNESS OPTIMIZATION AGENT TEST")
    print("=" * 60)
    
    try:
        # Initialize client
        client = build_openai_client()
        print("✅ OpenAI client created")
        
        # Initialize agent
        agent = WellnessOptimizationAgent(client)
        print(f"✅ Agent initialized: {agent.agent_name}")
        
        # Test with sample data - Healthy young dog
        test_profile = {
            "pet_name": "Luna",
            "pet_species": "dog",
            "breed": "australian shepherd",
            "age_years": 2,
            "exercise_level": "active",
            "living_situation": "house with yard",
            "behavioral_issues": []
        }
        
        test_ml_results = {
            "health_risk_score": 0.12,  # Very low risk
            "care_capability_score": 85.0
        }
        
        print("\n📤 Generating wellness optimization plan...")
        result = agent.generate_optimization_plan(test_profile, test_ml_results)
        
        print(f"\n📊 Optimization Plan Result:")
        print(f"Status: {result['status']}")
        
        if result['status'] == 'success':
            plan = result['wellness_optimization']
            
            print("\n🌟 WELLNESS OPTIMIZATION PLAN")
            print("=" * 40)
            
            print(f"\n📝 Optimization Overview:")
            print(f"  {plan.get('optimization_overview', 'N/A')}")
            
            print(f"\n✨ Wellness Enhancements:")
            for enhancement in plan.get('wellness_enhancements', []):
                print(f"  • {enhancement}")
            
            print(f"\n🏃 Activity Suggestions:")
            for activity in plan.get('activity_suggestions', []):
                print(f"  • {activity}")
            
            print(f"\n💕 Bonding Activities:")
            for activity in plan.get('bonding_activities', []):
                print(f"  • {activity}")
            
            # Test utility methods
            print(f"\n📝 Plan Summary:")
            print(f"  {agent.get_plan_summary(plan)}")
            
            print(f"\n✅ Top Enhancements:")
            for enhancement in agent.get_enhancements_list(plan)[:3]:
                print(f"  • {enhancement}")
        
        print("\n✅ Wellness Optimization Agent ready for use")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")