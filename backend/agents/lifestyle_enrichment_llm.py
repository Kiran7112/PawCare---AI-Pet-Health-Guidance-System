# agents/lifestyle_enrichment_llm.py
"""
Lifestyle Enrichment LLM Agent for PawCare+ (Wellness Path).
Generates lifestyle enrichment plans for healthy, low-risk pets.
"""

from typing import Dict, Any, List, Optional
import logging

from agents.base_llm_agent import BaseLLMAgent
from utils.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


class LifestyleEnrichmentAgent(BaseLLMAgent):
    """
    LLM agent for generating lifestyle enrichment plans for healthy, low-risk pets.
    
    This agent is part of the WELLNESS PATH and provides enrichment guidance including:
    - Overview of enrichment approach
    - Mental stimulation activities
    - Social interaction opportunities
    - Environmental enrichment suggestions
    """
    
    # Required fields in the output JSON
    REQUIRED_FIELDS = [
        "enrichment_overview",
        "mental_stimulation",
        "social_opportunities",
        "environmental_enrichment"
    ]
    
    # Species-specific enrichment ideas
    SPECIES_ENRICHMENT = {
        "dog": {
            "mental": [
                "Puzzle toys with hidden treats",
                "Scent work games (hide and seek with treats)",
                "Training new tricks using positive reinforcement",
                "Nose work mats or snuffle mats",
                "Interactive toys that dispense food"
            ],
            "social": [
                "Supervised playdates with compatible dogs",
                "Group training classes",
                "Dog park visits (if appropriate)",
                "Family game time with fetch or tug",
                "Greeting new people calmly with treats"
            ],
            "environmental": [
                "Rotating toy selection weekly",
                "Creating cozy den spaces",
                "Providing different textures (grass, sand, pavement)",
                "Setting up obstacle courses in yard",
                "Window perches for watching outdoor activity"
            ]
        },
        "cat": {
            "mental": [
                "Puzzle feeders for meals",
                "Treat-dispensing balls",
                "Clicker training for tricks",
                "Hide and seek with toys",
                "Watching bird videos (supervised)"
            ],
            "social": [
                "Interactive play with wand toys",
                "Gentle grooming sessions",
                "Cat-friendly visitors",
                "Play dates with friendly cats (if appropriate)",
                "Training sessions using positive reinforcement"
            ],
            "environmental": [
                "Vertical space with cat trees and shelves",
                "Window perches with outdoor views",
                "Cardboard boxes and paper bags",
                "Cat-safe plants (cat grass, catnip)",
                "Different scratching surfaces (sisal, cardboard, carpet)"
            ]
        },
        "rabbit": {
            "mental": [
                "Foraging toys with hidden treats",
                "Cardboard castles to explore and chew",
                "Treat balls that dispense pellets",
                "Maze challenges with hay",
                "Digging boxes with safe materials"
            ],
            "social": [
                "Daily supervised floor time",
                "Gentle handling and petting",
                "Bonding with other rabbits (if appropriate)",
                "Training simple behaviors (target training)",
                "Quiet time near family activities"
            ],
            "environmental": [
                "Tunnels and hideouts",
                "Different flooring textures",
                "Hay racks at various heights",
                "Safe chew toys (untreated wood, willow balls)",
                "Outdoor play in secure exercise pen"
            ]
        }
    }
    
    def __init__(self, client: OpenAIClient):
        """
        Initialize the lifestyle enrichment agent.
        
        Args:
            client: OpenAIClient instance for making API calls
        """
        super().__init__(
            client=client,
            agent_name="LifestyleEnrichment",
            default_temperature=0.6,  # Higher temperature for creative suggestions
            default_max_tokens=350     # As specified
        )
        logger.info("LifestyleEnrichmentAgent initialized")
    def generate(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Implementation of abstract generate method from BaseLLMAgent.
        Delegates to the specific generation method.
        """
        # Call the specific method based on what the agent does
        return self.generate_enrichment_plan(*args, **kwargs)
    def generate_enrichment_plan(
        self,
        profile: Dict[str, Any],
        ml_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate lifestyle enrichment plan for healthy, low-risk pets.
        
        Args:
            profile: Extracted pet profile containing:
                - pet_species: str
                - breed: str
                - age_years: int
                - living_situation: str
                - exercise_level: str
                - behavioral_issues: List[str] (if any)
                
            ml_results: ML predictions containing:
                - health_risk_score: float (0-1) - should be ≤ 0.3 for wellness path
                - care_capability_score: float (0-100)
        
        Returns:
            Dictionary containing:
                - lifestyle_enrichment: Dictionary with 4 required fields
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
            
            logger.info(f"Generating lifestyle enrichment plan for {pet_info['species']} {pet_info['breed']}")
            
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
                error_msg = result.get("_error", "Unknown error in enrichment plan generation")
                logger.error(f"Enrichment plan generation failed: {error_msg}")
                
                return {
                    "lifestyle_enrichment": self._get_fallback_plan(pet_info),
                    "status": "error",
                    "message": f"Plan generation failed: {error_msg}"
                }
            
            # Remove internal metadata
            plan = {k: v for k, v in result.items() if not k.startswith('_')}
            
            # Ensure all required fields are present with correct types
            plan = self._validate_and_format_plan(plan, pet_info)
            
            logger.info("Lifestyle enrichment plan generated successfully")
            
            return {
                "lifestyle_enrichment": plan,
                "status": "success",
                "message": "Enrichment plan generated successfully"
            }
            
        except Exception as e:
            logger.error(f"Unexpected error in enrichment plan: {str(e)}")
            
            # Provide fallback plan
            pet_info = self._extract_pet_info(profile)
            
            return {
                "lifestyle_enrichment": self._get_fallback_plan(pet_info),
                "status": "error",
                "message": f"Plan generation failed: {str(e)}"
            }
    
    def _extract_pet_info(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant pet information for lifestyle enrichment.
        
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
        living_situation = profile.get('living_situation', 'unknown')
        if isinstance(living_situation, str):
            living_situation = living_situation.capitalize()
        
        exercise_level = profile.get('exercise_level', 'moderate')
        
        # Behavioral context
        behavioral_issues = profile.get('behavioral_issues', [])
        if not isinstance(behavioral_issues, list):
            behavioral_issues = []
        
        # Get species-specific enrichment ideas
        species_key = species.lower()
        enrichment = self.SPECIES_ENRICHMENT.get(
            species_key, 
            self.SPECIES_ENRICHMENT.get('dog')  # Default to dog
        )
        
        return {
            "name": name,
            "species": species,
            "breed": breed,
            "age": age,
            "life_stage": life_stage,
            "living_situation": living_situation,
            "exercise_level": exercise_level,
            "behavioral_issues": behavioral_issues,
            "behavioral_issues_text": self._format_list_for_prompt(behavioral_issues, "None reported"),
            "enrichment": enrichment
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
        return """You are a pet lifestyle enrichment specialist. Your role is to provide creative, engaging suggestions that enhance the daily life of healthy pets through mental stimulation, social opportunities, and environmental enrichment.

For each case, you must provide a JSON response with exactly these four fields:

1. "enrichment_overview": A brief overview (1-2 sentences) explaining the importance of enrichment for this pet and your general approach.

2. "mental_stimulation": A list of 3-5 specific activities that challenge the pet mentally, such as puzzle toys, training games, or problem-solving activities.

3. "social_opportunities": A list of 3-5 suggestions for positive social interactions with humans, other animals, or novel experiences.

4. "environmental_enrichment": A list of 3-5 ideas for enhancing the pet's physical environment, such as habitat modifications, toy rotations, or sensory experiences.

Your response should be creative, practical, and tailored to the pet's species, age, and living situation. Focus on activities that are safe, enjoyable, and strengthen the human-animal bond."""
    
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
        return f"""Create a lifestyle enrichment plan for this healthy pet:

=== PET PROFILE ===
Name: {pet_info['name']}
Species: {pet_info['species']}
Breed: {pet_info['breed']}
Age: {pet_info['age']} years ({pet_info['life_stage']})
Living Situation: {pet_info['living_situation']}
Current Exercise Level: {pet_info['exercise_level']}
Behavioral Notes: {pet_info['behavioral_issues_text']}

=== WELLNESS STATUS ===
Health Risk Score: {risk_info['risk_score_percent']:.1f}% (LOW)

Generate a lifestyle enrichment plan with the four required fields, focusing on activities that are appropriate for this pet's age, living situation, and natural behaviors."""
    
    def _validate_and_format_plan(
        self,
        plan: Dict[str, Any],
        pet_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate and format the enrichment plan to ensure correct structure.
        
        Args:
            plan: Raw plan dictionary
            pet_info: Pet information for context
            
        Returns:
            Formatted plan with proper types
        """
        formatted = {}
        species = pet_info['species'].lower()
        life_stage = pet_info['life_stage']
        
        # Format enrichment_overview
        if "enrichment_overview" in plan:
            formatted["enrichment_overview"] = str(plan["enrichment_overview"])
        else:
            formatted["enrichment_overview"] = (
                f"Enrichment is essential for this {life_stage} {species}'s wellbeing. "
                f"These activities will provide mental stimulation, social engagement, "
                f"and environmental variety tailored to their needs."
            )
        
        # Format mental_stimulation (should be list)
        if "mental_stimulation" in plan:
            mental = plan["mental_stimulation"]
            if isinstance(mental, list):
                formatted["mental_stimulation"] = [str(m) for m in mental]
            elif isinstance(mental, str):
                # Split by lines or commas
                if '\n' in mental:
                    formatted["mental_stimulation"] = [m.strip() for m in mental.split('\n') if m.strip()]
                elif ',' in mental:
                    formatted["mental_stimulation"] = [m.strip() for m in mental.split(',')]
                else:
                    formatted["mental_stimulation"] = [mental]
            else:
                formatted["mental_stimulation"] = [str(mental)] if mental else []
        else:
            # Default mental activities from species mapping
            formatted["mental_stimulation"] = pet_info['enrichment'].get('mental', [])[:4]
        
        # Ensure at least 3 activities
        while len(formatted["mental_stimulation"]) < 3:
            formatted["mental_stimulation"].append(f"Try new puzzle toys for {species}s")
        
        # Limit to 5 activities
        formatted["mental_stimulation"] = formatted["mental_stimulation"][:5]
        
        # Format social_opportunities (should be list)
        if "social_opportunities" in plan:
            social = plan["social_opportunities"]
            if isinstance(social, list):
                formatted["social_opportunities"] = [str(s) for s in social]
            elif isinstance(social, str):
                # Split by lines or commas
                if '\n' in social:
                    formatted["social_opportunities"] = [s.strip() for s in social.split('\n') if s.strip()]
                elif ',' in social:
                    formatted["social_opportunities"] = [s.strip() for s in social.split(',')]
                else:
                    formatted["social_opportunities"] = [social]
            else:
                formatted["social_opportunities"] = [str(social)] if social else []
        else:
            # Default social activities from species mapping
            formatted["social_opportunities"] = pet_info['enrichment'].get('social', [])[:4]
        
        # Ensure at least 3 activities
        while len(formatted["social_opportunities"]) < 3:
            formatted["social_opportunities"].append(f"Daily one-on-one interaction time")
        
        # Limit to 5 activities
        formatted["social_opportunities"] = formatted["social_opportunities"][:5]
        
        # Format environmental_enrichment (should be list)
        if "environmental_enrichment" in plan:
            env = plan["environmental_enrichment"]
            if isinstance(env, list):
                formatted["environmental_enrichment"] = [str(e) for e in env]
            elif isinstance(env, str):
                # Split by lines or commas
                if '\n' in env:
                    formatted["environmental_enrichment"] = [e.strip() for e in env.split('\n') if e.strip()]
                elif ',' in env:
                    formatted["environmental_enrichment"] = [e.strip() for e in env.split(',')]
                else:
                    formatted["environmental_enrichment"] = [env]
            else:
                formatted["environmental_enrichment"] = [str(env)] if env else []
        else:
            # Default environmental ideas from species mapping
            formatted["environmental_enrichment"] = pet_info['enrichment'].get('environmental', [])[:4]
        
        # Ensure at least 3 ideas
        while len(formatted["environmental_enrichment"]) < 3:
            formatted["environmental_enrichment"].append(f"Rotate toys weekly to maintain novelty")
        
        # Limit to 5 ideas
        formatted["environmental_enrichment"] = formatted["environmental_enrichment"][:5]
        
        return formatted
    
    def _get_fallback_plan(self, pet_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate fallback enrichment plan when LLM fails.
        
        Args:
            pet_info: Pet information
            
        Returns:
            Dictionary with fallback enrichment plan
        """
        species = pet_info['species'].lower()
        life_stage = pet_info['life_stage']
        
        return {
            "enrichment_overview": (
                f"Your {life_stage} {species} will thrive with regular enrichment activities "
                f"that engage their natural instincts and provide variety in daily life."
            ),
            "mental_stimulation": [
                "Puzzle toys filled with treats or kibble",
                "Training sessions teaching new cues or tricks",
                "Scent games hiding treats around the house",
                "Interactive toys that require problem-solving",
                "Novel objects to explore (safely)"
            ],
            "social_opportunities": [
                "Daily dedicated play sessions",
                "Positive introductions to new people",
                "Supervised interactions with compatible animals",
                "Family activities that include your pet",
                "Gentle handling and grooming sessions"
            ],
            "environmental_enrichment": [
                "Rotate toys weekly to maintain interest",
                "Create cozy resting areas in different locations",
                "Provide species-appropriate climbing or perching spots",
                "Introduce new safe scents or textures",
                "Safe outdoor exploration (supervised)"
            ]
        }
    
    def get_plan_summary(self, plan: Dict[str, Any]) -> str:
        """
        Get a brief summary of the enrichment plan for display.
        
        Args:
            plan: Enrichment plan dictionary
            
        Returns:
            Brief summary string
        """
        if not plan:
            return "No enrichment plan available."
        
        overview = plan.get('enrichment_overview', '')
        mental = plan.get('mental_stimulation', [])
        
        summary = f"{overview}\n\nMental Activities:\n"
        
        if mental:
            for activity in mental[:3]:  # Show first 3
                summary += f"  • {activity}\n"
        
        return summary
    
    def get_mental_activities(self, plan: Dict[str, Any]) -> List[str]:
        """
        Extract mental stimulation activities.
        
        Args:
            plan: Enrichment plan dictionary
            
        Returns:
            List of mental activities
        """
        mental = plan.get('mental_stimulation', [])
        if isinstance(mental, list):
            return mental
        return []
    
    def get_social_opportunities(self, plan: Dict[str, Any]) -> List[str]:
        """
        Extract social opportunities list.
        
        Args:
            plan: Enrichment plan dictionary
            
        Returns:
            List of social opportunities
        """
        social = plan.get('social_opportunities', [])
        if isinstance(social, list):
            return social
        return []
    
    def get_environmental_ideas(self, plan: Dict[str, Any]) -> List[str]:
        """
        Extract environmental enrichment ideas.
        
        Args:
            plan: Enrichment plan dictionary
            
        Returns:
            List of environmental ideas
        """
        env = plan.get('environmental_enrichment', [])
        if isinstance(env, list):
            return env
        return []


# ==========================================
# CONVENIENCE FUNCTION
# ==========================================

def generate_lifestyle_enrichment(
    client: OpenAIClient,
    profile: Dict[str, Any],
    ml_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convenience function to generate lifestyle enrichment plan.
    
    Args:
        client: OpenAIClient instance
        profile: Extracted pet profile
        ml_results: ML prediction results
        
    Returns:
        Dictionary with enrichment plan and status
    """
    agent = LifestyleEnrichmentAgent(client)
    return agent.generate_enrichment_plan(profile, ml_results)


# ==========================================
# MODULE TESTING
# ==========================================

if __name__ == "__main__":
    import json
    from utils.openai_client import build_openai_client
    
    print("=" * 60)
    print("LIFESTYLE ENRICHMENT AGENT TEST")
    print("=" * 60)
    
    try:
        # Initialize client
        client = build_openai_client()
        print("✅ OpenAI client created")
        
        # Initialize agent
        agent = LifestyleEnrichmentAgent(client)
        print(f"✅ Agent initialized: {agent.agent_name}")
        
        # Test with sample data - Indoor cat
        test_profile = {
            "pet_name": "Whiskers",
            "pet_species": "cat",
            "breed": "domestic shorthair",
            "age_years": 4,
            "living_situation": "apartment",
            "exercise_level": "moderate",
            "behavioral_issues": []
        }
        
        test_ml_results = {
            "health_risk_score": 0.18,
            "care_capability_score": 80.0
        }
        
        print("\n📤 Generating lifestyle enrichment plan...")
        result = agent.generate_enrichment_plan(test_profile, test_ml_results)
        
        print(f"\n📊 Enrichment Plan Result:")
        print(f"Status: {result['status']}")
        
        if result['status'] == 'success':
            plan = result['lifestyle_enrichment']
            
            print("\n🌟 LIFESTYLE ENRICHMENT PLAN")
            print("=" * 40)
            
            print(f"\n📝 Enrichment Overview:")
            print(f"  {plan.get('enrichment_overview', 'N/A')}")
            
            print(f"\n🧠 Mental Stimulation:")
            for activity in plan.get('mental_stimulation', []):
                print(f"  • {activity}")
            
            print(f"\n👥 Social Opportunities:")
            for opportunity in plan.get('social_opportunities', []):
                print(f"  • {opportunity}")
            
            print(f"\n🏠 Environmental Enrichment:")
            for idea in plan.get('environmental_enrichment', []):
                print(f"  • {idea}")
            
            # Test utility methods
            print(f"\n📝 Plan Summary:")
            print(f"  {agent.get_plan_summary(plan)}")
            
            print(f"\n✅ Top Mental Activities:")
            for activity in agent.get_mental_activities(plan)[:3]:
                print(f"  • {activity}")
        
        print("\n✅ Lifestyle Enrichment Agent ready for use")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")