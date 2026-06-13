# agents/nutrition_critical_llm.py
"""
Nutrition Critical LLM Agent for PawCare+ (Critical Path).
Generates specialized nutrition plans for high-risk pets with serious health conditions.
"""

from typing import Dict, Any, List, Optional
import logging

from agents.base_llm_agent import BaseLLMAgent
from utils.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


class NutritionCriticalAgent(BaseLLMAgent):
    """
    LLM agent for generating critical nutrition plans for high-risk pets.
    
    This agent is part of the CRITICAL PATH and provides specialized nutrition guidance including:
    - Condition-specific dietary recommendations
    - Feeding schedules for medical management
    - Therapeutic supplements
    - Foods to avoid based on conditions
    - Weight management strategies
    - Meal transition plans
    """
    
    # Required fields in the output JSON
    REQUIRED_FIELDS = [
        "nutrition_overview",
        "recommended_diet",
        "feeding_schedule",
        "supplements",
        "foods_to_avoid",
        "weight_management",
        "transition_plan",
        "meal_prep_examples"
    ]
    
    # Condition-specific dietary considerations
    CONDITION_DIET_MAPPING = {
        "diabetes": "Low glycemic, consistent carbohydrate, high fiber",
        "kidney disease": "Reduced phosphorus, moderate protein, omega-3 fatty acids",
        "liver disease": "Moderate protein, high quality, easily digestible carbohydrates",
        "pancreatitis": "Very low fat, easily digestible, frequent small meals",
        "heart disease": "Reduced sodium, taurine supplementation, omega-3s",
        "allergies": "Novel protein or hydrolyzed protein, limited ingredients",
        "arthritis": "Joint supplements, omega-3s, weight management",
        "cancer": "High quality protein, omega-3s, antioxidants",
        "obesity": "Calorie restricted, high fiber, weight management formula",
        "gastrointestinal": "Easily digestible, low fat, possibly hydrolyzed",
        "urinary": "Controlled minerals, increased moisture, urinary health formula",
        "hyperthyroidism": "High quality protein, phosphorus restriction, iodine controlled"
    }
    
    def __init__(self, client: OpenAIClient):
        """
        Initialize the nutrition critical agent.
        
        Args:
            client: OpenAIClient instance for making API calls
        """
        super().__init__(
            client=client,
            agent_name="NutritionCritical",
            default_temperature=0.5,  # Balanced for dietary planning
            default_max_tokens=1200    # As specified
        )
        logger.info("NutritionCriticalAgent initialized")
    def generate(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Implementation of abstract generate method from BaseLLMAgent.
        Delegates to the specific generation method.
        """
        # Call the specific method based on what the agent does
        return self.generate_nutrition_plan(*args, **kwargs)
    def generate_nutrition_plan(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate critical nutrition plan for high-risk pets.
        
        Args:
            profile: Extracted pet profile containing:
                - pet_species: str
                - breed: str
                - age_years: int
                - weight_status: str
                - known_conditions: List[str]
                - allergies_known: List[str]
                - current_diet: str (optional)
                - diet_type: str
                - diet_quality: str
                - feeding_frequency: str (optional)
        
        Returns:
            Dictionary containing:
                - nutrition_critical: Dictionary with 8 required fields
                - status: String "success" or "error"
                - message: Optional status message
        """
        try:
            # Extract key information
            pet_info = self._extract_pet_info(profile)
            
            # Build prompts
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(pet_info)
            
            logger.info(f"Generating critical nutrition plan for {pet_info['species']} {pet_info['breed']} "
                       f"with conditions: {pet_info['conditions_text']}")
            
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
                error_msg = result.get("_error", "Unknown error in nutrition planning")
                logger.error(f"Nutrition plan generation failed: {error_msg}")
                
                return {
                    "nutrition_critical": self._get_fallback_plan(pet_info),
                    "status": "error",
                    "message": f"Planning failed: {error_msg}"
                }
            
            # Remove internal metadata
            plan = {k: v for k, v in result.items() if not k.startswith('_')}
            
            # Ensure all required fields are present with correct types
            plan = self._validate_and_format_plan(plan, pet_info)
            
            logger.info("Critical nutrition plan generated successfully")
            
            return {
                "nutrition_critical": plan,
                "status": "success",
                "message": "Nutrition plan generated successfully"
            }
            
        except Exception as e:
            logger.error(f"Unexpected error in nutrition planning: {str(e)}")
            
            # Provide fallback plan
            pet_info = self._extract_pet_info(profile)
            
            return {
                "nutrition_critical": self._get_fallback_plan(pet_info),
                "status": "error",
                "message": f"Planning failed: {str(e)}"
            }
    
    def _extract_pet_info(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant pet information for nutrition planning.
        
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
        
        # Age
        age = profile.get('age_years', 0)
        age_category = "senior" if age > 7 else "adult" if age > 1 else "young"
        
        # Weight
        weight_status = profile.get('weight_status', 'unknown')
        
        # Medical info
        conditions = profile.get('known_conditions', [])
        if not isinstance(conditions, list):
            conditions = []
        
        allergies = profile.get('allergies_known', [])
        if not isinstance(allergies, list):
            allergies = []
        
        # Current diet info
        diet_type = profile.get('diet_type', 'unknown')
        diet_quality = profile.get('diet_quality', 'unknown')
        current_diet = profile.get('current_diet', 'Not specified')
        feeding_freq = profile.get('feeding_frequency', 'Not specified')
        
        # Get condition-specific dietary considerations
        dietary_needs = self._get_dietary_considerations(conditions)
        
        return {
            "name": name,
            "species": species,
            "breed": breed,
            "age": age,
            "age_category": age_category,
            "weight_status": weight_status,
            "conditions": conditions,
            "conditions_text": self._format_list_for_prompt(conditions, "No specific conditions"),
            "allergies": allergies,
            "allergies_text": self._format_list_for_prompt(allergies, "No known allergies"),
            "diet_type": diet_type,
            "diet_quality": diet_quality,
            "current_diet": current_diet,
            "feeding_frequency": feeding_freq,
            "dietary_needs": dietary_needs,
            "dietary_needs_text": "\n".join([f"• {need}" for need in dietary_needs])
        }
    
    def _get_dietary_considerations(self, conditions: List[str]) -> List[str]:
        """
        Get dietary considerations based on medical conditions.
        
        Args:
            conditions: List of medical conditions
            
        Returns:
            List of dietary considerations
        """
        considerations = []
        
        for condition in conditions:
            condition_lower = condition.lower()
            
            # Check for partial matches in conditions
            for key, value in self.CONDITION_DIET_MAPPING.items():
                if key in condition_lower:
                    considerations.append(f"{condition}: {value}")
                    break
            else:
                # Generic consideration if condition not in mapping
                considerations.append(f"{condition}: Veterinary therapeutic diet may be beneficial")
        
        # Remove duplicates
        return list(dict.fromkeys(considerations))
    
    def _build_system_prompt(self) -> str:
        """
        Build the system prompt defining the agent's role and output format.
        
        Returns:
            System prompt string
        """
        return """You are a board-certified veterinary nutritionist with expertise in therapeutic diets for pets with serious medical conditions. Your role is to create specialized, condition-appropriate nutrition plans that support medical treatment and improve outcomes.

You understand that for high-risk pets, nutrition is not just about sustenance—it's a critical component of medical management. Your plans are precise, practical, and tailored to each pet's specific health challenges.

For each case, you must provide a JSON response with exactly these eight fields:

1. "nutrition_overview": A comprehensive overview (3-5 sentences) explaining the nutritional strategy and how it supports the pet's medical conditions.

2. "recommended_diet": Specific diet recommendations including type (commercial therapeutic diet, homemade, or mixed), brand suggestions if appropriate, and key nutritional components.

3. "feeding_schedule": Detailed feeding schedule including frequency, portion sizes, timing relative to medications, and any special considerations.

4. "supplements": List of recommended supplements with specific dosages, forms, and rationale for each based on the pet's conditions.

5. "foods_to_avoid": Comprehensive list of foods to avoid, organized by condition relevance and including both obvious and hidden sources.

6. "weight_management": Specific weight management strategy if needed, including target weight, monitoring frequency, and adjustment protocols.

7. "transition_plan": Step-by-step plan for transitioning to the new diet over 7-10 days to minimize gastrointestinal upset.

8. "meal_prep_examples": Practical examples of meal preparations, including portion measurements and preparation instructions.

Your response must be evidence-based, practical, and tailored to the pet's specific conditions and circumstances."""
    
    def _build_user_prompt(self, pet_info: Dict[str, Any]) -> str:
        """
        Build the user prompt with specific pet information.
        
        Args:
            pet_info: Extracted pet information
            
        Returns:
            User prompt string
        """
        return f"""Create a comprehensive critical nutrition plan for this high-risk pet:

=== PET PROFILE ===
Name: {pet_info['name']}
Species: {pet_info['species']}
Breed: {pet_info['breed']}
Age: {pet_info['age']} years ({pet_info['age_category']})
Weight Status: {pet_info['weight_status']}

=== MEDICAL CONDITIONS ===
{pet_info['conditions_text']}

=== ALLERGIES/INTOLERANCES ===
{pet_info['allergies_text']}

=== CURRENT DIET ===
Diet Type: {pet_info['diet_type']}
Diet Quality: {pet_info['diet_quality']}
Current Diet: {pet_info['current_diet']}
Feeding Frequency: {pet_info['feeding_frequency']}

=== DIETARY CONSIDERATIONS ===
{pet_info['dietary_needs_text']}

Based on this information, generate a comprehensive critical nutrition plan with all eight required fields. Focus on therapeutic dietary interventions that support the pet's medical conditions and overall health."""
    
    def _validate_and_format_plan(
        self,
        plan: Dict[str, Any],
        pet_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate and format the nutrition plan to ensure correct structure.
        
        Args:
            plan: Raw plan dictionary
            pet_info: Pet information for context
            
        Returns:
            Formatted plan with proper types
        """
        formatted = {}
        species = pet_info['species'].lower()
        
        # Format nutrition_overview
        if "nutrition_overview" in plan:
            formatted["nutrition_overview"] = str(plan["nutrition_overview"])
        else:
            formatted["nutrition_overview"] = (
                f"A therapeutic nutrition plan for this {species} addressing "
                f"{pet_info['conditions_text']}. The focus is on condition-specific "
                f"nutritional support while maintaining overall health."
            )
        
        # Format recommended_diet
        if "recommended_diet" in plan:
            formatted["recommended_diet"] = str(plan["recommended_diet"])
        else:
            formatted["recommended_diet"] = (
                f"Veterinary therapeutic diet recommended based on primary conditions. "
                f"Consult with your veterinarian for specific prescription diet options."
            )
        
        # Format feeding_schedule
        if "feeding_schedule" in plan:
            formatted["feeding_schedule"] = str(plan["feeding_schedule"])
        else:
            formatted["feeding_schedule"] = (
                f"Feed {pet_info['age_category']} {species} 2-3 times daily. "
                f"Consistent timing is crucial for condition management. "
                f"Divide daily portion into equal meals."
            )
        
        # Format supplements (should be list)
        if "supplements" in plan:
            supplements = plan["supplements"]
            if isinstance(supplements, list):
                formatted["supplements"] = [str(s) for s in supplements]
            else:
                formatted["supplements"] = [str(supplements)] if supplements else []
        else:
            formatted["supplements"] = [
                f"Omega-3 fatty acids: Support inflammation and {pet_info['conditions_text']}",
                "Probiotics: Support digestive health during medical treatment",
                "Joint supplements: If arthritis or mobility issues present"
            ]
        
        # Format foods_to_avoid (should be list)
        if "foods_to_avoid" in plan:
            foods = plan["foods_to_avoid"]
            if isinstance(foods, list):
                formatted["foods_to_avoid"] = [str(f) for f in foods]
            else:
                formatted["foods_to_avoid"] = [str(foods)] if foods else []
        else:
            formatted["foods_to_avoid"] = [
                "High-fat foods (especially if pancreatitis or obesity)",
                f"Foods containing allergens: {pet_info['allergies_text']}",
                "Table scraps and human food (unless specifically approved)",
                "Sudden diet changes without transition"
            ]
        
        # Format weight_management
        if "weight_management" in plan:
            formatted["weight_management"] = str(plan["weight_management"])
        else:
            formatted["weight_management"] = (
                f"Current weight status: {pet_info['weight_status']}. "
                f"Weigh weekly and adjust portions to maintain ideal body condition. "
                f"Target: Visible waist, easily palpable ribs."
            )
        
        # Format transition_plan
        if "transition_plan" in plan:
            formatted["transition_plan"] = str(plan["transition_plan"])
        else:
            formatted["transition_plan"] = (
                "Days 1-2: 25% new diet / 75% old diet\n"
                "Days 3-4: 50% new diet / 50% old diet\n"
                "Days 5-6: 75% new diet / 25% old diet\n"
                "Day 7+: 100% new diet\n\n"
                "Monitor for GI upset and slow transition if needed."
            )
        
        # Format meal_prep_examples
        if "meal_prep_examples" in plan:
            formatted["meal_prep_examples"] = str(plan["meal_prep_examples"])
        else:
            formatted["meal_prep_examples"] = (
                f"For a {pet_info['weight_status']} {species} of {pet_info['age']} years:\n"
                f"Morning: [Specific portion of recommended diet]\n"
                f"Evening: [Specific portion of recommended diet]\n"
                f"Adjust portions based on body condition and activity level."
            )
        
        return formatted
    
    def _get_fallback_plan(self, pet_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate fallback nutrition plan when LLM fails.
        
        Args:
            pet_info: Pet information
            
        Returns:
            Dictionary with fallback nutrition plan
        """
        species = pet_info['species'].lower()
        
        return {
            "nutrition_overview": (
                f"This {species} requires careful nutritional management due to "
                f"{pet_info['conditions_text']}. The following plan provides general "
                f"guidelines that should be discussed with your veterinarian."
            ),
            "recommended_diet": (
                f"Based on the primary conditions ({pet_info['conditions_text']}), "
                f"a veterinary therapeutic diet is recommended. Consult your vet about "
                f"specific prescription diets from brands like Hill's Prescription Diet, "
                f"Royal Canin Veterinary Diet, or Purina Pro Plan Veterinary Diets."
            ),
            "feeding_schedule": (
                f"Feed {pet_info['age_category']} {species} 2-3 times daily at consistent times. "
                f"Total daily amount should be divided equally among meals. "
                f"Measure portions accurately using a kitchen scale for precision."
            ),
            "supplements": [
                f"Omega-3 fatty acids (EPA/DHA): Supports inflammation management",
                f"Probiotics: Supports digestive health, especially if on medications",
                f"Joint supplements (glucosamine/chondroitin): If mobility issues present",
                "Always consult your vet before adding supplements"
            ],
            "foods_to_avoid": [
                "High-fat foods and table scraps",
                f"Foods containing: {pet_info['allergies_text']}",
                "Sudden changes in diet",
                "Foods with high sodium or phosphorus (if kidney/cardiac concerns)",
                "Raw diets (immunocompromised pets)"
            ],
            "weight_management": (
                f"Current weight status: {pet_info['weight_status']}. "
                f"Monitor weight weekly and adjust portions to maintain ideal body condition. "
                f"Work with your vet to establish target weight and monitoring schedule."
            ),
            "transition_plan": (
                "Day 1-2: 25% new diet / 75% old diet\n"
                "Day 3-4: 50% new diet / 50% old diet\n"
                "Day 5-6: 75% new diet / 25% old diet\n"
                "Day 7: 100% new diet\n\n"
                "Monitor for vomiting, diarrhea, or decreased appetite. "
                "If signs occur, slow the transition and consult your vet."
            ),
            "meal_prep_examples": (
                "Morning Meal (approx. 1/3 of daily calories):\n"
                "- [X] cups/grams of recommended diet\n"
                "- Add supplements as directed\n\n"
                "Evening Meal (approx. 1/3 of daily calories):\n"
                "- [X] cups/grams of recommended diet\n\n"
                "Mid-day Meal (if feeding 3x daily):\n"
                "- [X] cups/grams of recommended diet\n\n"
                "Adjust portions based on body condition score and activity level."
            )
        }
    
    def get_plan_summary(self, plan: Dict[str, Any]) -> str:
        """
        Get a brief summary of the nutrition plan for display.
        
        Args:
            plan: Nutrition plan dictionary
            
        Returns:
            Brief summary string
        """
        if not plan:
            return "No nutrition plan available."
        
        overview = plan.get('nutrition_overview', '')
        diet = plan.get('recommended_diet', '')
        
        summary = f"{overview}\n\n"
        summary += f"Recommended: {diet[:150]}..."
        
        return summary
    
    def get_supplements_list(self, plan: Dict[str, Any]) -> List[str]:
        """
        Extract supplements list for quick reference.
        
        Args:
            plan: Nutrition plan dictionary
            
        Returns:
            List of supplement strings
        """
        supplements = plan.get('supplements', [])
        if isinstance(supplements, list):
            return supplements
        return []
    
    def get_foods_to_avoid_list(self, plan: Dict[str, Any]) -> List[str]:
        """
        Extract foods to avoid list for quick reference.
        
        Args:
            plan: Nutrition plan dictionary
            
        Returns:
            List of foods to avoid
        """
        foods = plan.get('foods_to_avoid', [])
        if isinstance(foods, list):
            return foods
        return []
    
    def get_condition_specific_guidance(self, condition: str) -> str:
        """
        Get specific dietary guidance for a condition.
        
        Args:
            condition: Medical condition name
            
        Returns:
            Dietary guidance string
        """
        condition_lower = condition.lower()
        
        for key, value in self.CONDITION_DIET_MAPPING.items():
            if key in condition_lower:
                return f"For {condition}: {value}"
        
        return f"For {condition}: Consult your veterinarian for specific dietary recommendations."


# ==========================================
# CONVENIENCE FUNCTION
# ==========================================

def generate_critical_nutrition_plan(
    client: OpenAIClient,
    profile: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convenience function to generate critical nutrition plan.
    
    Args:
        client: OpenAIClient instance
        profile: Extracted pet profile
        
    Returns:
        Dictionary with nutrition plan and status
    """
    agent = NutritionCriticalAgent(client)
    return agent.generate_nutrition_plan(profile)


# ==========================================
# MODULE TESTING
# ==========================================

if __name__ == "__main__":
    import json
    from utils.openai_client import build_openai_client
    
    print("=" * 60)
    print("NUTRITION CRITICAL AGENT TEST")
    print("=" * 60)
    
    try:
        # Initialize client
        client = build_openai_client()
        print("✅ OpenAI client created")
        
        # Initialize agent
        agent = NutritionCriticalAgent(client)
        print(f"✅ Agent initialized: {agent.agent_name}")
        
        # Test with sample data - Diabetic dog
        test_profile = {
            "pet_name": "Max",
            "pet_species": "dog",
            "breed": "labrador retriever",
            "age_years": 12,
            "weight_status": "overweight",
            "known_conditions": ["diabetes mellitus", "arthritis", "heart murmur"],
            "allergies_known": ["chicken"],
            "diet_type": "kibble",
            "diet_quality": "good",
            "current_diet": "Adult maintenance kibble, twice daily",
            "feeding_frequency": "2 times daily"
        }
        
        print("\n📤 Generating critical nutrition plan...")
        result = agent.generate_nutrition_plan(test_profile)
        
        print(f"\n📊 Nutrition Plan Result:")
        print(f"Status: {result['status']}")
        
        if result['status'] == 'success':
            plan = result['nutrition_critical']
            
            print("\n🥗 CRITICAL NUTRITION PLAN")
            print("=" * 40)
            
            print(f"\n📋 Overview:")
            print(f"  {plan.get('nutrition_overview', 'N/A')}")
            
            print(f"\n🍖 Recommended Diet:")
            print(f"  {plan.get('recommended_diet', 'N/A')}")
            
            print(f"\n⏰ Feeding Schedule:")
            print(f"  {plan.get('feeding_schedule', 'N/A')}")
            
            print(f"\n💊 Supplements:")
            for supplement in plan.get('supplements', []):
                print(f"  • {supplement}")
            
            print(f"\n🚫 Foods to Avoid:")
            for food in plan.get('foods_to_avoid', []):
                print(f"  • {food}")
            
            print(f"\n⚖️ Weight Management:")
            print(f"  {plan.get('weight_management', 'N/A')}")
            
            print(f"\n🔄 Transition Plan:")
            print(f"  {plan.get('transition_plan', 'N/A')}")
            
            print(f"\n🍳 Meal Prep Examples:")
            print(f"  {plan.get('meal_prep_examples', 'N/A')}")
            
            # Test utility methods
            print(f"\n📝 Plan Summary:")
            print(f"  {agent.get_plan_summary(plan)}")
            
            print(f"\n✅ Supplements List:")
            for supp in agent.get_supplements_list(plan):
                print(f"  • {supp}")
        
        print("\n✅ Nutrition Critical Agent ready for use")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")