# agents/nutrition_wellness_llm.py
"""
Nutrition Wellness LLM Agent for PawCare+ (Wellness Path).
Generates nutrition enhancement guidance for healthy, low-risk pets.
"""

from typing import Dict, Any, List, Optional
import logging

from agents.base_llm_agent import BaseLLMAgent
from utils.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


class NutritionWellnessAgent(BaseLLMAgent):
    """
    LLM agent for generating nutrition enhancement guidance for healthy, low-risk pets.
    
    This agent is part of the WELLNESS PATH and provides enhancement guidance including:
    - Overview of current nutrition
    - Enhancement tips for optimal nutrition
    - Diet variety suggestions
    - Optional supplement considerations
    """
    
    # Required fields in the output JSON
    REQUIRED_FIELDS = [
        "nutrition_overview",
        "enhancement_tips",
        "variety_suggestions",
        "supplement_options"
    ]
    
    # Species-specific nutritional enhancement ideas
    SPECIES_ENHANCEMENTS = {
        "dog": {
            "food_toppers": ["Pumpkin puree", "Plain yogurt", "Bone broth", "Fresh vegetables"],
            "rotational_proteins": ["Chicken", "Beef", "Fish", "Lamb", "Turkey"],
            "healthy_additions": ["Blueberries", "Green beans", "Carrots", "Apples (no seeds)"],
            "supplements": ["Fish oil (Omega-3)", "Probiotics", "Joint support (glucosamine)"]
        },
        "cat": {
            "food_toppers": ["Bone broth", "Pureed pumpkin", "Fish oil", "Crushed treats"],
            "rotational_proteins": ["Chicken", "Turkey", "Fish", "Rabbit"],
            "healthy_additions": ["Cat grass", "Small vegetable bits", "Egg (cooked)"],
            "supplements": ["Taurine", "Omega-3s", "Probiotics", "Hairball remedies"]
        },
        "rabbit": {
            "food_toppers": ["Fresh herbs", "Small fruit bits", "Vegetable tops"],
            "rotational_proteins": ["Timothy hay", "Meadow hay", "Orchard grass"],
            "healthy_additions": ["Dandelion greens", "Basil", "Cilantro", "Parsley"],
            "supplements": ["Vitamin D", "Oxbow supplements", "Digestive support"]
        }
    }
    
    def __init__(self, client: OpenAIClient):
        """
        Initialize the nutrition wellness agent.
        
        Args:
            client: OpenAIClient instance for making API calls
        """
        super().__init__(
            client=client,
            agent_name="NutritionWellness",
            default_temperature=0.5,  # Balanced for creative suggestions
            default_max_tokens=350     # As specified
        )
        logger.info("NutritionWellnessAgent initialized")
    def generate(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Implementation of abstract generate method from BaseLLMAgent.
        Delegates to the specific generation method.
        """
        # Call the specific method based on what the agent does
        return self.generate_nutrition_enhancement(*args, **kwargs)
    def generate_nutrition_enhancement(
        self,
        profile: Dict[str, Any],
        ml_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate nutrition enhancement guidance for healthy, low-risk pets.
        
        Args:
            profile: Extracted pet profile containing:
                - pet_species: str
                - breed: str
                - age_years: int
                - weight_status: str
                - diet_type: str
                - diet_quality: str
                - allergies_known: List[str] (if any)
                
            ml_results: ML predictions containing:
                - health_risk_score: float (0-1) - should be ≤ 0.3 for wellness path
                - care_capability_score: float (0-100)
        
        Returns:
            Dictionary containing:
                - nutrition_wellness: Dictionary with 4 required fields
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
            
            logger.info(f"Generating nutrition enhancement for {pet_info['species']} {pet_info['breed']}")
            
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
                error_msg = result.get("_error", "Unknown error in nutrition enhancement generation")
                logger.error(f"Nutrition enhancement generation failed: {error_msg}")
                
                return {
                    "nutrition_wellness": self._get_fallback_enhancement(pet_info),
                    "status": "error",
                    "message": f"Enhancement generation failed: {error_msg}"
                }
            
            # Remove internal metadata
            enhancement = {k: v for k, v in result.items() if not k.startswith('_')}
            
            # Ensure all required fields are present with correct types
            enhancement = self._validate_and_format_enhancement(enhancement, pet_info)
            
            logger.info("Nutrition enhancement generated successfully")
            
            return {
                "nutrition_wellness": enhancement,
                "status": "success",
                "message": "Nutrition enhancement generated successfully"
            }
            
        except Exception as e:
            logger.error(f"Unexpected error in nutrition enhancement: {str(e)}")
            
            # Provide fallback enhancement
            pet_info = self._extract_pet_info(profile)
            
            return {
                "nutrition_wellness": self._get_fallback_enhancement(pet_info),
                "status": "error",
                "message": f"Enhancement generation failed: {str(e)}"
            }
    
    def _extract_pet_info(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant pet information for nutrition enhancement.
        
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
        
        # Current diet
        weight_status = profile.get('weight_status', 'normal')
        diet_type = profile.get('diet_type', 'unknown')
        diet_quality = profile.get('diet_quality', 'unknown')
        
        # Allergies and restrictions
        allergies = profile.get('allergies_known', [])
        if not isinstance(allergies, list):
            allergies = []
        
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
            "weight_status": weight_status,
            "diet_type": diet_type,
            "diet_quality": diet_quality,
            "allergies": allergies,
            "allergies_text": self._format_list_for_prompt(allergies, "No known allergies"),
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
        return """You are a pet nutrition specialist focusing on wellness optimization. Your role is to provide creative, safe suggestions for enhancing the nutrition of healthy pets.

For each case, you must provide a JSON response with exactly these four fields:

1. "nutrition_overview": A brief overview (1-2 sentences) of the pet's current nutritional status and approach.

2. "enhancement_tips": A list of 3-5 specific tips for enhancing the pet's nutrition (e.g., food toppers, preparation methods, feeding strategies).

3. "variety_suggestions": A list of 3-5 suggestions for adding safe variety to the pet's diet (e.g., protein rotation, vegetable additions).

4. "supplement_options": A list of 2-4 optional supplements that may benefit this pet, with brief explanations of their potential benefits.

Your response should be positive, practical, and focused on enhancement rather than correction. Always consider food safety and species-appropriate options."""
    
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
        return f"""Provide nutrition enhancement suggestions for this healthy pet:

=== PET PROFILE ===
Name: {pet_info['name']}
Species: {pet_info['species']}
Breed: {pet_info['breed']}
Age: {pet_info['age']} years ({pet_info['life_stage']})
Weight Status: {pet_info['weight_status']}
Current Diet: {pet_info['diet_type']} ({pet_info['diet_quality']})
Allergies/Restrictions: {pet_info['allergies_text']}

=== WELLNESS STATUS ===
Health Risk Score: {risk_info['risk_score_percent']:.1f}% (LOW)

Generate nutrition enhancement suggestions with the four required fields."""
    
    def _validate_and_format_enhancement(
        self,
        enhancement: Dict[str, Any],
        pet_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate and format the nutrition enhancement to ensure correct structure.
        
        Args:
            enhancement: Raw enhancement dictionary
            pet_info: Pet information for context
            
        Returns:
            Formatted enhancement with proper types
        """
        formatted = {}
        species = pet_info['species'].lower()
        life_stage = pet_info['life_stage']
        
        # Format nutrition_overview
        if "nutrition_overview" in enhancement:
            formatted["nutrition_overview"] = str(enhancement["nutrition_overview"])
        else:
            formatted["nutrition_overview"] = (
                f"Your {life_stage} {species} is on a {pet_info['diet_quality']} quality "
                f"{pet_info['diet_type']} diet. Here are some ways to enhance their nutrition."
            )
        
        # Format enhancement_tips (should be list)
        if "enhancement_tips" in enhancement:
            tips = enhancement["enhancement_tips"]
            if isinstance(tips, list):
                formatted["enhancement_tips"] = [str(tip) for tip in tips]
            elif isinstance(tips, str):
                # Split by lines or commas
                if '\n' in tips:
                    formatted["enhancement_tips"] = [t.strip() for t in tips.split('\n') if t.strip()]
                elif ',' in tips:
                    formatted["enhancement_tips"] = [t.strip() for t in tips.split(',')]
                else:
                    formatted["enhancement_tips"] = [tips]
            else:
                formatted["enhancement_tips"] = [str(tips)] if tips else []
        else:
            # Default tips from species mapping
            toppers = pet_info['enhancements'].get('food_toppers', [])
            formatted["enhancement_tips"] = [
                f"Add {toppers[0] if toppers else 'healthy toppers'} for variety",
                f"Use puzzle feeders for mental stimulation during meals",
                f"Consider {pet_info['enhancements'].get('healthy_additions', ['fresh foods'])[0]} as occasional additions",
                "Establish consistent meal times for routine"
            ][:4]
        
        # Ensure at least 3 tips
        while len(formatted["enhancement_tips"]) < 3:
            formatted["enhancement_tips"].append("Offer fresh water changed daily")
        
        # Limit to 5 tips
        formatted["enhancement_tips"] = formatted["enhancement_tips"][:5]
        
        # Format variety_suggestions (should be list)
        if "variety_suggestions" in enhancement:
            variety = enhancement["variety_suggestions"]
            if isinstance(variety, list):
                formatted["variety_suggestions"] = [str(v) for v in variety]
            elif isinstance(variety, str):
                # Split by lines or commas
                if '\n' in variety:
                    formatted["variety_suggestions"] = [v.strip() for v in variety.split('\n') if v.strip()]
                elif ',' in variety:
                    formatted["variety_suggestions"] = [v.strip() for v in variety.split(',')]
                else:
                    formatted["variety_suggestions"] = [variety]
            else:
                formatted["variety_suggestions"] = [str(variety)] if variety else []
        else:
            # Default variety suggestions
            proteins = pet_info['enhancements'].get('rotational_proteins', [])
            additions = pet_info['enhancements'].get('healthy_additions', [])
            
            formatted["variety_suggestions"] = [
                f"Rotate proteins: {', '.join(proteins[:3])}",
                f"Add variety with {additions[0] if additions else 'fresh foods'}",
                "Change texture (mix wet and dry if appropriate)",
                "Seasonal vegetable rotations",
                "Different treat types for training"
            ][:4]
        
        # Ensure at least 3 suggestions
        while len(formatted["variety_suggestions"]) < 3:
            formatted["variety_suggestions"].append("Introduce new foods gradually")
        
        # Limit to 5 suggestions
        formatted["variety_suggestions"] = formatted["variety_suggestions"][:5]
        
        # Format supplement_options (should be list)
        if "supplement_options" in enhancement:
            supplements = enhancement["supplement_options"]
            if isinstance(supplements, list):
                formatted["supplement_options"] = [str(s) for s in supplements]
            elif isinstance(supplements, str):
                # Split by lines or commas
                if '\n' in supplements:
                    formatted["supplement_options"] = [s.strip() for s in supplements.split('\n') if s.strip()]
                elif ',' in supplements:
                    formatted["supplement_options"] = [s.strip() for s in supplements.split(',')]
                else:
                    formatted["supplement_options"] = [supplements]
            else:
                formatted["supplement_options"] = [str(supplements)] if supplements else []
        else:
            # Default supplement options
            supps = pet_info['enhancements'].get('supplements', [])
            formatted["supplement_options"] = [
                f"{supps[0] if supps else 'Omega-3s'}: Supports skin, coat, and joint health",
                f"{supps[1] if len(supps) > 1 else 'Probiotics'}: Promotes digestive health",
                f"{supps[2] if len(supps) > 2 else 'Joint support'}: For long-term mobility",
                "Always consult your veterinarian before starting supplements"
            ][:3]
        
        # Ensure at least 2 options
        while len(formatted["supplement_options"]) < 2:
            formatted["supplement_options"].append("Consult your vet about species-specific supplements")
        
        # Limit to 4 options
        formatted["supplement_options"] = formatted["supplement_options"][:4]
        
        return formatted
    
    def _get_fallback_enhancement(self, pet_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate fallback nutrition enhancement when LLM fails.
        
        Args:
            pet_info: Pet information
            
        Returns:
            Dictionary with fallback nutrition enhancement
        """
        species = pet_info['species'].lower()
        life_stage = pet_info['life_stage']
        
        return {
            "nutrition_overview": (
                f"Your {life_stage} {species} is on a healthy track. These enhancements "
                f"can add variety and nutritional benefits to their diet."
            ),
            "enhancement_tips": [
                "Add small amounts of fresh, pet-safe vegetables as toppers",
                "Use food puzzles to make mealtime more engaging",
                "Consider mixing wet and dry food for texture variety",
                "Establish consistent feeding times for routine"
            ],
            "variety_suggestions": [
                "Rotate protein sources gradually over weeks",
                "Introduce new healthy treats one at a time",
                "Try seasonal vegetables in small amounts",
                "Alternate between different formulations of quality brands"
            ],
            "supplement_options": [
                "Omega-3 fatty acids: Supports skin, coat, and joint health",
                "Probiotics: Promotes digestive health and immunity",
                "Joint supplements: Consider for long-term support",
                "Always consult your vet before adding supplements"
            ]
        }
    
    def get_enhancement_summary(self, enhancement: Dict[str, Any]) -> str:
        """
        Get a brief summary of the nutrition enhancement for display.
        
        Args:
            enhancement: Nutrition enhancement dictionary
            
        Returns:
            Brief summary string
        """
        if not enhancement:
            return "No nutrition enhancement available."
        
        overview = enhancement.get('nutrition_overview', '')
        tips = enhancement.get('enhancement_tips', [])
        
        summary = f"{overview}\n\nKey Tips:\n"
        
        if tips:
            for tip in tips[:3]:  # Show first 3 tips
                summary += f"  • {tip}\n"
        
        return summary
    
    def get_enhancement_tips(self, enhancement: Dict[str, Any]) -> List[str]:
        """
        Extract enhancement tips list.
        
        Args:
            enhancement: Nutrition enhancement dictionary
            
        Returns:
            List of enhancement tips
        """
        tips = enhancement.get('enhancement_tips', [])
        if isinstance(tips, list):
            return tips
        return []
    
    def get_variety_suggestions(self, enhancement: Dict[str, Any]) -> List[str]:
        """
        Extract variety suggestions list.
        
        Args:
            enhancement: Nutrition enhancement dictionary
            
        Returns:
            List of variety suggestions
        """
        variety = enhancement.get('variety_suggestions', [])
        if isinstance(variety, list):
            return variety
        return []
    
    def get_supplement_options(self, enhancement: Dict[str, Any]) -> List[str]:
        """
        Extract supplement options list.
        
        Args:
            enhancement: Nutrition enhancement dictionary
            
        Returns:
            List of supplement options
        """
        supplements = enhancement.get('supplement_options', [])
        if isinstance(supplements, list):
            return supplements
        return []


# ==========================================
# CONVENIENCE FUNCTION
# ==========================================

def generate_nutrition_wellness(
    client: OpenAIClient,
    profile: Dict[str, Any],
    ml_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convenience function to generate nutrition wellness enhancement.
    
    Args:
        client: OpenAIClient instance
        profile: Extracted pet profile
        ml_results: ML prediction results
        
    Returns:
        Dictionary with nutrition enhancement and status
    """
    agent = NutritionWellnessAgent(client)
    return agent.generate_nutrition_enhancement(profile, ml_results)


# ==========================================
# MODULE TESTING
# ==========================================

if __name__ == "__main__":
    import json
    from utils.openai_client import build_openai_client
    
    print("=" * 60)
    print("NUTRITION WELLNESS AGENT TEST")
    print("=" * 60)
    
    try:
        # Initialize client
        client = build_openai_client()
        print("✅ OpenAI client created")
        
        # Initialize agent
        agent = NutritionWellnessAgent(client)
        print(f"✅ Agent initialized: {agent.agent_name}")
        
        # Test with sample data - Healthy adult dog
        test_profile = {
            "pet_name": "Bailey",
            "pet_species": "dog",
            "breed": "golden retriever",
            "age_years": 3,
            "weight_status": "normal",
            "diet_type": "kibble",
            "diet_quality": "premium",
            "allergies_known": []
        }
        
        test_ml_results = {
            "health_risk_score": 0.15,
            "care_capability_score": 85.0
        }
        
        print("\n📤 Generating nutrition wellness enhancement...")
        result = agent.generate_nutrition_enhancement(test_profile, test_ml_results)
        
        print(f"\n📊 Nutrition Enhancement Result:")
        print(f"Status: {result['status']}")
        
        if result['status'] == 'success':
            enhancement = result['nutrition_wellness']
            
            print("\n🥗 NUTRITION WELLNESS ENHANCEMENT")
            print("=" * 40)
            
            print(f"\n📝 Nutrition Overview:")
            print(f"  {enhancement.get('nutrition_overview', 'N/A')}")
            
            print(f"\n✨ Enhancement Tips:")
            for tip in enhancement.get('enhancement_tips', []):
                print(f"  • {tip}")
            
            print(f"\n🔄 Variety Suggestions:")
            for suggestion in enhancement.get('variety_suggestions', []):
                print(f"  • {suggestion}")
            
            print(f"\n💊 Supplement Options:")
            for option in enhancement.get('supplement_options', []):
                print(f"  • {option}")
            
            # Test utility methods
            print(f"\n📝 Enhancement Summary:")
            print(f"  {agent.get_enhancement_summary(enhancement)}")
            
            print(f"\n✅ Top Tips:")
            for tip in agent.get_enhancement_tips(enhancement)[:3]:
                print(f"  • {tip}")
        
        print("\n✅ Nutrition Wellness Agent ready for use")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")