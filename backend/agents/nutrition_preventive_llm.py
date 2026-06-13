# agents/nutrition_preventive_llm.py
"""
Nutrition Preventive LLM Agent for PawCare+ (Preventive Path).
Generates preventive nutrition guidance for moderate-risk pets.
"""

from typing import Dict, Any, List, Optional
import logging

from agents.base_llm_agent import BaseLLMAgent
from utils.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


class NutritionPreventiveAgent(BaseLLMAgent):
    """
    LLM agent for generating preventive nutrition guidance for moderate-risk pets.
    
    This agent is part of the PREVENTIVE PATH and provides focused nutrition guidance including:
    - Nutritional approach overview
    - Diet recommendations
    - Portion size guidance
    - Healthy treat options
    """
    
    # Required fields in the output JSON
    REQUIRED_FIELDS = [
        "nutrition_overview",
        "recommended_diet",
        "portion_guidance",
        "healthy_treats"
    ]
    
    # Species-specific dietary considerations
    SPECIES_CONSIDERATIONS = {
        "dog": {
            "primary": "Complete and balanced commercial diet appropriate for life stage",
            "protein": "Moderate to high quality protein",
            "fat": "Moderate fat for energy",
            "fiber": "Adequate fiber for digestive health",
            "considerations": "Avoid grapes, raisins, onions, garlic, xylitol"
        },
        "cat": {
            "primary": "Obligate carnivore - requires meat-based diet",
            "protein": "High quality animal protein",
            "fat": "Moderate to high fat",
            "fiber": "Limited fiber needs",
            "considerations": "Avoid onions, garlic, grapes, raisins; ensure adequate taurine"
        },
        "rabbit": {
            "primary": "Unlimited grass hay, limited pellets, fresh vegetables",
            "protein": "Moderate protein from hay and limited pellets",
            "fat": "Low fat",
            "fiber": "High fiber essential for GI health",
            "considerations": "Avoid muesli-style mixes, high-carb treats"
        }
    }
    
    def __init__(self, client: OpenAIClient):
        """
        Initialize the preventive nutrition agent.
        
        Args:
            client: OpenAIClient instance for making API calls
        """
        super().__init__(
            client=client,
            agent_name="NutritionPreventive",
            default_temperature=0.4,  # Focused/deterministic for nutrition guidance
            default_max_tokens=350     # As specified
        )
        logger.info("NutritionPreventiveAgent initialized")
    
    # ==========================================
    # IMPLEMENTATION OF ABSTRACT GENERATE METHOD
    # ==========================================
    def generate(self, profile: Dict[str, Any], ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implementation of abstract generate method from BaseLLMAgent.
        Delegates to generate_nutrition_guide.
        
        Args:
            profile: Extracted pet profile
            ml_results: ML prediction results
            
        Returns:
            Dictionary with nutrition guide results
        """
        return self.generate_nutrition_guide(profile, ml_results)
    
    def generate_nutrition_guide(
        self,
        profile: Dict[str, Any],
        ml_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate preventive nutrition guide for moderate-risk pets.
        
        Args:
            profile: Extracted pet profile containing:
                - pet_species: str
                - breed: str
                - age_years: int
                - weight_status: str
                - known_conditions: List[str] (if any)
                - diet_type: str
                - diet_quality: str
                - allergies_known: List[str]
                
            ml_results: ML predictions containing:
                - health_risk_score: float (0-1)
                - care_capability_score: float (0-100)
        
        Returns:
            Dictionary containing:
                - nutrition_preventive: Dictionary with 4 required fields
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
            
            logger.info(f"Generating preventive nutrition guide for {pet_info['species']} {pet_info['breed']}")
            
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
                error_msg = result.get("_error", "Unknown error in nutrition guide generation")
                logger.error(f"Nutrition guide generation failed: {error_msg}")
                
                return {
                    "nutrition_preventive": self._get_fallback_guide(pet_info),
                    "status": "error",
                    "message": f"Guide generation failed: {error_msg}"
                }
            
            # Remove internal metadata
            guide = {k: v for k, v in result.items() if not k.startswith('_')}
            
            # Ensure all required fields are present with correct types
            guide = self._validate_and_format_guide(guide, pet_info)
            
            logger.info("Preventive nutrition guide generated successfully")
            
            return {
                "nutrition_preventive": guide,
                "status": "success",
                "message": "Nutrition guide generated successfully"
            }
            
        except Exception as e:
            logger.error(f"Unexpected error in nutrition guide generation: {str(e)}")
            
            # Provide fallback guide
            pet_info = self._extract_pet_info(profile)
            
            return {
                "nutrition_preventive": self._get_fallback_guide(pet_info),
                "status": "error",
                "message": f"Guide generation failed: {str(e)}"
            }
    
    def _extract_pet_info(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant pet information for nutrition guidance.
        
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
        
        # Weight status
        weight_status = profile.get('weight_status', 'unknown')
        
        # Current diet
        diet_type = profile.get('diet_type', 'unknown')
        diet_quality = profile.get('diet_quality', 'unknown')
        
        # Health considerations
        conditions = profile.get('known_conditions', [])
        if not isinstance(conditions, list):
            conditions = []
        
        allergies = profile.get('allergies_known', [])
        if not isinstance(allergies, list):
            allergies = []
        
        # Get species-specific considerations
        species_key = species.lower()
        considerations = self.SPECIES_CONSIDERATIONS.get(
            species_key, 
            self.SPECIES_CONSIDERATIONS.get('dog')  # Default to dog
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
            "conditions": conditions,
            "conditions_text": self._format_list_for_prompt(conditions, "No specific conditions"),
            "allergies": allergies,
            "allergies_text": self._format_list_for_prompt(allergies, "No known allergies"),
            "considerations": considerations
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
        return """You are a veterinary nutritionist specializing in preventive care. Your role is to provide practical, evidence-based nutrition guidance that helps maintain optimal health and prevents nutrition-related issues.

For each case, you must provide a JSON response with exactly these four fields:

1. "nutrition_overview": A brief overview (1-2 sentences) of the nutritional approach for this pet, considering their species, age, and health status.

2. "recommended_diet": Specific diet recommendations including type (commercial, homemade, mixed), quality considerations, and any condition-specific adjustments.

3. "portion_guidance": Clear guidance on portion sizes, how to determine appropriate amounts, and any feeding schedule considerations.

4. "healthy_treats": A list of 3-5 healthy treat options appropriate for this pet, with frequency and quantity guidance.

Your response should be concise, practical, and tailored to the pet's specific needs. Focus on prevention rather than treatment."""
    
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
        return f"""Provide preventive nutrition guidance for this pet:

=== PET PROFILE ===
Species: {pet_info['species']}
Breed: {pet_info['breed']}
Age: {pet_info['age']} years ({pet_info['life_stage']})
Weight Status: {pet_info['weight_status']}
Current Diet: {pet_info['diet_type']} ({pet_info['diet_quality']})

=== HEALTH CONSIDERATIONS ===
Conditions: {pet_info['conditions_text']}
Allergies: {pet_info['allergies_text']}

=== RISK CONTEXT ===
Health Risk Score: {risk_info['risk_score_percent']:.1f}%

Based on this information, generate a concise preventive nutrition guide with the four required fields."""
    
    def _validate_and_format_guide(
        self,
        guide: Dict[str, Any],
        pet_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate and format the nutrition guide to ensure correct structure.
        
        Args:
            guide: Raw guide dictionary
            pet_info: Pet information for context
            
        Returns:
            Formatted guide with proper types
        """
        formatted = {}
        species = pet_info['species'].lower()
        life_stage = pet_info['life_stage']
        
        # Format nutrition_overview
        if "nutrition_overview" in guide:
            formatted["nutrition_overview"] = str(guide["nutrition_overview"])
        else:
            formatted["nutrition_overview"] = (
                f"A balanced, {life_stage}-appropriate diet is essential for this {species}'s "
                f"preventive health. Focus on high-quality ingredients and appropriate portions."
            )
        
        # Format recommended_diet
        if "recommended_diet" in guide:
            formatted["recommended_diet"] = str(guide["recommended_diet"])
        else:
            considerations = pet_info['considerations']
            formatted["recommended_diet"] = (
                f"High-quality commercial {life_stage} formula from reputable brands. "
                f"{considerations.get('primary', 'Complete and balanced diet.')} "
                f"Look for AAFCO statement of nutritional adequacy."
            )
        
        # Format portion_guidance
        if "portion_guidance" in guide:
            formatted["portion_guidance"] = str(guide["portion_guidance"])
        else:
            formatted["portion_guidance"] = (
                f"Follow package guidelines based on ideal weight, not current weight. "
                f"Adjust based on body condition score. Divide into 2 meals daily. "
                f"Measure portions accurately using a kitchen scale for precision."
            )
        
        # Format healthy_treats (should be list)
        if "healthy_treats" in guide:
            treats = guide["healthy_treats"]
            if isinstance(treats, list):
                formatted["healthy_treats"] = [str(t) for t in treats]
            elif isinstance(treats, str):
                # Split by lines or commas
                if '\n' in treats:
                    formatted["healthy_treats"] = [t.strip() for t in treats.split('\n') if t.strip()]
                elif ',' in treats:
                    formatted["healthy_treats"] = [t.strip() for t in treats.split(',')]
                else:
                    formatted["healthy_treats"] = [treats]
            else:
                formatted["healthy_treats"] = [str(treats)] if treats else []
        else:
            # Default treats based on species
            if species == 'dog':
                formatted["healthy_treats"] = [
                    "Small pieces of cooked lean meat (chicken, turkey)",
                    "Baby carrots or green beans",
                    "Plain, air-popped popcorn (no salt/butter)",
                    "Blueberries or apple slices (no seeds)",
                    "Commercial low-calorie training treats"
                ]
            elif species == 'cat':
                formatted["healthy_treats"] = [
                    "Freeze-dried meat treats",
                    "Small pieces of cooked chicken or fish",
                    "Commercial catnip treats",
                    "Small amount of plain pumpkin puree"
                ]
            else:
                formatted["healthy_treats"] = [
                    "Small pieces of favorite vegetables",
                    "Commercial treats for species",
                    "Occasional fruit bits (in moderation)"
                ]
        
        # Limit to 5 treats
        formatted["healthy_treats"] = formatted["healthy_treats"][:5]
        
        return formatted
    
    def _get_fallback_guide(self, pet_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate fallback nutrition guide when LLM fails.
        
        Args:
            pet_info: Pet information
            
        Returns:
            Dictionary with fallback nutrition guide
        """
        species = pet_info['species'].lower()
        life_stage = pet_info['life_stage']
        
        return {
            "nutrition_overview": (
                f"A balanced, species-appropriate diet is the foundation of preventive health "
                f"for this {pet_info['age']}-year-old {species}. Focus on quality ingredients "
                f"and appropriate portion control."
            ),
            "recommended_diet": (
                f"High-quality commercial {life_stage} diet from a reputable brand "
                f"that meets AAFCO standards. Choose formulas appropriate for "
                f"{'weight management' if pet_info['weight_status'] in ['overweight', 'obese'] else 'maintenance'}."
            ),
            "portion_guidance": (
                f"Feed according to package guidelines based on IDEAL body weight. "
                f"Divide daily portion into 2 meals. Adjust by 10% based on body condition. "
                f"Use a measuring cup or kitchen scale for accuracy."
            ),
            "healthy_treats": [
                "Small pieces of cooked lean meat",
                "Fresh vegetables (carrots, green beans)",
                "Commercial low-calorie training treats",
                "Small fruit pieces (blueberries, apple)",
                "Occasional dental chews"
            ][:4]
        }
    
    def get_guide_summary(self, guide: Dict[str, Any]) -> str:
        """
        Get a brief summary of the nutrition guide for display.
        
        Args:
            guide: Nutrition guide dictionary
            
        Returns:
            Brief summary string
        """
        if not guide:
            return "No nutrition guide available."
        
        overview = guide.get('nutrition_overview', '')
        diet = guide.get('recommended_diet', '')
        
        return f"{overview}\n\nDiet: {diet[:100]}..."
    
    def get_treats_list(self, guide: Dict[str, Any]) -> List[str]:
        """
        Extract healthy treats list for quick reference.
        
        Args:
            guide: Nutrition guide dictionary
            
        Returns:
            List of treat options
        """
        treats = guide.get('healthy_treats', [])
        if isinstance(treats, list):
            return treats
        return []


# ==========================================
# CONVENIENCE FUNCTION
# ==========================================

def generate_preventive_nutrition_guide(
    client: OpenAIClient,
    profile: Dict[str, Any],
    ml_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convenience function to generate preventive nutrition guide.
    
    Args:
        client: OpenAIClient instance
        profile: Extracted pet profile
        ml_results: ML prediction results
        
    Returns:
        Dictionary with nutrition guide and status
    """
    agent = NutritionPreventiveAgent(client)
    return agent.generate_nutrition_guide(profile, ml_results)


# ==========================================
# MODULE TESTING
# ==========================================

if __name__ == "__main__":
    import json
    from utils.openai_client import build_openai_client
    
    print("=" * 60)
    print("NUTRITION PREVENTIVE AGENT TEST")
    print("=" * 60)
    
    try:
        # Initialize client
        client = build_openai_client()
        print("✅ OpenAI client created")
        
        # Initialize agent
        agent = NutritionPreventiveAgent(client)
        print(f"✅ Agent initialized: {agent.agent_name}")
        
        # Test with sample data
        test_profile = {
            "pet_name": "Luna",
            "pet_species": "cat",
            "breed": "domestic shorthair",
            "age_years": 4,
            "weight_status": "overweight",
            "diet_type": "kibble",
            "diet_quality": "average",
            "known_conditions": [],
            "allergies_known": []
        }
        
        test_ml_results = {
            "health_risk_score": 0.42,
            "care_capability_score": 75.0
        }
        
        print("\n📤 Generating preventive nutrition guide...")
        result = agent.generate_nutrition_guide(test_profile, test_ml_results)
        
        print(f"\n📊 Nutrition Guide Result:")
        print(f"Status: {result['status']}")
        
        if result['status'] == 'success':
            guide = result['nutrition_preventive']
            
            print("\n🥗 PREVENTIVE NUTRITION GUIDE")
            print("=" * 40)
            
            print(f"\n📋 Overview:")
            print(f"  {guide.get('nutrition_overview', 'N/A')}")
            
            print(f"\n🍖 Recommended Diet:")
            print(f"  {guide.get('recommended_diet', 'N/A')}")
            
            print(f"\n⚖️ Portion Guidance:")
            print(f"  {guide.get('portion_guidance', 'N/A')}")
            
            print(f"\n🍪 Healthy Treats:")
            for treat in guide.get('healthy_treats', []):
                print(f"  • {treat}")
            
            # Test utility methods
            print(f"\n📝 Guide Summary:")
            print(f"  {agent.get_guide_summary(guide)}")
        
        print("\n✅ Nutrition Preventive Agent ready for use")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")