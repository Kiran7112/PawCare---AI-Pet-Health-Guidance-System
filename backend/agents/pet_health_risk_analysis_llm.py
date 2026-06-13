# agents/pet_health_risk_analysis_llm.py
"""
Pet Health Risk Analysis LLM Agent for PawCare+ (Critical Path).
Generates comprehensive health risk analysis for high-risk pets.
"""

from typing import Dict, Any, List, Optional
import logging

from agents.base_llm_agent import BaseLLMAgent
from utils.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


class PetHealthRiskAnalysisAgent(BaseLLMAgent):
    """
    LLM agent for generating comprehensive health risk analysis for high-risk pets.
    
    This agent is part of the CRITICAL PATH and provides detailed analysis including:
    - Honest risk assessment
    - Critical risk factors
    - Success probability
    - Warning signs to monitor
    - Urgency timeline for interventions
    """
    
    # Required fields in the output JSON
    REQUIRED_FIELDS = [
        "honest_risk_assessment",
        "critical_risk_factors",
        "success_probability",
        "warning_signs",
        "urgency_timeline"
    ]
    
    def __init__(self, client: OpenAIClient):
        """
        Initialize the pet health risk analysis agent.
        
        Args:
            client: OpenAIClient instance for making API calls
        """
        super().__init__(
            client=client,
            agent_name="PetHealthRiskAnalysis",
            default_temperature=0.5,  # Balanced for analysis
            default_max_tokens=1000    # As specified
        )
        logger.info("PetHealthRiskAnalysisAgent initialized")
    def generate(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Implementation of abstract generate method from BaseLLMAgent.
        Delegates to the specific generation method.
        """
        # Call the specific method based on what the agent does
        return self.generate_risk_analysis(*args, **kwargs)
    def generate_risk_analysis(
        self,
        profile: Dict[str, Any],
        ml_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive health risk analysis for high-risk pets.
        
        Args:
            profile: Extracted pet profile dictionary containing:
                - pet_species: str
                - breed: str
                - age_years: int
                - known_conditions: List[str]
                - weight_status: str
                - recent_symptoms: List[str] (if available)
                - symptom_duration_days: int (if available)
                - medications_current: List[str]
                
            ml_results: Dictionary containing ML predictions:
                - health_risk_score: float (0-1)
                - care_capability_score: float (0-100)
                - health_risk_factors: Optional[Dict] feature importance
        
        Returns:
            Dictionary containing:
                - health_risk_analysis: Dictionary with 5 required fields
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
            
            logger.info(f"Generating risk analysis for {pet_info['species']} {pet_info['breed']}, "
                       f"age {pet_info['age']}, risk score {risk_info['risk_score']:.2f}")
            
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
                error_msg = result.get("_error", "Unknown error in risk analysis")
                logger.error(f"Risk analysis generation failed: {error_msg}")
                
                return {
                    "health_risk_analysis": self._get_fallback_analysis(pet_info, risk_info),
                    "status": "error",
                    "message": f"Analysis failed: {error_msg}"
                }
            
            # Remove internal metadata
            analysis = {k: v for k, v in result.items() if not k.startswith('_')}
            
            # Ensure all required fields are present with correct types
            analysis = self._validate_and_format_analysis(analysis)
            
            logger.info("Risk analysis generated successfully")
            
            return {
                "health_risk_analysis": analysis,
                "status": "success",
                "message": "Risk analysis generated successfully"
            }
            
        except Exception as e:
            logger.error(f"Unexpected error in risk analysis: {str(e)}")
            
            # Provide fallback analysis
            pet_info = self._extract_pet_info(profile)
            risk_info = self._extract_risk_info(ml_results)
            
            return {
                "health_risk_analysis": self._get_fallback_analysis(pet_info, risk_info),
                "status": "error",
                "message": f"Analysis failed: {str(e)}"
            }
    
    def _extract_pet_info(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant pet information from profile.
        
        Args:
            profile: Raw extracted profile
            
        Returns:
            Dictionary with formatted pet information
        """
        # Get species with proper formatting
        species = profile.get('pet_species', 'unknown')
        if isinstance(species, str):
            species = species.capitalize()
        
        # Get breed
        breed = profile.get('breed', 'unknown')
        if isinstance(breed, str):
            breed = breed.capitalize()
        
        # Get age
        age = profile.get('age_years', 0)
        if isinstance(age, (int, float)):
            age_text = f"{int(age)} years" if age > 0 else "unknown age"
        else:
            age_text = "unknown age"
        
        # Get conditions
        conditions = profile.get('known_conditions', [])
        if not isinstance(conditions, list):
            conditions = []
        
        # Get symptoms if available
        symptoms = profile.get('recent_symptoms', [])
        if not isinstance(symptoms, list):
            symptoms = []
        
        # Get symptom duration
        duration = profile.get('symptom_duration_days', None)
        duration_text = f"{duration} days" if duration else "recently"
        
        # Get medications
        medications = profile.get('medications_current', [])
        if not isinstance(medications, list):
            medications = []
        
        # Get weight status
        weight = profile.get('weight_status', 'unknown')
        
        return {
            "species": species,
            "breed": breed,
            "age": age,
            "age_text": age_text,
            "conditions": conditions,
            "conditions_text": self._format_list_for_prompt(conditions, "No known conditions"),
            "symptoms": symptoms,
            "symptoms_text": self._format_list_for_prompt(symptoms, "No specific symptoms reported"),
            "duration": duration,
            "duration_text": duration_text,
            "medications": medications,
            "medications_text": self._format_list_for_prompt(medications, "No current medications"),
            "weight": weight
        }
    
    def _extract_risk_info(self, ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract risk information from ML results.
        
        Args:
            ml_results: ML prediction results
            
        Returns:
            Dictionary with formatted risk information
        """
        risk_score = ml_results.get('health_risk_score', 0.5)
        care_score = ml_results.get('care_capability_score', 50.0)
        
        # Determine risk level description
        if risk_score > 0.8:
            risk_level = "CRITICAL"
            risk_description = "immediate veterinary attention required"
        elif risk_score > 0.6:
            risk_level = "HIGH"
            risk_description = "urgent veterinary consultation needed"
        else:
            risk_level = "ELEVATED"
            risk_description = "requires prompt attention"
        
        # Get feature contributions if available
        contributions = ml_results.get('feature_contributions', {})
        top_factors = []
        if contributions:
            # Sort by importance and get top 3
            sorted_factors = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
            top_factors = [f"{factor}" for factor, _ in sorted_factors[:3]]
        
        return {
            "risk_score": risk_score,
            "risk_score_percent": risk_score * 100,
            "risk_level": risk_level,
            "risk_description": risk_description,
            "care_score": care_score,
            "top_factors": top_factors,
            "factors_text": ", ".join(top_factors) if top_factors else "multiple contributing factors"
        }
    
    def _build_system_prompt(self) -> str:
        """
        Build the system prompt defining the agent's role and output format.
        
        Returns:
            System prompt string
        """
        return """You are a senior veterinary specialist with 20 years of experience in emergency and critical care. Your role is to provide honest, direct, and comprehensive health risk assessments for pets with serious health concerns.

You communicate with clarity and empathy while never sugar-coating serious conditions. Your assessments are designed to help pet owners understand the severity of the situation and take appropriate action.

For each case, you must provide a JSON response with exactly these five fields:

1. "honest_risk_assessment": A detailed paragraph (3-5 sentences) explaining the overall health situation, severity, and potential outcomes. Be direct but compassionate.

2. "critical_risk_factors": A list of exactly 4 specific factors that are most concerning in this case (e.g., ["Advanced age combined with multiple conditions", "Rapid weight loss", "Persistent symptoms despite treatment"]).

3. "success_probability": A paragraph describing the likelihood of successful management or treatment, considering the pet's condition, owner's care capability, and available interventions. Include timeframe expectations.

4. "warning_signs": A list of 3-5 specific signs that indicate worsening condition and require immediate re-evaluation or emergency care.

5. "urgency_timeline": A clear timeline specifying when interventions are needed (e.g., "within 24 hours", "emergency now", "next 3-5 days") and what specific actions are required.

Your response must be professional, evidence-based, and tailored to the specific pet's situation. Use the provided health risk score and pet profile to inform your assessment."""
    
    def _build_user_prompt(
        self,
        pet_info: Dict[str, Any],
        risk_info: Dict[str, Any]
    ) -> str:
        """
        Build the user prompt with specific case information.
        
        Args:
            pet_info: Extracted pet information
            risk_info: Extracted risk information
            
        Returns:
            User prompt string
        """
        return f"""Provide a comprehensive health risk analysis for this pet:

=== PET PROFILE ===
Species: {pet_info['species']}
Breed: {pet_info['breed']}
Age: {pet_info['age_text']}
Weight Status: {pet_info['weight']}
Known Conditions: {pet_info['conditions_text']}
Current Medications: {pet_info['medications_text']}
Recent Symptoms: {pet_info['symptoms_text']}
Symptom Onset: {pet_info['duration_text']}

=== RISK ASSESSMENT ===
Health Risk Score: {risk_info['risk_score_percent']:.1f}% (Scale: 0-100%)
Risk Level: {risk_info['risk_level']}
Key Risk Factors: {risk_info['factors_text']}
Owner Care Capability: {risk_info['care_score']:.1f}/100

Based on this information, generate a comprehensive risk analysis with the five required fields. Be honest and direct while providing actionable guidance."""
    
    def _validate_and_format_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and format the analysis to ensure correct structure.
        
        Args:
            analysis: Raw analysis dictionary
            
        Returns:
            Formatted analysis with proper types
        """
        formatted = {}
        
        # Format honest_risk_assessment (should be string)
        if "honest_risk_assessment" in analysis:
            formatted["honest_risk_assessment"] = str(analysis["honest_risk_assessment"])
        else:
            formatted["honest_risk_assessment"] = "Risk assessment not available."
        
        # Format critical_risk_factors (should be list)
        if "critical_risk_factors" in analysis:
            factors = analysis["critical_risk_factors"]
            if isinstance(factors, list):
                formatted["critical_risk_factors"] = [str(f) for f in factors[:4]]  # Take first 4
            else:
                formatted["critical_risk_factors"] = [str(factors)] if factors else []
        else:
            formatted["critical_risk_factors"] = []
        
        # Ensure we have exactly 4 factors (pad with defaults if needed)
        while len(formatted["critical_risk_factors"]) < 4:
            formatted["critical_risk_factors"].append("Further diagnostic evaluation needed")
        
        # Format success_probability (should be string)
        if "success_probability" in analysis:
            formatted["success_probability"] = str(analysis["success_probability"])
        else:
            formatted["success_probability"] = "Success probability depends on timely intervention and response to treatment."
        
        # Format warning_signs (should be list)
        if "warning_signs" in analysis:
            signs = analysis["warning_signs"]
            if isinstance(signs, list):
                formatted["warning_signs"] = [str(s) for s in signs]
            else:
                formatted["warning_signs"] = [str(signs)] if signs else []
        else:
            formatted["warning_signs"] = []
        
        # Ensure we have at least 3 signs
        while len(formatted["warning_signs"]) < 3:
            formatted["warning_signs"].append("Monitor for any sudden changes in behavior or condition")
        
        # Format urgency_timeline (should be string)
        if "urgency_timeline" in analysis:
            formatted["urgency_timeline"] = str(analysis["urgency_timeline"])
        else:
            formatted["urgency_timeline"] = "Seek veterinary attention immediately."
        
        return formatted
    
    def _get_fallback_analysis(
        self,
        pet_info: Dict[str, Any],
        risk_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate fallback analysis when LLM fails.
        
        Args:
            pet_info: Pet information
            risk_info: Risk information
            
        Returns:
            Dictionary with fallback analysis
        """
        risk_level = risk_info['risk_level']
        
        return {
            "honest_risk_assessment": (
                f"Based on the {risk_level.lower()} health risk score ({risk_info['risk_score_percent']:.1f}%), "
                f"this pet requires immediate veterinary attention. The combination of {pet_info['conditions_text']} "
                f"and reported symptoms suggests a potentially serious condition that needs professional evaluation."
            ),
            "critical_risk_factors": [
                f"Health risk score of {risk_info['risk_score_percent']:.1f}% indicates {risk_level.lower()} concern",
                f"Presence of: {pet_info['conditions_text']}",
                f"Symptoms observed: {pet_info['symptoms_text']}",
                f"Owner care capability of {risk_info['care_score']:.1f}/100"
            ][:4],
            "success_probability": (
                f"Success depends on prompt veterinary intervention. With immediate care, many conditions can be managed "
                f"effectively. The prognosis will become clearer after diagnostic testing."
            ),
            "warning_signs": [
                "Sudden worsening of symptoms",
                "Difficulty breathing",
                "Collapse or inability to stand",
                "Severe pain or distress",
                "Refusal to eat or drink for >12 hours"
            ][:3],
            "urgency_timeline": "Seek emergency veterinary care IMMEDIATELY. Do not wait."
        }
    
    def get_analysis_summary(self, analysis: Dict[str, Any]) -> str:
        """
        Get a brief summary of the analysis for display.
        
        Args:
            analysis: Analysis dictionary
            
        Returns:
            Brief summary string
        """
        if not analysis:
            return "No analysis available."
        
        assessment = analysis.get('honest_risk_assessment', '')
        timeline = analysis.get('urgency_timeline', '')
        
        # Truncate assessment for summary
        if len(assessment) > 150:
            assessment = assessment[:147] + "..."
        
        return f"{assessment}\n\nUrgency: {timeline}"
    
    def get_risk_level_color(self, risk_score: float) -> str:
        """
        Get color code for risk level display.
        
        Args:
            risk_score: Health risk score (0-1)
            
        Returns:
            Color code for UI
        """
        if risk_score > 0.8:
            return "#d32f2f"  # Dark red
        elif risk_score > 0.6:
            return "#f44336"  # Red
        elif risk_score > 0.4:
            return "#ff9800"  # Orange
        else:
            return "#ffc107"  # Yellow


# ==========================================
# CONVENIENCE FUNCTION
# ==========================================

def analyze_health_risk(
    client: OpenAIClient,
    profile: Dict[str, Any],
    ml_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convenience function to generate health risk analysis.
    
    Args:
        client: OpenAIClient instance
        profile: Extracted pet profile
        ml_results: ML prediction results
        
    Returns:
        Dictionary with analysis and status
    """
    agent = PetHealthRiskAnalysisAgent(client)
    return agent.generate_risk_analysis(profile, ml_results)


# ==========================================
# MODULE TESTING
# ==========================================

if __name__ == "__main__":
    import json
    from utils.openai_client import build_openai_client
    
    print("=" * 60)
    print("PET HEALTH RISK ANALYSIS AGENT TEST")
    print("=" * 60)
    
    try:
        # Initialize client
        client = build_openai_client()
        print("✅ OpenAI client created")
        
        # Initialize agent
        agent = PetHealthRiskAnalysisAgent(client)
        print(f"✅ Agent initialized: {agent.agent_name}")
        
        # Test with sample data
        test_profile = {
            "pet_species": "dog",
            "breed": "labrador retriever",
            "age_years": 12,
            "weight_status": "overweight",
            "known_conditions": ["arthritis", "heart murmur"],
            "recent_symptoms": ["lethargy", "increased thirst", "coughing"],
            "symptom_duration_days": 5,
            "medications_current": ["carprofen"],
            "behavioral_issues": []
        }
        
        test_ml_results = {
            "health_risk_score": 0.85,
            "care_capability_score": 75.0,
            "feature_contributions": {
                "age": 0.4,
                "conditions_count": 0.3,
                "symptoms": 0.2,
                "weight": 0.1
            }
        }
        
        print("\n📤 Generating risk analysis...")
        result = agent.generate_risk_analysis(test_profile, test_ml_results)
        
        print(f"\n📊 Analysis Result:")
        print(f"Status: {result['status']}")
        
        if result['status'] == 'success':
            analysis = result['health_risk_analysis']
            print("\n📋 Risk Analysis:")
            print(f"\n🔴 Honest Risk Assessment:")
            print(f"  {analysis.get('honest_risk_assessment', 'N/A')}")
            
            print(f"\n⚠️ Critical Risk Factors:")
            for i, factor in enumerate(analysis.get('critical_risk_factors', []), 1):
                print(f"  {i}. {factor}")
            
            print(f"\n📈 Success Probability:")
            print(f"  {analysis.get('success_probability', 'N/A')}")
            
            print(f"\n🚨 Warning Signs:")
            for sign in analysis.get('warning_signs', []):
                print(f"  • {sign}")
            
            print(f"\n⏱️ Urgency Timeline:")
            print(f"  {analysis.get('urgency_timeline', 'N/A')}")
            
            # Test summary
            print(f"\n📝 Summary:")
            print(f"  {agent.get_analysis_summary(analysis)}")
        
        print("\n✅ Pet Health Risk Analysis Agent ready for use")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")