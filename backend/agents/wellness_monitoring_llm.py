# agents/wellness_monitoring_llm.py
"""
Wellness Monitoring LLM Agent for PawCare+ (Critical Path).
Generates comprehensive wellness monitoring plans for high-risk pets.
"""

from typing import Dict, Any, List, Optional
import logging

from agents.base_llm_agent import BaseLLMAgent
from utils.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


class WellnessMonitoringAgent(BaseLLMAgent):
    """
    LLM agent for generating wellness monitoring plans for high-risk pets.
    
    This agent is part of the CRITICAL PATH and provides comprehensive monitoring including:
    - Daily monitoring checklists
    - Weekly assessment tasks
    - Monthly comprehensive checks
    - Veterinary visit scheduling
    - Health tracking methods
    - Red flag identification
    - Progress milestones
    - Long-term monitoring timeline
    """
    
    # Required fields in the output JSON
    REQUIRED_FIELDS = [
        "monitoring_overview",
        "daily_monitoring",
        "weekly_assessment",
        "monthly_detailed_check",
        "vet_visit_schedule",
        "health_tracking_method",
        "red_flags",
        "progress_milestones",
        "long_term_timeline"
    ]
    
    # Species-specific normal ranges (for reference)
    SPECIES_NORMAL_RANGES = {
        "dog": {
            "temperature": "38.3-39.2°C",
            "heart_rate": "60-140 bpm (varies by size)",
            "respiratory_rate": "10-30 breaths/min"
        },
        "cat": {
            "temperature": "38.1-39.2°C",
            "heart_rate": "140-220 bpm",
            "respiratory_rate": "20-30 breaths/min"
        },
        "rabbit": {
            "temperature": "38.3-39.4°C",
            "heart_rate": "120-150 bpm",
            "respiratory_rate": "30-60 breaths/min"
        }
    }
    
    def __init__(self, client: OpenAIClient):
        """
        Initialize the wellness monitoring agent.
        
        Args:
            client: OpenAIClient instance for making API calls
        """
        super().__init__(
            client=client,
            agent_name="WellnessMonitoring",
            default_temperature=0.5,  # Balanced for monitoring plans
            default_max_tokens=1000    # As specified
        )
        logger.info("WellnessMonitoringAgent initialized")
    def generate(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Implementation of abstract generate method from BaseLLMAgent.
        Delegates to the specific generation method.
        """
        # Call the specific method based on what the agent does
        return self.generate_monitoring_plan(*args, **kwargs)
    def generate_monitoring_plan(
        self,
        profile: Dict[str, Any],
        ml_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive wellness monitoring plan for high-risk pets.
        
        Args:
            profile: Extracted pet profile containing:
                - pet_species: str
                - breed: str
                - age_years: int
                - known_conditions: List[str]
                - medications_current: List[str]
                - recent_symptoms: List[str]
                - weight_status: str
                
            ml_results: ML prediction scores containing:
                - health_risk_score: float (0-1)
                - care_capability_score: float (0-100)
                - health_risk_factors: Optional[Dict]
        
        Returns:
            Dictionary containing:
                - wellness_monitoring: Dictionary with 9 required fields
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
            
            logger.info(f"Generating wellness monitoring plan for {pet_info['species']} {pet_info['breed']} "
                       f"with risk score {risk_info['risk_score']:.2f}")
            
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
                error_msg = result.get("_error", "Unknown error in monitoring planning")
                logger.error(f"Monitoring plan generation failed: {error_msg}")
                
                return {
                    "wellness_monitoring": self._get_fallback_plan(pet_info, risk_info),
                    "status": "error",
                    "message": f"Planning failed: {error_msg}"
                }
            
            # Remove internal metadata
            plan = {k: v for k, v in result.items() if not k.startswith('_')}
            
            # Ensure all required fields are present with correct types
            plan = self._validate_and_format_plan(plan, pet_info, risk_info)
            
            logger.info("Wellness monitoring plan generated successfully")
            
            return {
                "wellness_monitoring": plan,
                "status": "success",
                "message": "Monitoring plan generated successfully"
            }
            
        except Exception as e:
            logger.error(f"Unexpected error in monitoring planning: {str(e)}")
            
            # Provide fallback plan
            pet_info = self._extract_pet_info(profile)
            risk_info = self._extract_risk_info(ml_results)
            
            return {
                "wellness_monitoring": self._get_fallback_plan(pet_info, risk_info),
                "status": "error",
                "message": f"Planning failed: {str(e)}"
            }
    
    def _extract_pet_info(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant pet information for monitoring planning.
        
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
        
        # Medical info
        conditions = profile.get('known_conditions', [])
        if not isinstance(conditions, list):
            conditions = []
        
        medications = profile.get('medications_current', [])
        if not isinstance(medications, list):
            medications = []
        
        symptoms = profile.get('recent_symptoms', [])
        if not isinstance(symptoms, list):
            symptoms = []
        
        # Weight
        weight_status = profile.get('weight_status', 'unknown')
        
        # Get species-specific normal ranges
        normal_ranges = self.SPECIES_NORMAL_RANGES.get(species.lower(), {
            "temperature": "Check with your veterinarian",
            "heart_rate": "Check with your veterinarian",
            "respiratory_rate": "Check with your veterinarian"
        })
        
        return {
            "name": name,
            "species": species,
            "breed": breed,
            "age": age,
            "age_category": age_category,
            "conditions": conditions,
            "conditions_text": self._format_list_for_prompt(conditions, "No specific conditions"),
            "medications": medications,
            "medications_text": self._format_list_for_prompt(medications, "No current medications"),
            "symptoms": symptoms,
            "symptoms_text": self._format_list_for_prompt(symptoms, "No recent symptoms"),
            "weight_status": weight_status,
            "normal_ranges": normal_ranges
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
        
        # Determine monitoring intensity based on risk
        if risk_score > 0.8:
            intensity = "INTENSIVE"
            frequency = "multiple times daily"
        elif risk_score > 0.6:
            intensity = "ELEVATED"
            frequency = "daily"
        else:
            intensity = "STANDARD"
            frequency = "daily with weekly assessments"
        
        # Get risk factors if available
        risk_factors = ml_results.get('health_risk_factors', {})
        top_factors = []
        if risk_factors and isinstance(risk_factors, dict):
            sorted_factors = sorted(risk_factors.items(), key=lambda x: abs(x[1]), reverse=True)
            top_factors = [factor for factor, _ in sorted_factors[:3]]
        
        return {
            "risk_score": risk_score,
            "risk_score_percent": risk_score * 100,
            "care_score": care_score,
            "monitoring_intensity": intensity,
            "monitoring_frequency": frequency,
            "risk_factors": top_factors,
            "risk_factors_text": ", ".join(top_factors) if top_factors else "general health status"
        }
    
    def _build_system_prompt(self) -> str:
        """
        Build the system prompt defining the agent's role and output format.
        
        Returns:
            System prompt string
        """
        return """You are a veterinary wellness specialist with expertise in monitoring high-risk pets. Your role is to create comprehensive, practical monitoring plans that help owners track their pet's health and detect problems early.

You understand that for high-risk pets, consistent monitoring can be life-saving. Your plans are detailed, actionable, and tailored to each pet's specific conditions and risk factors.

For each case, you must provide a JSON response with exactly these nine fields:

1. "monitoring_overview": A brief overview (2-3 sentences) of the monitoring strategy and why it's important for this pet.

2. "daily_monitoring": A checklist of specific items to monitor daily, including appetite, behavior, symptoms, and vital signs where appropriate. Format as a list of actionable items.

3. "weekly_assessment": A list of weekly assessment tasks such as weight checks, symptom review, and medication adherence verification.

4. "monthly_detailed_check": A comprehensive monthly checklist including body condition scoring, dental checks, coat/skin assessment, and any condition-specific evaluations.

5. "vet_visit_schedule": A recommended veterinary visit schedule including routine checkups, medication rechecks, and condition-specific monitoring appointments.

6. "health_tracking_method": Specific recommendations for tracking health metrics, including tools (apps, journals), what to record, and how to spot trends.

7. "red_flags": A list of warning signs that require immediate veterinary attention, prioritized by urgency.

8. "progress_milestones": Specific, measurable milestones to track improvement over time, with realistic timeframes.

9. "long_term_timeline": A long-term monitoring timeline (3-12 months) showing expected progression and adjustment points.

Your response must be practical, measurable, and tailored to the pet's specific health status and risk level."""
    
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
        return f"""Create a comprehensive wellness monitoring plan for this high-risk pet:

=== PET PROFILE ===
Name: {pet_info['name']}
Species: {pet_info['species']}
Breed: {pet_info['breed']}
Age: {pet_info['age']} years ({pet_info['age_category']})
Weight Status: {pet_info['weight_status']}

=== MEDICAL INFORMATION ===
Known Conditions:
{pet_info['conditions_text']}

Current Medications:
{pet_info['medications_text']}

Recent Symptoms:
{pet_info['symptoms_text']}

=== RISK ASSESSMENT ===
Health Risk Score: {risk_info['risk_score_percent']:.1f}%
Risk Level: {risk_info['monitoring_intensity']}
Key Risk Factors: {risk_info['risk_factors_text']}
Owner Care Capability: {risk_info['care_score']:.1f}/100

=== NORMAL RANGES (Reference) ===
Temperature: {pet_info['normal_ranges'].get('temperature', 'N/A')}
Heart Rate: {pet_info['normal_ranges'].get('heart_rate', 'N/A')}
Respiratory Rate: {pet_info['normal_ranges'].get('respiratory_rate', 'N/A')}

Based on this information, generate a comprehensive wellness monitoring plan with all nine required fields. Focus on practical, actionable monitoring tasks appropriate for the pet's risk level."""
    
    def _validate_and_format_plan(
        self,
        plan: Dict[str, Any],
        pet_info: Dict[str, Any],
        risk_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate and format the monitoring plan to ensure correct structure.
        
        Args:
            plan: Raw plan dictionary
            pet_info: Pet information for context
            risk_info: Risk information for context
            
        Returns:
            Formatted plan with proper types
        """
        formatted = {}
        species = pet_info['species'].lower()
        intensity = risk_info['monitoring_intensity'].lower()
        
        # Format monitoring_overview
        if "monitoring_overview" in plan:
            formatted["monitoring_overview"] = str(plan["monitoring_overview"])
        else:
            formatted["monitoring_overview"] = (
                f"This {intensity} monitoring plan is designed for {pet_info['name']}, a "
                f"{pet_info['age_category']} {species} with {pet_info['conditions_text']}. "
                f"Consistent monitoring will help detect changes early and guide treatment decisions."
            )
        
        # Format daily_monitoring (should be list)
        if "daily_monitoring" in plan:
            daily = plan["daily_monitoring"]
            if isinstance(daily, list):
                formatted["daily_monitoring"] = [str(d) for d in daily]
            elif isinstance(daily, str):
                # Split by lines or bullets
                formatted["daily_monitoring"] = [line.strip() for line in daily.split('\n') if line.strip()]
            else:
                formatted["daily_monitoring"] = [str(daily)] if daily else []
        else:
            formatted["daily_monitoring"] = [
                "☐ Appetite: Record amount eaten and any changes",
                "☐ Water intake: Note increased/decreased drinking",
                "☐ Energy level: Compare to baseline (1-5 scale)",
                "☐ Symptoms: Check for coughing, vomiting, diarrhea",
                "☐ Medications: Verify all doses given",
                "☐ Behavior: Note any unusual behaviors or signs of pain"
            ]
        
        # Format weekly_assessment (should be list)
        if "weekly_assessment" in plan:
            weekly = plan["weekly_assessment"]
            if isinstance(weekly, list):
                formatted["weekly_assessment"] = [str(w) for w in weekly]
            elif isinstance(weekly, str):
                formatted["weekly_assessment"] = [line.strip() for line in weekly.split('\n') if line.strip()]
            else:
                formatted["weekly_assessment"] = [str(weekly)] if weekly else []
        else:
            formatted["weekly_assessment"] = [
                "⚖️ Weight check: Record weight, track trends",
                "📋 Symptom review: Note any recurring patterns",
                "💊 Medication review: Verify refills needed",
                "📝 Journal review: Identify any concerns",
                "📸 Photo documentation: Take weekly photos for comparison"
            ]
        
        # Format monthly_detailed_check (should be list)
        if "monthly_detailed_check" in plan:
            monthly = plan["monthly_detailed_check"]
            if isinstance(monthly, list):
                formatted["monthly_detailed_check"] = [str(m) for m in monthly]
            elif isinstance(monthly, str):
                formatted["monthly_detailed_check"] = [line.strip() for line in monthly.split('\n') if line.strip()]
            else:
                formatted["monthly_detailed_check"] = [str(monthly)] if monthly else []
        else:
            formatted["monthly_detailed_check"] = [
                "🦷 Dental check: Examine teeth and gums",
                "🫀 Body condition scoring: Assess weight and muscle mass",
                "🪮 Coat and skin evaluation: Check for abnormalities",
                f"🔬 Condition-specific assessments for: {pet_info['conditions_text']}",
                "📊 Review all tracking data for trends"
            ]
        
        # Format vet_visit_schedule
        if "vet_visit_schedule" in plan:
            formatted["vet_visit_schedule"] = str(plan["vet_visit_schedule"])
        else:
            formatted["vet_visit_schedule"] = (
                f"• Routine checkup: Every 3-6 months\n"
                f"• Medication rechecks: As prescribed\n"
                f"• Condition monitoring: Based on {pet_info['conditions_text']}\n"
                f"• Immediate visit: Any red flags appear"
            )
        
        # Format health_tracking_method
        if "health_tracking_method" in plan:
            formatted["health_tracking_method"] = str(plan["health_tracking_method"])
        else:
            formatted["health_tracking_method"] = (
                "Use a dedicated health journal or app to track:\n"
                "• Daily observations (appetite, behavior, symptoms)\n"
                "• Weekly weight and trends\n"
                "• Medication administration\n"
                "• Photos for visual comparison\n"
                "Share this log with your veterinarian at each visit."
            )
        
        # Format red_flags (should be list)
        if "red_flags" in plan:
            red_flags = plan["red_flags"]
            if isinstance(red_flags, list):
                formatted["red_flags"] = [str(r) for r in red_flags]
            elif isinstance(red_flags, str):
                formatted["red_flags"] = [line.strip() for line in red_flags.split('\n') if line.strip()]
            else:
                formatted["red_flags"] = [str(red_flags)] if red_flags else []
        else:
            formatted["red_flags"] = [
                "🚨 Difficulty breathing or severe coughing",
                "🚨 Collapse or inability to stand",
                "🚨 Seizures",
                "🚨 Severe pain (vocalizing, restlessness, aggression)",
                "🚨 Vomiting >2 times in 24 hours or bloody vomit",
                "🚨 No urination for >12 hours",
                "🚨 Sudden worsening of any monitored symptom"
            ]
        
        # Format progress_milestones
        if "progress_milestones" in plan:
            formatted["progress_milestones"] = str(plan["progress_milestones"])
        else:
            formatted["progress_milestones"] = (
                f"Short-term (1 month): {pet_info['conditions_text']} symptoms stabilize\n"
                f"Medium-term (3 months): Improved quality of life indicators\n"
                f"Long-term (6 months): Achievement of treatment goals\n"
                f"Review and adjust milestones with your veterinarian."
            )
        
        # Format long_term_timeline
        if "long_term_timeline" in plan:
            formatted["long_term_timeline"] = str(plan["long_term_timeline"])
        else:
            formatted["long_term_timeline"] = (
                f"Months 1-3: Intensive monitoring ({risk_info['monitoring_frequency']})\n"
                f"Months 4-6: Gradual extension if stable\n"
                f"Months 7-12: Maintenance monitoring with condition-specific checks\n"
                f"Adjust timeline based on response to treatment."
            )
        
        return formatted
    
    def _get_fallback_plan(
        self,
        pet_info: Dict[str, Any],
        risk_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate fallback monitoring plan when LLM fails.
        
        Args:
            pet_info: Pet information
            risk_info: Risk information
            
        Returns:
            Dictionary with fallback monitoring plan
        """
        species = pet_info['species'].lower()
        
        return {
            "monitoring_overview": (
                f"This monitoring plan is designed for {pet_info['name']}, a {pet_info['age_category']} "
                f"{species} with {pet_info['conditions_text']}. With a health risk score of "
                f"{risk_info['risk_score_percent']:.1f}%, {risk_info['monitoring_frequency']} "
                f"monitoring is recommended to track progress and detect changes early."
            ),
            "daily_monitoring": [
                "☐ Appetite: Note amount eaten and any changes",
                "☐ Water intake: Track increased/decreased drinking",
                "☐ Energy/activity level: Compare to normal",
                "☐ Specific symptoms related to conditions",
                "☐ Medication administration verification",
                "☐ Pain assessment (if applicable)"
            ],
            "weekly_assessment": [
                "⚖️ Weight: Record and track trends",
                "📋 Symptom review: Identify patterns",
                "💊 Medication inventory: Check supplies",
                "📝 Journal review: Note concerns",
                "📸 Progress photos: Take weekly photos"
            ],
            "monthly_detailed_check": [
                "🦷 Dental health assessment",
                "🫀 Body condition scoring",
                "🪮 Coat and skin condition",
                f"🔍 {pet_info['conditions_text']} specific checks",
                "📊 Data review for trends"
            ],
            "vet_visit_schedule": (
                f"• Regular checkups: Every 3-6 months\n"
                f"• Condition monitoring: As recommended for {pet_info['conditions_text']}\n"
                f"• Medication reviews: With your veterinarian\n"
                f"• Emergency: Any red flags appear"
            ),
            "health_tracking_method": (
                "Maintain a daily health log (paper or app) recording:\n"
                "• Daily observations and metrics\n"
                "• Weekly weight measurements\n"
                "• Medication administration\n"
                "• Photos for visual comparison\n"
                "Share log with veterinarian at each visit."
            ),
            "red_flags": [
                "🚨 Difficulty breathing",
                "🚨 Collapse or inability to stand",
                "🚨 Seizures",
                "🚨 Severe pain",
                "🚨 Persistent vomiting/diarrhea",
                "🚨 No urination >12 hours",
                "🚨 Sudden worsening of any symptom"
            ],
            "progress_milestones": (
                f"Short-term (1 month): Stabilization of {pet_info['conditions_text']}\n"
                f"Medium-term (3 months): Improved daily function\n"
                f"Long-term (6 months): Achievement of treatment goals\n"
                f"Review and adjust with veterinarian."
            ),
            "long_term_timeline": (
                f"Months 1-3: {risk_info['monitoring_frequency']} monitoring\n"
                f"Months 4-6: Gradual reduction if stable\n"
                f"Months 7-12: Maintenance monitoring\n"
                f"Timeline adjusts based on response."
            )
        }
    
    def get_plan_summary(self, plan: Dict[str, Any]) -> str:
        """
        Get a brief summary of the monitoring plan for display.
        
        Args:
            plan: Monitoring plan dictionary
            
        Returns:
            Brief summary string
        """
        if not plan:
            return "No monitoring plan available."
        
        overview = plan.get('monitoring_overview', '')
        daily = plan.get('daily_monitoring', [])
        
        summary = f"{overview}\n\nDaily Tasks:\n"
        
        if daily:
            for task in daily[:3]:  # Show first 3 daily tasks
                summary += f"  {task}\n"
        
        return summary
    
    def get_daily_checklist(self, plan: Dict[str, Any]) -> List[str]:
        """
        Extract daily monitoring checklist.
        
        Args:
            plan: Monitoring plan dictionary
            
        Returns:
            List of daily monitoring tasks
        """
        daily = plan.get('daily_monitoring', [])
        if isinstance(daily, list):
            return daily
        return []
    
    def get_red_flags_list(self, plan: Dict[str, Any]) -> List[str]:
        """
        Extract red flags list for quick reference.
        
        Args:
            plan: Monitoring plan dictionary
            
        Returns:
            List of red flag warnings
        """
        red_flags = plan.get('red_flags', [])
        if isinstance(red_flags, list):
            return red_flags
        return []
    
    def get_monitoring_frequency_label(self, risk_score: float) -> str:
        """
        Get human-readable monitoring frequency label.
        
        Args:
            risk_score: Health risk score (0-1)
            
        Returns:
            Frequency label
        """
        if risk_score > 0.8:
            return "Intensive (multiple times daily)"
        elif risk_score > 0.6:
            return "Elevated (daily)"
        else:
            return "Standard (daily with weekly checks)"


# ==========================================
# CONVENIENCE FUNCTION
# ==========================================

def generate_monitoring_plan(
    client: OpenAIClient,
    profile: Dict[str, Any],
    ml_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convenience function to generate wellness monitoring plan.
    
    Args:
        client: OpenAIClient instance
        profile: Extracted pet profile
        ml_results: ML prediction results
        
    Returns:
        Dictionary with monitoring plan and status
    """
    agent = WellnessMonitoringAgent(client)
    return agent.generate_monitoring_plan(profile, ml_results)


# ==========================================
# MODULE TESTING
# ==========================================

if __name__ == "__main__":
    import json
    from utils.openai_client import build_openai_client
    
    print("=" * 60)
    print("WELLNESS MONITORING AGENT TEST")
    print("=" * 60)
    
    try:
        # Initialize client
        client = build_openai_client()
        print("✅ OpenAI client created")
        
        # Initialize agent
        agent = WellnessMonitoringAgent(client)
        print(f"✅ Agent initialized: {agent.agent_name}")
        
        # Test with sample data
        test_profile = {
            "pet_name": "Max",
            "pet_species": "dog",
            "breed": "labrador retriever",
            "age_years": 12,
            "weight_status": "overweight",
            "known_conditions": ["diabetes mellitus", "arthritis", "heart murmur"],
            "medications_current": ["insulin", "carprofen"],
            "recent_symptoms": ["increased thirst", "lethargy", "occasional cough"]
        }
        
        test_ml_results = {
            "health_risk_score": 0.85,
            "care_capability_score": 75.0,
            "health_risk_factors": {
                "age": 0.4,
                "conditions_count": 0.3,
                "symptoms": 0.2
            }
        }
        
        print("\n📤 Generating wellness monitoring plan...")
        result = agent.generate_monitoring_plan(test_profile, test_ml_results)
        
        print(f"\n📊 Monitoring Plan Result:")
        print(f"Status: {result['status']}")
        
        if result['status'] == 'success':
            plan = result['wellness_monitoring']
            
            print("\n📋 WELLNESS MONITORING PLAN")
            print("=" * 40)
            
            print(f"\n📝 Overview:")
            print(f"  {plan.get('monitoring_overview', 'N/A')}")
            
            print(f"\n📅 Daily Monitoring:")
            for task in plan.get('daily_monitoring', []):
                print(f"  {task}")
            
            print(f"\n📊 Weekly Assessment:")
            for task in plan.get('weekly_assessment', []):
                print(f"  {task}")
            
            print(f"\n📆 Monthly Detailed Check:")
            for task in plan.get('monthly_detailed_check', []):
                print(f"  {task}")
            
            print(f"\n🏥 Vet Visit Schedule:")
            print(f"  {plan.get('vet_visit_schedule', 'N/A')}")
            
            print(f"\n📱 Health Tracking Method:")
            print(f"  {plan.get('health_tracking_method', 'N/A')}")
            
            print(f"\n🚨 Red Flags:")
            for flag in plan.get('red_flags', []):
                print(f"  {flag}")
            
            print(f"\n🎯 Progress Milestones:")
            print(f"  {plan.get('progress_milestones', 'N/A')}")
            
            print(f"\n⏱️ Long-term Timeline:")
            print(f"  {plan.get('long_term_timeline', 'N/A')}")
            
            # Test utility methods
            print(f"\n📝 Plan Summary:")
            print(f"  {agent.get_plan_summary(plan)}")
            
            print(f"\n✅ Daily Checklist Sample:")
            for task in agent.get_daily_checklist(plan)[:3]:
                print(f"  {task}")
        
        print("\n✅ Wellness Monitoring Agent ready for use")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")