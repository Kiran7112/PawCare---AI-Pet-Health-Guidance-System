# agents/input_validator_agent.py
"""
Input Validation Agent for PawCare+.
Validates user input text fields without requiring LLM or external dependencies.
"""

from typing import Dict, List, Any, Optional
import re


class InputValidatorAgent:
    """
    Simple validation class for pet health assessment inputs.
    
    Validates the three text input fields (about_pet, daily_routine, health_concerns)
    for basic requirements like non-emptiness and reasonable length.
    
    Class Attributes:
        MIN_LENGTH (int): Minimum characters required for each field
        MAX_LENGTH (int): Maximum characters allowed for each field
    """
    
    # Constants for validation
    MIN_LENGTH = 10
    """Minimum characters required for each text field"""
    
    MAX_LENGTH = 3000
    """Maximum characters allowed for each text field"""
    
    # Optional: Add field-specific validation rules
    FIELD_DESCRIPTIONS = {
        "about_pet": "description of your pet (age, breed, weight, etc.)",
        "daily_routine": "description of daily activities (diet, exercise, routine)",
        "health_concerns": "description of health issues or concerns"
    }
    
    def __init__(self):
        """Initialize the input validator agent."""
        pass  # No initialization needed for this simple validator
    
    def validate_inputs(self, raw_inputs: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate all three user input fields.
        
        Args:
            raw_inputs: Dictionary with keys about_pet, daily_routine, health_concerns
                       Example: {
                           "about_pet": "My dog is a 5-year-old Labrador...",
                           "daily_routine": "Walks twice daily, eats kibble...",
                           "health_concerns": "Seems lethargic lately..."
                       }
        
        Returns:
            Dictionary containing:
                - validation_errors: List of error message strings
                - is_valid: Boolean indicating if validation passed
                - status: String "success" if valid, "error" if invalid
                - validated_inputs: Cleaned inputs (whitespace stripped)
                - field_status: Individual status per field (optional)
        
        Example:
            >>> validator = InputValidatorAgent()
            >>> result = validator.validate_inputs({
            ...     "about_pet": "My dog Max",
            ...     "daily_routine": "Walks daily",
            ...     "health_concerns": ""
            ... })
            >>> result["is_valid"]
            False
            >>> result["validation_errors"]
            ['health_concerns is required']
        """
        validation_errors = []
        validated_inputs = {}
        field_status = {}
        
        # Define required fields
        required_fields = ["about_pet", "daily_routine", "health_concerns"]
        
        # Validate each field
        for field_name in required_fields:
            field_value = raw_inputs.get(field_name, "")
            
            # Clean the input: strip whitespace
            cleaned_value = field_value.strip() if field_value else ""
            validated_inputs[field_name] = cleaned_value
            
            # Individual field validation result
            field_errors = []
            
            # Check if field exists and is not empty
            if not cleaned_value:
                error_msg = f"{field_name} is required"
                validation_errors.append(error_msg)
                field_errors.append(error_msg)
            
            # Check minimum length (if field is not empty)
            elif len(cleaned_value) < self.MIN_LENGTH:
                error_msg = f"{field_name} must be at least {self.MIN_LENGTH} characters (currently {len(cleaned_value)})"
                validation_errors.append(error_msg)
                field_errors.append(error_msg)
            
            # Check maximum length
            elif len(cleaned_value) > self.MAX_LENGTH:
                error_msg = f"{field_name} exceeds maximum length of {self.MAX_LENGTH} characters"
                validation_errors.append(error_msg)
                field_errors.append(error_msg)
            
            # Additional optional validation: check for placeholder text
            if cleaned_value and self._is_placeholder_text(cleaned_value):
                error_msg = f"{field_name} appears to contain placeholder text. Please provide real information about your pet."
                validation_errors.append(error_msg)
                field_errors.append(error_msg)
            
            # Record field status
            field_status[field_name] = {
                "valid": len(field_errors) == 0,
                "errors": field_errors,
                "length": len(cleaned_value)
            }
        
        # Determine overall validity
        is_valid = len(validation_errors) == 0
        
        # Return structured result
        return {
            "validation_errors": validation_errors,
            "is_valid": is_valid,
            "status": "success" if is_valid else "error",
            "validated_inputs": validated_inputs,
            "field_status": field_status,
            "stats": {
                "total_fields": len(required_fields),
                "valid_fields": sum(1 for status in field_status.values() if status["valid"]),
                "invalid_fields": sum(1 for status in field_status.values() if not status["valid"])
            }
        }
    
    def _is_placeholder_text(self, text: str) -> bool:
        """
        Check if text appears to be placeholder/demo content.
        
        Args:
            text: Input text to check
            
        Returns:
            True if text appears to be placeholder content
        """
        placeholder_patterns = [
            r"^\s*test\s*$",
            r"^\s*asdf\s*$",
            r"^\s*placeholder\s*$",
            r"^\s*sample\s*$",
            r"^\s*demo\s*$",
            r"^\s*123\s*$",
            r"^\s*xyz\s*$"
        ]
        
        text_lower = text.lower().strip()
        
        # Check against patterns
        for pattern in placeholder_patterns:
            if re.match(pattern, text_lower):
                return True
        
        # Check for very short nonsense (less than 3 meaningful characters)
        if len(text_lower) < 5 and text_lower.isalpha() and len(set(text_lower)) < 3:
            return True
        
        return False
    
    def validate_single_field(self, field_name: str, field_value: str) -> Dict[str, Any]:
        """
        Validate a single input field.
        
        Args:
            field_name: Name of the field being validated
            field_value: Value to validate
            
        Returns:
            Dictionary with validation result for the single field
        """
        cleaned = field_value.strip() if field_value else ""
        errors = []
        
        if not cleaned:
            errors.append(f"{field_name} is required")
        elif len(cleaned) < self.MIN_LENGTH:
            errors.append(f"{field_name} must be at least {self.MIN_LENGTH} characters")
        elif len(cleaned) > self.MAX_LENGTH:
            errors.append(f"{field_name} exceeds maximum length")
        
        return {
            "field": field_name,
            "valid": len(errors) == 0,
            "errors": errors,
            "length": len(cleaned),
            "value": cleaned
        }
    
    def get_validation_guidance(self) -> Dict[str, Any]:
        """
        Get guidance text for users about input requirements.
        
        Returns:
            Dictionary with guidance information for UI display
        """
        return {
            "min_length": self.MIN_LENGTH,
            "max_length": self.MAX_LENGTH,
            "required_fields": list(self.FIELD_DESCRIPTIONS.keys()),
            "field_descriptions": self.FIELD_DESCRIPTIONS,
            "tips": [
                f"Each field must be at least {self.MIN_LENGTH} characters",
                f"Maximum length is {self.MAX_LENGTH} characters per field",
                "Be specific about your pet's age, breed, and symptoms",
                "Include details about diet, exercise, and daily routine",
                "Describe any behavioral changes or concerns"
            ]
        }


# ==========================================
# CONVENIENCE FUNCTION
# ==========================================

def validate_pet_inputs(
    about_pet: str,
    daily_routine: str,
    health_concerns: str
) -> Dict[str, Any]:
    """
    Convenience function to validate pet health assessment inputs.
    
    Args:
        about_pet: Description of pet characteristics
        daily_routine: Description of daily activities
        health_concerns: Description of health issues
        
    Returns:
        Dictionary with validation results
    """
    validator = InputValidatorAgent()
    return validator.validate_inputs({
        "about_pet": about_pet,
        "daily_routine": daily_routine,
        "health_concerns": health_concerns
    })


# ==========================================
# MODULE TESTING
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("INPUT VALIDATOR AGENT TEST")
    print("=" * 60)
    
    validator = InputValidatorAgent()
    
    # Test Case 1: Valid inputs
    print("\n📝 Test Case 1: Valid Inputs")
    valid_inputs = {
        "about_pet": "My dog is a 5-year-old Labrador Retriever named Max. He weighs about 30kg and is generally healthy.",
        "daily_routine": "Max eats twice daily (morning and evening), goes for 30-minute walks twice a day, and sleeps indoors.",
        "health_concerns": "Recently noticed he's drinking more water than usual and seems slightly lethargic after walks."
    }
    result1 = validator.validate_inputs(valid_inputs)
    print(f"Valid: {result1['is_valid']}")
    print(f"Errors: {result1['validation_errors']}")
    print(f"Status: {result1['status']}")
    
    # Test Case 2: Empty fields
    print("\n📝 Test Case 2: Empty Fields")
    empty_inputs = {
        "about_pet": "",
        "daily_routine": "  ",
        "health_concerns": ""
    }
    result2 = validator.validate_inputs(empty_inputs)
    print(f"Valid: {result2['is_valid']}")
    print(f"Errors: {result2['validation_errors']}")
    
    # Test Case 3: Too short
    print("\n📝 Test Case 3: Too Short")
    short_inputs = {
        "about_pet": "My dog",
        "daily_routine": "Walks",
        "health_concerns": "Sick"
    }
    result3 = validator.validate_inputs(short_inputs)
    print(f"Valid: {result3['is_valid']}")
    print(f"Errors: {result3['validation_errors']}")
    
    # Test Case 4: Placeholder text
    print("\n📝 Test Case 4: Placeholder Text")
    placeholder_inputs = {
        "about_pet": "test",
        "daily_routine": "asdf",
        "health_concerns": "placeholder"
    }
    result4 = validator.validate_inputs(placeholder_inputs)
    print(f"Valid: {result4['is_valid']}")
    print(f"Errors: {result4['validation_errors']}")
    
    # Test Case 5: Mixed results
    print("\n📝 Test Case 5: Mixed Results")
    mixed_inputs = {
        "about_pet": "My cat Whiskers is 8 years old, indoor only, Persian mix.",
        "daily_routine": "",  # Missing
        "health_concerns": "Has been coughing occasionally for the past few days. Appetite seems normal though."
    }
    result5 = validator.validate_inputs(mixed_inputs)
    print(f"Valid: {result5['is_valid']}")
    print(f"Errors: {result5['validation_errors']}")
    print(f"Field Status: {result5['field_status']['daily_routine']}")
    
    # Display guidance
    print("\n📋 Validation Guidance:")
    guidance = validator.get_validation_guidance()
    print(f"Minimum Length: {guidance['min_length']} characters")
    print(f"Maximum Length: {guidance['max_length']} characters")
    print("Tips:")
    for tip in guidance['tips']:
        print(f"  • {tip}")
    
    print("\n✅ Input Validator Agent ready for use")