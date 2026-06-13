#!/usr/bin/env python3
"""
OpenAI client adapter for PawCare+.
Provides unified interface to OpenAI API with key rotation, JSON parsing, and robust error handling.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)

# Load environment variables
try:
    from dotenv import load_dotenv

    env_file = Path.cwd() / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass

# Import OpenAI
try:
    from openai import OpenAI, APIError, RateLimitError, AuthenticationError
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    APIError = Exception
    RateLimitError = Exception
    AuthenticationError = Exception
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not installed. Install with: pip install openai>=1.0.0")


# ==========================================
# API KEY MANAGEMENT FUNCTIONS
# ==========================================

def _get_all_api_keys() -> List[str]:
    """
    Retrieve all available OpenAI API keys from the environment.

    Checks in order:
    1. Environment variable OPENAI_API_KEY
    2. Environment variables OPENAI_API_KEY_1 through OPENAI_API_KEY_4

    Keys are loaded from the process environment (typically populated from a
    .env file at startup; see api/server.py).

    Returns:
        List of unique API key strings in priority order
    """
    keys = []

    # Check environment variables
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key and env_key not in keys:
        keys.append(env_key)
    
    # Check alternate spelling (common typo)
    env_key_alt = os.getenv("OPENAL_API_KEY")
    if env_key_alt and env_key_alt not in keys:
        keys.append(env_key_alt)

    # Check numbered environment variables
    for i in range(1, 5):
        env_key_i = os.getenv(f"OPENAI_API_KEY_{i}")
        if env_key_i and env_key_i not in keys:
            keys.append(env_key_i)
        
        # Alternate spelling for numbered keys
        env_key_i_alt = os.getenv(f"OPENAL_API_KEY_{i}")
        if env_key_i_alt and env_key_i_alt not in keys:
            keys.append(env_key_i_alt)

    # Remove any empty strings
    keys = [k for k in keys if k and k.strip()]
    
    return keys


def _get_first_api_key() -> Optional[str]:
    """
    Get the first available API key for backwards compatibility.
    
    Returns:
        First API key string or None if no keys found
    """
    keys = _get_all_api_keys()
    return keys[0] if keys else None


# ==========================================
# JSON PARSING UTILITIES
# ==========================================

def _clean_json_text(text: str) -> str:
    """
    Remove markdown formatting and extract clean JSON text.
    
    Args:
        text: Raw response text that may contain markdown code fences
        
    Returns:
        Cleaned text with markdown fences removed
    """
    cleaned = text.strip()
    # Remove ```json ... ``` blocks (case insensitive)
    cleaned = re.sub(r"```json\s*", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    # Remove ``` ... ``` blocks (any language)
    cleaned = re.sub(r"```\w*\s*", "", cleaned, flags=re.MULTILINE)
    # Remove any remaining backticks
    cleaned = re.sub(r"```", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def _strip_trailing_commas(text: str) -> str:
    """
    Remove trailing commas before closing braces/brackets.
    
    Args:
        text: JSON string that might have trailing commas
        
    Returns:
        JSON string with trailing commas removed
    """
    previous = None
    current = text
    # Pattern matches comma followed by optional whitespace then closing brace or bracket
    pattern = re.compile(r",\s*(?=[\]}])")
    
    # Keep applying until no more changes (handles nested structures)
    while current != previous:
        previous = current
        current = pattern.sub("", current)
    return current


def _repair_unterminated_strings(text: str) -> str:
    """
    Repair JSON with unterminated strings.
    
    Args:
        text: Potentially malformed JSON string
        
    Returns:
        Repaired JSON string
    """
    try:
        # Remove any remaining markdown markers
        repaired = re.sub(r'^\s*```.*$', '', text, flags=re.MULTILINE)

        # Try to extract just the JSON object if there's surrounding text
        # This pattern matches balanced braces
        json_match = re.search(r'\{[^{}]*?(?:\{[^{}]*?\}[^{}]*?)*\}', repaired, re.DOTALL)
        if json_match:
            repaired = json_match.group(0)

        return repaired.strip('[]')
    except Exception:
        return text


def _extract_json_by_braces(text: str) -> str:
    """
    Extract JSON by matching opening and closing braces.
    
    Args:
        text: Text that may contain JSON with surrounding content
        
    Returns:
        Extracted JSON object string
    """
    try:
        # Find the first opening brace
        start_idx = text.find('{')
        if start_idx == -1:
            return text

        # Count braces to find the matching closing brace
        brace_count = 0
        i = start_idx
        while i < len(text):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found the matching closing brace
                    return text[start_idx:i+1]
            i += 1

        # If we get here, braces are unmatched
        return text
    except Exception:
        return text


def _fix_unescaped_quotes_in_values(text: str) -> str:
    """
    Fix unescaped quotes and special characters within JSON string values.
    
    Args:
        text: JSON string with potential unescaped quotes
        
    Returns:
        JSON string with quotes properly escaped
    """
    try:
        result = []
        i = 0
        in_string = False
        escape_next = False

        while i < len(text):
            char = text[i]

            # Handle escaped characters
            if escape_next:
                result.append(char)
                escape_next = False
                i += 1
                continue

            # If we see a backslash in a string, mark next char as escaped
            if char == '\\' and in_string:
                result.append(char)
                escape_next = True
                i += 1
                continue

            # Toggle in_string state when we see an unescaped quote
            if char == '"':
                in_string = not in_string
                result.append(char)
                i += 1
                continue

            # If we're inside a string, handle special characters
            if in_string:
                if char == '\n':
                    result.append('\\n')
                elif char == '\r':
                    result.append('\\r')
                elif char == '\t':
                    result.append('\\t')
                elif char == '\b':
                    result.append('\\b')
                elif char == '\f':
                    result.append('\\f')
                else:
                    result.append(char)
            else:
                result.append(char)

            i += 1

        return ''.join(result)
    except Exception:
        return text


# ==========================================
# OPENAI CLIENT CLASS
# ==========================================

class OpenAIClient:
    """
    OpenAI client wrapper with automatic API key rotation and robust JSON parsing.
    
    Provides unified interface to OpenAI API with:
    - Multiple API key support with automatic rotation on quota exhaustion
    - Configurable model, temperature, and max tokens
    - Robust JSON extraction with multiple repair strategies
    - Response field validation
    - Structured JSON generation with required field validation
    
    Attributes:
        model (str): Name of OpenAI model (default: gpt-3.5-turbo)
        api_keys (List[str]): List of available API keys for rotation
        current_key_index (int): Index of currently active key
        client (OpenAI): Active OpenAI client instance
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI client with first available API key.
        
        Args:
            model: OpenAI model to use (default: gpt-3.5-turbo)
                  Options: gpt-3.5-turbo, gpt-4, gpt-3.5-turbo, gpt-3.5-turbo-16k
        
        Raises:
            ImportError: If OpenAI package not installed
            ValueError: If no API keys found
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package is not installed. "
                "Install with: pip install openai>=1.0.0"
            )
        
        self.model = model
        self.api_keys = _get_all_api_keys()
        self.current_key_index = 0
        
        if not self.api_keys:
            raise ValueError(
                "No OpenAI API keys found. "
                "Set OPENAI_API_KEY or OPENAI_API_KEY_1..4 in .env file"
            )
        
        self.client = None
        self._initialize_client()
        
        logger.info(f"OpenAI client initialized with model {self.model} "
                   f"(key {self.current_key_index + 1}/{len(self.api_keys)})")
    
    def _initialize_client(self) -> None:
        """
        Set up OpenAI client with current API key.
        
        Raises:
            ValueError: If current_key_index is out of bounds
        """
        if self.current_key_index >= len(self.api_keys):
            raise ValueError(f"API key index {self.current_key_index} out of bounds "
                           f"(only {len(self.api_keys)} keys available)")
        
        current_key = self.api_keys[self.current_key_index]
        
        try:
            # Initialize OpenAI client with the current key
            self.client = OpenAI(api_key=current_key)
            logger.debug(f"Initialized OpenAI client with API key "
                        f"{self.current_key_index + 1}/{len(self.api_keys)}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise ValueError(f"Failed to initialize OpenAI client: {str(e)}")
    
    def _rotate_api_key(self) -> None:
        """
        Rotate to next available API key when quota exceeded.
        
        Raises:
            ValueError: If all API keys have been exhausted
        """
        self.current_key_index += 1
        
        if self.current_key_index >= len(self.api_keys):
            raise ValueError("All API keys exhausted")
        
        self._initialize_client()
        logger.info(f"Rotated to API key {self.current_key_index + 1}/{len(self.api_keys)}")
    
    def generate_content(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Generate content from OpenAI with automatic key rotation on quota exceeded.
        
        Args:
            prompt: User prompt for the LLM
            system_prompt: Optional system prompt for context
            temperature: Float between 0 and 1 controlling randomness
                        (0=deterministic, 1=creative)
            max_tokens: Maximum output tokens
            response_format: Optional format specification, e.g., {"type": "json_object"}
        
        Returns:
            String containing generated content from LLM
            
        Raises:
            ValueError: If generation fails or all API keys exhausted
        """
        max_retries = len(self.api_keys)
        attempt = 0
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        while attempt < max_retries:
            try:
                # Prepare completion parameters
                completion_params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                
                # Add response format if specified (for JSON mode)
                if response_format:
                    completion_params["response_format"] = response_format
                
                # Make the API call
                response = self.client.chat.completions.create(**completion_params)
                
                # Extract response text
                if response.choices and len(response.choices) > 0:
                    response_text = response.choices[0].message.content
                    if response_text:
                        return response_text
                    else:
                        raise ValueError("Empty response from OpenAI")
                else:
                    raise ValueError("No choices in OpenAI response")
                
            except RateLimitError as e:
                # Rate limit or quota exceeded - rotate key
                logger.warning(
                    f"Rate limit/quota exceeded on key {self.current_key_index + 1}: {str(e)}"
                )
                attempt += 1
                
                if attempt < max_retries:
                    try:
                        self._rotate_api_key()
                        logger.info(f"Retrying with next API key "
                                  f"({self.current_key_index + 1}/{len(self.api_keys)})")
                        continue
                    except ValueError as rotate_error:
                        raise ValueError(f"All API keys exhausted: {rotate_error}")
                else:
                    raise ValueError(
                        f"All {len(self.api_keys)} API keys rate limited/quota exceeded. "
                        "Please wait before retrying."
                    )
                    
            except AuthenticationError as e:
                # Invalid API key - rotate
                logger.warning(
                    f"Authentication failed on key {self.current_key_index + 1}: {str(e)}"
                )
                attempt += 1
                
                if attempt < max_retries:
                    try:
                        self._rotate_api_key()
                        logger.info("Rotating to next API key due to auth failure")
                        continue
                    except ValueError:
                        raise ValueError("All API keys invalid")
                else:
                    raise ValueError("All API keys invalid")
                    
            except APIError as e:
                # Other API errors (server errors, etc.)
                logger.error(f"OpenAI API error: {str(e)}")
                
                # Check if it's a server error (5xx) - retry without rotating
                if hasattr(e, 'status_code') and e.status_code >= 500:
                    attempt += 1
                    if attempt < max_retries:
                        # Exponential backoff for server errors
                        wait_time = 2 ** attempt
                        logger.info(f"Server error, retrying in {wait_time}s "
                                  f"(attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                
                # Non-recoverable error
                raise ValueError(f"OpenAI API error: {str(e)}")
                
            except Exception as e:
                # Unexpected errors
                logger.error(f"Unexpected error: {str(e)}")
                raise ValueError(f"Failed to generate content from OpenAI: {str(e)}")
        
        raise ValueError("Failed to generate content after all retries")
    
    def extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response with multi-strategy error recovery.
        
        Attempts multiple parsing strategies in sequence:
        1. Clean markdown and try standard parse
        2. Fix trailing commas
        3. Repair unterminated strings
        4. Extract JSON by brace matching
        5. Try nested JSON objects
        
        Args:
            response_text: Raw response text that may contain JSON
            
        Returns:
            Dictionary containing parsed JSON
            
        Raises:
            ValueError: If JSON cannot be extracted after all strategies
        """
        try:
            # Initial cleaning
            cleaned = _clean_json_text(response_text)
            
            # Extract JSON by matching braces first (handles surrounding text)
            brace_extracted = _extract_json_by_braces(cleaned)
            candidates = [cleaned, brace_extracted]
            
            # If the response contains other text, try extracting the first JSON object
            json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if json_match:
                candidates.append(json_match.group(0))
            
            last_error = None
            for candidate in candidates:
                if not candidate or '{' not in candidate:
                    continue
                
                try:
                    # First try standard repair (remove trailing commas)
                    repaired = _strip_trailing_commas(candidate)
                    parsed_json = json.loads(repaired)
                    if not isinstance(parsed_json, dict):
                        raise ValueError("Response is not a valid JSON object")
                    return parsed_json
                    
                except json.JSONDecodeError as parse_error:
                    # Try fixing unescaped quotes in values
                    try:
                        repaired_quotes = _fix_unescaped_quotes_in_values(candidate)
                        repaired = _strip_trailing_commas(repaired_quotes)
                        parsed_json = json.loads(repaired)
                        if isinstance(parsed_json, dict):
                            return parsed_json
                    except Exception:
                        pass
                    
                    # Try aggressive string repair for unterminated strings
                    try:
                        repaired_strings = _repair_unterminated_strings(candidate)
                        repaired_quotes = _fix_unescaped_quotes_in_values(repaired_strings)
                        repaired = _strip_trailing_commas(repaired_quotes)
                        parsed_json = json.loads(repaired)
                        if isinstance(parsed_json, dict):
                            return parsed_json
                    except Exception:
                        pass
                    
                    # Try brace extraction again with fixes applied
                    try:
                        brace_extracted = _extract_json_by_braces(candidate)
                        repaired_quotes = _fix_unescaped_quotes_in_values(brace_extracted)
                        repaired = _strip_trailing_commas(repaired_quotes)
                        parsed_json = json.loads(repaired)
                        if isinstance(parsed_json, dict):
                            return parsed_json
                    except Exception:
                        pass
                    
                    # Try to find nested JSON objects if main parsing fails
                    try:
                        # Find all { } pairs and try parsing from innermost
                        nested_matches = re.findall(r"\{[^{}]*\}", candidate)
                        for nested in nested_matches:
                            try:
                                parsed_json = json.loads(nested)
                                if isinstance(parsed_json, dict):
                                    return parsed_json
                            except Exception:
                                continue
                    except Exception:
                        pass
                    
                    last_error = parse_error
                    continue
                    
                except Exception as parse_error:
                    last_error = parse_error
                    continue
            
            raise ValueError(f"Invalid JSON in LLM response: {last_error}")
            
        except Exception as e:
            logger.error(f"Response extraction error: {str(e)}")
            raise ValueError(f"Failed to extract JSON from LLM response: {str(e)}")
    
    def validate_response_fields(
        self, 
        response: Dict[str, Any], 
        required_fields: List[str]
    ) -> None:
        """
        Verify response contains all required fields.
        
        Args:
            response: Dictionary to validate
            required_fields: List of required field names
            
        Raises:
            ValueError: If any required fields are missing
        """
        missing_fields = [field for field in required_fields if field not in response]
        
        if missing_fields:
            raise ValueError(
                f"LLM response missing required fields: {missing_fields}"
            )
    
    def generate_structured_json(
        self,
        prompt: str,
        required_fields: List[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> Dict[str, Any]:
        """
        Generate and validate JSON response with specified required fields.
        
        Uses OpenAI's native JSON mode for more reliable parsing,
        with fallback to manual parsing if JSON mode fails.
        
        Args:
            prompt: User prompt for the LLM
            required_fields: List of required field names in response
            system_prompt: Optional system prompt for context
            temperature: Float controlling randomness (0-1)
            max_tokens: Maximum output tokens
            
        Returns:
            Dictionary containing valid JSON with all required fields
            
        Raises:
            ValueError: If JSON generation or validation fails
        """
        try:
            # First attempt with JSON mode (more reliable)
            response_text = self.generate_content(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            result = self.extract_json_from_response(response_text)
            
            # Validate required fields
            self.validate_response_fields(result, required_fields)
            
            return result
            
        except Exception as e:
            # If JSON mode fails, try without it (some models/versions may not support it)
            logger.warning(f"JSON mode failed, falling back to manual parsing: {str(e)}")
            
            # Add explicit instruction to respond with JSON
            json_prompt = f"{prompt}\n\nRespond with valid JSON only. Do not include any other text."
            
            response_text = self.generate_content(
                prompt=json_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            result = self.extract_json_from_response(response_text)
            self.validate_response_fields(result, required_fields)
            
            return result
    
    def generate_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
        base_delay: float = 1.0,
        **kwargs
    ) -> str:
        """
        Generate content with exponential backoff retry for transient errors.
        
        Args:
            prompt: User prompt
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
            **kwargs: Additional arguments passed to generate_content
            
        Returns:
            Generated content string
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return self.generate_content(prompt, **kwargs)
            except (RateLimitError, APIError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Transient error, retrying in {delay}s: {str(e)}")
                    time.sleep(delay)
                continue
            except Exception as e:
                # Non-transient error, don't retry
                raise
        
        raise ValueError(f"Failed after {max_retries} retries. Last error: {last_error}")


# ==========================================
# FACTORY FUNCTION
# ==========================================

def build_openai_client(model: str = "gpt-3.5-turbo") -> OpenAIClient:
    """
    Factory function to create a configured OpenAI client.
    
    Args:
        model: OpenAI model name to use
        
    Returns:
        Configured OpenAIClient instance
        
    Raises:
        ValueError: If OpenAI package not installed or no API keys found
    """
    if not OPENAI_AVAILABLE:
        raise ImportError(
            "OpenAI package is not installed. "
            "Install with: pip install openai>=1.0.0"
        )
    
    return OpenAIClient(model=model)


# ==========================================
# CONVENIENCE FUNCTION
# ==========================================

def extract_json(text: str) -> Dict[str, Any]:
    """
    Simple utility to extract JSON from text.
    
    Args:
        text: Text containing JSON
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        ValueError: If JSON cannot be extracted
    """
    try:
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in response")
        return json.loads(json_match.group(0))
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {str(e)}")


# ==========================================
# MODULE TESTING
# ==========================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("OPENAI CLIENT MODULE TEST")
    print("=" * 60)
    
    # Test API key loading
    keys = _get_all_api_keys()
    print(f"\n✅ Found {len(keys)} API keys")
    
    if keys:
        try:
            # Test client initialization
            client = build_openai_client()
            print(f"✅ Client initialized with model: {client.model}")
            
            # Test simple generation
            response = client.generate_content(
                prompt="Say 'Hello, PawCare+!' in JSON format",
                system_prompt="You are a helpful assistant that responds in JSON.",
                max_tokens=50
            )
            print(f"✅ Generation successful: {response[:50]}...")
            
            # Test JSON parsing
            json_result = client.extract_json_from_response(response)
            print(f"✅ JSON parsing successful")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    else:
        print("❌ No API keys found. Set OPENAI_API_KEY in .env file")
    
    print("\n✅ Module ready for import")