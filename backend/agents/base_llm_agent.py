# agents/base_llm_agent.py

from typing import Dict, Any, Optional, List
import logging
from abc import ABC, abstractmethod
from utils.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


class BaseLLMAgent(ABC):
    def __init__(
        self,
        client: OpenAIClient,
        agent_name: str = "BaseLLMAgent",
        default_temperature: float = 0.3,
        default_max_tokens: int = 1500
    ):
        self.client = client
        self.agent_name = agent_name
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

    @abstractmethod
    def generate(self, *args, **kwargs) -> Dict[str, Any]:
        pass

    # ================= CORE METHODS =================

    def _generate_with_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        required_fields: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:

        temp = temperature or self.default_temperature
        tokens = max_tokens or self.default_max_tokens

        try:
            result = self.client.generate_structured_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                required_fields=required_fields or [],
                temperature=temp,
                max_tokens=tokens,
                **kwargs
            )

            result.update({
                "_agent": self.agent_name,
                "_generation_success": True
            })

            return result

        except Exception as e:
            logger.error(f"{self.agent_name}: {e}")
            return {
                "_agent": self.agent_name,
                "_generation_success": False,
                "_error": str(e),
                "fallback_used": True
            }

    def _generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:

        temp = temperature or self.default_temperature
        tokens = max_tokens or self.default_max_tokens

        try:
            return self.client.generate_content(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=temp,
                max_tokens=tokens,
                **kwargs
            )
        except Exception as e:
            logger.error(f"{self.agent_name}: {e}")
            return ""

    # ================= UTILITIES =================

    def _build_prompt_with_context(
        self,
        base_prompt: str,
        context: Dict[str, Any],
        include_fields: Optional[List[str]] = None
    ) -> str:

        data = (
            {k: context.get(k, "") for k in include_fields}
            if include_fields else context
        )

        return base_prompt.format(**data)

    def _safe_get(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        return data.get(key, default)

    def _format_list_for_prompt(
        self,
        items: List[str],
        empty_text: str = "None"
    ) -> str:
        return ", ".join(items) if items else empty_text

    def _truncate_text(
        self,
        text: str,
        max_length: int = 500
    ) -> str:
        return text if len(text) <= max_length else text[:max_length] + "..."