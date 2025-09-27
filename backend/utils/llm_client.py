"""LLM client for OpenRouter integration using Langchain OpenAI."""

import logging
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with LLMs via OpenRouter using Langchain OpenAI."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.openrouter_api_key
        self.base_url = settings.openrouter_base_url
        self.model_name = settings.llm_model_name
        
        if not self.api_key:
            logger.warning("OpenRouter API key not found. LLM features will be limited.")
            self.llm = None
        else:
            # Initialize ChatOpenAI with OpenRouter configuration
            self.llm = ChatOpenAI(
                model=self.model_name,
                openai_api_key=self.api_key,
                openai_api_base=self.base_url,
                temperature=0.7,
                max_tokens=1000,
                timeout=60
            )
    
    async def generate_response(
        self,
        prompt_template: ChatPromptTemplate,
        variables: Dict[str, Any],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Optional[str]:
        """Generate response using the LLM."""
        if not self.llm:
            logger.warning("LLM not initialized. Check API key configuration.")
            return None
            
        try:
            # Create a new LLM instance with custom parameters if different from default
            llm = self.llm
            if temperature != 0.7 or max_tokens != 1000:
                llm = ChatOpenAI(
                    model=self.model_name,
                    openai_api_key=self.api_key,
                    openai_api_base=self.base_url,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=60
                )
            
            # Format the prompt and invoke the LLM
            chain = prompt_template | llm
            response = await chain.ainvoke(variables)
            
            return response.content
                
        except Exception as e:
            logger.error(f"Failed to generate LLM response: {e}")
            return None
    
    async def analyze_image_with_context(
        self,
        image_description: str,
        analysis_context: str,
        specific_instructions: str,
        temperature: float = 0.7
    ) -> Optional[str]:
        """Analyze image with specific context and instructions."""
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert UI/UX design analyst. Analyze the provided image description and context to provide detailed insights.

Your analysis should be:
- Specific and actionable
- Based on design principles and best practices
- Focused on user experience and usability
- Include concrete recommendations

Format your response as structured insights with clear categories."""),
            ("human", """Image Description: {image_description}

Analysis Context: {analysis_context}

Specific Instructions: {specific_instructions}

Please provide a detailed analysis following the instructions above.""")
        ])
        
        return await self.generate_response(
            prompt_template,
            {
                "image_description": image_description,
                "analysis_context": analysis_context,
                "specific_instructions": specific_instructions
            },
            temperature=temperature
        )
    
    def create_custom_llm(self, temperature: float = 0.7, max_tokens: int = 1000) -> Optional[ChatOpenAI]:
        """Create a custom LLM instance with specific parameters."""
        if not self.api_key:
            return None
            
        return ChatOpenAI(
            model=self.model_name,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=60
        )


# Global LLM client instance
_llm_client: Optional[LLMClient] = None


def get_llm_client(api_key: Optional[str] = None) -> LLMClient:
    """Get or create the global LLM client instance."""
    global _llm_client
    # Create new instance if API key is provided or if no instance exists
    if _llm_client is None or api_key:
        _llm_client = LLMClient(api_key)
    return _llm_client
