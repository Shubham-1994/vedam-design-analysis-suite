"""Base agent class for the multimodal design analysis suite."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import time
import logging
from PIL import Image
import numpy as np
from langchain_core.prompts import ChatPromptTemplate

from ..models.schemas import Finding, AgentResult, AnalysisType
from ..utils.embeddings import get_embeddings
from ..utils.vector_store import get_vector_store
from ..utils.llm_client import get_llm_client

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all analysis agents."""
    
    def __init__(self, agent_name: str, analysis_type: AnalysisType, api_keys: Optional[Dict[str, str]] = None):
        self.agent_name = agent_name
        self.analysis_type = analysis_type
        self.api_keys = api_keys or {}
        
        # Initialize services with API keys if provided
        huggingface_key = self.api_keys.get('huggingface')
        openrouter_key = self.api_keys.get('openrouter')
        
        self.embeddings = get_embeddings(huggingface_key)
        self.vector_store = get_vector_store()
        self.llm_client = get_llm_client(openrouter_key)
        self.logger = logging.getLogger(f"{__name__}.{agent_name}")
    
    @abstractmethod
    async def analyze(
        self,
        image: Image.Image,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """Perform analysis on the given image."""
        pass
    
    def _create_finding(
        self,
        category: str,
        severity: str,
        title: str,
        description: str,
        recommendation: str,
        confidence_score: float,
        supporting_evidence: Optional[List[str]] = None
    ) -> Finding:
        """Helper method to create a Finding object."""
        return Finding(
            category=category,
            severity=severity,
            title=title,
            description=description,
            recommendation=recommendation,
            confidence_score=confidence_score,
            supporting_evidence=supporting_evidence or []
        )
    
    def _get_similar_patterns(
        self,
        image: Image.Image,
        category_filter: Optional[str] = None,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Get similar design patterns from the vector store."""
        try:
            # Encode the image
            image_embedding = self.embeddings.encode_image(image)
            
            # Prepare filter
            filter_metadata = {}
            if category_filter:
                filter_metadata["category"] = category_filter
            
            # Search for similar patterns
            similar_patterns = self.vector_store.search_similar_patterns(
                query_embedding=image_embedding,
                n_results=n_results,
                filter_metadata=filter_metadata if filter_metadata else None
            )
            
            return similar_patterns
            
        except Exception as e:
            self.logger.error(f"Failed to get similar patterns: {e}")
            return []
    
    def _search_patterns_by_text(
        self,
        query_text: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for patterns using text query."""
        try:
            return self.vector_store.search_by_text_query(
                text_query=query_text,
                n_results=n_results
            )
        except Exception as e:
            self.logger.error(f"Failed to search patterns by text: {e}")
            return []
    
    def _calculate_overall_score(self, findings: List[Finding]) -> float:
        """Calculate overall score based on findings."""
        if not findings:
            return 5.0  # Neutral score
        
        # Weight scores by severity
        severity_weights = {
            "low": 0.1,
            "medium": 0.3,
            "high": 0.6,
            "critical": 1.0
        }
        
        total_weight = 0
        weighted_score = 0
        
        for finding in findings:
            weight = severity_weights.get(finding.severity, 0.3)
            # Convert confidence to impact (higher confidence = more impact on score)
            impact = finding.confidence_score * weight
            
            # Negative findings reduce score, positive findings increase it
            if finding.severity in ["high", "critical"]:
                score_impact = -impact * 3  # Negative impact
            elif finding.severity == "medium":
                score_impact = -impact * 1.5
            else:
                score_impact = impact * 0.5  # Slight positive for low severity (good practices)
            
            weighted_score += score_impact
            total_weight += weight
        
        # Base score of 7.0, adjusted by findings
        base_score = 7.0
        if total_weight > 0:
            adjustment = weighted_score / total_weight
            final_score = base_score + adjustment
        else:
            final_score = base_score
        
        # Clamp between 0 and 10
        return max(0.0, min(10.0, final_score))
    
    async def _generate_llm_insights(
        self,
        prompt_template: ChatPromptTemplate,
        variables: Dict[str, Any],
        temperature: float = 0.7
    ) -> Optional[str]:
        """Generate insights using LLM."""
        try:
            return await self.llm_client.generate_response(
                prompt_template=prompt_template,
                variables=variables,
                temperature=temperature
            )
        except Exception as e:
            self.logger.error(f"Failed to generate LLM insights: {e}")
            return None
    
    def _create_image_description(self, image: Image.Image, context: Optional[Dict[str, Any]] = None) -> str:
        """Create a detailed description of the image for LLM analysis."""
        # Get basic image properties
        width, height = image.size
        aspect_ratio = width / height
        
        # Determine image type based on aspect ratio and size
        if aspect_ratio > 1.5:
            layout_type = "wide/landscape layout"
        elif aspect_ratio < 0.7:
            layout_type = "tall/portrait layout"
        else:
            layout_type = "square/balanced layout"
        
        description = f"UI/UX design image with {layout_type}, dimensions {width}x{height} pixels"
        
        # Add context information if available
        if context:
            if context.get("platform"):
                description += f", designed for {context['platform']} platform"
            if context.get("industry"):
                description += f", in the {context['industry']} industry"
            if context.get("target_audience"):
                description += f", targeting {context['target_audience']} users"
        
        return description
    
    async def _execute_analysis(
        self,
        image: Image.Image,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """Execute the analysis and return results with timing."""
        start_time = time.time()
        
        try:
            # Perform the actual analysis
            result = await self.analyze(image, context)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Update the result with execution time
            result.execution_time = execution_time
            
            self.logger.info(
                f"Agent {self.agent_name} completed analysis in {execution_time:.2f}s "
                f"with score {result.overall_score:.1f}"
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Agent {self.agent_name} failed after {execution_time:.2f}s: {e}")
            
            # Return error result
            return AgentResult(
                agent_name=self.agent_name,
                analysis_type=self.analysis_type,
                findings=[
                    self._create_finding(
                        category="error",
                        severity="critical",
                        title="Analysis Failed",
                        description=f"Agent failed to complete analysis: {str(e)}",
                        recommendation="Please try again or contact support",
                        confidence_score=1.0
                    )
                ],
                overall_score=0.0,
                execution_time=execution_time,
                metadata={"error": str(e)}
            )
