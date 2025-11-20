"""Pydantic models for structured input/output."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class AnalysisType(str, Enum):
    """Types of analysis to perform."""
    VISUAL = "visual"
    UX_CRITIQUE = "ux_critique"
    MARKET_RESEARCH = "market_research"
    ACCESSIBILITY = "accessibility"
    STYLE_CONSISTENCY = "style_consistency"


class DesignContext(BaseModel):
    """Context information for the design analysis."""
    project_name: Optional[str] = None
    target_audience: Optional[str] = None
    design_goals: Optional[str] = None
    industry: Optional[str] = None
    platform: Optional[str] = Field(None, description="web, mobile, desktop, etc.")
    additional_notes: Optional[str] = None


class UploadRequest(BaseModel):
    """Request model for design upload."""
    context: Optional[DesignContext] = None
    analysis_types: List[AnalysisType] = Field(default_factory=lambda: list(AnalysisType))


class Finding(BaseModel):
    """Individual finding from analysis."""
    category: str
    severity: str = Field(..., description="low, medium, high, critical")
    title: str
    description: str
    recommendation: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    supporting_evidence: Optional[List[str]] = None


class AgentResult(BaseModel):
    """Result from a single agent."""
    agent_name: str
    analysis_type: AnalysisType
    findings: List[Finding]
    overall_score: float = Field(..., ge=0.0, le=10.0)
    execution_time: float
    metadata: Optional[Dict[str, Any]] = None


class MarketComparison(BaseModel):
    """Market comparison data."""
    similar_designs: List[Dict[str, Any]]
    industry_trends: List[str]
    competitive_analysis: Optional[str] = None


class VisualAnalysisMetrics(BaseModel):
    """Visual analysis specific metrics."""
    layout_score: float = Field(..., ge=0.0, le=10.0)
    visual_hierarchy_score: float = Field(..., ge=0.0, le=10.0)
    color_harmony_score: float = Field(..., ge=0.0, le=10.0)
    typography_score: float = Field(..., ge=0.0, le=10.0)
    whitespace_usage_score: float = Field(..., ge=0.0, le=10.0)


class UXMetrics(BaseModel):
    """UX analysis specific metrics."""
    usability_score: float = Field(..., ge=0.0, le=10.0)
    navigation_clarity: float = Field(..., ge=0.0, le=10.0)
    information_architecture: float = Field(..., ge=0.0, le=10.0)
    user_flow_efficiency: float = Field(..., ge=0.0, le=10.0)


class AccessibilityMetrics(BaseModel):
    """Accessibility analysis metrics."""
    wcag_compliance_score: float = Field(..., ge=0.0, le=10.0)
    color_contrast_score: float = Field(..., ge=0.0, le=10.0)
    text_readability_score: float = Field(..., ge=0.0, le=10.0)
    keyboard_navigation_score: float = Field(..., ge=0.0, le=10.0)


class AnalysisResult(BaseModel):
    """Complete analysis result."""
    analysis_id: str
    upload_filename: str
    context: Optional[DesignContext] = None
    agent_results: List[AgentResult]
    
    # Aggregated metrics
    overall_score: float = Field(..., ge=0.0, le=10.0)
    visual_metrics: Optional[VisualAnalysisMetrics] = None
    ux_metrics: Optional[UXMetrics] = None
    accessibility_metrics: Optional[AccessibilityMetrics] = None
    market_comparison: Optional[MarketComparison] = None
    
    # Summary
    key_strengths: List[str]
    priority_improvements: List[Finding]
    actionable_recommendations: List[str]
    
    # Metadata
    analysis_timestamp: str
    processing_time: float
    confidence_level: float = Field(..., ge=0.0, le=1.0)


class AnalysisStatus(BaseModel):
    """Status of an ongoing analysis."""
    analysis_id: str
    status: str = Field(..., description="pending, processing, completed, failed")
    progress: float = Field(..., ge=0.0, le=1.0)
    current_stage: Optional[str] = None
    estimated_completion: Optional[str] = None
    error_message: Optional[str] = None
    agent_states: Optional[Dict[str, str]] = Field(default=None, description="Individual agent states")


class AnalysisHistoryItem(BaseModel):
    """Analysis history item for dashboard."""
    analysis_id: str
    upload_filename: str
    analysis_timestamp: str
    overall_score: float = Field(..., ge=0.0, le=10.0)
    status: str
    processing_time: Optional[float] = None
    context: Optional[DesignContext] = None
    thumbnail_path: Optional[str] = None


class AnalysisHistoryResponse(BaseModel):
    """Response for analysis history."""
    analyses: List[AnalysisHistoryItem]
    total_count: int
    page: int
    page_size: int


class DownloadFormat(str, Enum):
    """Available download formats."""
    JSON = "json"
    PDF = "pdf"
    PNG = "png"
    CSV = "csv"


class ImageGenerationRequest(BaseModel):
    """Request model for image generation."""
    analysis_id: str
    generation_options: Optional[Dict[str, Any]] = None
    num_variants: int = Field(default=1, ge=1, le=5)
    focus_area: Optional[str] = None


class ImageGenerationResult(BaseModel):
    """Result from image generation."""
    success: bool
    generated_image_base64: Optional[str] = None
    prompt_used: Optional[str] = None
    generation_options: Optional[Dict[str, Any]] = None
    improvements_applied: Optional[List[str]] = None
    variant_name: Optional[str] = None
    focus_area: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    """Response for image generation API."""
    analysis_id: str
    original_filename: str
    variants: List[ImageGenerationResult]
    generation_timestamp: str
    processing_time: float
