"""LangGraph orchestrator for coordinating multiple analysis agents."""

import asyncio
import uuid
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime
import logging

from langgraph.graph import StateGraph, END
from PIL import Image

from .visual_analysis_agent import VisualAnalysisAgent
from .ux_critique_agent import UXCritiqueAgent
from .market_research_agent import MarketResearchAgent
from ..models.schemas import (
    AnalysisResult, AgentResult, AnalysisType, DesignContext,
    VisualAnalysisMetrics, UXMetrics, AccessibilityMetrics, MarketComparison,
    Finding
)

logger = logging.getLogger(__name__)


class AnalysisState(TypedDict):
    """State for the analysis workflow."""
    analysis_id: str
    image: Image.Image
    context: Optional[DesignContext]
    requested_analyses: List[AnalysisType]
    api_keys: Optional[Dict[str, str]]  # API keys for external services
    
    # Agent results
    visual_result: Optional[AgentResult]
    ux_result: Optional[AgentResult]
    market_result: Optional[AgentResult]
    
    # Agent states tracking
    agent_states: Dict[str, str]  # agent_name -> status (pending, running, completed, failed)
    
    # Aggregated results
    all_findings: List[Finding]
    overall_score: float
    
    # Metadata
    start_time: datetime
    current_stage: str
    error_message: Optional[str]
    progress_callback: Optional[callable]


class DesignAnalysisOrchestrator:
    """Orchestrates multiple agents using LangGraph."""
    
    def __init__(self):
        # Agents will be created dynamically with API keys
        self.visual_agent = None
        self.ux_agent = None
        self.market_agent = None
        
        # Create the workflow graph
        self.workflow = self._create_workflow()
        # Remove checkpointer to avoid PIL Image serialization issues
        self.app = self.workflow.compile()
        
        logger.info("Design Analysis Orchestrator initialized")
    
    def _get_or_create_agents(self, api_keys: Optional[Dict[str, str]] = None):
        """Get or create agents with the provided API keys."""
        # Always create new agents if API keys are provided to ensure they use the latest keys
        if api_keys or not self.visual_agent:
            self.visual_agent = VisualAnalysisAgent(api_keys)
        if api_keys or not self.ux_agent:
            self.ux_agent = UXCritiqueAgent(api_keys)
        if api_keys or not self.market_agent:
            self.market_agent = MarketResearchAgent(api_keys)
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow."""
        workflow = StateGraph(AnalysisState)
        
        # Add nodes
        workflow.add_node("initialize", self._initialize_analysis)
        workflow.add_node("visual_analysis", self._run_visual_analysis)
        workflow.add_node("ux_analysis", self._run_ux_analysis)
        workflow.add_node("market_analysis", self._run_market_analysis)
        workflow.add_node("synthesize", self._synthesize_results)
        
        # Define the workflow
        workflow.set_entry_point("initialize")
        
        # Add conditional edges based on requested analyses
        workflow.add_conditional_edges(
            "initialize",
            self._route_analyses,
            {
                "visual_only": "visual_analysis",
                "ux_only": "ux_analysis", 
                "market_only": "market_analysis",
                "parallel": "visual_analysis",  # Start with visual for parallel execution
                "synthesize": "synthesize"  # If no specific analyses requested
            }
        )
        
        # Parallel execution paths
        workflow.add_conditional_edges(
            "visual_analysis",
            self._check_next_analysis,
            {
                "ux_analysis": "ux_analysis",
                "market_analysis": "market_analysis",
                "synthesize": "synthesize"
            }
        )
        
        workflow.add_conditional_edges(
            "ux_analysis", 
            self._check_next_analysis,
            {
                "market_analysis": "market_analysis",
                "synthesize": "synthesize"
            }
        )
        
        workflow.add_edge("market_analysis", "synthesize")
        workflow.add_edge("synthesize", END)
        
        return workflow
    
    async def analyze_design(
        self,
        image: Image.Image,
        context: Optional[DesignContext] = None,
        requested_analyses: Optional[List[AnalysisType]] = None,
        progress_callback: Optional[callable] = None,
        analysis_id: Optional[str] = None,
        api_keys: Optional[Dict[str, str]] = None
    ) -> AnalysisResult:
        """Analyze a design using the orchestrated workflow."""
        
        if analysis_id is None:
            analysis_id = str(uuid.uuid4())
        
        # Default to all analyses if none specified
        if not requested_analyses:
            requested_analyses = [
                AnalysisType.VISUAL,
                AnalysisType.UX_CRITIQUE,
                AnalysisType.MARKET_RESEARCH
            ]
        
        # Initialize agent states
        agent_states = {}
        for analysis_type in requested_analyses:
            if analysis_type == AnalysisType.VISUAL:
                agent_states["Visual Analysis Agent"] = "pending"
            elif analysis_type == AnalysisType.UX_CRITIQUE:
                agent_states["UX Critique Agent"] = "pending"
            elif analysis_type == AnalysisType.MARKET_RESEARCH:
                agent_states["Market Research Agent"] = "pending"
        
        # Initialize state
        initial_state = AnalysisState(
            analysis_id=analysis_id,
            image=image,
            context=context,
            requested_analyses=requested_analyses,
            api_keys=api_keys,
            visual_result=None,
            ux_result=None,
            market_result=None,
            agent_states=agent_states,
            all_findings=[],
            overall_score=0.0,
            start_time=datetime.now(),
            current_stage="initializing",
            error_message=None
        )
        
        # Store progress callback in state for use by agent methods
        initial_state["progress_callback"] = progress_callback
        
        try:
            logger.info(f"Starting analysis {analysis_id} with agents: {requested_analyses}")
            
            # Create agents with API keys
            self._get_or_create_agents(api_keys)
            
            # Initialize global state tracking
            update_analysis_agent_states(analysis_id, agent_states)
            
            # Run the workflow
            final_state = await self.app.ainvoke(initial_state)
            
            # Ensure final agent states are updated
            if final_state.get("agent_states"):
                update_analysis_agent_states(analysis_id, final_state["agent_states"])
            
            # Create the final result
            result = self._create_analysis_result(final_state)
            
            logger.info(f"Analysis {analysis_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Analysis {analysis_id} failed: {e}")
            
            # Return error result
            return AnalysisResult(
                analysis_id=analysis_id,
                upload_filename="unknown",
                context=context,
                agent_results=[],
                overall_score=0.0,
                key_strengths=[],
                priority_improvements=[],
                actionable_recommendations=[],
                analysis_timestamp=datetime.now().isoformat(),
                processing_time=0.0,
                confidence_level=0.0
            )
    
    async def _initialize_analysis(self, state: AnalysisState) -> AnalysisState:
        """Initialize the analysis workflow."""
        logger.info(f"Initializing analysis {state['analysis_id']}")
        
        state["current_stage"] = "initialized"
        
        # Update agent states to show initialization
        for agent_name in state["agent_states"]:
            state["agent_states"][agent_name] = "pending"
        update_analysis_agent_states(state["analysis_id"], state["agent_states"])
        
        # Call progress callback for initialization
        if state.get("progress_callback"):
            state["progress_callback"](0.1, "Analysis initialized", state["agent_states"])
        
        # Add a small delay to ensure initialization is visible
        await asyncio.sleep(1)
        
        logger.info(f"Analysis {state['analysis_id']} initialized with agents: {list(state['agent_states'].keys())}")
        
        return state
    
    def _route_analyses(self, state: AnalysisState) -> str:
        """Route to appropriate analysis path based on requested analyses."""
        requested = state["requested_analyses"]
        
        if not requested:
            return "synthesize"
        
        if len(requested) == 1:
            if AnalysisType.VISUAL in requested:
                return "visual_only"
            elif AnalysisType.UX_CRITIQUE in requested:
                return "ux_only"
            elif AnalysisType.MARKET_RESEARCH in requested:
                return "market_only"
        
        return "parallel"
    
    def _check_next_analysis(self, state: AnalysisState) -> str:
        """Determine the next analysis to run."""
        requested = state["requested_analyses"]
        
        # Check what's been completed and what's still needed
        if AnalysisType.UX_CRITIQUE in requested and state["ux_result"] is None:
            return "ux_analysis"
        elif AnalysisType.MARKET_RESEARCH in requested and state["market_result"] is None:
            return "market_analysis"
        else:
            return "synthesize"
    
    async def _run_visual_analysis(self, state: AnalysisState) -> AnalysisState:
        """Run visual analysis agent."""
        if AnalysisType.VISUAL not in state["requested_analyses"]:
            return state
        
        try:
            logger.info(f"Running visual analysis for {state['analysis_id']}")
            state["current_stage"] = "visual_analysis"
            state["agent_states"]["Visual Analysis Agent"] = "running"
            update_analysis_agent_states(state["analysis_id"], state["agent_states"])
            
            # Call progress callback if available
            if state.get("progress_callback"):
                state["progress_callback"](0.4, "Running visual analysis...", state["agent_states"])
            
            # Add a small delay to ensure status is visible
            await asyncio.sleep(1)
            
            context_dict = state["context"].dict() if state["context"] else None
            result = await self.visual_agent._execute_analysis(state["image"], context_dict)
            
            state["visual_result"] = result
            state["all_findings"].extend(result.findings)
            state["agent_states"]["Visual Analysis Agent"] = "completed"
            update_analysis_agent_states(state["analysis_id"], state["agent_states"])
            
            # Call progress callback for completion
            if state.get("progress_callback"):
                state["progress_callback"](0.5, "Visual analysis completed", state["agent_states"])
            
            # Add a small delay to ensure completion status is visible
            await asyncio.sleep(0.5)
            
            logger.info(f"Visual analysis completed for {state['analysis_id']}")
            
        except Exception as e:
            logger.error(f"Visual analysis failed for {state['analysis_id']}: {e}")
            state["error_message"] = f"Visual analysis failed: {str(e)}"
            state["agent_states"]["Visual Analysis Agent"] = "failed"
            update_analysis_agent_states(state["analysis_id"], state["agent_states"])
            
            # Call progress callback for failure
            if state.get("progress_callback"):
                state["progress_callback"](0.5, "Visual analysis failed", state["agent_states"])
        
        return state
    
    async def _run_ux_analysis(self, state: AnalysisState) -> AnalysisState:
        """Run UX critique agent."""
        if AnalysisType.UX_CRITIQUE not in state["requested_analyses"]:
            return state
        
        try:
            logger.info(f"Running UX analysis for {state['analysis_id']}")
            state["current_stage"] = "ux_analysis"
            state["agent_states"]["UX Critique Agent"] = "running"
            update_analysis_agent_states(state["analysis_id"], state["agent_states"])
            
            # Call progress callback if available
            if state.get("progress_callback"):
                state["progress_callback"](0.6, "Running UX analysis...", state["agent_states"])
            
            # Add a small delay to ensure status is visible
            await asyncio.sleep(1)
            
            context_dict = state["context"].dict() if state["context"] else None
            result = await self.ux_agent._execute_analysis(state["image"], context_dict)
            
            state["ux_result"] = result
            state["all_findings"].extend(result.findings)
            state["agent_states"]["UX Critique Agent"] = "completed"
            update_analysis_agent_states(state["analysis_id"], state["agent_states"])
            
            # Call progress callback for completion
            if state.get("progress_callback"):
                state["progress_callback"](0.7, "UX analysis completed", state["agent_states"])
            
            # Add a small delay to ensure completion status is visible
            await asyncio.sleep(0.5)
            
            logger.info(f"UX analysis completed for {state['analysis_id']}")
            
        except Exception as e:
            logger.error(f"UX analysis failed for {state['analysis_id']}: {e}")
            state["error_message"] = f"UX analysis failed: {str(e)}"
            state["agent_states"]["UX Critique Agent"] = "failed"
            update_analysis_agent_states(state["analysis_id"], state["agent_states"])
            
            # Call progress callback for failure
            if state.get("progress_callback"):
                state["progress_callback"](0.7, "UX analysis failed", state["agent_states"])
        
        return state
    
    async def _run_market_analysis(self, state: AnalysisState) -> AnalysisState:
        """Run market research agent."""
        if AnalysisType.MARKET_RESEARCH not in state["requested_analyses"]:
            return state
        
        try:
            logger.info(f"Running market analysis for {state['analysis_id']}")
            state["current_stage"] = "market_analysis"
            state["agent_states"]["Market Research Agent"] = "running"
            update_analysis_agent_states(state["analysis_id"], state["agent_states"])
            
            # Call progress callback if available
            if state.get("progress_callback"):
                state["progress_callback"](0.8, "Running market analysis...", state["agent_states"])
            
            # Add a small delay to ensure status is visible
            await asyncio.sleep(1)
            
            context_dict = state["context"].dict() if state["context"] else None
            result = await self.market_agent._execute_analysis(state["image"], context_dict)
            
            state["market_result"] = result
            state["all_findings"].extend(result.findings)
            state["agent_states"]["Market Research Agent"] = "completed"
            update_analysis_agent_states(state["analysis_id"], state["agent_states"])
            
            # Call progress callback for completion
            if state.get("progress_callback"):
                state["progress_callback"](0.9, "Market analysis completed", state["agent_states"])
            
            # Add a small delay to ensure completion status is visible
            await asyncio.sleep(0.5)
            
            logger.info(f"Market analysis completed for {state['analysis_id']}")
            
        except Exception as e:
            logger.error(f"Market analysis failed for {state['analysis_id']}: {e}")
            state["error_message"] = f"Market analysis failed: {str(e)}"
            state["agent_states"]["Market Research Agent"] = "failed"
            update_analysis_agent_states(state["analysis_id"], state["agent_states"])
            
            # Call progress callback for failure
            if state.get("progress_callback"):
                state["progress_callback"](0.9, "Market analysis failed", state["agent_states"])
        
        return state
    
    async def _synthesize_results(self, state: AnalysisState) -> AnalysisState:
        """Synthesize results from all agents."""
        try:
            logger.info(f"Synthesizing results for {state['analysis_id']}")
            state["current_stage"] = "synthesizing"
            
            # Calculate overall score
            agent_results = [
                result for result in [
                    state["visual_result"],
                    state["ux_result"], 
                    state["market_result"]
                ] if result is not None
            ]
            
            if agent_results:
                state["overall_score"] = sum(r.overall_score for r in agent_results) / len(agent_results)
            else:
                state["overall_score"] = 0.0
            
            state["current_stage"] = "completed"
            logger.info(f"Synthesis completed for {state['analysis_id']}")
            
        except Exception as e:
            logger.error(f"Synthesis failed for {state['analysis_id']}: {e}")
            state["error_message"] = f"Synthesis failed: {str(e)}"
            state["current_stage"] = "failed"
        
        return state
    
    def _create_analysis_result(self, state: AnalysisState) -> AnalysisResult:
        """Create the final analysis result from the workflow state."""
        
        # Collect agent results
        agent_results = []
        for result in [state["visual_result"], state["ux_result"], state["market_result"]]:
            if result is not None:
                agent_results.append(result)
        
        # Extract metrics
        visual_metrics = None
        ux_metrics = None
        accessibility_metrics = None
        market_comparison = None
        
        if state["visual_result"]:
            visual_metrics_data = state["visual_result"].metadata.get("visual_metrics")
            if visual_metrics_data:
                visual_metrics = VisualAnalysisMetrics(**visual_metrics_data)
        
        if state["ux_result"]:
            ux_metrics_data = state["ux_result"].metadata.get("ux_metrics")
            if ux_metrics_data:
                ux_metrics = UXMetrics(**ux_metrics_data)
        
        if state["market_result"]:
            market_data = state["market_result"].metadata.get("market_comparison")
            if market_data:
                market_comparison = MarketComparison(**market_data)
        
        # Identify key strengths and priority improvements
        key_strengths = []
        priority_improvements = []
        
        for finding in state["all_findings"]:
            if finding.severity in ["low"] and finding.confidence_score > 0.7:
                if "excellent" in finding.title.lower() or "good" in finding.title.lower():
                    key_strengths.append(finding.title)
            elif finding.severity in ["high", "critical"] and finding.confidence_score > 0.6:
                priority_improvements.append(finding)
        
        # Generate actionable recommendations
        actionable_recommendations = []
        high_confidence_findings = [
            f for f in state["all_findings"] 
            if f.confidence_score > 0.7 and f.severity in ["medium", "high", "critical"]
        ]
        
        for finding in high_confidence_findings[:5]:  # Top 5 recommendations
            actionable_recommendations.append(finding.recommendation)
        
        # Calculate processing time
        processing_time = (datetime.now() - state["start_time"]).total_seconds()
        
        # Calculate confidence level
        if agent_results:
            avg_confidence = sum(
                sum(f.confidence_score for f in result.findings) / len(result.findings)
                if result.findings else 0.5
                for result in agent_results
            ) / len(agent_results)
        else:
            avg_confidence = 0.0
        
        return AnalysisResult(
            analysis_id=state["analysis_id"],
            upload_filename="uploaded_design.png",  # This would be set by the API
            context=state["context"],
            agent_results=agent_results,
            overall_score=state["overall_score"],
            visual_metrics=visual_metrics,
            ux_metrics=ux_metrics,
            accessibility_metrics=accessibility_metrics,
            market_comparison=market_comparison,
            key_strengths=key_strengths,
            priority_improvements=priority_improvements,
            actionable_recommendations=actionable_recommendations,
            analysis_timestamp=datetime.now().isoformat(),
            processing_time=processing_time,
            confidence_level=avg_confidence
        )


# Global orchestrator instance and state tracking
_orchestrator_instance: Optional[DesignAnalysisOrchestrator] = None
_analysis_states: Dict[str, Dict[str, str]] = {}  # analysis_id -> agent_states


def get_orchestrator() -> DesignAnalysisOrchestrator:
    """Get or create the global orchestrator instance."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = DesignAnalysisOrchestrator()
    return _orchestrator_instance


def get_analysis_agent_states(analysis_id: str) -> Optional[Dict[str, str]]:
    """Get the current agent states for an analysis."""
    return _analysis_states.get(analysis_id)


def update_analysis_agent_states(analysis_id: str, agent_states: Dict[str, str]):
    """Update the agent states for an analysis."""
    logger.info(f"[STATE UPDATE] Analysis {analysis_id}: {agent_states}")
    _analysis_states[analysis_id] = agent_states.copy()
    logger.info(f"[GLOBAL STATE] All analyses: {_analysis_states}")
    
    # Log individual agent status changes
    for agent_name, status in agent_states.items():
        logger.info(f"[AGENT STATUS] {analysis_id} - {agent_name}: {status}")


def cleanup_analysis_states(analysis_id: str):
    """Clean up agent states for a completed analysis."""
    if analysis_id in _analysis_states:
        del _analysis_states[analysis_id]
