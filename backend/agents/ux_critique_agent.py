"""UX Critique Agent for usability and user experience analysis."""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple

from .base_agent import BaseAgent
from ..models.schemas import Finding, AgentResult, AnalysisType, UXMetrics
from langchain_core.prompts import ChatPromptTemplate


class UXCritiqueAgent(BaseAgent):
    """Agent for analyzing UX and usability aspects."""
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        super().__init__("UX Experience Critic", AnalysisType.UX_CRITIQUE, api_keys)
        
        # UX heuristics and patterns
        self.usability_heuristics = [
            "visibility_of_system_status",
            "match_system_real_world",
            "user_control_freedom",
            "consistency_standards",
            "error_prevention",
            "recognition_vs_recall",
            "flexibility_efficiency",
            "aesthetic_minimalist_design",
            "help_users_recognize_errors",
            "help_documentation"
        ]
        
        # LLM Prompt Templates
        self.ux_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a UX expert specializing in usability analysis and user experience design. Analyze the provided UX metrics and design context to provide comprehensive usability insights.

Your analysis should focus on:
- Nielsen's 10 usability heuristics
- User flow and task efficiency
- Information architecture and findability
- Navigation patterns and clarity
- Accessibility and inclusive design
- Cognitive load and mental models

Provide specific, actionable recommendations based on UX best practices."""),
            ("human", """Image Description: {image_description}

UX Analysis Metrics:
- Usability Score: {usability_score}/10
- Navigation Clarity: {navigation_score}/10
- Information Architecture: {ia_score}/10
- User Flow Efficiency: {flow_score}/10

Context: {context}
Target Users: {target_users}
Platform: {platform}

Please provide detailed UX critique and recommendations based on these metrics and context.""")
        ])
        
        self.heuristic_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a usability expert specializing in Nielsen's 10 usability heuristics. Evaluate the design against these principles:

1. Visibility of system status
2. Match between system and real world
3. User control and freedom
4. Consistency and standards
5. Error prevention
6. Recognition rather than recall
7. Flexibility and efficiency of use
8. Aesthetic and minimalist design
9. Help users recognize, diagnose, and recover from errors
10. Help and documentation

Provide specific examples and actionable improvements."""),
            ("human", """Design Context: {context}
Usability Findings: {usability_findings}
Platform: {platform}

Analyze this design against Nielsen's heuristics and provide detailed recommendations.""")
        ])
    
    async def analyze(
        self,
        image: Image.Image,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """Perform UX critique analysis on the design."""
        findings = []
        
        # Convert PIL image to numpy array
        image_array = np.array(image)
        
        # Perform UX analyses
        usability_score, usability_findings = self._analyze_usability_heuristics(image_array, context)
        navigation_score, navigation_findings = self._analyze_navigation_clarity(image_array)
        ia_score, ia_findings = self._analyze_information_architecture(image_array)
        flow_score, flow_findings = self._analyze_user_flow_efficiency(image_array, context)
        
        # Combine all findings
        findings.extend(usability_findings)
        findings.extend(navigation_findings)
        findings.extend(ia_findings)
        findings.extend(flow_findings)
        
        # Generate LLM-powered UX insights
        llm_insights = await self._generate_llm_ux_insights(
            image, context, usability_score, navigation_score, ia_score, flow_score
        )
        
        # Add LLM insights to findings if available
        if llm_insights:
            findings.append(self._create_finding(
                category="llm_insights",
                severity="low",
                title="AI-Powered UX Analysis",
                description=llm_insights,
                recommendation="Consider the AI UX analysis alongside technical metrics",
                confidence_score=0.8,
                supporting_evidence=["LLM UX Analysis"]
            ))
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(findings)
        
        # Create UX metrics
        ux_metrics = UXMetrics(
            usability_score=usability_score,
            navigation_clarity=navigation_score,
            information_architecture=ia_score,
            user_flow_efficiency=flow_score
        )
        
        return AgentResult(
            agent_name=self.agent_name,
            analysis_type=self.analysis_type,
            findings=findings,
            overall_score=overall_score,
            execution_time=0.0,  # Will be set by base class
            metadata={
                "ux_metrics": ux_metrics.dict(),
                "llm_insights": llm_insights
            }
        )
    
    def _analyze_usability_heuristics(
        self, 
        image_array: np.ndarray, 
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, List[Finding]]:
        """Analyze based on Nielsen's usability heuristics."""
        findings = []
        
        try:
            # Get similar patterns for reference
            similar_patterns = self._get_similar_patterns(
                Image.fromarray(image_array),
                category_filter="usability",
                n_results=3
            )
            
            # Analyze interface elements
            elements_score = self._detect_ui_elements(image_array)
            
            # Check for common usability issues
            consistency_score = self._check_consistency(image_array)
            feedback_score = self._check_feedback_elements(image_array)
            error_prevention_score = self._check_error_prevention(image_array)
            
            usability_score = (elements_score + consistency_score + feedback_score + error_prevention_score) / 4
            
            # Generate findings based on analysis
            if consistency_score < 6.0:
                findings.append(self._create_finding(
                    category="usability",
                    severity="medium",
                    title="Consistency Issues Detected",
                    description="Interface elements may lack visual or functional consistency",
                    recommendation="Ensure consistent styling, spacing, and behavior across similar elements",
                    confidence_score=0.7,
                    supporting_evidence=[f"Found {len(similar_patterns)} similar patterns for reference"]
                ))
            
            if feedback_score < 5.0:
                findings.append(self._create_finding(
                    category="usability",
                    severity="high",
                    title="Insufficient User Feedback",
                    description="Interface may not provide adequate feedback for user actions",
                    recommendation="Add visual feedback for interactive elements (hover states, loading indicators, etc.)",
                    confidence_score=0.8
                ))
            
            if error_prevention_score < 6.0:
                findings.append(self._create_finding(
                    category="usability",
                    severity="medium",
                    title="Error Prevention Opportunities",
                    description="Interface could better prevent user errors",
                    recommendation="Add input validation, confirmation dialogs, and clear constraints",
                    confidence_score=0.6
                ))
            
            # Check against best practices from similar patterns
            if similar_patterns:
                best_practices_score = self._compare_with_best_practices(similar_patterns)
                if best_practices_score < 7.0:
                    findings.append(self._create_finding(
                        category="best_practices",
                        severity="low",
                        title="Deviation from Best Practices",
                        description="Design differs from established patterns in this category",
                        recommendation="Consider adopting proven patterns from similar successful interfaces",
                        confidence_score=0.6,
                        supporting_evidence=[p["description"] for p in similar_patterns[:2]]
                    ))
            
            return usability_score, findings
            
        except Exception as e:
            self.logger.error(f"Usability heuristics analysis failed: {e}")
            return 5.0, []
    
    def _analyze_navigation_clarity(self, image_array: np.ndarray) -> Tuple[float, List[Finding]]:
        """Analyze navigation clarity and structure."""
        findings = []
        
        try:
            # Search for navigation patterns
            nav_patterns = self._search_patterns_by_text("navigation menu header")
            
            # Detect potential navigation elements
            nav_score = self._detect_navigation_elements(image_array)
            
            # Check for navigation hierarchy
            hierarchy_score = self._analyze_navigation_hierarchy(image_array)
            
            navigation_score = (nav_score + hierarchy_score) / 2
            
            if navigation_score < 6.0:
                findings.append(self._create_finding(
                    category="navigation",
                    severity="medium",
                    title="Navigation Clarity Issues",
                    description="Navigation structure may be unclear or hard to find",
                    recommendation="Ensure navigation is prominently placed and clearly structured",
                    confidence_score=0.7
                ))
            
            if hierarchy_score < 5.0:
                findings.append(self._create_finding(
                    category="navigation",
                    severity="high",
                    title="Poor Navigation Hierarchy",
                    description="Navigation lacks clear hierarchical structure",
                    recommendation="Organize navigation items in logical groups with clear hierarchy",
                    confidence_score=0.8
                ))
            
            # Compare with navigation best practices
            if nav_patterns:
                findings.append(self._create_finding(
                    category="navigation",
                    severity="low",
                    title="Navigation Pattern Reference",
                    description="Consider these proven navigation patterns",
                    recommendation="Review successful navigation implementations for inspiration",
                    confidence_score=0.5,
                    supporting_evidence=[p["description"] for p in nav_patterns[:2]]
                ))
            
            return navigation_score, findings
            
        except Exception as e:
            self.logger.error(f"Navigation analysis failed: {e}")
            return 5.0, []
    
    def _analyze_information_architecture(self, image_array: np.ndarray) -> Tuple[float, List[Finding]]:
        """Analyze information architecture and content organization."""
        findings = []
        
        try:
            # Analyze content grouping
            grouping_score = self._analyze_content_grouping(image_array)
            
            # Check for clear content hierarchy
            content_hierarchy_score = self._analyze_content_hierarchy(image_array)
            
            # Analyze scanning patterns (F-pattern, Z-pattern)
            scanning_score = self._analyze_scanning_patterns(image_array)
            
            ia_score = (grouping_score + content_hierarchy_score + scanning_score) / 3
            
            if grouping_score < 6.0:
                findings.append(self._create_finding(
                    category="information_architecture",
                    severity="medium",
                    title="Content Grouping Issues",
                    description="Related content may not be properly grouped together",
                    recommendation="Group related information using proximity, borders, or background colors",
                    confidence_score=0.7
                ))
            
            if content_hierarchy_score < 5.0:
                findings.append(self._create_finding(
                    category="information_architecture",
                    severity="high",
                    title="Unclear Content Hierarchy",
                    description="Content lacks clear hierarchical organization",
                    recommendation="Use typography, spacing, and visual weight to create clear content hierarchy",
                    confidence_score=0.8
                ))
            
            if scanning_score < 6.0:
                findings.append(self._create_finding(
                    category="information_architecture",
                    severity="medium",
                    title="Poor Scanning Pattern",
                    description="Layout doesn't support natural reading/scanning patterns",
                    recommendation="Organize content to support F-pattern or Z-pattern scanning",
                    confidence_score=0.6
                ))
            
            return ia_score, findings
            
        except Exception as e:
            self.logger.error(f"Information architecture analysis failed: {e}")
            return 5.0, []
    
    def _analyze_user_flow_efficiency(
        self, 
        image_array: np.ndarray, 
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, List[Finding]]:
        """Analyze user flow and task efficiency."""
        findings = []
        
        try:
            # Detect interactive elements
            interactive_score = self._detect_interactive_elements(image_array)
            
            # Analyze call-to-action prominence
            cta_score = self._analyze_cta_prominence(image_array)
            
            # Check for form usability (if forms are present)
            form_score = self._analyze_form_usability(image_array)
            
            flow_score = (interactive_score + cta_score + form_score) / 3
            
            if interactive_score < 6.0:
                findings.append(self._create_finding(
                    category="user_flow",
                    severity="medium",
                    title="Interactive Elements Unclear",
                    description="Interactive elements may not be clearly identifiable",
                    recommendation="Make interactive elements more obvious through visual cues",
                    confidence_score=0.7
                ))
            
            if cta_score < 5.0:
                findings.append(self._create_finding(
                    category="user_flow",
                    severity="high",
                    title="Weak Call-to-Action",
                    description="Primary actions are not prominent enough",
                    recommendation="Make primary CTAs more prominent with color, size, and positioning",
                    confidence_score=0.8
                ))
            
            if form_score < 6.0 and form_score > 0:  # Only if forms detected
                findings.append(self._create_finding(
                    category="user_flow",
                    severity="medium",
                    title="Form Usability Issues",
                    description="Forms may have usability problems",
                    recommendation="Improve form layout, labeling, and validation feedback",
                    confidence_score=0.6
                ))
            
            # Context-specific analysis
            if context and context.get("platform") == "mobile":
                mobile_score = self._analyze_mobile_usability(image_array)
                if mobile_score < 6.0:
                    findings.append(self._create_finding(
                        category="mobile_usability",
                        severity="medium",
                        title="Mobile Usability Issues",
                        description="Interface may not be optimized for mobile use",
                        recommendation="Ensure touch targets are at least 44px, optimize for thumb navigation",
                        confidence_score=0.7
                    ))
            
            return flow_score, findings
            
        except Exception as e:
            self.logger.error(f"User flow analysis failed: {e}")
            return 5.0, []
    
    def _detect_ui_elements(self, image_array: np.ndarray) -> float:
        """Detect and analyze UI elements."""
        try:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Detect rectangular elements (buttons, cards, etc.)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Count potential UI elements
            ui_elements = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Filter small noise
                    # Check if it's roughly rectangular
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                    if len(approx) >= 4:
                        ui_elements += 1
            
            # Score based on reasonable number of elements
            if ui_elements == 0:
                return 3.0
            elif ui_elements < 5:
                return 6.0
            elif ui_elements < 15:
                return 8.0
            else:
                return 6.0  # Too many elements might be cluttered
                
        except Exception as e:
            self.logger.error(f"UI elements detection failed: {e}")
            return 5.0
    
    def _check_consistency(self, image_array: np.ndarray) -> float:
        """Check visual consistency across elements."""
        try:
            # This is a simplified consistency check
            # In a real implementation, you'd analyze color consistency,
            # spacing patterns, typography consistency, etc.
            
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Analyze color distribution consistency
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            # Check for consistent spacing patterns
            edges = cv2.Canny(gray, 50, 150)
            horizontal_projection = np.sum(edges, axis=1)
            vertical_projection = np.sum(edges, axis=0)
            
            # Analyze spacing consistency (simplified)
            h_peaks = np.where(horizontal_projection > np.mean(horizontal_projection))[0]
            v_peaks = np.where(vertical_projection > np.mean(vertical_projection))[0]
            
            if len(h_peaks) > 1:
                h_spacing = np.diff(h_peaks)
                h_consistency = 1.0 - (np.std(h_spacing) / np.mean(h_spacing)) if np.mean(h_spacing) > 0 else 0
            else:
                h_consistency = 1.0
            
            if len(v_peaks) > 1:
                v_spacing = np.diff(v_peaks)
                v_consistency = 1.0 - (np.std(v_spacing) / np.mean(v_spacing)) if np.mean(v_spacing) > 0 else 0
            else:
                v_consistency = 1.0
            
            consistency_score = (h_consistency + v_consistency) * 5
            return max(0.0, min(10.0, consistency_score))
            
        except Exception as e:
            self.logger.error(f"Consistency check failed: {e}")
            return 5.0
    
    def _check_feedback_elements(self, image_array: np.ndarray) -> float:
        """Check for user feedback elements."""
        # This is a placeholder implementation
        # In reality, you'd look for loading indicators, hover states,
        # progress bars, status messages, etc.
        return 6.0
    
    def _check_error_prevention(self, image_array: np.ndarray) -> float:
        """Check for error prevention mechanisms."""
        # This is a placeholder implementation
        # In reality, you'd look for form validation, confirmation dialogs,
        # input constraints, etc.
        return 6.0
    
    def _compare_with_best_practices(self, similar_patterns: List[Dict[str, Any]]) -> float:
        """Compare design with best practices from similar patterns."""
        if not similar_patterns:
            return 7.0
        
        # Calculate average usability score from similar patterns
        scores = []
        for pattern in similar_patterns:
            if "usability_score" in pattern.get("metadata", {}):
                scores.append(pattern["metadata"]["usability_score"])
        
        if scores:
            avg_score = np.mean(scores)
            # Return a score based on how well it compares
            return min(10.0, avg_score)
        
        return 7.0
    
    def _detect_navigation_elements(self, image_array: np.ndarray) -> float:
        """Detect navigation elements in the interface."""
        # Simplified navigation detection
        # Look for horizontal bars at the top, menu icons, etc.
        height, width = image_array.shape[:2]
        
        # Check top area for navigation
        top_area = image_array[:int(height * 0.15), :]
        gray_top = cv2.cvtColor(top_area, cv2.COLOR_RGB2GRAY)
        
        # Look for horizontal structures
        edges = cv2.Canny(gray_top, 50, 150)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        
        nav_score = min(10.0, np.sum(horizontal_lines) / 1000)
        return nav_score
    
    def _analyze_navigation_hierarchy(self, image_array: np.ndarray) -> float:
        """Analyze navigation hierarchy."""
        # Simplified hierarchy analysis
        return 6.0
    
    def _analyze_content_grouping(self, image_array: np.ndarray) -> float:
        """Analyze how content is grouped."""
        # Simplified content grouping analysis
        return 6.0
    
    def _analyze_content_hierarchy(self, image_array: np.ndarray) -> float:
        """Analyze content hierarchy."""
        # Simplified content hierarchy analysis
        return 6.0
    
    def _analyze_scanning_patterns(self, image_array: np.ndarray) -> float:
        """Analyze support for natural scanning patterns."""
        # Simplified scanning pattern analysis
        return 6.0
    
    def _detect_interactive_elements(self, image_array: np.ndarray) -> float:
        """Detect interactive elements."""
        # Simplified interactive elements detection
        return 6.0
    
    def _analyze_cta_prominence(self, image_array: np.ndarray) -> float:
        """Analyze call-to-action prominence."""
        # Simplified CTA analysis
        return 6.0
    
    def _analyze_form_usability(self, image_array: np.ndarray) -> float:
        """Analyze form usability if forms are present."""
        # Simplified form analysis
        return 6.0
    
    def _analyze_mobile_usability(self, image_array: np.ndarray) -> float:
        """Analyze mobile-specific usability."""
        # Simplified mobile usability analysis
        return 6.0
    
    async def _generate_llm_ux_insights(
        self,
        image: Image.Image,
        context: Optional[Dict[str, Any]],
        usability_score: float,
        navigation_score: float,
        ia_score: float,
        flow_score: float
    ) -> Optional[str]:
        """Generate LLM-powered UX insights."""
        try:
            # Create image description
            image_description = self._create_image_description(image, context)
            
            # Extract context information
            context_str = str(context) if context else "No specific context provided"
            target_users = context.get("target_audience", "General users") if context else "General users"
            platform = context.get("platform", "Web") if context else "Web"
            
            # Generate comprehensive UX insights
            insights = await self._generate_llm_insights(
                self.ux_analysis_prompt,
                {
                    "image_description": image_description,
                    "usability_score": usability_score,
                    "navigation_score": navigation_score,
                    "ia_score": ia_score,
                    "flow_score": flow_score,
                    "context": context_str,
                    "target_users": target_users,
                    "platform": platform
                },
                temperature=0.7
            )
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to generate LLM UX insights: {e}")
            return None
    
    async def _generate_heuristic_analysis(
        self,
        context: Optional[Dict[str, Any]],
        usability_findings: List[str]
    ) -> Optional[str]:
        """Generate LLM-powered heuristic analysis."""
        try:
            context_str = str(context) if context else "No specific context provided"
            platform = context.get("platform", "Web") if context else "Web"
            findings_text = "; ".join(usability_findings) if usability_findings else "No specific findings"
            
            insights = await self._generate_llm_insights(
                self.heuristic_analysis_prompt,
                {
                    "context": context_str,
                    "usability_findings": findings_text,
                    "platform": platform
                },
                temperature=0.6
            )
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to generate heuristic analysis: {e}")
            return None
