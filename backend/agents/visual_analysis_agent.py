"""Visual Analysis Agent for UI/UX design evaluation."""

import cv2
import numpy as np
from PIL import Image, ImageStat
from typing import Dict, Any, List, Optional, Tuple
import colorsys
from collections import Counter

from .base_agent import BaseAgent
from ..models.schemas import Finding, AgentResult, AnalysisType, VisualAnalysisMetrics
from langchain_core.prompts import ChatPromptTemplate


class VisualAnalysisAgent(BaseAgent):
    """Agent for analyzing visual design aspects."""
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        super().__init__("Visual Analysis Agent", AnalysisType.VISUAL, api_keys)
        
        # LLM Prompt Templates
        self.visual_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert UI/UX visual design analyst. Analyze the provided design metrics and image description to provide detailed visual design insights.

Your analysis should focus on:
- Layout structure and grid systems
- Visual hierarchy and information flow
- Color harmony and accessibility
- Typography and readability
- Whitespace usage and balance

Provide specific, actionable recommendations based on visual design principles."""),
            ("human", """Image Description: {image_description}

Visual Analysis Metrics:
- Layout Score: {layout_score}/10
- Visual Hierarchy Score: {hierarchy_score}/10  
- Color Harmony Score: {color_score}/10
- Typography Score: {typography_score}/10
- Whitespace Usage Score: {whitespace_score}/10

Context: {context}

Please provide detailed visual design insights and recommendations based on these metrics and the image description.""")
        ])
        
        self.color_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a color theory expert specializing in UI/UX design. Analyze the color palette and provide insights on color harmony, accessibility, and psychological impact.

Focus on:
- Color harmony principles (complementary, triadic, analogous)
- Accessibility and contrast ratios
- Brand alignment and emotional impact
- Cultural considerations"""),
            ("human", """Dominant Colors: {dominant_colors}
Color Harmony Score: {harmony_score}/10
Contrast Score: {contrast_score}/10
Context: {context}

Provide detailed color analysis and recommendations.""")
        ])
    
    async def analyze(
        self,
        image: Image.Image,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """Perform visual analysis on the design."""
        findings = []
        
        # Convert PIL image to numpy array for OpenCV operations
        image_array = np.array(image)
        
        # Perform various visual analyses
        layout_score, layout_findings = self._analyze_layout(image_array)
        hierarchy_score, hierarchy_findings = self._analyze_visual_hierarchy(image_array)
        color_score, color_findings = self._analyze_color_harmony(image)
        typography_score, typography_findings = self._analyze_typography(image_array)
        whitespace_score, whitespace_findings = self._analyze_whitespace(image_array)
        
        # Combine all findings
        findings.extend(layout_findings)
        findings.extend(hierarchy_findings)
        findings.extend(color_findings)
        findings.extend(typography_findings)
        findings.extend(whitespace_findings)
        
        # Generate LLM-powered insights
        llm_insights = await self._generate_llm_visual_insights(
            image, context, layout_score, hierarchy_score, color_score, 
            typography_score, whitespace_score
        )
        
        # Add LLM insights to findings if available
        if llm_insights:
            findings.append(self._create_finding(
                category="llm_insights",
                severity="low",
                title="AI-Powered Visual Analysis",
                description=llm_insights,
                recommendation="Consider the AI analysis alongside technical metrics",
                confidence_score=0.8,
                supporting_evidence=["LLM Analysis"]
            ))
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(findings)
        
        # Create visual metrics
        visual_metrics = VisualAnalysisMetrics(
            layout_score=layout_score,
            visual_hierarchy_score=hierarchy_score,
            color_harmony_score=color_score,
            typography_score=typography_score,
            whitespace_usage_score=whitespace_score
        )
        
        return AgentResult(
            agent_name=self.agent_name,
            analysis_type=self.analysis_type,
            findings=findings,
            overall_score=overall_score,
            execution_time=0.0,  # Will be set by base class
            metadata={
                "visual_metrics": visual_metrics.dict(),
                "llm_insights": llm_insights
            }
        )
    
    def _analyze_layout(self, image_array: np.ndarray) -> Tuple[float, List[Finding]]:
        """Analyze layout structure and grid alignment."""
        findings = []
        
        try:
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours to analyze layout structure
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze alignment and grid structure
            alignment_score = self._calculate_alignment_score(contours, image_array.shape)
            
            # Check for proper margins
            margin_score = self._check_margins(edges)
            
            # Overall layout score
            layout_score = (alignment_score + margin_score) / 2
            
            if layout_score < 6.0:
                findings.append(self._create_finding(
                    category="layout",
                    severity="medium" if layout_score > 4.0 else "high",
                    title="Layout Structure Issues",
                    description=f"Layout alignment and structure could be improved (score: {layout_score:.1f}/10)",
                    recommendation="Consider using a grid system for better alignment and visual structure",
                    confidence_score=0.8
                ))
            elif layout_score > 8.0:
                findings.append(self._create_finding(
                    category="layout",
                    severity="low",
                    title="Excellent Layout Structure",
                    description="Layout demonstrates good alignment and grid structure",
                    recommendation="Maintain current layout principles",
                    confidence_score=0.9
                ))
            
            return layout_score, findings
            
        except Exception as e:
            self.logger.error(f"Layout analysis failed: {e}")
            return 5.0, []
    
    def _analyze_visual_hierarchy(self, image_array: np.ndarray) -> Tuple[float, List[Finding]]:
        """Analyze visual hierarchy and focal points."""
        findings = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Calculate contrast variations
            contrast_map = cv2.Laplacian(gray, cv2.CV_64F)
            contrast_variance = np.var(contrast_map)
            
            # Analyze size variations (using contours)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                areas = [cv2.contourArea(contour) for contour in contours if cv2.contourArea(contour) > 100]
                size_variance = np.var(areas) if areas else 0
            else:
                size_variance = 0
            
            # Calculate hierarchy score based on contrast and size variations
            hierarchy_score = min(10.0, (contrast_variance / 1000 + size_variance / 10000) * 2)
            
            if hierarchy_score < 5.0:
                findings.append(self._create_finding(
                    category="visual_hierarchy",
                    severity="medium",
                    title="Weak Visual Hierarchy",
                    description="The design lacks clear visual hierarchy and focal points",
                    recommendation="Use size, color, and contrast to create clear information hierarchy",
                    confidence_score=0.7
                ))
            elif hierarchy_score > 7.5:
                findings.append(self._create_finding(
                    category="visual_hierarchy",
                    severity="low",
                    title="Strong Visual Hierarchy",
                    description="Design demonstrates clear visual hierarchy",
                    recommendation="Continue using effective hierarchy principles",
                    confidence_score=0.8
                ))
            
            return hierarchy_score, findings
            
        except Exception as e:
            self.logger.error(f"Visual hierarchy analysis failed: {e}")
            return 5.0, []
    
    def _analyze_color_harmony(self, image: Image.Image) -> Tuple[float, List[Finding]]:
        """Analyze color palette and harmony."""
        findings = []
        
        try:
            # Get dominant colors
            colors = self._extract_dominant_colors(image, num_colors=8)
            
            # Analyze color harmony
            harmony_score = self._calculate_color_harmony(colors)
            
            # Check contrast ratios
            contrast_score = self._analyze_color_contrast(image)
            
            # Overall color score
            color_score = (harmony_score + contrast_score) / 2
            
            if color_score < 6.0:
                findings.append(self._create_finding(
                    category="color",
                    severity="medium",
                    title="Color Harmony Issues",
                    description=f"Color palette could be more harmonious (score: {color_score:.1f}/10)",
                    recommendation="Consider using color theory principles for better harmony",
                    confidence_score=0.7
                ))
            
            if contrast_score < 5.0:
                findings.append(self._create_finding(
                    category="accessibility",
                    severity="high",
                    title="Poor Color Contrast",
                    description="Some color combinations may not meet accessibility standards",
                    recommendation="Ensure sufficient contrast ratios (4.5:1 for normal text, 3:1 for large text)",
                    confidence_score=0.8
                ))
            
            return color_score, findings
            
        except Exception as e:
            self.logger.error(f"Color analysis failed: {e}")
            return 5.0, []
    
    def _analyze_typography(self, image_array: np.ndarray) -> Tuple[float, List[Finding]]:
        """Analyze typography and text elements."""
        findings = []
        
        try:
            # This is a simplified analysis - in a real implementation,
            # you'd use OCR and text detection libraries
            
            # Convert to grayscale
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Detect potential text regions using morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            # Find contours that might be text
            edges = cv2.Canny(morph, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze text-like regions
            text_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                area = cv2.contourArea(contour)
                
                # Filter for text-like shapes
                if 0.1 < aspect_ratio < 10 and area > 50:
                    text_regions.append((x, y, w, h))
            
            # Calculate typography score based on text region distribution
            if text_regions:
                # Analyze spacing and alignment of text regions
                y_positions = [y for x, y, w, h in text_regions]
                spacing_consistency = 1.0 - (np.std(y_positions) / np.mean(y_positions)) if y_positions else 0
                typography_score = min(10.0, spacing_consistency * 10)
            else:
                typography_score = 7.0  # Neutral score if no text detected
            
            if typography_score < 6.0:
                findings.append(self._create_finding(
                    category="typography",
                    severity="medium",
                    title="Typography Consistency Issues",
                    description="Text elements may lack consistent spacing or alignment",
                    recommendation="Ensure consistent typography scale and spacing",
                    confidence_score=0.6
                ))
            
            return typography_score, findings
            
        except Exception as e:
            self.logger.error(f"Typography analysis failed: {e}")
            return 7.0, []
    
    def _analyze_whitespace(self, image_array: np.ndarray) -> Tuple[float, List[Finding]]:
        """Analyze whitespace usage and breathing room."""
        findings = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Threshold to identify content vs whitespace
            _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            
            # Calculate whitespace ratio
            total_pixels = binary.size
            white_pixels = np.sum(binary == 255)
            whitespace_ratio = white_pixels / total_pixels
            
            # Analyze whitespace distribution
            # Use morphological operations to find content clusters
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            content_mask = cv2.morphologyEx(255 - binary, cv2.MORPH_CLOSE, kernel)
            
            # Find content regions
            contours, _ = cv2.findContours(content_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate spacing between content regions
            if len(contours) > 1:
                centers = []
                for contour in contours:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        centers.append((cx, cy))
                
                if len(centers) > 1:
                    distances = []
                    for i in range(len(centers)):
                        for j in range(i + 1, len(centers)):
                            dist = np.sqrt((centers[i][0] - centers[j][0])**2 + 
                                         (centers[i][1] - centers[j][1])**2)
                            distances.append(dist)
                    
                    spacing_consistency = 1.0 - (np.std(distances) / np.mean(distances)) if distances else 0
                else:
                    spacing_consistency = 1.0
            else:
                spacing_consistency = 1.0
            
            # Calculate whitespace score
            optimal_whitespace = 0.3  # 30% whitespace is often considered good
            whitespace_score_component = 1.0 - abs(whitespace_ratio - optimal_whitespace) / optimal_whitespace
            whitespace_score = (whitespace_score_component + spacing_consistency) * 5
            whitespace_score = max(0.0, min(10.0, whitespace_score))
            
            if whitespace_ratio < 0.15:
                findings.append(self._create_finding(
                    category="whitespace",
                    severity="medium",
                    title="Insufficient Whitespace",
                    description="Design appears cluttered with insufficient breathing room",
                    recommendation="Add more whitespace to improve readability and visual comfort",
                    confidence_score=0.8
                ))
            elif whitespace_ratio > 0.6:
                findings.append(self._create_finding(
                    category="whitespace",
                    severity="low",
                    title="Excessive Whitespace",
                    description="Design may have too much empty space",
                    recommendation="Consider adding more content or reducing whitespace",
                    confidence_score=0.7
                ))
            elif whitespace_score > 7.5:
                findings.append(self._create_finding(
                    category="whitespace",
                    severity="low",
                    title="Good Whitespace Usage",
                    description="Whitespace is well-balanced and enhances readability",
                    recommendation="Maintain current whitespace principles",
                    confidence_score=0.8
                ))
            
            return whitespace_score, findings
            
        except Exception as e:
            self.logger.error(f"Whitespace analysis failed: {e}")
            return 5.0, []
    
    def _extract_dominant_colors(self, image: Image.Image, num_colors: int = 8) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from the image."""
        try:
            # Resize image for faster processing
            image = image.resize((150, 150))
            
            # Convert to RGB if not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get pixel data
            pixels = list(image.getdata())
            
            # Use k-means clustering to find dominant colors
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            colors = kmeans.cluster_centers_.astype(int)
            return [tuple(color) for color in colors]
            
        except ImportError:
            # Fallback method without sklearn
            pixel_count = Counter(image.getdata())
            return [color for color, count in pixel_count.most_common(num_colors)]
        except Exception as e:
            self.logger.error(f"Color extraction failed: {e}")
            return [(128, 128, 128)]  # Return gray as fallback
    
    def _calculate_color_harmony(self, colors: List[Tuple[int, int, int]]) -> float:
        """Calculate color harmony score."""
        if len(colors) < 2:
            return 7.0
        
        try:
            # Convert RGB to HSV for better color analysis
            hsv_colors = []
            for r, g, b in colors:
                h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
                hsv_colors.append((h * 360, s, v))
            
            # Analyze hue relationships
            hues = [h for h, s, v in hsv_colors]
            hue_differences = []
            
            for i in range(len(hues)):
                for j in range(i + 1, len(hues)):
                    diff = abs(hues[i] - hues[j])
                    # Handle circular nature of hue
                    diff = min(diff, 360 - diff)
                    hue_differences.append(diff)
            
            if not hue_differences:
                return 7.0
            
            # Check for common harmony patterns
            harmony_score = 5.0
            
            # Complementary colors (around 180 degrees)
            complementary = any(170 <= diff <= 190 for diff in hue_differences)
            # Triadic colors (around 120 degrees)
            triadic = any(110 <= diff <= 130 for diff in hue_differences)
            # Analogous colors (within 30 degrees)
            analogous = any(diff <= 30 for diff in hue_differences)
            
            if complementary or triadic:
                harmony_score += 2.0
            elif analogous:
                harmony_score += 1.5
            
            # Penalize too many conflicting colors
            if len(set(int(h/30) for h in hues)) > 4:
                harmony_score -= 1.0
            
            return max(0.0, min(10.0, harmony_score))
            
        except Exception as e:
            self.logger.error(f"Color harmony calculation failed: {e}")
            return 5.0
    
    def _analyze_color_contrast(self, image: Image.Image) -> float:
        """Analyze color contrast in the image."""
        try:
            # Convert to grayscale to analyze luminance
            gray_image = image.convert('L')
            
            # Calculate statistics
            stat = ImageStat.Stat(gray_image)
            
            # Get mean and standard deviation
            mean_luminance = stat.mean[0]
            std_luminance = stat.stddev[0]
            
            # Higher standard deviation indicates better contrast
            # Normalize to 0-10 scale
            contrast_score = min(10.0, (std_luminance / 64.0) * 10)
            
            return contrast_score
            
        except Exception as e:
            self.logger.error(f"Contrast analysis failed: {e}")
            return 5.0
    
    def _calculate_alignment_score(self, contours, image_shape) -> float:
        """Calculate alignment score based on contour positions."""
        if not contours:
            return 5.0
        
        try:
            # Get bounding rectangles
            rects = [cv2.boundingRect(contour) for contour in contours 
                    if cv2.contourArea(contour) > 100]
            
            if len(rects) < 2:
                return 7.0
            
            # Analyze horizontal and vertical alignment
            left_edges = [x for x, y, w, h in rects]
            right_edges = [x + w for x, y, w, h in rects]
            top_edges = [y for x, y, w, h in rects]
            bottom_edges = [y + h for x, y, w, h in rects]
            
            # Calculate alignment consistency
            left_std = np.std(left_edges) / image_shape[1] if left_edges else 0
            right_std = np.std(right_edges) / image_shape[1] if right_edges else 0
            top_std = np.std(top_edges) / image_shape[0] if top_edges else 0
            bottom_std = np.std(bottom_edges) / image_shape[0] if bottom_edges else 0
            
            # Lower standard deviation means better alignment
            alignment_score = 10.0 - (left_std + right_std + top_std + bottom_std) * 10
            
            return max(0.0, min(10.0, alignment_score))
            
        except Exception as e:
            self.logger.error(f"Alignment calculation failed: {e}")
            return 5.0
    
    def _check_margins(self, edges) -> float:
        """Check for proper margins around content."""
        try:
            height, width = edges.shape
            
            # Check edges of the image for content
            top_margin = np.sum(edges[:int(height * 0.1), :]) == 0
            bottom_margin = np.sum(edges[int(height * 0.9):, :]) == 0
            left_margin = np.sum(edges[:, :int(width * 0.1)]) == 0
            right_margin = np.sum(edges[:, int(width * 0.9):]) == 0
            
            margin_score = sum([top_margin, bottom_margin, left_margin, right_margin]) * 2.5
            
            return margin_score
            
        except Exception as e:
            self.logger.error(f"Margin check failed: {e}")
            return 5.0
    
    async def _generate_llm_visual_insights(
        self,
        image: Image.Image,
        context: Optional[Dict[str, Any]],
        layout_score: float,
        hierarchy_score: float,
        color_score: float,
        typography_score: float,
        whitespace_score: float
    ) -> Optional[str]:
        """Generate LLM-powered visual design insights."""
        try:
            # Create image description
            image_description = self._create_image_description(image, context)
            
            # Prepare context string
            context_str = str(context) if context else "No specific context provided"
            
            # Generate comprehensive visual insights
            insights = await self._generate_llm_insights(
                self.visual_analysis_prompt,
                {
                    "image_description": image_description,
                    "layout_score": layout_score,
                    "hierarchy_score": hierarchy_score,
                    "color_score": color_score,
                    "typography_score": typography_score,
                    "whitespace_score": whitespace_score,
                    "context": context_str
                },
                temperature=0.7
            )
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to generate LLM visual insights: {e}")
            return None
    
    async def _generate_color_insights(
        self,
        dominant_colors: List[Tuple[int, int, int]],
        harmony_score: float,
        contrast_score: float,
        context: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Generate LLM-powered color analysis insights."""
        try:
            # Format colors for LLM
            color_strings = [f"RGB({r}, {g}, {b})" for r, g, b in dominant_colors]
            colors_text = ", ".join(color_strings)
            
            context_str = str(context) if context else "No specific context provided"
            
            insights = await self._generate_llm_insights(
                self.color_analysis_prompt,
                {
                    "dominant_colors": colors_text,
                    "harmony_score": harmony_score,
                    "contrast_score": contrast_score,
                    "context": context_str
                },
                temperature=0.6
            )
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to generate color insights: {e}")
            return None
