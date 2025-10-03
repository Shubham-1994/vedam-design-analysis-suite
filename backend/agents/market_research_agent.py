"""Market Research Agent for competitive analysis and trend identification."""

import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple

from .base_agent import BaseAgent
from ..models.schemas import Finding, AgentResult, AnalysisType, MarketComparison
from langchain_core.prompts import ChatPromptTemplate


class MarketResearchAgent(BaseAgent):
    """Agent for market research and competitive analysis."""
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        super().__init__("Market Research Specialist", AnalysisType.MARKET_RESEARCH, api_keys)
        
        # Industry trends and patterns
        self.current_trends = [
            "minimalist_design",
            "dark_mode_support",
            "micro_interactions",
            "glassmorphism",
            "neumorphism",
            "bold_typography",
            "gradient_overlays",
            "custom_illustrations",
            "accessibility_first",
            "mobile_first_design"
        ]
        
        # LLM Prompt Templates
        self.market_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a market research expert specializing in UI/UX design trends and competitive analysis. Analyze the provided market metrics and design context to provide comprehensive market insights.

Your analysis should focus on:
- Competitive positioning and differentiation
- Current design trends and market alignment
- Innovation opportunities and market gaps
- Industry-specific design patterns
- User expectations and market demands
- Future trend predictions

Provide strategic recommendations for market positioning."""),
            ("human", """Image Description: {image_description}

Market Analysis Metrics:
- Market Similarity Score: {similarity_score}/10
- Trend Alignment Score: {trends_score}/10
- Competitive Positioning Score: {competitive_score}/10
- Innovation Potential Score: {innovation_score}/10

Industry: {industry}
Target Market: {target_market}
Competitive Landscape: {competitive_landscape}
Context: {context}

Please provide detailed market analysis and strategic recommendations.""")
        ])
        
        self.competitive_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a competitive analysis expert in the design industry. Analyze competitive positioning and provide strategic insights for market differentiation.

Focus on:
- Direct and indirect competitors
- Competitive advantages and weaknesses
- Market positioning strategies
- Differentiation opportunities
- Competitive threats and opportunities"""),
            ("human", """Similar Designs Found: {similar_designs_count}
High Similarity Competitors: {high_similarity_count}
Industry: {industry}
Competitive Insights: {competitive_insights}

Provide detailed competitive analysis and differentiation strategies.""")
        ])
        
        self.trend_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a design trend analyst with expertise in UI/UX evolution and market dynamics. Analyze current trends and predict future directions.

Focus on:
- Current design trend adoption
- Emerging patterns and innovations
- Industry-specific trend variations
- User behavior and preference shifts
- Technology impact on design trends"""),
            ("human", """Current Trends Analysis: {trends_analysis}
Industry: {industry}
Design Alignment: {trend_alignment}
Innovation Areas: {innovation_areas}

Provide comprehensive trend analysis and future predictions.""")
        ])
    
    async def analyze(
        self,
        image: Image.Image,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """Perform market research analysis on the design."""
        findings = []
        
        # Perform market research analyses
        similarity_score, similarity_findings = await self._analyze_market_similarity(image, context)
        trends_score, trends_findings = self._analyze_trend_alignment(image, context)
        competitive_score, competitive_findings = await self._analyze_competitive_positioning(image, context)
        innovation_score, innovation_findings = self._analyze_innovation_potential(image, context)
        
        # Combine all findings
        findings.extend(similarity_findings)
        findings.extend(trends_findings)
        findings.extend(competitive_findings)
        findings.extend(innovation_findings)
        
        # Generate LLM-powered market insights
        llm_insights = await self._generate_llm_market_insights(
            image, context, similarity_score, trends_score, competitive_score, innovation_score
        )
        
        # Add LLM insights to findings if available
        if llm_insights:
            findings.append(self._create_finding(
                category="llm_insights",
                severity="low",
                title="AI-Powered Market Analysis",
                description=llm_insights,
                recommendation="Consider the AI market analysis alongside technical metrics",
                confidence_score=0.8,
                supporting_evidence=["LLM Market Analysis"]
            ))
        
        # Calculate overall score
        overall_score = (similarity_score + trends_score + competitive_score + innovation_score) / 4
        
        # Create market comparison data
        market_comparison = await self._create_market_comparison(image, context)
        
        return AgentResult(
            agent_name=self.agent_name,
            analysis_type=self.analysis_type,
            findings=findings,
            overall_score=overall_score,
            execution_time=0.0,  # Will be set by base class
            metadata={
                "market_comparison": market_comparison.dict() if market_comparison else None,
                "similarity_score": similarity_score,
                "trends_score": trends_score,
                "competitive_score": competitive_score,
                "innovation_score": innovation_score,
                "llm_insights": llm_insights
            }
        )
    
    async def _analyze_market_similarity(
        self, 
        image: Image.Image, 
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, List[Finding]]:
        """Analyze similarity to existing market solutions."""
        findings = []
        
        try:
            # Get similar designs from vector store
            similar_patterns = self._get_similar_patterns(image, n_results=10)
            
            if not similar_patterns:
                findings.append(self._create_finding(
                    category="market_analysis",
                    severity="medium",
                    title="Limited Market Reference Data",
                    description="Insufficient similar designs found for comprehensive market analysis",
                    recommendation="Consider expanding the reference database with more industry examples",
                    confidence_score=0.5
                ))
                return 5.0, findings
            
            # Analyze similarity scores
            similarity_scores = [pattern["similarity"] for pattern in similar_patterns]
            avg_similarity = np.mean(similarity_scores)
            max_similarity = max(similarity_scores)
            
            # Determine uniqueness vs familiarity balance
            if max_similarity > 0.9:
                findings.append(self._create_finding(
                    category="market_analysis",
                    severity="high",
                    title="Very High Similarity to Existing Designs",
                    description=f"Design is very similar to existing solutions (max similarity: {max_similarity:.2f})",
                    recommendation="Consider adding unique elements to differentiate from competitors",
                    confidence_score=0.8,
                    supporting_evidence=[similar_patterns[0]["description"]]
                ))
                similarity_score = 4.0
            elif max_similarity > 0.7:
                findings.append(self._create_finding(
                    category="market_analysis",
                    severity="medium",
                    title="High Similarity to Market Standards",
                    description=f"Design follows established patterns closely (max similarity: {max_similarity:.2f})",
                    recommendation="Good adherence to conventions, consider subtle differentiation",
                    confidence_score=0.7,
                    supporting_evidence=[p["description"] for p in similar_patterns[:2]]
                ))
                similarity_score = 7.0
            elif max_similarity > 0.4:
                findings.append(self._create_finding(
                    category="market_analysis",
                    severity="low",
                    title="Balanced Familiarity and Uniqueness",
                    description=f"Design strikes good balance between familiar and unique (similarity: {max_similarity:.2f})",
                    recommendation="Maintain this balance while refining unique elements",
                    confidence_score=0.8
                ))
                similarity_score = 8.5
            else:
                findings.append(self._create_finding(
                    category="market_analysis",
                    severity="medium",
                    title="Highly Unique Design Approach",
                    description=f"Design is very different from market standards (max similarity: {max_similarity:.2f})",
                    recommendation="Ensure uniqueness doesn't compromise usability and user expectations",
                    confidence_score=0.7
                ))
                similarity_score = 6.0
            
            # Analyze industry-specific patterns
            if context and context.get("industry"):
                industry_findings = self._analyze_industry_patterns(similar_patterns, context["industry"])
                findings.extend(industry_findings)
            
            return similarity_score, findings
            
        except Exception as e:
            self.logger.error(f"Market similarity analysis failed: {e}")
            return 5.0, []
    
    def _analyze_trend_alignment(
        self, 
        image: Image.Image, 
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, List[Finding]]:
        """Analyze alignment with current design trends."""
        findings = []
        
        try:
            # Search for trend-related patterns
            trend_scores = {}
            
            for trend in self.current_trends:
                trend_patterns = self._search_patterns_by_text(trend)
                if trend_patterns:
                    # Calculate how well the design aligns with this trend
                    image_embedding = self.embeddings.encode_image(image)
                    trend_similarities = []
                    
                    for pattern in trend_patterns[:3]:  # Top 3 results
                        # This would need the pattern's embedding stored in metadata
                        # For now, use a simplified approach
                        trend_similarities.append(pattern["similarity"])
                    
                    if trend_similarities:
                        trend_scores[trend] = max(trend_similarities)
            
            # Analyze trend alignment
            if trend_scores:
                avg_trend_score = np.mean(list(trend_scores.values()))
                top_trends = sorted(trend_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                
                trends_score = avg_trend_score * 10
                
                if avg_trend_score > 0.7:
                    findings.append(self._create_finding(
                        category="trends",
                        severity="low",
                        title="Strong Trend Alignment",
                        description=f"Design aligns well with current trends: {', '.join([t[0] for t in top_trends])}",
                        recommendation="Continue leveraging current trends while maintaining timeless elements",
                        confidence_score=0.8
                    ))
                elif avg_trend_score > 0.4:
                    findings.append(self._create_finding(
                        category="trends",
                        severity="low",
                        title="Moderate Trend Alignment",
                        description=f"Design shows some alignment with trends: {', '.join([t[0] for t in top_trends[:2]])}",
                        recommendation="Consider incorporating more current design trends",
                        confidence_score=0.6
                    ))
                else:
                    findings.append(self._create_finding(
                        category="trends",
                        severity="medium",
                        title="Limited Trend Alignment",
                        description="Design shows minimal alignment with current trends",
                        recommendation="Consider updating design to reflect current market preferences",
                        confidence_score=0.7
                    ))
            else:
                trends_score = 5.0
                findings.append(self._create_finding(
                    category="trends",
                    severity="medium",
                    title="Trend Analysis Inconclusive",
                    description="Unable to determine trend alignment due to limited reference data",
                    recommendation="Expand trend analysis with more comprehensive data",
                    confidence_score=0.4
                ))
            
            return trends_score, findings
            
        except Exception as e:
            self.logger.error(f"Trend analysis failed: {e}")
            return 5.0, []
    
    async def _analyze_competitive_positioning(
        self, 
        image: Image.Image, 
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, List[Finding]]:
        """Analyze competitive positioning and differentiation."""
        findings = []
        
        try:
            # Get competitive designs (similar patterns with high similarity)
            competitive_patterns = self._get_similar_patterns(image, n_results=5)
            
            if not competitive_patterns:
                return 5.0, findings
            
            # Analyze competitive landscape
            high_similarity_count = sum(1 for p in competitive_patterns if p["similarity"] > 0.8)
            medium_similarity_count = sum(1 for p in competitive_patterns if 0.6 <= p["similarity"] <= 0.8)
            
            # Determine competitive positioning
            if high_similarity_count >= 3:
                competitive_score = 4.0
                findings.append(self._create_finding(
                    category="competitive_analysis",
                    severity="high",
                    title="Crowded Competitive Space",
                    description=f"Found {high_similarity_count} very similar designs in the market",
                    recommendation="Differentiate significantly to stand out in crowded market",
                    confidence_score=0.8,
                    supporting_evidence=[p["description"] for p in competitive_patterns[:2]]
                ))
            elif high_similarity_count >= 1:
                competitive_score = 6.0
                findings.append(self._create_finding(
                    category="competitive_analysis",
                    severity="medium",
                    title="Some Direct Competition",
                    description=f"Found {high_similarity_count} very similar design(s)",
                    recommendation="Consider strategic differentiation while maintaining usability",
                    confidence_score=0.7
                ))
            elif medium_similarity_count >= 2:
                competitive_score = 7.5
                findings.append(self._create_finding(
                    category="competitive_analysis",
                    severity="low",
                    title="Moderate Competition",
                    description="Design has moderate similarity to existing solutions",
                    recommendation="Good positioning with room for further differentiation",
                    confidence_score=0.7
                ))
            else:
                competitive_score = 8.5
                findings.append(self._create_finding(
                    category="competitive_analysis",
                    severity="low",
                    title="Strong Differentiation",
                    description="Design shows good differentiation from existing solutions",
                    recommendation="Maintain differentiation while ensuring user familiarity",
                    confidence_score=0.8
                ))
            
            # Analyze competitive advantages
            advantages = self._identify_competitive_advantages(competitive_patterns, context)
            if advantages:
                findings.append(self._create_finding(
                    category="competitive_analysis",
                    severity="low",
                    title="Competitive Advantages Identified",
                    description=f"Potential advantages: {', '.join(advantages)}",
                    recommendation="Leverage and emphasize these competitive advantages",
                    confidence_score=0.6
                ))
            
            return competitive_score, findings
            
        except Exception as e:
            self.logger.error(f"Competitive analysis failed: {e}")
            return 5.0, []
    
    def _analyze_innovation_potential(
        self, 
        image: Image.Image, 
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, List[Finding]]:
        """Analyze innovation potential and uniqueness."""
        findings = []
        
        try:
            # Get similar patterns to assess uniqueness
            similar_patterns = self._get_similar_patterns(image, n_results=10)
            
            if not similar_patterns:
                innovation_score = 7.0
                findings.append(self._create_finding(
                    category="innovation",
                    severity="low",
                    title="Potentially Innovative Approach",
                    description="Design appears unique due to limited similar references",
                    recommendation="Validate innovation with user testing",
                    confidence_score=0.5
                ))
                return innovation_score, findings
            
            # Calculate innovation metrics
            similarity_scores = [p["similarity"] for p in similar_patterns]
            avg_similarity = np.mean(similarity_scores)
            uniqueness_score = 1.0 - avg_similarity
            
            # Analyze pattern diversity
            categories = [p.get("metadata", {}).get("category", "unknown") for p in similar_patterns]
            category_diversity = len(set(categories)) / len(categories) if categories else 0
            
            # Calculate innovation score
            innovation_score = (uniqueness_score * 7 + category_diversity * 3) * 10
            innovation_score = max(0.0, min(10.0, innovation_score))
            
            if innovation_score > 8.0:
                findings.append(self._create_finding(
                    category="innovation",
                    severity="low",
                    title="High Innovation Potential",
                    description=f"Design shows strong innovation potential (score: {innovation_score:.1f}/10)",
                    recommendation="Consider patent protection and first-mover advantage strategies",
                    confidence_score=0.8
                ))
            elif innovation_score > 6.0:
                findings.append(self._create_finding(
                    category="innovation",
                    severity="low",
                    title="Moderate Innovation",
                    description=f"Design has good innovation elements (score: {innovation_score:.1f}/10)",
                    recommendation="Enhance unique aspects while maintaining usability",
                    confidence_score=0.7
                ))
            else:
                findings.append(self._create_finding(
                    category="innovation",
                    severity="medium",
                    title="Limited Innovation",
                    description=f"Design follows conventional patterns (score: {innovation_score:.1f}/10)",
                    recommendation="Consider adding innovative elements to differentiate",
                    confidence_score=0.7
                ))
            
            # Analyze specific innovation opportunities
            opportunities = self._identify_innovation_opportunities(similar_patterns, context)
            if opportunities:
                findings.append(self._create_finding(
                    category="innovation",
                    severity="low",
                    title="Innovation Opportunities Identified",
                    description=f"Potential areas for innovation: {', '.join(opportunities)}",
                    recommendation="Explore these areas for breakthrough improvements",
                    confidence_score=0.6
                ))
            
            return innovation_score, findings
            
        except Exception as e:
            self.logger.error(f"Innovation analysis failed: {e}")
            return 5.0, []
    
    async def _create_market_comparison(
        self, 
        image: Image.Image, 
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[MarketComparison]:
        """Create comprehensive market comparison data."""
        try:
            # Get similar designs
            similar_patterns = self._get_similar_patterns(image, n_results=5)
            
            # Format similar designs
            similar_designs = []
            for pattern in similar_patterns:
                similar_designs.append({
                    "id": pattern["id"],
                    "description": pattern["description"],
                    "similarity_score": pattern["similarity"],
                    "category": pattern.get("metadata", {}).get("category", "unknown"),
                    "usability_score": pattern.get("metadata", {}).get("usability_score", 0)
                })
            
            # Identify industry trends
            industry_trends = []
            if context and context.get("industry"):
                industry = context["industry"]
                industry_patterns = self._search_patterns_by_text(f"{industry} design trends")
                industry_trends = [p["description"] for p in industry_patterns[:5]]
            
            if not industry_trends:
                industry_trends = [
                    "Mobile-first responsive design",
                    "Minimalist and clean interfaces",
                    "Dark mode compatibility",
                    "Accessibility-focused design",
                    "Micro-interactions and animations"
                ]
            
            # Generate competitive analysis
            competitive_analysis = self._generate_competitive_analysis(similar_patterns, context)
            
            return MarketComparison(
                similar_designs=similar_designs,
                industry_trends=industry_trends,
                competitive_analysis=competitive_analysis
            )
            
        except Exception as e:
            self.logger.error(f"Market comparison creation failed: {e}")
            return None
    
    def _analyze_industry_patterns(
        self, 
        similar_patterns: List[Dict[str, Any]], 
        industry: str
    ) -> List[Finding]:
        """Analyze industry-specific design patterns."""
        findings = []
        
        try:
            # Search for industry-specific patterns
            industry_patterns = self._search_patterns_by_text(f"{industry} interface design")
            
            if industry_patterns:
                # Compare with industry standards
                industry_similarities = [p["similarity"] for p in industry_patterns[:3]]
                avg_industry_similarity = np.mean(industry_similarities)
                
                if avg_industry_similarity > 0.7:
                    findings.append(self._create_finding(
                        category="industry_analysis",
                        severity="low",
                        title=f"Strong {industry} Industry Alignment",
                        description=f"Design aligns well with {industry} industry standards",
                        recommendation=f"Continue following {industry} best practices",
                        confidence_score=0.8
                    ))
                elif avg_industry_similarity < 0.4:
                    findings.append(self._create_finding(
                        category="industry_analysis",
                        severity="medium",
                        title=f"Deviation from {industry} Standards",
                        description=f"Design differs significantly from {industry} industry norms",
                        recommendation=f"Consider {industry} user expectations and conventions",
                        confidence_score=0.7
                    ))
            
        except Exception as e:
            self.logger.error(f"Industry pattern analysis failed: {e}")
        
        return findings
    
    def _identify_competitive_advantages(
        self, 
        competitive_patterns: List[Dict[str, Any]], 
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Identify potential competitive advantages."""
        advantages = []
        
        try:
            # Analyze metadata for competitive insights
            if competitive_patterns:
                # Compare usability scores
                competitor_scores = [
                    p.get("metadata", {}).get("usability_score", 0) 
                    for p in competitive_patterns
                ]
                
                if competitor_scores:
                    avg_competitor_score = np.mean(competitor_scores)
                    if avg_competitor_score < 7.0:
                        advantages.append("Superior usability potential")
                
                # Analyze categories
                categories = [p.get("metadata", {}).get("category") for p in competitive_patterns]
                if "accessibility" not in categories:
                    advantages.append("Accessibility focus opportunity")
                
                if "mobile" not in categories and context and context.get("platform") == "mobile":
                    advantages.append("Mobile-first advantage")
            
        except Exception as e:
            self.logger.error(f"Competitive advantage identification failed: {e}")
        
        return advantages
    
    def _identify_innovation_opportunities(
        self, 
        similar_patterns: List[Dict[str, Any]], 
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Identify innovation opportunities."""
        opportunities = []
        
        try:
            # Analyze gaps in existing patterns
            categories = [p.get("metadata", {}).get("category", "") for p in similar_patterns]
            
            missing_categories = set(["accessibility", "personalization", "ai_integration"]) - set(categories)
            
            for category in missing_categories:
                if category == "accessibility":
                    opportunities.append("Enhanced accessibility features")
                elif category == "personalization":
                    opportunities.append("Personalized user experience")
                elif category == "ai_integration":
                    opportunities.append("AI-powered interactions")
            
            # Context-based opportunities
            if context:
                if context.get("platform") == "mobile" and "gesture" not in " ".join(categories):
                    opportunities.append("Advanced gesture controls")
                
                if context.get("target_audience") == "elderly" and "accessibility" not in categories:
                    opportunities.append("Senior-friendly design innovations")
            
        except Exception as e:
            self.logger.error(f"Innovation opportunity identification failed: {e}")
        
        return opportunities
    
    def _generate_competitive_analysis(
        self, 
        similar_patterns: List[Dict[str, Any]], 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate competitive analysis summary."""
        try:
            if not similar_patterns:
                return "Limited competitive data available for analysis."
            
            num_competitors = len(similar_patterns)
            high_similarity = sum(1 for p in similar_patterns if p["similarity"] > 0.8)
            
            analysis = f"Analysis of {num_competitors} similar designs reveals "
            
            if high_similarity >= 3:
                analysis += "a highly competitive market with several very similar solutions. "
                analysis += "Differentiation through unique features or superior execution will be critical."
            elif high_similarity >= 1:
                analysis += "moderate competition with some similar solutions. "
                analysis += "Strategic positioning and feature differentiation recommended."
            else:
                analysis += "limited direct competition, suggesting good market positioning. "
                analysis += "Focus on execution excellence and user adoption."
            
            # Add industry context if available
            if context and context.get("industry"):
                analysis += f" In the {context['industry']} sector, "
                analysis += "consider industry-specific user expectations and regulatory requirements."
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Competitive analysis generation failed: {e}")
            return "Unable to generate competitive analysis due to technical issues."
    
    async def _generate_llm_market_insights(
        self,
        image: Image.Image,
        context: Optional[Dict[str, Any]],
        similarity_score: float,
        trends_score: float,
        competitive_score: float,
        innovation_score: float
    ) -> Optional[str]:
        """Generate LLM-powered market insights."""
        try:
            # Create image description
            image_description = self._create_image_description(image, context)
            
            # Extract context information
            context_str = str(context) if context else "No specific context provided"
            industry = context.get("industry", "General") if context else "General"
            target_market = context.get("target_audience", "General users") if context else "General users"
            
            # Create competitive landscape summary
            similar_patterns = self._get_similar_patterns(image, n_results=5)
            competitive_landscape = f"Found {len(similar_patterns)} similar designs" if similar_patterns else "Limited competitive data"
            
            # Generate comprehensive market insights
            insights = await self._generate_llm_insights(
                self.market_analysis_prompt,
                {
                    "image_description": image_description,
                    "similarity_score": similarity_score,
                    "trends_score": trends_score,
                    "competitive_score": competitive_score,
                    "innovation_score": innovation_score,
                    "industry": industry,
                    "target_market": target_market,
                    "competitive_landscape": competitive_landscape,
                    "context": context_str
                },
                temperature=0.7
            )
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to generate LLM market insights: {e}")
            return None
    
    async def _generate_competitive_insights(
        self,
        similar_designs_count: int,
        high_similarity_count: int,
        industry: str,
        competitive_insights: str
    ) -> Optional[str]:
        """Generate LLM-powered competitive analysis."""
        try:
            insights = await self._generate_llm_insights(
                self.competitive_analysis_prompt,
                {
                    "similar_designs_count": similar_designs_count,
                    "high_similarity_count": high_similarity_count,
                    "industry": industry,
                    "competitive_insights": competitive_insights
                },
                temperature=0.6
            )
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to generate competitive insights: {e}")
            return None
    
    async def _generate_trend_insights(
        self,
        trends_analysis: str,
        industry: str,
        trend_alignment: float,
        innovation_areas: List[str]
    ) -> Optional[str]:
        """Generate LLM-powered trend analysis."""
        try:
            innovation_text = ", ".join(innovation_areas) if innovation_areas else "No specific areas identified"
            
            insights = await self._generate_llm_insights(
                self.trend_analysis_prompt,
                {
                    "trends_analysis": trends_analysis,
                    "industry": industry,
                    "trend_alignment": trend_alignment,
                    "innovation_areas": innovation_text
                },
                temperature=0.6
            )
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to generate trend insights: {e}")
            return None
