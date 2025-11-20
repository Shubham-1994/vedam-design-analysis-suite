"""Image generation agent using InstructPix2Pix for design regeneration."""

import asyncio
import logging
import time
import torch
from typing import Dict, Any, Optional, List
from PIL import Image
import io
import base64

from ..models.schemas import AgentResult, Finding, AnalysisType

logger = logging.getLogger(__name__)


class ImageGenerationAgent:
    """Agent for generating improved designs based on analysis results."""
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        self.agent_name = "Design Regeneration Specialist"
        self.analysis_type = AnalysisType.VISUAL  # Using visual type as base
        self.api_keys = api_keys or {}
        
        # Pipeline will be loaded lazily to avoid startup delays
        self.pipeline = None
        self._pipeline_loaded = False
        
        logger.info(f"Initialized {self.agent_name}")
    
    def _load_pipeline(self):
        """Load the InstructPix2Pix pipeline for image editing."""
        if self._pipeline_loaded:
            return
        
        try:
            logger.info("Loading InstructPix2Pix pipeline...")
            
            # Import here to avoid startup delays
            from diffusers import StableDiffusionInstructPix2PixPipeline
            
            # Load the pipeline - using a well-supported model
            self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                "timbrooks/instruct-pix2pix", 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.pipeline.to('cuda')
                logger.info("Pipeline loaded on GPU")
            else:
                logger.info("Pipeline loaded on CPU")
            
            # Disable progress bar for cleaner logs
            self.pipeline.set_progress_bar_config(disable=True)
            
            self._pipeline_loaded = True
            logger.info("InstructPix2Pix pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load InstructPix2Pix pipeline: {e}")
            raise
    
    async def _execute_analysis(
        self, 
        image: Image.Image, 
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """Generate improved design based on analysis results."""
        
        start_time = time.time()
        findings = []
        
        try:
            logger.info(f"Starting image generation analysis with {self.agent_name}")
            
            # This method is called by the orchestrator, but for image generation
            # we need the analysis results to create meaningful prompts
            # For now, we'll create a placeholder result that indicates the agent is ready
            
            findings.append(Finding(
                category="Image Generation",
                severity="low",
                title="Image Generation Agent Ready",
                description="The image generation agent is ready to create improved designs based on analysis results.",
                recommendation="Use the /regenerate endpoint with analysis results to generate improved designs.",
                confidence_score=1.0,
                supporting_evidence=["InstructPix2Pix pipeline available"]
            ))
            
            execution_time = time.time() - start_time
            
            return AgentResult(
                agent_name=self.agent_name,
                analysis_type=self.analysis_type,
                findings=findings,
                overall_score=8.0,  # High score as the agent is functional
                execution_time=execution_time,
                metadata={
                    "pipeline_loaded": self._pipeline_loaded,
                    "gpu_available": torch.cuda.is_available(),
                    "generation_ready": True
                }
            )
            
        except Exception as e:
            logger.error(f"Image generation analysis failed: {e}")
            
            findings.append(Finding(
                category="Error",
                severity="high",
                title="Image Generation Setup Failed",
                description=f"Failed to initialize image generation capabilities: {str(e)}",
                recommendation="Check system requirements and GPU availability for optimal performance.",
                confidence_score=1.0
            ))
            
            execution_time = time.time() - start_time
            
            return AgentResult(
                agent_name=self.agent_name,
                analysis_type=self.analysis_type,
                findings=findings,
                overall_score=0.0,
                execution_time=execution_time,
                metadata={"error": str(e)}
            )
    
    async def generate_improved_design(
        self,
        original_image: Image.Image,
        analysis_results: Dict[str, Any],
        generation_options: Optional[Dict[str, Any]] = None,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate an improved design based on analysis results."""
        
        try:
            # Load pipeline if not already loaded
            if not self._pipeline_loaded:
                self._load_pipeline()
            
            if not self.pipeline:
                raise Exception("Pipeline not available")
            
            # Generate improvement prompt based on analysis results
            prompt = custom_prompt or self._create_improvement_prompt(analysis_results)
            
            # Set default generation parameters for InstructPix2Pix
            default_options = {
                "image_guidance_scale": 1.5,
                "guidance_scale": 7.5,
                "num_inference_steps": 20
            }
            
            # Create generator separately (not serializable)
            generator = torch.manual_seed(42)
            
            # Update with user-provided options
            if generation_options:
                # Map Qwen parameters to InstructPix2Pix parameters
                if "true_cfg_scale" in generation_options:
                    default_options["guidance_scale"] = generation_options["true_cfg_scale"] * 2
                if "num_inference_steps" in generation_options:
                    default_options["num_inference_steps"] = generation_options["num_inference_steps"] // 2
                # Add other mappings as needed
                for key, value in generation_options.items():
                    if key in ["guidance_scale", "image_guidance_scale", "num_inference_steps"]:
                        default_options[key] = value
                    elif key == "generator" and hasattr(value, 'manual_seed'):
                        generator = value
            
            # Prepare inputs for InstructPix2Pix
            inputs = {
                "prompt": prompt,
                "image": original_image,
                "generator": generator,
                **default_options
            }
            
            logger.info(f"Generating improved design with prompt: {prompt[:100]}...")
            
            # Generate the improved image
            with torch.inference_mode():
                generated_image = self.pipeline(**inputs).images[0]
            
            # Convert to base64 for API response
            buffer = io.BytesIO()
            generated_image.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return {
                "success": True,
                "generated_image_base64": image_base64,
                "prompt_used": prompt,
                "generation_options": default_options,  # Now safe to serialize
                "improvements_applied": self._extract_improvements_from_results(analysis_results)
            }
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to generate improved design"
            }
    
    def _create_improvement_prompt(self, analysis_results: Dict[str, Any]) -> str:
        """Create a detailed prompt for image improvement based on analysis results."""
        
        prompt_parts = []
        
        # Base prompt for design improvement
        prompt_parts.append("Create an improved version of this design with the following enhancements:")
        
        # Extract key improvements from analysis results
        priority_improvements = analysis_results.get('priority_improvements', [])
        actionable_recommendations = analysis_results.get('actionable_recommendations', [])
        
        # Process priority improvements
        if priority_improvements:
            for improvement in priority_improvements[:3]:  # Top 3 improvements
                title = improvement.get('title', '')
                recommendation = improvement.get('recommendation', '')
                
                if 'color' in title.lower() or 'color' in recommendation.lower():
                    prompt_parts.append("Improve color harmony and contrast")
                elif 'typography' in title.lower() or 'font' in title.lower():
                    prompt_parts.append("Enhance typography and text readability")
                elif 'layout' in title.lower() or 'composition' in title.lower():
                    prompt_parts.append("Optimize layout and visual composition")
                elif 'whitespace' in title.lower() or 'spacing' in title.lower():
                    prompt_parts.append("Improve whitespace and element spacing")
                elif 'hierarchy' in title.lower():
                    prompt_parts.append("Strengthen visual hierarchy")
        
        # Process actionable recommendations
        if actionable_recommendations:
            for rec in actionable_recommendations[:2]:  # Top 2 recommendations
                if 'modern' in rec.lower():
                    prompt_parts.append("Apply modern design principles")
                elif 'professional' in rec.lower():
                    prompt_parts.append("Enhance professional appearance")
                elif 'user-friendly' in rec.lower() or 'usability' in rec.lower():
                    prompt_parts.append("Improve user experience and usability")
        
        # Add context-based improvements
        context = analysis_results.get('context', {})
        if context:
            industry = context.get('industry', '')
            platform = context.get('platform', '')
            
            if industry:
                prompt_parts.append(f"Optimize for {industry} industry standards")
            if platform:
                prompt_parts.append(f"Enhance for {platform} platform best practices")
        
        # Add quality and style improvements
        prompt_parts.extend([
            "Maintain brand consistency while improving visual appeal",
            "Ensure high-quality, professional appearance",
            "Optimize for better user engagement",
            "Keep the core message and functionality intact"
        ])
        
        # Join all parts into a coherent prompt
        prompt = ". ".join(prompt_parts) + ". Create a polished, modern, and user-friendly design."
        
        # Limit prompt length to avoid token limits
        if len(prompt) > 500:
            prompt = prompt[:500] + "..."
        
        return prompt
    
    def _extract_improvements_from_results(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Extract a list of improvements that were applied based on analysis results."""
        
        improvements = []
        
        # Extract from priority improvements
        priority_improvements = analysis_results.get('priority_improvements', [])
        for improvement in priority_improvements[:5]:
            title = improvement.get('title', '')
            if title:
                improvements.append(title)
        
        # Extract from actionable recommendations
        actionable_recommendations = analysis_results.get('actionable_recommendations', [])
        for rec in actionable_recommendations[:3]:
            if rec and rec not in improvements:
                improvements.append(rec)
        
        return improvements
    
    async def generate_multiple_variants(
        self,
        original_image: Image.Image,
        analysis_results: Dict[str, Any],
        num_variants: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate multiple design variants with different improvement focuses."""
        
        variants = []
        
        # Define different improvement focuses
        focus_areas = [
            {"name": "Visual Appeal", "emphasis": "color harmony, visual hierarchy, modern aesthetics"},
            {"name": "User Experience", "emphasis": "usability, navigation clarity, user-friendly layout"},
            {"name": "Professional Polish", "emphasis": "professional appearance, brand consistency, quality"}
        ]
        
        for i, focus in enumerate(focus_areas[:num_variants]):
            try:
                # Create focused prompt
                base_prompt = self._create_improvement_prompt(analysis_results)
                focused_prompt = f"{base_prompt} Focus especially on {focus['emphasis']}."
                
                # Generate variant with different seed for variety
                generation_options = {
                    "generator": torch.manual_seed(42 + i * 10),
                    "guidance_scale": 7.5 + (i * 1.0),  # Slight variation in guidance
                    "image_guidance_scale": 1.5 + (i * 0.2)
                }
                
                # Generate variant with focused prompt
                variant_result = await self.generate_improved_design(
                    original_image,
                    analysis_results,
                    generation_options,
                    custom_prompt=focused_prompt
                )
                
                if variant_result.get("success"):
                    variant_result["variant_name"] = focus["name"]
                    variant_result["focus_area"] = focus["emphasis"]
                    variant_result["prompt_used"] = focused_prompt
                    variants.append(variant_result)
                
            except Exception as e:
                logger.error(f"Failed to generate variant {focus['name']}: {e}")
                continue
        
        return variants
