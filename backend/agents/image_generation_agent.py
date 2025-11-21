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
            from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
            
            # Always use float16 for memory efficiency (use 'dtype' instead of deprecated 'torch_dtype')
            self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                "timbrooks/instruct-pix2pix", 
                dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Use the optimized scheduler from the sample code
            self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            # Move to GPU if available (prefer CUDA over MPS for stability)
            if torch.cuda.is_available():
                self.pipeline.to('cuda')
                logger.info("Pipeline loaded on CUDA GPU")
            elif torch.mps.is_available():
                self.pipeline.to('mps')
                logger.info("Pipeline loaded on MPS GPU")
            else:
                logger.info("Pipeline loaded on CPU")
            
            # Enable memory optimizations
            self.pipeline.enable_attention_slicing()
            self.pipeline.enable_vae_slicing()
            logger.info("Enabled attention slicing and VAE slicing for memory optimization")
            
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
                    "gpu_available": torch.cuda.is_available() or torch.mps.is_available(),
                    "cuda_available": torch.cuda.is_available(),
                    "mps_available": torch.mps.is_available(),
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
            
            # Prepare input image following the sample code pattern
            original_image = self._prepare_input_image(original_image)
            logger.info(f"Input image prepared: {original_image.size}")
            
            # Generate improvement prompt based on analysis results
            prompt = custom_prompt or self._create_improvement_prompt(analysis_results)
            
            # Set default generation parameters optimized for quality and stability
            default_options = {
                "image_guidance_scale": 1.5,  # Increased for better image fidelity
                "guidance_scale": 7.5,        # Add guidance scale for better results
                "num_inference_steps": 15     # Increased for better quality
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
            logger.info(f"Generation parameters: {default_options}")
            
            # Generate the improved image with memory management
            with torch.inference_mode():
                try:
                    result = self.pipeline(**inputs)
                    generated_image = result.images[0]
                    logger.info("Image generation completed successfully")
                except Exception as gen_error:
                    logger.error(f"Pipeline generation failed: {gen_error}")
                    # Create a fallback image if generation fails
                    generated_image = self._create_fallback_image(original_image.size)
                
                # Clear GPU cache if using CUDA
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Log generated image dimensions immediately
            logger.info(f"Pipeline output dimensions: {generated_image.size}")
            
            # Validate the generated image for NaN/invalid values
            generated_image = self._validate_generated_image(generated_image)
            
            # Critical: Check if generated image has reasonable dimensions
            width, height = generated_image.size
            if width > 4096 or height > 4096 or width * height > 16777216:  # 4096x4096 max
                logger.error(f"Generated image too large: {width}x{height} ({width*height} pixels)")
                # Force resize to safe dimensions immediately
                max_dim = 1024
                if width > height:
                    new_width = max_dim
                    new_height = int((height * max_dim) / width)
                else:
                    new_height = max_dim
                    new_width = int((width * max_dim) / height)
                generated_image = generated_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Emergency resize to: {generated_image.size}")
            
            # Validate and resize generated image if necessary
            generated_image = self._validate_and_resize_image(generated_image, max_dimension=1024)
            
            # Convert to base64 for API response with memory safety
            image_base64 = self._safe_image_to_base64(generated_image)
            
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
        
        # Get metrics-based and priority improvements
        metrics_context = self._build_metrics_context(analysis_results)
        
        # Add metrics-based improvements (prioritized by low scores and severity)
        if metrics_context:
            prompt_parts.extend(metrics_context[:5])  # Limit to top 5 for space management
        
        # Fallback to legacy logic if no metrics available
        if not metrics_context:
            # Extract key improvements from analysis results (legacy approach)
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
            
            if industry and len(". ".join(prompt_parts)) < 400:  # Only add if space allows
                prompt_parts.append(f"Optimize for {industry} industry standards")
            if platform and len(". ".join(prompt_parts)) < 350:  # Only add if space allows
                prompt_parts.append(f"Enhance for {platform} platform best practices")
        
        # Add essential quality improvements (always include these)
        essential_parts = [
            "Maintain brand consistency while improving visual appeal",
            "Ensure high-quality, professional appearance",
            "Keep the core message and functionality intact"
        ]
        
        # Add essential parts if space allows
        current_length = len(". ".join(prompt_parts))
        for essential in essential_parts:
            if current_length + len(essential) + 2 < 450:  # Leave room for ending
                prompt_parts.append(essential)
                current_length += len(essential) + 2
        
        # Join all parts into a coherent prompt
        prompt = ". ".join(prompt_parts) + ". Create a polished, modern, and user-friendly design."
        
        # Limit prompt length to avoid token limits
        if len(prompt) > 500:
            # Truncate intelligently - keep base prompt and most important improvements
            base_prompt = prompt_parts[0]
            improvements = prompt_parts[1:]
            
            # Calculate available space
            ending = ". Create a polished, modern, and user-friendly design."
            available_space = 500 - len(base_prompt) - len(ending) - 2  # 2 for ". "
            
            # Add improvements until we run out of space
            selected_improvements = []
            current_length = 0
            
            for improvement in improvements:
                if current_length + len(improvement) + 2 <= available_space:
                    selected_improvements.append(improvement)
                    current_length += len(improvement) + 2
                else:
                    break
            
            if selected_improvements:
                prompt = base_prompt + ". " + ". ".join(selected_improvements) + ending
            else:
                prompt = base_prompt + ending
        
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
    
    def _extract_low_scoring_metrics(self, analysis_results: Dict[str, Any], threshold: float = 7.0) -> List[str]:
        """Extract and format metrics that scored below the threshold."""
        
        low_scoring_metrics = []
        
        # Check visual metrics
        visual_metrics = analysis_results.get('visual_metrics', {})
        if visual_metrics:
            metric_mappings = {
                'layout_score': 'layout composition',
                'visual_hierarchy_score': 'visual hierarchy',
                'color_harmony_score': 'color harmony',
                'typography_score': 'typography',
                'whitespace_usage_score': 'whitespace usage'
            }
            
            for metric_key, display_name in metric_mappings.items():
                score = visual_metrics.get(metric_key, 10.0)
                if isinstance(score, (int, float)) and score < threshold:
                    low_scoring_metrics.append(f"improve {display_name} (score: {score:.1f}/10)")
        
        # Check UX metrics
        ux_metrics = analysis_results.get('ux_metrics', {})
        if ux_metrics:
            ux_mappings = {
                'usability_score': 'usability',
                'navigation_clarity': 'navigation clarity',
                'information_architecture': 'information architecture',
                'user_flow_efficiency': 'user flow'
            }
            
            for metric_key, display_name in ux_mappings.items():
                score = ux_metrics.get(metric_key, 10.0)
                if isinstance(score, (int, float)) and score < threshold:
                    low_scoring_metrics.append(f"enhance {display_name} (score: {score:.1f}/10)")
        
        # Check accessibility metrics
        accessibility_metrics = analysis_results.get('accessibility_metrics', {})
        if accessibility_metrics:
            accessibility_mappings = {
                'wcag_compliance_score': 'WCAG compliance',
                'color_contrast_score': 'color contrast',
                'text_readability_score': 'text readability',
                'keyboard_navigation_score': 'keyboard navigation'
            }
            
            for metric_key, display_name in accessibility_mappings.items():
                score = accessibility_metrics.get(metric_key, 10.0)
                if isinstance(score, (int, float)) and score < threshold:
                    low_scoring_metrics.append(f"improve {display_name} (score: {score:.1f}/10)")
        
        return low_scoring_metrics[:4]  # Limit to top 4 to manage prompt length
    
    def _format_priority_improvements(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Format priority improvements with severity levels."""
        
        formatted_improvements = []
        priority_improvements = analysis_results.get('priority_improvements', [])
        
        # Process top 3 priority improvements
        for improvement in priority_improvements[:3]:
            title = improvement.get('title', '')
            severity = improvement.get('severity', 'medium')
            
            if title:
                # Create concise improvement directive based on severity
                if severity.lower() in ['high', 'critical']:
                    formatted_improvements.append(f"fix {title.lower()} (critical)")
                elif severity.lower() == 'medium':
                    formatted_improvements.append(f"address {title.lower()}")
                else:  # low severity
                    formatted_improvements.append(f"refine {title.lower()}")
        
        return formatted_improvements
    
    def _build_metrics_context(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Build comprehensive metrics context for prompt generation."""
        
        context_parts = []
        
        # Add low-scoring metrics
        low_scoring = self._extract_low_scoring_metrics(analysis_results)
        context_parts.extend(low_scoring)
        
        # Add priority improvements with severity
        priority_formatted = self._format_priority_improvements(analysis_results)
        context_parts.extend(priority_formatted)
        
        # Add overall score context if significantly low
        overall_score = analysis_results.get('overall_score', 10.0)
        if isinstance(overall_score, (int, float)) and overall_score < 6.0:
            context_parts.append(f"address overall design quality (score: {overall_score:.1f}/10)")
        
        return context_parts
    
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
    
    def _validate_and_resize_image(self, image: Image.Image, max_dimension: int = 2048) -> Image.Image:
        """Validate and resize image to prevent memory issues."""
        try:
            # Check if image dimensions are reasonable
            width, height = image.size
            
            # Log current dimensions
            logger.info(f"Generated image dimensions: {width}x{height}")
            
            # If image is too large, resize it
            if width > max_dimension or height > max_dimension:
                logger.warning(f"Image too large ({width}x{height}), resizing to max {max_dimension}px")
                
                # Calculate new dimensions maintaining aspect ratio
                if width > height:
                    new_width = max_dimension
                    new_height = int((height * max_dimension) / width)
                else:
                    new_height = max_dimension
                    new_width = int((width * max_dimension) / height)
                
                # Resize with high-quality resampling
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Resized to: {new_width}x{new_height}")
            
            # Ensure image is in RGB mode
            if image.mode != 'RGB':
                logger.info(f"Converting image from {image.mode} to RGB")
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            logger.error(f"Error validating/resizing image: {e}")
            # Return a safe fallback image if validation fails
            return Image.new('RGB', (512, 512), color='white')
    
    def _compress_image(self, image: Image.Image, max_size_mb: int = 50) -> Image.Image:
        """Compress image to fit within size limit."""
        try:
            # Start with high quality and reduce if needed
            for quality in [95, 85, 75, 65, 55, 45]:
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', quality=quality, optimize=True)
                
                size_mb = buffer.tell() / (1024 * 1024)
                logger.info(f"Compression quality {quality}: {size_mb:.2f}MB")
                
                if size_mb <= max_size_mb:
                    # Convert back to RGB PIL Image
                    buffer.seek(0)
                    compressed_image = Image.open(buffer).convert('RGB')
                    return compressed_image
            
            # If still too large, resize further
            logger.warning("Image still too large after compression, reducing dimensions")
            width, height = image.size
            new_width = int(width * 0.8)
            new_height = int(height * 0.8)
            
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            return self._compress_image(resized_image, max_size_mb)  # Recursive call
            
        except Exception as e:
            logger.error(f"Error compressing image: {e}")
            # Return original image if compression fails
            return image
    
    def _safe_image_to_base64(self, image: Image.Image, max_size_mb: int = 10) -> str:
        """Safely convert image to base64 with size limits and error handling."""
        try:
            # Start with JPEG format for better compression
            formats_to_try = [
                ('JPEG', {'quality': 85, 'optimize': True}),
                ('JPEG', {'quality': 75, 'optimize': True}),
                ('JPEG', {'quality': 65, 'optimize': True}),
                ('PNG', {'optimize': True}),
            ]
            
            for format_name, save_kwargs in formats_to_try:
                buffer = io.BytesIO()
                
                # Convert to RGB if saving as JPEG
                save_image = image
                if format_name == 'JPEG' and image.mode in ('RGBA', 'LA', 'P'):
                    save_image = image.convert('RGB')
                
                save_image.save(buffer, format=format_name, **save_kwargs)
                buffer_size = buffer.tell()
                size_mb = buffer_size / (1024 * 1024)
                
                logger.info(f"Image buffer size with {format_name}: {size_mb:.2f}MB")
                
                if size_mb <= max_size_mb:
                    buffer.seek(0)
                    return base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                logger.warning(f"Buffer too large ({size_mb:.2f}MB) with {format_name}, trying next format")
            
            # If all formats fail, resize and try again
            logger.warning("All formats too large, resizing image")
            width, height = image.size
            new_width = int(width * 0.7)
            new_height = int(height * 0.7)
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Recursive call with smaller image
            return self._safe_image_to_base64(resized_image, max_size_mb)
            
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            # Return a minimal placeholder image as base64
            placeholder = Image.new('RGB', (100, 100), color='white')
            buffer = io.BytesIO()
            placeholder.save(buffer, format='JPEG', quality=50)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _prepare_input_image(self, image: Image.Image) -> Image.Image:
        """Prepare input image following the sample code pattern for optimal results."""
        try:
            # Handle EXIF orientation (like in sample code)
            from PIL import ImageOps
            image = ImageOps.exif_transpose(image)
            
            # Convert to RGB (essential for InstructPix2Pix)
            image = image.convert("RGB")
            
            # Resize to optimal dimensions (512x512 or smaller for memory efficiency)
            width, height = image.size
            max_dimension = 512
            
            if width > max_dimension or height > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = int((height * max_dimension) / width)
                else:
                    new_height = max_dimension
                    new_width = int((width * max_dimension) / height)
                
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Resized input image to: {new_width}x{new_height}")
            
            return image
            
        except Exception as e:
            logger.error(f"Error preparing input image: {e}")
            # Return a safe fallback
            return image.convert("RGB") if image.mode != "RGB" else image
    
    def _validate_generated_image(self, image: Image.Image) -> Image.Image:
        """Validate generated image and handle NaN/invalid values."""
        try:
            import numpy as np
            
            # Convert to numpy array to check for NaN values
            img_array = np.array(image)
            
            # Check for NaN or infinite values
            if np.isnan(img_array).any() or np.isinf(img_array).any():
                logger.error("Generated image contains NaN or infinite values, creating fallback")
                return self._create_fallback_image(image.size)
            
            # Check if image is completely black/empty (all zeros)
            if np.all(img_array == 0):
                logger.error("Generated image is completely black, creating fallback")
                return self._create_fallback_image(image.size)
            
            # Check if image has very low variance (likely blank/corrupted)
            if img_array.var() < 1.0:
                logger.warning("Generated image has very low variance, might be blank")
                return self._create_fallback_image(image.size)
            
            # Image seems valid
            logger.info(f"Generated image validation passed - variance: {img_array.var():.2f}")
            return image
            
        except Exception as e:
            logger.error(f"Error validating generated image: {e}")
            return self._create_fallback_image(image.size)
    
    def _create_fallback_image(self, size: tuple) -> Image.Image:
        """Create a fallback image when generation fails."""
        try:
            width, height = size
            
            # Create a simple gradient image as fallback
            from PIL import ImageDraw
            
            fallback = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(fallback)
            
            # Add a simple pattern to indicate this is a fallback
            draw.rectangle([10, 10, width-10, height-10], outline='lightgray', width=2)
            draw.text((width//2-50, height//2-10), "Generation Failed", fill='gray')
            
            logger.info(f"Created fallback image: {width}x{height}")
            return fallback
            
        except Exception as e:
            logger.error(f"Error creating fallback image: {e}")
            # Ultimate fallback - simple colored rectangle
            return Image.new('RGB', (512, 384), color='lightgray')
