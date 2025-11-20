# AI Design Regeneration Feature

## Overview

The AI Design Regeneration feature uses the powerful **Qwen-Image-Edit-2509** model from Hugging Face to automatically generate improved design variants based on the analysis results from your three specialized agents (Visual Analysis, UX Critique, and Market Research).

## Features

### 🎨 Intelligent Design Improvement
- **Smart Prompt Generation**: Automatically creates detailed improvement prompts based on analysis findings
- **Multi-Variant Generation**: Generate 1-3 design variants with different improvement focuses
- **Contextual Enhancements**: Incorporates industry-specific and platform-specific improvements

### 🎯 Focused Improvement Areas
- **Visual Appeal**: Enhances color harmony, visual hierarchy, and modern aesthetics
- **User Experience**: Improves usability, navigation clarity, and user-friendly layouts
- **Professional Polish**: Adds brand consistency, professional appearance, and quality refinements

### ⚙️ Advanced Controls
- **CFG Scale**: Control how closely the AI follows the improvement prompt (1-10)
- **Inference Steps**: Adjust quality vs speed trade-off (20-80 steps)
- **Negative Prompts**: Specify what to avoid in generated designs
- **Custom Parameters**: Fine-tune generation settings for specific needs

## Installation

### Quick Installation
Run the provided installation script:
```bash
python install_image_generation.py
```

### Manual Installation
If you prefer to install manually:
```bash
# Activate your virtual environment first
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install required packages
pip install diffusers>=0.30.0
pip install accelerate>=0.25.0

# Install PyTorch (GPU version recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### System Requirements

#### Minimum Requirements
- **RAM**: 8GB system RAM
- **Storage**: 10GB free space (for model downloads)
- **Python**: 3.8 or higher

#### Recommended Requirements
- **RAM**: 16GB+ system RAM
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070 or better)
- **Storage**: 20GB+ free space on SSD
- **CUDA**: Version 11.8 or higher

## Usage

### Basic Usage
1. **Complete an Analysis**: First, run a design analysis to get recommendations
2. **Access Regeneration**: Scroll to the "AI Design Regeneration" section in results
3. **Select Variants**: Choose how many design variants to generate (1-3)
4. **Generate**: Click "Generate Improved Designs" and wait for processing
5. **Download**: Save your favorite generated variants

### Advanced Usage

#### Custom Generation Options
```javascript
// Example advanced options
{
  "true_cfg_scale": 4.0,        // How closely to follow the prompt
  "num_inference_steps": 40,    // Quality vs speed (more = better quality)
  "guidance_scale": 1.0,        // Guidance strength
  "negative_prompt": "blurry, low quality, distorted, ugly, bad composition"
}
```

#### API Usage
```javascript
// Direct API call
const response = await ApiClient.regenerateDesign(analysisId, {
  analysis_id: "your-analysis-id",
  num_variants: 3,
  generation_options: {
    true_cfg_scale: 5.0,
    num_inference_steps: 50
  }
});
```

## How It Works

### 1. Analysis Integration
The system analyzes your design analysis results to identify:
- **Priority Improvements**: High-impact issues that need addressing
- **Key Strengths**: Elements to preserve in the regenerated design
- **Contextual Factors**: Industry, platform, and audience considerations

### 2. Prompt Engineering
Based on the analysis, the system creates detailed prompts like:
```
Create an improved version of this design with the following enhancements:
Improve color harmony and contrast. Enhance typography and text readability. 
Optimize layout and visual composition. Apply modern design principles. 
Optimize for e-commerce industry standards. Enhance for web platform best practices.
```

### 3. AI Generation
The Qwen-Image-Edit-2509 model processes:
- **Original Image**: Your uploaded design
- **Improvement Prompt**: Generated based on analysis
- **Generation Parameters**: Quality and style controls

### 4. Multi-Variant Output
Each variant focuses on different aspects:
- **Variant 1**: Visual appeal and aesthetics
- **Variant 2**: User experience and usability  
- **Variant 3**: Professional polish and branding

## Performance Optimization

### GPU Acceleration
- **Automatic Detection**: System automatically uses GPU if available
- **CUDA Support**: Optimized for NVIDIA GPUs with CUDA
- **Memory Management**: Efficient VRAM usage with torch.inference_mode()

### Generation Speed
- **CPU**: 2-5 minutes per variant
- **GPU (RTX 3070+)**: 30-60 seconds per variant
- **GPU (RTX 4080+)**: 15-30 seconds per variant

### Quality Settings
- **Fast**: 20 steps, CFG 3.0 (~30% faster)
- **Balanced**: 40 steps, CFG 4.0 (default)
- **High Quality**: 60+ steps, CFG 5.0+ (~50% slower)

## Troubleshooting

### Common Issues

#### "Pipeline not available" Error
```bash
# Reinstall diffusers
pip uninstall diffusers
pip install diffusers>=0.30.0
```

#### Out of Memory Errors
- Reduce `num_inference_steps` to 20-30
- Close other applications
- Use CPU mode if GPU memory is insufficient

#### Slow Generation
- Enable GPU acceleration
- Reduce number of variants
- Lower inference steps for faster generation

#### Model Download Issues
- Ensure stable internet connection
- Check available disk space (10GB+ required)
- Try clearing Hugging Face cache: `rm -rf ~/.cache/huggingface/`

### Performance Tips

1. **First Run**: Model download may take 10-15 minutes
2. **Subsequent Runs**: Much faster as model is cached
3. **Batch Generation**: Generate multiple variants at once for efficiency
4. **GPU Memory**: Monitor VRAM usage with `nvidia-smi`

## API Reference

### Endpoints

#### POST `/analysis/{analysis_id}/regenerate`
Generate improved design variants.

**Request Body:**
```json
{
  "analysis_id": "string",
  "num_variants": 1,
  "generation_options": {
    "true_cfg_scale": 4.0,
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "negative_prompt": "string"
  },
  "focus_area": "string"
}
```

**Response:**
```json
{
  "analysis_id": "string",
  "original_filename": "string",
  "variants": [
    {
      "success": true,
      "generated_image_base64": "string",
      "prompt_used": "string",
      "improvements_applied": ["string"],
      "variant_name": "string",
      "focus_area": "string"
    }
  ],
  "generation_timestamp": "string",
  "processing_time": 0.0
}
```

## Model Information

### Qwen-Image-Edit-2509
- **Developer**: Qwen Team (Alibaba)
- **Model Type**: Diffusion-based image editing
- **Capabilities**: Multi-image editing, single-image consistency, ControlNet support
- **License**: Apache 2.0
- **Model Size**: ~7GB download

### Key Improvements in 2509 Version
- **Multi-image Support**: Can work with multiple input images
- **Enhanced Consistency**: Better preservation of identity and style
- **Text Editing**: Improved text rendering and font handling
- **ControlNet Integration**: Native support for depth maps, edge maps, keypoints

## Best Practices

### For Best Results
1. **Complete Analysis First**: Ensure thorough analysis with all three agents
2. **Provide Context**: Include industry, platform, and audience information
3. **Use Specific Prompts**: More detailed analysis leads to better regeneration
4. **Iterate**: Try different variant counts and settings

### Quality Guidelines
- **High-Resolution Input**: Use images with good resolution (1024px+ recommended)
- **Clear Composition**: Avoid overly complex or cluttered designs
- **Consistent Branding**: Maintain brand elements while improving design
- **User Testing**: Validate generated designs with actual users

### Performance Guidelines
- **GPU Recommended**: Significant speed improvement with dedicated GPU
- **Batch Processing**: Generate multiple variants simultaneously
- **Monitor Resources**: Watch CPU/GPU usage during generation
- **Cache Management**: Periodically clear model cache if disk space is low

## Integration Examples

### Frontend Integration
```svelte
<script>
  import ImageRegeneration from './ImageRegeneration.svelte';
  
  // Use in analysis results
  let analysisResult = { /* your analysis data */ };
  let analysisId = "analysis-uuid";
</script>

<ImageRegeneration 
  {analysisResult} 
  {analysisId} 
/>
```

### Backend Integration
```python
from backend.agents.image_generation_agent import ImageGenerationAgent

# Initialize agent
agent = ImageGenerationAgent()

# Generate improved design
result = await agent.generate_improved_design(
    original_image=image,
    analysis_results=analysis_data,
    generation_options={
        "true_cfg_scale": 4.0,
        "num_inference_steps": 40
    }
)
```

## Future Enhancements

### Planned Features
- **Style Transfer**: Apply specific design styles (minimalist, corporate, etc.)
- **Brand Guidelines**: Automatic brand compliance checking
- **A/B Testing**: Generate variants optimized for conversion testing
- **Batch Processing**: Process multiple designs simultaneously
- **Custom Models**: Support for fine-tuned industry-specific models

### Community Contributions
- **Custom Prompts**: User-contributed improvement templates
- **Model Variants**: Support for specialized editing models
- **Performance Optimizations**: Community-driven speed improvements
- **Integration Plugins**: Third-party design tool integrations

## Support

For issues, questions, or feature requests:
1. Check this documentation first
2. Review the troubleshooting section
3. Check system requirements and installation
4. Submit issues with detailed error messages and system information

## License

This feature uses the Qwen-Image-Edit-2509 model under Apache 2.0 license. The integration code follows the same license as the main project.
