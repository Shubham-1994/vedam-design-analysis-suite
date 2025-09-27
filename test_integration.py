#!/usr/bin/env python3
"""
Integration test script for the Multimodal Design Analysis Suite.
"""

import asyncio
import sys
import logging
from pathlib import Path
from PIL import Image, ImageDraw
import io

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.agents.orchestrator import get_orchestrator
from backend.models.schemas import DesignContext, AnalysisType
from backend.utils.vector_store import get_vector_store
from backend.utils.embeddings import get_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_image() -> Image.Image:
    """Create a simple test image for analysis."""
    # Create a 800x600 image with some basic UI elements
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    # Header
    draw.rectangle([0, 0, 800, 80], fill='#3B82F6')
    draw.text((20, 30), "Test UI Design", fill='white')
    
    # Navigation
    draw.rectangle([20, 100, 780, 140], fill='#F3F4F6')
    draw.text((30, 115), "Home | About | Services | Contact", fill='black')
    
    # Main content area
    draw.rectangle([20, 160, 520, 400], fill='#FFFFFF', outline='#E5E7EB')
    draw.text((30, 180), "Main Content Area", fill='black')
    
    # Sidebar
    draw.rectangle([540, 160, 780, 400], fill='#F9FAFB', outline='#E5E7EB')
    draw.text((550, 180), "Sidebar", fill='black')
    
    # Footer
    draw.rectangle([20, 420, 780, 480], fill='#374151')
    draw.text((30, 440), "Footer Content", fill='white')
    
    # Add some buttons
    draw.rectangle([30, 220, 130, 250], fill='#10B981')
    draw.text((50, 230), "Button", fill='white')
    
    draw.rectangle([150, 220, 250, 250], fill='#EF4444')
    draw.text((170, 230), "Cancel", fill='white')
    
    return img

async def test_components():
    """Test individual components."""
    logger.info("Testing individual components...")
    
    try:
        # Test embeddings
        logger.info("Testing embeddings...")
        embeddings = get_embeddings()
        test_embedding = embeddings.encode_text("test ui design")
        logger.info(f"Embedding dimension: {len(test_embedding)}")
        
        # Test vector store
        logger.info("Testing vector store...")
        vector_store = get_vector_store()
        stats = vector_store.get_collection_stats()
        logger.info(f"Vector store stats: {stats}")
        
        # Populate sample data if empty
        if stats["total_patterns"] == 0:
            logger.info("Populating sample data...")
            vector_store.populate_sample_data()
            stats = vector_store.get_collection_stats()
            logger.info(f"Updated stats: {stats}")
        
        logger.info("✅ Component tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Component test failed: {e}")
        return False

async def test_full_analysis():
    """Test the full analysis workflow."""
    logger.info("Testing full analysis workflow...")
    
    try:
        # Create test image
        test_image = create_test_image()
        
        # Create test context
        context = DesignContext(
            project_name="Test UI Design",
            target_audience="General users",
            design_goals="Test the analysis system",
            industry="technology",
            platform="web"
        )
        
        # Get orchestrator
        orchestrator = get_orchestrator()
        
        # Run analysis
        logger.info("Starting analysis...")
        result = await orchestrator.analyze_design(
            image=test_image,
            context=context,
            requested_analyses=[
                AnalysisType.VISUAL,
                AnalysisType.UX_CRITIQUE,
                AnalysisType.MARKET_RESEARCH
            ]
        )
        
        # Validate result
        assert result is not None, "Analysis result is None"
        assert result.overall_score >= 0, "Invalid overall score"
        assert len(result.agent_results) > 0, "No agent results"
        
        logger.info(f"✅ Analysis completed successfully!")
        logger.info(f"   Overall score: {result.overall_score:.1f}/10")
        logger.info(f"   Agent results: {len(result.agent_results)}")
        logger.info(f"   Total findings: {sum(len(ar.findings) for ar in result.agent_results)}")
        logger.info(f"   Processing time: {result.processing_time:.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Full analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    logger.info("🚀 Starting Multimodal Design Analysis Suite Integration Tests")
    
    # Test components
    components_ok = await test_components()
    if not components_ok:
        logger.error("❌ Component tests failed - stopping")
        return False
    
    # Test full workflow
    analysis_ok = await test_full_analysis()
    if not analysis_ok:
        logger.error("❌ Analysis tests failed")
        return False
    
    logger.info("🎉 All integration tests passed!")
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
