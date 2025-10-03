"""FastAPI main application for the multimodal design analysis suite."""

import os
import uuid
import asyncio
import json
import csv
import re
from pathlib import Path
from typing import List, Optional
import logging
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from PIL import Image
import io
import base64

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional PDF generation imports
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab not available. PDF download functionality will be disabled.")

from ..config import settings
from ..models.schemas import (
    AnalysisResult, AnalysisStatus, DesignContext, AnalysisType, UploadRequest,
    AnalysisHistoryItem, AnalysisHistoryResponse, DownloadFormat
)
from pydantic import BaseModel
from ..agents.orchestrator import get_orchestrator, get_analysis_agent_states, cleanup_analysis_states
from ..utils.vector_store import get_vector_store
from ..utils.llm_client import get_llm_client

# Logger already configured above

# Create FastAPI app
app = FastAPI(
    title="Multimodal Design Analysis Suite",
    description="AI-powered UI/UX design analysis using multimodal agents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for tracking analyses
analysis_status: dict = {}


class ChatRequest(BaseModel):
    message: str
    analysis_result: Optional[dict] = None


class ChatResponse(BaseModel):
    response: str

# Initialize components on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the application components."""
    try:
        logger.info("Initializing application components...")
        
        # Initialize vector store and populate with sample data
        vector_store = get_vector_store()
        stats = vector_store.get_collection_stats()
        
        if stats["total_patterns"] == 0:
            logger.info("Populating vector store with sample data...")
            vector_store.populate_sample_data()
            logger.info("Sample data populated successfully")
        
        # Initialize orchestrator
        orchestrator = get_orchestrator()
        
        logger.info("Application initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Multimodal Design Analysis Suite API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/analyze",
            "status": "/analysis/{analysis_id}/status",
            "result": "/analysis/{analysis_id}/result",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check vector store
        vector_store = get_vector_store()
        stats = vector_store.get_collection_stats()
        
        return {
            "status": "healthy",
            "components": {
                "vector_store": "operational",
                "total_patterns": stats["total_patterns"]
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.post("/analyze", response_model=dict)
async def analyze_design(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    project_name: Optional[str] = Form(None),
    target_audience: Optional[str] = Form(None),
    design_goals: Optional[str] = Form(None),
    industry: Optional[str] = Form(None),
    platform: Optional[str] = Form(None),
    additional_notes: Optional[str] = Form(None),
    analysis_types: Optional[str] = Form(None),  # Comma-separated string
    x_openrouter_key: Optional[str] = Header(None, alias="X-OpenRouter-Key"),
    x_huggingface_key: Optional[str] = Header(None, alias="X-HuggingFace-Key")
):
    """Upload and analyze a design file."""
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, etc.)"
        )
    
    # Check file size
    if file.size and file.size > settings.get_max_file_size_bytes():
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds maximum allowed size of {settings.get_max_file_size_bytes()} bytes"
        )
    
    try:
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Read and validate image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save uploaded file
        file_path = settings.upload_dir / f"{analysis_id}_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(image_data)
        
        # Parse context
        context = None
        if any([project_name, target_audience, design_goals, industry, platform, additional_notes]):
            context = DesignContext(
                project_name=project_name,
                target_audience=target_audience,
                design_goals=design_goals,
                industry=industry,
                platform=platform,
                additional_notes=additional_notes
            )
        
        # Parse requested analysis types
        requested_analyses = []
        if analysis_types:
            type_names = [t.strip().upper() for t in analysis_types.split(',')]
            for type_name in type_names:
                try:
                    analysis_type = AnalysisType(type_name.lower())
                    requested_analyses.append(analysis_type)
                except ValueError:
                    logger.warning(f"Unknown analysis type: {type_name}")
        
        # If no specific types requested, use all
        if not requested_analyses:
            requested_analyses = [
                AnalysisType.VISUAL,
                AnalysisType.UX_CRITIQUE,
                AnalysisType.MARKET_RESEARCH
            ]
        
        # Initialize analysis status
        analysis_status[analysis_id] = AnalysisStatus(
            analysis_id=analysis_id,
            status="pending",
            progress=0.0,
            current_stage="queued",
            estimated_completion=None,
            error_message=None
        )
        
        # Start analysis in background
        background_tasks.add_task(
            run_analysis,
            analysis_id,
            image,
            context,
            requested_analyses,
            file.filename or "uploaded_design.png",
            x_openrouter_key,
            x_huggingface_key
        )
        
        logger.info(f"Analysis {analysis_id} queued for file: {file.filename}")
        
        return {
            "analysis_id": analysis_id,
            "status": "queued",
            "message": "Analysis started successfully",
            "estimated_time": "2-5 minutes"
        }
        
    except Exception as e:
        logger.error(f"Failed to start analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start analysis: {str(e)}"
        )


@app.get("/analysis/{analysis_id}/status", response_model=AnalysisStatus)
async def get_analysis_status(analysis_id: str):
    """Get the status of an analysis."""
    
    if analysis_id not in analysis_status:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found"
        )
    
    status = analysis_status[analysis_id]
    
    # Get current agent states from orchestrator
    agent_states = get_analysis_agent_states(analysis_id)
    logger.info(f"[API STATUS] Retrieved agent states for {analysis_id}: {agent_states}")
    logger.info(f"[API STATUS] Current analysis status: {status.status}, progress: {status.progress}")
    
    # Always set agent states if available, otherwise initialize with pending states
    if agent_states:
        status.agent_states = agent_states
        logger.info(f"[API STATUS] Using orchestrator agent states: {agent_states}")
    elif status.status in ["pending", "processing"]:
        # Initialize with pending states for all agents
        status.agent_states = {
            "Visual Design Analyst": "pending",
            "UX Experience Critic": "pending", 
            "Market Research Specialist": "pending"
        }
        logger.info(f"[API STATUS] Using default pending states: {status.agent_states}")
    
    return status


@app.get("/analysis/{analysis_id}/result", response_model=AnalysisResult)
async def get_analysis_result(analysis_id: str):
    """Get the result of a completed analysis."""
    
    if analysis_id not in analysis_status:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found"
        )
    
    status = analysis_status[analysis_id]
    
    if status.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Analysis not completed. Current status: {status.status}"
        )
    
    # Load result from file or cache
    result_file = settings.upload_dir / f"{analysis_id}_result.json"
    
    if not result_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Analysis result not found"
        )
    
    try:
        import json
        with open(result_file, 'r') as f:
            result_data = json.load(f)
        
        return AnalysisResult(**result_data)
        
    except Exception as e:
        logger.error(f"Failed to load analysis result: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to load analysis result"
        )


@app.delete("/analysis/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """Delete an analysis and its associated files."""
    
    # Check if analysis files exist instead of just checking in-memory status
    result_file = settings.upload_dir / f"{analysis_id}_result.json"
    analysis_files = list(settings.upload_dir.glob(f"{analysis_id}_*"))
    
    if not result_file.exists() and not analysis_files:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found"
        )
    
    try:
        # Remove from status tracking if it exists
        if analysis_id in analysis_status:
            del analysis_status[analysis_id]
        
        # Clean up agent states
        cleanup_analysis_states(analysis_id)
        
        # Clean up files
        files_deleted = 0
        for pattern in [f"{analysis_id}_*", f"*_{analysis_id}_*"]:
            for file_path in settings.upload_dir.glob(pattern):
                file_path.unlink(missing_ok=True)
                files_deleted += 1
        
        logger.info(f"Analysis {analysis_id} deleted successfully ({files_deleted} files removed)")
        
        return {"message": "Analysis deleted successfully"}
        
    except Exception as e:
        logger.error(f"Failed to delete analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete analysis"
        )


@app.get("/vector-store/stats")
async def get_vector_store_stats():
    """Get vector store statistics."""
    try:
        vector_store = get_vector_store()
        stats = vector_store.get_collection_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get vector store stats: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get vector store statistics"
        )


@app.get("/analysis/history", response_model=AnalysisHistoryResponse)
async def get_analysis_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    search: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None)
):
    """Get analysis history with pagination and filtering."""
    try:
        # Get all completed analyses from the uploads directory
        analyses = []
        
        for result_file in settings.upload_dir.glob("*_result.json"):
            try:
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                
                # Extract analysis ID from filename
                analysis_id = result_file.stem.replace('_result', '')
                
                # Find corresponding image file
                image_files = list(settings.upload_dir.glob(f"{analysis_id}_*"))
                image_file = None
                for img_file in image_files:
                    if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        image_file = img_file
                        break
                
                # Create history item
                history_item = AnalysisHistoryItem(
                    analysis_id=analysis_id,
                    upload_filename=result_data.get('upload_filename', 'Unknown'),
                    analysis_timestamp=result_data.get('analysis_timestamp', ''),
                    overall_score=result_data.get('overall_score', 0.0),
                    status='completed',
                    processing_time=result_data.get('processing_time'),
                    context=result_data.get('context'),
                    thumbnail_path=str(image_file) if image_file else None
                )
                
                # Apply filters
                if search and search.lower() not in history_item.upload_filename.lower():
                    continue
                    
                if status_filter and status_filter != history_item.status:
                    continue
                
                analyses.append(history_item)
                
            except Exception as e:
                logger.warning(f"Failed to parse result file {result_file}: {e}")
                continue
        
        # Sort by timestamp (newest first)
        analyses.sort(key=lambda x: x.analysis_timestamp, reverse=True)
        
        # Apply pagination
        total_count = len(analyses)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_analyses = analyses[start_idx:end_idx]
        
        return AnalysisHistoryResponse(
            analyses=paginated_analyses,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Failed to get analysis history: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get analysis history"
        )


@app.get("/analysis/{analysis_id}/download")
async def download_analysis(
    analysis_id: str,
    format: DownloadFormat = Query(DownloadFormat.JSON)
):
    """Download analysis in various formats."""
    
    # Check if analysis exists
    result_file = settings.upload_dir / f"{analysis_id}_result.json"
    if not result_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Analysis not found"
        )
    
    try:
        # Load analysis result
        with open(result_file, 'r') as f:
            result_data = json.load(f)
        
        if format == DownloadFormat.JSON:
            return download_as_json(result_data, analysis_id)
        elif format == DownloadFormat.PDF:
            return await download_as_pdf(result_data, analysis_id)
        elif format == DownloadFormat.PNG:
            return await download_as_png(result_data, analysis_id)
        elif format == DownloadFormat.CSV:
            return download_as_csv(result_data, analysis_id)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {format}"
            )
            
    except Exception as e:
        logger.error(f"Failed to download analysis {analysis_id} as {format}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate {format.upper()} download"
        )


@app.post("/analysis/{analysis_id}/chat", response_model=ChatResponse)
async def chat_with_analysis(
    analysis_id: str,
    chat_request: ChatRequest,
    x_openrouter_key: Optional[str] = Header(None, alias="X-OpenRouter-Key")
):
    """Chat with AI about analysis results."""
    
    try:
        # Get analysis result if not provided in request
        analysis_result = chat_request.analysis_result
        
        if not analysis_result:
            # Load result from file
            result_file = settings.upload_dir / f"{analysis_id}_result.json"
            
            if not result_file.exists():
                raise HTTPException(
                    status_code=404,
                    detail="Analysis not found"
                )
            
            with open(result_file, 'r') as f:
                analysis_result = json.load(f)
        
        # Get LLM client
        llm_client = get_llm_client(x_openrouter_key)
        
        if not llm_client.llm:
            raise HTTPException(
                status_code=503,
                detail="LLM service not available. Please check API key configuration."
            )
        
        # Create context from analysis result
        context = create_analysis_context(analysis_result)
        
        # Generate response using LLM
        response = await generate_chat_response(
            llm_client,
            chat_request.message,
            context,
            analysis_result
        )
        
        if not response:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate response"
            )
        
        return ChatResponse(response=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error for analysis {analysis_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process chat message"
        )


async def run_analysis(
    analysis_id: str,
    image: Image.Image,
    context: Optional[DesignContext],
    requested_analyses: List[AnalysisType],
    filename: str,
    openrouter_key: Optional[str] = None,
    huggingface_key: Optional[str] = None
):
    """Run the analysis workflow in the background."""
    
    try:
        logger.info(f"Starting analysis {analysis_id}")
        
        # Update status
        analysis_status[analysis_id].status = "processing"
        analysis_status[analysis_id].progress = 0.1
        analysis_status[analysis_id].current_stage = "initializing"
        
        # Get orchestrator and run analysis
        orchestrator = get_orchestrator()
        
        # Update progress during analysis
        analysis_status[analysis_id].progress = 0.3
        analysis_status[analysis_id].current_stage = "running_agents"
        
        # Define progress callback to update analysis status
        def progress_callback(progress: float, stage: str, agent_states: dict):
            analysis_status[analysis_id].progress = progress
            analysis_status[analysis_id].current_stage = stage
            logger.info(f"Analysis {analysis_id} progress: {progress:.1%} - {stage}")
        
        result = await orchestrator.analyze_design(
            image=image,
            context=context,
            requested_analyses=requested_analyses,
            progress_callback=progress_callback,
            analysis_id=analysis_id,
            api_keys={
                'openrouter': openrouter_key or settings.openrouter_api_key,
                'huggingface': huggingface_key or settings.huggingface_api_token
            }
        )
        
        # Update filename in result
        result.upload_filename = filename
        
        # Save result to file
        result_file = settings.upload_dir / f"{analysis_id}_result.json"
        with open(result_file, 'w') as f:
            import json
            json.dump(result.model_dump(), f, indent=2)
        
        # Update status to completed
        analysis_status[analysis_id].status = "completed"
        analysis_status[analysis_id].progress = 1.0
        analysis_status[analysis_id].current_stage = "completed"
        
        # Clean up agent states
        cleanup_analysis_states(analysis_id)
        
        logger.info(f"Analysis {analysis_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis {analysis_id} failed: {e}")
        
        # Update status to failed
        analysis_status[analysis_id].status = "failed"
        analysis_status[analysis_id].error_message = str(e)
        analysis_status[analysis_id].current_stage = "failed"
        
        # Clean up agent states
        cleanup_analysis_states(analysis_id)


def clean_markdown_text(text: str) -> str:
    """Clean markdown formatting from text for PDF/CSV output."""
    if not text:
        return text
    
    # Remove markdown headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Remove bold/italic formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
    text = re.sub(r'__(.*?)__', r'\1', text)      # Bold
    text = re.sub(r'_(.*?)_', r'\1', text)        # Italic
    
    # Remove code blocks and inline code
    text = re.sub(r'```[\s\S]*?```', '', text)    # Code blocks
    text = re.sub(r'`(.*?)`', r'\1', text)        # Inline code
    
    # Remove links but keep text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Remove bullet points and list formatting
    text = re.sub(r'^[\s]*[-*+]\s+', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '• ', text, flags=re.MULTILINE)
    
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    
    return text


def download_as_json(result_data: dict, analysis_id: str):
    """Download analysis as JSON."""
    json_content = json.dumps(result_data, indent=2)
    
    return StreamingResponse(
        io.StringIO(json_content),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=analysis_{analysis_id}.json"}
    )


async def download_as_pdf(result_data: dict, analysis_id: str):
    """Download analysis as PDF."""
    if not REPORTLAB_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="PDF generation not available. ReportLab dependency not installed."
        )
    
    try:
        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2563eb')
        )
        story.append(Paragraph(f"Design Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Basic Info
        story.append(Paragraph(f"<b>Analysis ID:</b> {analysis_id}", styles['Normal']))
        story.append(Paragraph(f"<b>File:</b> {result_data.get('upload_filename', 'Unknown')}", styles['Normal']))
        story.append(Paragraph(f"<b>Date:</b> {result_data.get('analysis_timestamp', 'Unknown')}", styles['Normal']))
        story.append(Paragraph(f"<b>Overall Score:</b> {result_data.get('overall_score', 0):.2f}/10", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Key Strengths
        if result_data.get('key_strengths'):
            story.append(Paragraph("Key Strengths", styles['Heading2']))
            for strength in result_data['key_strengths']:
                story.append(Paragraph(f"• {strength}", styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Priority Improvements
        if result_data.get('priority_improvements'):
            story.append(Paragraph("Priority Improvements", styles['Heading2']))
            for improvement in result_data['priority_improvements']:
                story.append(Paragraph(f"<b>{improvement.get('title', 'N/A')}</b>", styles['Normal']))
                story.append(Paragraph(f"Severity: {improvement.get('severity', 'N/A')}", styles['Normal']))
                story.append(Paragraph(f"{improvement.get('description', 'N/A')}", styles['Normal']))
                story.append(Paragraph(f"<i>Recommendation: {improvement.get('recommendation', 'N/A')}</i>", styles['Normal']))
                story.append(Spacer(1, 10))
        
        # Agent Results
        if result_data.get('agent_results'):
            story.append(Paragraph("Detailed Analysis", styles['Heading2']))
            for agent_result in result_data['agent_results']:
                story.append(Paragraph(f"<b>{agent_result.get('agent_name', 'Unknown Agent')}</b>", styles['Heading3']))
                story.append(Paragraph(f"Score: {agent_result.get('overall_score', 0):.2f}/10", styles['Normal']))
                
                # Findings
                if agent_result.get('findings'):
                    for finding in agent_result['findings'][:3]:  # Limit to first 3 findings
                        title = clean_markdown_text(finding.get('title', 'N/A'))
                        description = clean_markdown_text(finding.get('description', 'N/A'))
                        story.append(Paragraph(f"• <b>{title}</b>", styles['Normal']))
                        story.append(Paragraph(f"  {description}", styles['Normal']))
                story.append(Spacer(1, 15))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(buffer.read()),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=analysis_{analysis_id}.pdf"}
        )
        
    except Exception as e:
        logger.error(f"Failed to generate PDF: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate PDF")


async def download_as_png(result_data: dict, analysis_id: str):
    """Download analysis as PNG image."""
    try:
        # Find the original image file
        image_files = list(settings.upload_dir.glob(f"{analysis_id}_*"))
        image_file = None
        
        for img_file in image_files:
            if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                image_file = img_file
                break
        
        if not image_file:
            raise HTTPException(status_code=404, detail="Original image not found")
        
        return FileResponse(
            path=image_file,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=analysis_{analysis_id}_original.png"}
        )
        
    except Exception as e:
        logger.error(f"Failed to download PNG: {e}")
        raise HTTPException(status_code=500, detail="Failed to download image")


def download_as_csv(result_data: dict, analysis_id: str):
    """Download analysis as CSV."""
    try:
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(['Analysis Report'])
        writer.writerow(['Analysis ID', analysis_id])
        writer.writerow(['File', result_data.get('upload_filename', 'Unknown')])
        writer.writerow(['Date', result_data.get('analysis_timestamp', 'Unknown')])
        writer.writerow(['Overall Score', f"{result_data.get('overall_score', 0):.2f}/10"])
        writer.writerow([])
        
        # Agent Results
        if result_data.get('agent_results'):
            writer.writerow(['Agent Results'])
            writer.writerow(['Agent Name', 'Score', 'Execution Time'])
            for agent_result in result_data['agent_results']:
                writer.writerow([
                    agent_result.get('agent_name', 'Unknown'),
                    f"{agent_result.get('overall_score', 0):.2f}/10",
                    f"{agent_result.get('execution_time', 0):.2f}s"
                ])
            writer.writerow([])
        
        # Findings
        if result_data.get('agent_results'):
            writer.writerow(['Findings'])
            writer.writerow(['Agent', 'Category', 'Severity', 'Title', 'Description', 'Recommendation'])
            for agent_result in result_data['agent_results']:
                agent_name = agent_result.get('agent_name', 'Unknown')
                for finding in agent_result.get('findings', []):
                    writer.writerow([
                        agent_name,
                        finding.get('category', 'N/A'),
                        finding.get('severity', 'N/A'),
                        clean_markdown_text(finding.get('title', 'N/A')),
                        clean_markdown_text(finding.get('description', 'N/A')),
                        clean_markdown_text(finding.get('recommendation', 'N/A'))
                    ])
        
        csv_content = output.getvalue()
        output.close()
        
        return StreamingResponse(
            io.StringIO(csv_content),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=analysis_{analysis_id}.csv"}
        )
        
    except Exception as e:
        logger.error(f"Failed to generate CSV: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate CSV")


def create_analysis_context(analysis_result: dict) -> str:
    """Create a structured context from analysis result for chat."""
    try:
        context_parts = []
        
        # Basic info
        context_parts.append(f"Analysis ID: {analysis_result.get('analysis_id', 'Unknown')}")
        context_parts.append(f"File: {analysis_result.get('upload_filename', 'Unknown')}")
        context_parts.append(f"Overall Score: {analysis_result.get('overall_score', 0):.2f}/10")
        context_parts.append(f"Processing Time: {analysis_result.get('processing_time', 0):.2f}s")
        
        # Key strengths
        if analysis_result.get('key_strengths'):
            context_parts.append("\nKey Strengths:")
            for strength in analysis_result['key_strengths']:
                context_parts.append(f"- {strength}")
        
        # Priority improvements
        if analysis_result.get('priority_improvements'):
            context_parts.append("\nPriority Improvements:")
            for improvement in analysis_result['priority_improvements']:
                context_parts.append(f"- {improvement.get('title', 'N/A')}: {improvement.get('description', 'N/A')}")
                context_parts.append(f"  Severity: {improvement.get('severity', 'N/A')}")
                context_parts.append(f"  Recommendation: {improvement.get('recommendation', 'N/A')}")
        
        # Agent results summary
        if analysis_result.get('agent_results'):
            context_parts.append("\nAgent Analysis Results:")
            for agent_result in analysis_result['agent_results']:
                agent_name = agent_result.get('agent_name', 'Unknown Agent')
                score = agent_result.get('overall_score', 0)
                context_parts.append(f"- {agent_name}: {score:.2f}/10")
                
                # Top findings
                findings = agent_result.get('findings', [])
                if findings:
                    context_parts.append(f"  Top findings:")
                    for finding in findings[:3]:  # Limit to top 3 findings
                        title = finding.get('title', 'N/A')
                        severity = finding.get('severity', 'N/A')
                        context_parts.append(f"    • {title} (Severity: {severity})")
        
        # Metrics
        if analysis_result.get('visual_metrics'):
            context_parts.append("\nVisual Metrics:")
            for key, value in analysis_result['visual_metrics'].items():
                if isinstance(value, (int, float)):
                    context_parts.append(f"- {key}: {value:.2f}")
        
        if analysis_result.get('ux_metrics'):
            context_parts.append("\nUX Metrics:")
            for key, value in analysis_result['ux_metrics'].items():
                if isinstance(value, (int, float)):
                    context_parts.append(f"- {key}: {value:.2f}")
        
        # Market comparison
        if analysis_result.get('market_comparison'):
            market_comp = analysis_result['market_comparison']
            if market_comp.get('competitive_analysis'):
                context_parts.append(f"\nMarket Analysis: {market_comp['competitive_analysis']}")
            
            if market_comp.get('industry_trends'):
                context_parts.append("\nIndustry Trends:")
                for trend in market_comp['industry_trends'][:3]:  # Limit to 3 trends
                    context_parts.append(f"- {trend}")
        
        return "\n".join(context_parts)
        
    except Exception as e:
        logger.error(f"Failed to create analysis context: {e}")
        return f"Analysis for {analysis_result.get('upload_filename', 'unknown file')} with overall score {analysis_result.get('overall_score', 0):.2f}/10"


async def generate_chat_response(
    llm_client,
    user_message: str,
    analysis_context: str,
    analysis_result: dict
) -> Optional[str]:
    """Generate chat response using LLM with analysis context."""
    try:
        from langchain_core.prompts import ChatPromptTemplate
        
        # Create a specialized prompt for analysis chat
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert UI/UX design analysis assistant. You have access to detailed analysis results for a design and can help users understand:

1. Specific findings and their implications
2. How to implement recommendations
3. Design scores and what they mean
4. Best practices and industry standards
5. Clarification on technical terms

Guidelines:
- Be conversational and helpful
- Provide specific, actionable advice
- Reference the analysis data when relevant
- Keep responses concise but informative
- If asked about something not in the analysis, acknowledge the limitation
- Use bullet points for lists and recommendations
- Be encouraging while being honest about areas for improvement

Analysis Context:
{analysis_context}"""),
            ("human", "{user_message}")
        ])
        
        response = await llm_client.generate_response(
            prompt_template,
            {
                "analysis_context": analysis_context,
                "user_message": user_message
            },
            temperature=0.7,
            max_tokens=800
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to generate chat response: {e}")
        return None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
