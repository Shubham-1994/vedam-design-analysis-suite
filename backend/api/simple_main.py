"""Simplified FastAPI main application for demo purposes."""

import os
import uuid
import asyncio
from pathlib import Path
from typing import List, Optional
import logging
import time
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Multimodal Design Analysis Suite (Demo)",
    description="AI-powered UI/UX design analysis - Simplified Demo Version",
    version="1.0.0-demo"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory storage for demo
analysis_status = {}
upload_dir = Path("uploads")
upload_dir.mkdir(exist_ok=True)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Multimodal Design Analysis Suite API (Demo Mode)",
        "version": "1.0.0-demo",
        "note": "This is a simplified demo version. Full ML features require additional dependencies.",
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
    return {
        "status": "healthy",
        "mode": "demo",
        "components": {
            "api": "operational",
            "ml_agents": "demo_mode"
        }
    }

@app.post("/analyze")
async def analyze_design(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    project_name: Optional[str] = Form(None),
    target_audience: Optional[str] = Form(None),
    design_goals: Optional[str] = Form(None),
    industry: Optional[str] = Form(None),
    platform: Optional[str] = Form(None),
    additional_notes: Optional[str] = Form(None),
    analysis_types: Optional[str] = Form(None)
):
    """Upload and analyze design files (demo version) - supports multiple files."""
    
    # Validate we have files
    if not files:
        raise HTTPException(
            status_code=400,
            detail="At least one file must be uploaded"
        )
    
    # Validate file types and sizes
    max_size = 50 * 1024 * 1024
    processed_files = []
    
    for file in files:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' must be an image (JPEG, PNG, etc.)"
            )
        
        if file.size and file.size > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File '{file.filename}' exceeds maximum allowed size of {max_size} bytes"
            )
    
    try:
        # Generate batch analysis ID
        batch_id = str(uuid.uuid4())
        
        # Process each file
        for i, file in enumerate(files):
            # Read and validate image
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save uploaded file
            file_path = upload_dir / f"{batch_id}_{i}_{file.filename}"
            with open(file_path, "wb") as f:
                f.write(image_data)
            
            processed_files.append({
                "index": i,
                "filename": file.filename,
                "image": image,
                "file_path": file_path
            })
        
        # Initialize batch analysis status
        analysis_status[batch_id] = {
            "analysis_id": batch_id,
            "status": "pending",
            "progress": 0.0,
            "current_stage": "queued",
            "estimated_completion": None,
            "error_message": None,
            "batch_info": {
                "total_files": len(processed_files),
                "files": [{"filename": f["filename"], "index": f["index"]} for f in processed_files]
            }
        }
        
        # Start batch demo analysis in background
        background_tasks.add_task(
            run_batch_demo_analysis,
            batch_id,
            processed_files,
            {
                "project_name": project_name,
                "target_audience": target_audience,
                "design_goals": design_goals,
                "industry": industry,
                "platform": platform,
                "additional_notes": additional_notes
            },
            analysis_types.split(',') if analysis_types else ['visual', 'ux_critique', 'market_research']
        )
        
        filenames = [f["filename"] for f in processed_files]
        logger.info(f"Demo batch analysis {batch_id} queued for {len(processed_files)} files: {', '.join(filenames)}")
        
        return {
            "analysis_id": batch_id,
            "status": "queued",
            "message": f"Batch analysis started successfully for {len(processed_files)} files",
            "estimated_time": f"{30 + len(processed_files) * 15}-{60 + len(processed_files) * 20} seconds (demo mode)",
            "batch_info": {
                "total_files": len(processed_files),
                "files": filenames
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to start analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start analysis: {str(e)}"
        )

@app.get("/analysis/{analysis_id}/status")
async def get_analysis_status(analysis_id: str):
    """Get the status of an analysis."""
    
    if analysis_id not in analysis_status:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found"
        )
    
    return analysis_status[analysis_id]

@app.get("/analysis/{analysis_id}/result")
async def get_analysis_result(analysis_id: str):
    """Get the result of a completed analysis."""
    
    if analysis_id not in analysis_status:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found"
        )
    
    status = analysis_status[analysis_id]
    
    if status["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Analysis not completed. Current status: {status['status']}"
        )
    
    # Check for batch result first, then single result
    batch_result_file = upload_dir / f"{analysis_id}_batch_result.json"
    single_result_file = upload_dir / f"{analysis_id}_result.json"
    
    result_file = None
    if batch_result_file.exists():
        result_file = batch_result_file
    elif single_result_file.exists():
        result_file = single_result_file
    
    if not result_file:
        raise HTTPException(
            status_code=404,
            detail="Analysis result not found"
        )
    
    try:
        with open(result_file, 'r') as f:
            result_data = json.load(f)
        
        return result_data
        
    except Exception as e:
        logger.error(f"Failed to load analysis result: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to load analysis result"
        )

async def run_batch_demo_analysis(
    batch_id: str,
    processed_files: List[dict],
    context: dict,
    requested_analyses: List[str]
):
    """Run a batch demo analysis workflow for multiple files."""
    
    try:
        logger.info(f"Starting batch demo analysis {batch_id} for {len(processed_files)} files")
        
        # Update status
        analysis_status[batch_id]["status"] = "processing"
        analysis_status[batch_id]["progress"] = 0.1
        analysis_status[batch_id]["current_stage"] = "initializing batch"
        
        await asyncio.sleep(2)  # Simulate initialization
        
        # Process each file
        batch_results = []
        total_files = len(processed_files)
        
        for i, file_info in enumerate(processed_files):
            file_progress = (i + 1) / total_files
            base_progress = 0.1 + (file_progress * 0.8)  # 10% to 90% for processing files
            
            # Update progress for current file
            analysis_status[batch_id]["progress"] = base_progress - 0.2
            analysis_status[batch_id]["current_stage"] = f"analyzing file {i+1}/{total_files}: {file_info['filename']}"
            
            # Simulate analysis for this file
            await asyncio.sleep(2)  # Visual analysis
            analysis_status[batch_id]["progress"] = base_progress - 0.15
            
            await asyncio.sleep(2)  # UX analysis  
            analysis_status[batch_id]["progress"] = base_progress - 0.1
            
            await asyncio.sleep(1)  # Market research
            analysis_status[batch_id]["progress"] = base_progress - 0.05
            
            # Generate result for this file
            file_result = create_demo_result(
                f"{batch_id}_file_{i}",
                file_info["filename"],
                context,
                file_info["image"]
            )
            
            batch_results.append({
                "file_index": i,
                "filename": file_info["filename"],
                "result": file_result
            })
            
            analysis_status[batch_id]["progress"] = base_progress
        
        # Generate batch summary
        analysis_status[batch_id]["progress"] = 0.95
        analysis_status[batch_id]["current_stage"] = "generating batch summary"
        
        batch_result = create_batch_demo_result(batch_id, batch_results, context)
        
        # Save batch result to file
        result_file = upload_dir / f"{batch_id}_batch_result.json"
        with open(result_file, 'w') as f:
            json.dump(batch_result, f, indent=2)
        
        # Update status to completed
        analysis_status[batch_id]["status"] = "completed"
        analysis_status[batch_id]["progress"] = 1.0
        analysis_status[batch_id]["current_stage"] = "completed"
        
        logger.info(f"Demo batch analysis {batch_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Demo batch analysis {batch_id} failed: {e}")
        
        # Update status to failed
        analysis_status[batch_id]["status"] = "failed"
        analysis_status[batch_id]["error_message"] = str(e)
        analysis_status[batch_id]["current_stage"] = "failed"

async def run_demo_analysis(
    analysis_id: str,
    image: Image.Image,
    context: dict,
    requested_analyses: List[str],
    filename: str
):
    """Run a demo analysis workflow for single file (legacy support)."""
    
    try:
        logger.info(f"Starting demo analysis {analysis_id}")
        
        # Update status
        analysis_status[analysis_id]["status"] = "processing"
        analysis_status[analysis_id]["progress"] = 0.1
        analysis_status[analysis_id]["current_stage"] = "initializing"
        
        await asyncio.sleep(2)  # Simulate initialization
        
        # Simulate visual analysis
        analysis_status[analysis_id]["progress"] = 0.3
        analysis_status[analysis_id]["current_stage"] = "visual_analysis"
        await asyncio.sleep(3)
        
        # Simulate UX analysis
        analysis_status[analysis_id]["progress"] = 0.6
        analysis_status[analysis_id]["current_stage"] = "ux_analysis"
        await asyncio.sleep(3)
        
        # Simulate market research
        analysis_status[analysis_id]["progress"] = 0.8
        analysis_status[analysis_id]["current_stage"] = "market_analysis"
        await asyncio.sleep(2)
        
        # Generate demo result
        analysis_status[analysis_id]["progress"] = 0.9
        analysis_status[analysis_id]["current_stage"] = "synthesizing"
        
        # Create demo analysis result
        result = create_demo_result(analysis_id, filename, context, image)
        
        # Save result to file
        result_file = upload_dir / f"{analysis_id}_result.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Update status to completed
        analysis_status[analysis_id]["status"] = "completed"
        analysis_status[analysis_id]["progress"] = 1.0
        analysis_status[analysis_id]["current_stage"] = "completed"
        
        logger.info(f"Demo analysis {analysis_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Demo analysis {analysis_id} failed: {e}")
        
        # Update status to failed
        analysis_status[analysis_id]["status"] = "failed"
        analysis_status[analysis_id]["error_message"] = str(e)
        analysis_status[analysis_id]["current_stage"] = "failed"

def create_demo_result(analysis_id: str, filename: str, context: dict, image: Image.Image) -> dict:
    """Create a demo analysis result."""
    
    # Get image dimensions for basic analysis
    width, height = image.size
    aspect_ratio = width / height
    
    # Generate demo findings based on image properties
    findings = []
    
    # Visual analysis findings
    if aspect_ratio > 2.0:
        findings.append({
            "category": "layout",
            "severity": "medium",
            "title": "Wide Aspect Ratio Detected",
            "description": f"The design has a very wide aspect ratio ({aspect_ratio:.2f}:1), which may not work well on mobile devices.",
            "recommendation": "Consider creating responsive breakpoints for different screen sizes.",
            "confidence_score": 0.8,
            "supporting_evidence": [f"Image dimensions: {width}x{height}"]
        })
    
    if width < 800:
        findings.append({
            "category": "visual_hierarchy",
            "severity": "low",
            "title": "Compact Design Layout",
            "description": "The design appears to be optimized for smaller screens or compact layouts.",
            "recommendation": "Ensure text remains readable and interactive elements are appropriately sized.",
            "confidence_score": 0.7,
            "supporting_evidence": [f"Image width: {width}px"]
        })
    
    # UX findings based on context
    if context.get("platform") == "mobile":
        findings.append({
            "category": "usability",
            "severity": "medium",
            "title": "Mobile Platform Optimization",
            "description": "Design is intended for mobile platform. Ensure touch targets meet minimum size requirements.",
            "recommendation": "Use minimum 44px touch targets and consider thumb-friendly navigation patterns.",
            "confidence_score": 0.9,
            "supporting_evidence": ["Platform specified as mobile"]
        })
    
    # Market research findings
    if context.get("industry"):
        findings.append({
            "category": "market_analysis",
            "severity": "low",
            "title": f"Industry Best Practices for {context['industry'].title()}",
            "description": f"Design should align with {context['industry']} industry standards and user expectations.",
            "recommendation": f"Research current {context['industry']} design trends and competitor analysis.",
            "confidence_score": 0.6,
            "supporting_evidence": [f"Industry context: {context['industry']}"]
        })
    
    # Calculate demo scores
    visual_score = 7.5 + (width / 1000) * 0.5  # Simple scoring based on width
    ux_score = 8.0 if context.get("platform") == "mobile" else 7.2
    market_score = 7.8 if context.get("industry") else 6.5
    
    overall_score = (visual_score + ux_score + market_score) / 3
    
    return {
        "analysis_id": analysis_id,
        "upload_filename": filename,
        "context": context,
        "agent_results": [
            {
                "agent_name": "Visual Analysis Agent (Demo)",
                "analysis_type": "visual",
                "findings": [f for f in findings if f["category"] in ["layout", "visual_hierarchy", "color"]],
                "overall_score": visual_score,
                "execution_time": 3.2,
                "metadata": {
                    "visual_metrics": {
                        "layout_score": visual_score,
                        "visual_hierarchy_score": visual_score - 0.3,
                        "color_harmony_score": 7.8,
                        "typography_score": 7.5,
                        "whitespace_usage_score": 8.1
                    }
                }
            },
            {
                "agent_name": "UX Critique Agent (Demo)",
                "analysis_type": "ux_critique",
                "findings": [f for f in findings if f["category"] in ["usability", "navigation", "accessibility"]],
                "overall_score": ux_score,
                "execution_time": 2.8,
                "metadata": {
                    "ux_metrics": {
                        "usability_score": ux_score,
                        "navigation_clarity": ux_score - 0.2,
                        "information_architecture": 7.9,
                        "user_flow_efficiency": 8.3
                    }
                }
            },
            {
                "agent_name": "Market Research Agent (Demo)",
                "analysis_type": "market_research",
                "findings": [f for f in findings if f["category"] in ["market_analysis", "trends", "competitive"]],
                "overall_score": market_score,
                "execution_time": 4.1,
                "metadata": {
                    "market_comparison": {
                        "similar_designs": [
                            {
                                "id": "demo_pattern_1",
                                "description": "Modern minimalist interface design",
                                "similarity_score": 0.78,
                                "category": "layout",
                                "usability_score": 8.2
                            },
                            {
                                "id": "demo_pattern_2", 
                                "description": "Clean card-based layout system",
                                "similarity_score": 0.65,
                                "category": "visual_hierarchy",
                                "usability_score": 7.9
                            }
                        ],
                        "industry_trends": [
                            "Minimalist design approach",
                            "Mobile-first responsive design",
                            "Accessibility-focused interfaces",
                            "Dark mode compatibility",
                            "Micro-interactions and animations"
                        ],
                        "competitive_analysis": "Design shows good alignment with current market trends. Consider enhancing unique differentiators while maintaining usability standards."
                    }
                }
            }
        ],
        "overall_score": overall_score,
        "visual_metrics": {
            "layout_score": visual_score,
            "visual_hierarchy_score": visual_score - 0.3,
            "color_harmony_score": 7.8,
            "typography_score": 7.5,
            "whitespace_usage_score": 8.1
        },
        "ux_metrics": {
            "usability_score": ux_score,
            "navigation_clarity": ux_score - 0.2,
            "information_architecture": 7.9,
            "user_flow_efficiency": 8.3
        },
        "key_strengths": [
            "Clean and modern visual design",
            "Good use of whitespace",
            "Appropriate sizing for target platform"
        ],
        "priority_improvements": [f for f in findings if f["severity"] in ["high", "critical"]],
        "actionable_recommendations": [
            "Ensure responsive design across all device sizes",
            "Test with real users for usability validation",
            "Consider accessibility guidelines (WCAG 2.1)",
            "Optimize loading performance for better user experience",
            "Add micro-interactions to enhance user engagement"
        ],
        "analysis_timestamp": datetime.now().isoformat(),
        "processing_time": 10.2,
        "confidence_level": 0.75
    }

def create_batch_demo_result(batch_id: str, batch_results: List[dict], context: dict) -> dict:
    """Create a batch analysis result summary."""
    
    total_files = len(batch_results)
    
    # Calculate aggregate scores
    overall_scores = [result["result"]["overall_score"] for result in batch_results]
    avg_overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0
    
    # Aggregate findings by severity
    all_findings = []
    for result in batch_results:
        for agent_result in result["result"]["agent_results"]:
            for finding in agent_result["findings"]:
                finding["source_file"] = result["filename"]
                all_findings.append(finding)
    
    # Count findings by severity
    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for finding in all_findings:
        severity_counts[finding["severity"]] = severity_counts.get(finding["severity"], 0) + 1
    
    # Get top recommendations across all files
    all_recommendations = []
    for result in batch_results:
        all_recommendations.extend(result["result"]["actionable_recommendations"])
    
    # Remove duplicates and get top 10
    unique_recommendations = list(dict.fromkeys(all_recommendations))[:10]
    
    # Create batch summary
    batch_summary = {
        "batch_id": batch_id,
        "analysis_type": "batch_analysis",
        "total_files": total_files,
        "context": context,
        
        # Aggregate metrics
        "batch_metrics": {
            "average_overall_score": avg_overall_score,
            "score_range": {
                "min": min(overall_scores) if overall_scores else 0,
                "max": max(overall_scores) if overall_scores else 0
            },
            "findings_summary": severity_counts,
            "total_findings": len(all_findings)
        },
        
        # Individual file results with full analysis data
        "file_results": [
            {
                "file_index": result["file_index"],
                "filename": result["filename"],
                "overall_score": result["result"]["overall_score"],
                "key_findings_count": len(result["result"]["agent_results"][0]["findings"]) if result["result"]["agent_results"] else 0,
                "analysis_id": result["result"]["analysis_id"],
                "result": result["result"]  # Include full analysis result
            }
            for result in batch_results
        ],
        
        # Batch-level insights
        "batch_insights": {
            "consistency_analysis": analyze_batch_consistency(batch_results),
            "common_issues": identify_common_issues(all_findings),
            "best_performing_file": max(batch_results, key=lambda x: x["result"]["overall_score"])["filename"] if batch_results else None,
            "areas_for_improvement": identify_improvement_areas(all_findings)
        },
        
        # Aggregated recommendations
        "batch_recommendations": unique_recommendations,
        
        # Metadata
        "analysis_timestamp": datetime.now().isoformat(),
        "processing_time": total_files * 3.5,  # Estimated based on file count
        "confidence_level": sum(result["result"]["confidence_level"] for result in batch_results) / total_files if batch_results else 0
    }
    
    return batch_summary

def analyze_batch_consistency(batch_results: List[dict]) -> dict:
    """Analyze consistency across batch files."""
    if len(batch_results) < 2:
        return {"status": "insufficient_data", "message": "Need at least 2 files for consistency analysis"}
    
    scores = [result["result"]["overall_score"] for result in batch_results]
    score_variance = sum((score - sum(scores)/len(scores))**2 for score in scores) / len(scores)
    
    consistency_level = "high" if score_variance < 1.0 else "medium" if score_variance < 4.0 else "low"
    
    return {
        "consistency_level": consistency_level,
        "score_variance": score_variance,
        "analysis": f"Design consistency across files is {consistency_level} (variance: {score_variance:.2f})"
    }

def identify_common_issues(all_findings: List[dict]) -> List[dict]:
    """Identify common issues across multiple files."""
    issue_counts = {}
    
    for finding in all_findings:
        category = finding["category"]
        if category in issue_counts:
            issue_counts[category]["count"] += 1
            issue_counts[category]["files"].add(finding.get("source_file", "unknown"))
        else:
            issue_counts[category] = {
                "count": 1,
                "category": category,
                "files": {finding.get("source_file", "unknown")},
                "example_title": finding["title"]
            }
    
    # Convert to list and sort by frequency
    common_issues = []
    for category, data in issue_counts.items():
        if data["count"] > 1:  # Only include issues found in multiple instances
            common_issues.append({
                "category": category,
                "frequency": data["count"],
                "affected_files": len(data["files"]),
                "example_issue": data["example_title"]
            })
    
    return sorted(common_issues, key=lambda x: x["frequency"], reverse=True)[:5]

def identify_improvement_areas(all_findings: List[dict]) -> List[str]:
    """Identify key areas for improvement across the batch."""
    category_severity = {}
    
    for finding in all_findings:
        category = finding["category"]
        severity_weight = {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(finding["severity"], 1)
        
        if category in category_severity:
            category_severity[category] += severity_weight
        else:
            category_severity[category] = severity_weight
    
    # Sort by total severity weight and return top areas
    sorted_areas = sorted(category_severity.items(), key=lambda x: x[1], reverse=True)
    
    return [area[0] for area in sorted_areas[:5]]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "simple_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
