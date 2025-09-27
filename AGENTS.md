# AGENTS.md

## Project Overview

The Multimodal Design Analysis Suite is an AI-powered UI/UX design analysis platform that uses specialized agents to provide comprehensive feedback on visual design, user experience, and market positioning. The system leverages LangGraph for agent orchestration, Hugging Face for embeddings, and OpenRouter for LLM integration.

## Setup Commands

### Backend Setup
- Create virtual environment: `python -m venv venv`
- Activate environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
- Install dependencies: `pip install -r requirements.txt`
- Start backend server: `python run_backend.py`

### Frontend Setup
- Navigate to frontend: `cd frontend`
- Install dependencies: `npm install`
- Start development server: `npm run dev`

### Docker Setup (Alternative)
- Start all services: `docker-compose up --build`
- Backend will be available at `http://localhost:8000`
- Frontend will be available at `http://localhost:3000`

## Environment Configuration

Create a `.env` file in the project root with:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
HUGGINGFACE_API_TOKEN=your_huggingface_token_here
```

## Code Style & Architecture

### Backend (Python)
- **Framework**: FastAPI with async/await patterns
- **Agent Architecture**: LangGraph-based orchestration with specialized agents
- **Code Style**: PEP 8 compliant, type hints required
- **Error Handling**: Comprehensive logging and structured error responses
- **File Structure**: Modular design with clear separation of concerns

### Frontend (Svelte + TypeScript)
- **Framework**: Svelte 4 with TypeScript
- **Styling**: TailwindCSS with utility-first approach
- **State Management**: Svelte stores for reactive state
- **Code Style**: TypeScript strict mode, single quotes, no semicolons
- **Components**: Reusable component architecture

## Testing Instructions

### Backend Testing
- Run integration tests: `python test_integration.py`
- Test specific endpoints using the interactive docs at `http://localhost:8000/docs`
- Health check: `curl http://localhost:8000/health`

### Frontend Testing
- Type checking: `npm run check`
- Build verification: `npm run build`
- Preview production build: `npm run preview`

## Development Workflow

### Agent Development
- All agents inherit from `BaseAgent` in `backend/agents/base_agent.py`
- Implement `_execute_analysis` method for custom analysis logic
- Use structured schemas from `backend/models/schemas.py`
- Follow the orchestrator pattern in `backend/agents/orchestrator.py`

### API Development
- Add new endpoints to `backend/api/main.py`
- Use Pydantic models for request/response validation
- Implement proper error handling with HTTP status codes
- Add CORS configuration for frontend integration

### Frontend Development
- Create components in `frontend/src/lib/components/`
- Use stores in `frontend/src/lib/stores/` for state management
- Follow reactive patterns with Svelte's built-in reactivity
- Implement proper TypeScript interfaces

## Key Components

### Agents
1. **Visual Analysis Agent**: Analyzes layout, colors, typography, and visual hierarchy
2. **UX Critique Agent**: Evaluates usability heuristics, navigation, and accessibility
3. **Market Research Agent**: Performs competitive analysis and trend identification
4. **Orchestrator**: Coordinates agent execution using LangGraph workflows

### Data Flow
1. Image upload → FastAPI endpoint
2. Image preprocessing and validation
3. LangGraph orchestrator coordinates agents
4. RAG-enhanced analysis using ChromaDB vector store
5. Results synthesis and structured output
6. Frontend displays interactive results with charts and metrics

## File Upload Requirements

- **Supported formats**: PNG, JPG, JPEG, GIF, WebP
- **Maximum file size**: 50MB (configurable in `backend/config.py`)
- **Processing**: Automatic RGB conversion and validation
- **Storage**: Local filesystem with UUID-based naming

## Vector Store & RAG

- **Database**: ChromaDB for similarity search
- **Embeddings**: UIClip model for multimodal embeddings
- **Initialization**: Auto-populates with sample data on first run
- **Usage**: Enhances agent analysis with relevant design patterns

## API Key Management

- **OpenRouter**: Required for LLM functionality (GPT-4 Turbo)
- **Hugging Face**: Optional, for enhanced model access
- **Runtime Override**: API keys can be provided via headers
- **Fallback**: Uses environment variables as default

## Deployment Notes

### Production Considerations
- Set `debug: false` in `backend/config.py`
- Configure proper CORS origins in `backend/api/main.py`
- Use environment variables for all sensitive configuration
- Consider using a reverse proxy (nginx) for production
- Implement proper logging and monitoring

### Docker Deployment
- Multi-service architecture with separate containers
- Persistent volumes for uploads and vector database
- Network isolation with custom bridge network
- ChromaDB service for vector storage

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
2. **API Key Issues**: Verify environment variables are set correctly
3. **Port Conflicts**: Change ports in configuration if 8000 or 3000 are in use
4. **Memory Issues**: Large images may require more RAM; consider resizing
5. **Vector Store Issues**: Delete `vector_db/` directory to reset and repopulate

### Debug Mode
- Enable debug logging by setting `DEBUG=true` in environment
- Use FastAPI's interactive docs at `/docs` for API testing
- Check browser console for frontend errors
- Monitor backend logs for agent execution details

## Performance Optimization

- **Async Processing**: All agent operations are asynchronous
- **Background Tasks**: Analysis runs in background with progress tracking
- **Caching**: Vector embeddings are cached for performance
- **Parallel Execution**: Agents can run in parallel when possible
- **Resource Management**: Proper cleanup of analysis states

## Security Considerations

- **File Validation**: Strict image format and size validation
- **API Rate Limiting**: Consider implementing rate limiting for production
- **Input Sanitization**: All user inputs are validated and sanitized
- **Error Handling**: Sensitive information is not exposed in error messages
- **CORS**: Configure appropriate origins for production deployment
