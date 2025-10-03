# Multimodal Design Analysis Suite

Check out this video: [Pixora](https://app.heygen.com/videos/779f3cf692724aa8a4078f9f51143bdc)

A comprehensive AI-powered UI/UX design analysis platform that uses multimodal agents to provide detailed feedback on visual design, user experience, and market positioning.

## 🚀 Features

### Core Analysis
- **Multimodal AI Analysis**: Upload design images and get comprehensive analysis
- **LangGraph Agent Orchestration**: Coordinated analysis using specialized agents
- **RAG-Enhanced Insights**: Leverages Hugging Face datasets and UIClip embeddings
- **Visual Analysis Agent**: Layout, colors, typography, and visual hierarchy analysis
- **UX Critique Agent**: Usability heuristics, user flow, and accessibility evaluation
- **Market Research Agent**: Competitive analysis, trend identification, and positioning

### New Dashboard & Export Features ✨
- **Analysis Dashboard**: View and manage past analysis reports with search and filtering
- **Multiple Download Formats**: Export reports as JSON, PDF, CSV, or download original images
- **History Management**: Paginated view of all completed analyses with metadata
- **Enhanced UI**: Modern interface with dropdown menus and improved navigation
- **Real-time Progress**: Interactive dashboard with real-time analysis tracking

## 🏗️ Architecture

### Backend (Python)
- **FastAPI**: RESTful API with async support
- **LangGraph**: Agent orchestration and workflow management
- **Hugging Face**: UIClip embeddings and datasets for RAG
- **ChromaDB**: Vector database for similarity search
- **OpenRouter**: LLM integration with OpenAI models

### Frontend (Svelte)
- **Svelte + TypeScript**: Modern reactive UI framework
- **TailwindCSS**: Utility-first styling
- **Chart.js**: Interactive metrics visualization
- **Lucide Icons**: Beautiful icon library

### Agent Architecture
```
Input Layer → Preprocessing → Embedding Pipeline → Agent Orchestrator
                                                         ↓
┌─────────────────┬─────────────────┬─────────────────────────┐
│ Visual Analysis │ UX Critique     │ Market Research         │
│ Agent           │ Agent           │ Agent                   │
├─────────────────┼─────────────────┼─────────────────────────┤
│ • Layout        │ • Usability     │ • Competitive Analysis  │
│ • Colors        │ • Navigation    │ • Trend Alignment       │
│ • Typography    │ • Accessibility │ • Innovation Potential  │
│ • Whitespace    │ • User Flow     │ • Market Positioning    │
└─────────────────┴─────────────────┴─────────────────────────┘
                                ↓
                    RAG Retrieval & Synthesis
                                ↓
                        Structured Output
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git

### Backend Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd multimodal-design-analysis-suite
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp env.example .env
```

Edit `.env` file with your API keys:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
HUGGINGFACE_API_TOKEN=your_huggingface_token_here
```

### Frontend Setup

1. **Navigate to frontend directory**
```bash
cd frontend
```

2. **Install Node.js dependencies**
```bash
npm install
```

## 🚀 Running the Application

### Start Backend Server
```bash
# From project root
python run_backend.py
```
The backend will be available at `http://localhost:8000`

### Start Frontend Development Server
```bash
# From frontend directory
cd frontend
npm run dev
```
The frontend will be available at `http://localhost:3000`

## 📖 API Documentation

Once the backend is running, visit `http://localhost:8000/docs` for interactive API documentation.

### Key Endpoints

### Analysis
- `POST /analyze` - Upload and analyze a design
- `GET /analysis/{id}/status` - Check analysis status
- `GET /analysis/{id}/result` - Get analysis results
- `DELETE /analysis/{id}` - Delete analysis

### New Dashboard & Export Endpoints ✨
- `GET /analysis/history` - Get paginated analysis history with search and filtering
- `GET /analysis/{id}/download?format={json|pdf|csv|png}` - Download analysis in various formats

### System
- `GET /health` - Health check
- `GET /vector-store/stats` - Vector store statistics

### Analysis History Parameters
- `page` (int): Page number (default: 1)
- `page_size` (int): Items per page (default: 10, max: 100)
- `search` (string): Search by filename
- `status_filter` (string): Filter by status (completed, failed, processing)

## 🎯 Usage

1. **Upload Design**: Navigate to the web interface and upload your UI/UX design image
2. **Configure Analysis**: Select analysis types and provide optional context
3. **Monitor Progress**: Watch real-time progress as agents analyze your design
4. **Review Results**: Get comprehensive analysis with scores, findings, and recommendations
5. **Download Report**: Export detailed analysis results

### Supported File Types
- PNG, JPG, JPEG, GIF, WebP
- Maximum file size: 50MB

### Analysis Types
- **Visual Analysis**: Layout, visual hierarchy, color harmony, typography, whitespace
- **UX Critique**: Usability heuristics, navigation clarity, user flow efficiency
- **Market Research**: Competitive analysis, trend alignment, innovation potential

## 🔧 Configuration

### Backend Configuration
Edit `backend/config.py` or use environment variables:

- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `HUGGINGFACE_API_TOKEN`: Hugging Face API token
- `UPLOAD_DIR`: Directory for uploaded files (default: ./uploads)
- `VECTOR_DB_PATH`: ChromaDB storage path (default: ./vector_db)
- `MAX_FILE_SIZE`: Maximum upload size in bytes (default: 50MB)

### Frontend Configuration
Edit `frontend/vite.config.js` for proxy settings and development configuration.

## 🧪 Development

### Backend Development
```bash
# Run with auto-reload
python run_backend.py

# Run tests (when available)
pytest

# Format code
black backend/
```

### Frontend Development
```bash
cd frontend

# Development server with hot reload
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Type checking
npm run check
```

## 🏗️ Project Structure

```
multimodal-design-analysis-suite/
├── backend/
│   ├── agents/
│   │   ├── base_agent.py
│   │   ├── visual_analysis_agent.py
│   │   ├── ux_critique_agent.py
│   │   ├── market_research_agent.py
│   │   └── orchestrator.py
│   ├── api/
│   │   └── main.py
│   ├── models/
│   │   └── schemas.py
│   ├── utils/
│   │   ├── embeddings.py
│   │   └── vector_store.py
│   └── config.py
├── frontend/
│   ├── src/
│   │   ├── lib/
│   │   │   ├── components/
│   │   │   ├── stores/
│   │   │   └── api/
│   │   ├── App.svelte
│   │   └── main.ts
│   ├── package.json
│   └── vite.config.js
├── requirements.txt
├── run_backend.py
└── README.md
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [LangGraph](https://langchain-ai.github.io/langgraph/) for agent orchestration
- [Hugging Face](https://huggingface.co/) for embeddings and datasets
- [UIClip](https://huggingface.co/biglab/uiclip_jitteredwebsites-2-224-paraphrased_webpairs_humanpairs) for multimodal embeddings
- [OpenRouter](https://openrouter.ai/) for LLM access
- [Svelte](https://svelte.dev/) for the frontend framework

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
2. **API Key Issues**: Check that environment variables are set correctly
3. **Port Conflicts**: Change ports in configuration if 8000 or 3000 are in use
4. **Memory Issues**: Large images may require more RAM; consider resizing before upload

### Getting Help

- Check the [Issues](https://github.com/your-repo/issues) page
- Review API documentation at `/docs`
- Enable debug logging by setting `DEBUG=true` in environment

## 🔮 Future Enhancements

- [ ] Accessibility analysis agent
- [ ] Style consistency checker
- [ ] A/B testing recommendations
- [ ] Integration with design tools (Figma, Sketch)
- [ ] Batch analysis capabilities
- [ ] Custom model fine-tuning
- [ ] Real-time collaboration features
