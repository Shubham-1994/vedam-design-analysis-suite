# Multimodal Design Analysis Suite - Architecture Diagram

## System Architecture Overview

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[Svelte Frontend<br/>Port 3000]
        UI_COMP[Components<br/>- UploadSection<br/>- AnalysisResults<br/>- Dashboard<br/>- LoadingSpinner]
        UI_STORE[Stores<br/>- analysisStore<br/>- apiKeyStore]
        UI --> UI_COMP
        UI --> UI_STORE
    end

    subgraph "API Gateway"
        API[FastAPI Backend<br/>Port 8000]
        CORS[CORS Middleware]
        API --> CORS
    end

    subgraph "Core Services"
        ORCH[LangGraph Orchestrator<br/>Agent Coordination]
        CONFIG[Configuration<br/>Settings & Environment]
        UPLOAD[File Upload Handler<br/>Image Processing]
    end

    subgraph "AI Agents"
        VA[Visual Analysis Agent<br/>Layout, Colors, Typography]
        UX[UX Critique Agent<br/>Usability, Navigation, A11y]
        MR[Market Research Agent<br/>Competitive Analysis, Trends]
        BASE[Base Agent<br/>Common Functionality]
        
        VA --> BASE
        UX --> BASE
        MR --> BASE
    end

    subgraph "Data Layer"
        VECTOR[ChromaDB Vector Store<br/>Design Pattern Embeddings]
        UPLOADS[File System<br/>Uploaded Images & Results]
        EMBED[Embedding Service<br/>UIClip Model]
    end

    subgraph "External Services"
        OPENROUTER[OpenRouter API<br/>GPT-4 Turbo]
        HF[Hugging Face<br/>Models & Datasets]
    end

    subgraph "Docker Infrastructure"
        DOCKER[Docker Compose]
        CHROMA_CONTAINER[ChromaDB Container<br/>Port 8001]
        BACKEND_CONTAINER[Backend Container]
        FRONTEND_CONTAINER[Frontend Container]
        
        DOCKER --> CHROMA_CONTAINER
        DOCKER --> BACKEND_CONTAINER
        DOCKER --> FRONTEND_CONTAINER
    end

    %% Frontend to Backend
    UI --> API
    
    %% API to Core Services
    API --> ORCH
    API --> CONFIG
    API --> UPLOAD
    
    %% Orchestrator to Agents
    ORCH --> VA
    ORCH --> UX
    ORCH --> MR
    
    %% Agents to Data Layer
    VA --> VECTOR
    UX --> VECTOR
    MR --> VECTOR
    BASE --> EMBED
    
    %% Data Layer Connections
    UPLOAD --> UPLOADS
    EMBED --> VECTOR
    
    %% External Service Connections
    BASE --> OPENROUTER
    EMBED --> HF
    
    %% Docker Connections
    BACKEND_CONTAINER --> API
    FRONTEND_CONTAINER --> UI
    CHROMA_CONTAINER --> VECTOR

    %% Styling
    classDef frontend fill:#e1f5fe
    classDef backend fill:#f3e5f5
    classDef agents fill:#e8f5e8
    classDef data fill:#fff3e0
    classDef external fill:#ffebee
    classDef docker fill:#f5f5f5

    class UI,UI_COMP,UI_STORE frontend
    class API,CORS,ORCH,CONFIG,UPLOAD backend
    class VA,UX,MR,BASE agents
    class VECTOR,UPLOADS,EMBED data
    class OPENROUTER,HF external
    class DOCKER,CHROMA_CONTAINER,BACKEND_CONTAINER,FRONTEND_CONTAINER docker
```

## Data Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant API
    participant Orchestrator
    participant VisualAgent
    participant UXAgent
    participant MarketAgent
    participant VectorDB
    participant OpenRouter

    User->>Frontend: Upload design image
    Frontend->>API: POST /analyze (multipart/form-data)
    API->>API: Validate image & create analysis ID
    API->>Orchestrator: Start analysis workflow
    
    Note over Orchestrator: Initialize LangGraph workflow
    Orchestrator->>VisualAgent: Analyze visual elements
    VisualAgent->>VectorDB: Query similar patterns
    VectorDB-->>VisualAgent: Return relevant patterns
    VisualAgent->>OpenRouter: Generate analysis with LLM
    OpenRouter-->>VisualAgent: Return analysis results
    VisualAgent-->>Orchestrator: Visual analysis complete
    
    Orchestrator->>UXAgent: Analyze UX elements
    UXAgent->>VectorDB: Query UX patterns
    VectorDB-->>UXAgent: Return UX patterns
    UXAgent->>OpenRouter: Generate UX analysis
    OpenRouter-->>UXAgent: Return UX results
    UXAgent-->>Orchestrator: UX analysis complete
    
    Orchestrator->>MarketAgent: Analyze market positioning
    MarketAgent->>VectorDB: Query market trends
    VectorDB-->>MarketAgent: Return trend data
    MarketAgent->>OpenRouter: Generate market analysis
    OpenRouter-->>MarketAgent: Return market results
    MarketAgent-->>Orchestrator: Market analysis complete
    
    Orchestrator->>Orchestrator: Synthesize all results
    Orchestrator-->>API: Return complete analysis
    API->>API: Save results to file system
    
    loop Status Polling
        Frontend->>API: GET /analysis/{id}/status
        API-->>Frontend: Return current status & progress
    end
    
    Frontend->>API: GET /analysis/{id}/result
    API-->>Frontend: Return complete analysis
    Frontend->>User: Display interactive results
```

## Agent Workflow Diagram

```mermaid
stateDiagram-v2
    [*] --> Initialize
    Initialize --> RouteAnalysis
    
    RouteAnalysis --> VisualOnly : Single Visual
    RouteAnalysis --> UXOnly : Single UX
    RouteAnalysis --> MarketOnly : Single Market
    RouteAnalysis --> ParallelExecution : Multiple Agents
    
    VisualOnly --> VisualAnalysis
    UXOnly --> UXAnalysis
    MarketOnly --> MarketAnalysis
    
    ParallelExecution --> VisualAnalysis
    VisualAnalysis --> CheckNext1
    CheckNext1 --> UXAnalysis : UX Requested
    CheckNext1 --> MarketAnalysis : Market Requested
    CheckNext1 --> Synthesize : All Complete
    
    UXAnalysis --> CheckNext2
    CheckNext2 --> MarketAnalysis : Market Requested
    CheckNext2 --> Synthesize : All Complete
    
    MarketAnalysis --> Synthesize
    
    Synthesize --> [*]
    
    state VisualAnalysis {
        [*] --> LoadImage
        LoadImage --> ExtractFeatures
        ExtractFeatures --> QueryVectorDB
        QueryVectorDB --> GenerateAnalysis
        GenerateAnalysis --> [*]
    }
    
    state UXAnalysis {
        [*] --> AnalyzeUsability
        AnalyzeUsability --> CheckAccessibility
        CheckAccessibility --> EvaluateNavigation
        EvaluateNavigation --> [*]
    }
    
    state MarketAnalysis {
        [*] --> IdentifyTrends
        IdentifyTrends --> CompetitiveAnalysis
        CompetitiveAnalysis --> PositioningAnalysis
        PositioningAnalysis --> [*]
    }
```

## Component Architecture Diagram

```mermaid
graph LR
    subgraph "Frontend Components"
        APP[App.svelte<br/>Main Application]
        HEADER[Header.svelte<br/>Navigation]
        UPLOAD[UploadSection.svelte<br/>File Upload]
        RESULTS[AnalysisResults.svelte<br/>Results Display]
        DASHBOARD[AnalysisDashboard.svelte<br/>History Management]
        LOADING[LoadingSpinner.svelte<br/>Progress Indicator]
        CHART[MetricsChart.svelte<br/>Data Visualization]
        
        APP --> HEADER
        APP --> UPLOAD
        APP --> RESULTS
        APP --> DASHBOARD
        APP --> LOADING
        RESULTS --> CHART
    end

    subgraph "Backend Modules"
        MAIN[main.py<br/>FastAPI App]
        SCHEMAS[schemas.py<br/>Data Models]
        CONFIG_PY[config.py<br/>Settings]
        ORCH_PY[orchestrator.py<br/>Agent Coordination]
        
        subgraph "Agents"
            BASE_AGENT[base_agent.py<br/>Common Logic]
            VISUAL_AGENT[visual_analysis_agent.py<br/>Visual Analysis]
            UX_AGENT[ux_critique_agent.py<br/>UX Analysis]
            MARKET_AGENT[market_research_agent.py<br/>Market Analysis]
        end
        
        subgraph "Utils"
            EMBEDDINGS[embeddings.py<br/>Vector Processing]
            VECTOR_STORE[vector_store.py<br/>DB Operations]
            LLM_CLIENT[llm_client.py<br/>API Integration]
        end
        
        MAIN --> SCHEMAS
        MAIN --> CONFIG_PY
        MAIN --> ORCH_PY
        ORCH_PY --> BASE_AGENT
        BASE_AGENT --> VISUAL_AGENT
        BASE_AGENT --> UX_AGENT
        BASE_AGENT --> MARKET_AGENT
        BASE_AGENT --> EMBEDDINGS
        BASE_AGENT --> VECTOR_STORE
        BASE_AGENT --> LLM_CLIENT
    end

    %% Cross-layer connections
    APP -.->|HTTP API| MAIN
    UPLOAD -.->|File Upload| MAIN
    RESULTS -.->|Get Results| MAIN
    DASHBOARD -.->|History API| MAIN

    %% Styling
    classDef frontend fill:#e3f2fd
    classDef backend fill:#f1f8e9
    classDef agents fill:#fff3e0
    classDef utils fill:#fce4ec

    class APP,HEADER,UPLOAD,RESULTS,DASHBOARD,LOADING,CHART frontend
    class MAIN,SCHEMAS,CONFIG_PY,ORCH_PY backend
    class BASE_AGENT,VISUAL_AGENT,UX_AGENT,MARKET_AGENT agents
    class EMBEDDINGS,VECTOR_STORE,LLM_CLIENT utils
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Production Environment"
        LB[Load Balancer<br/>nginx/Traefik]
        
        subgraph "Application Tier"
            FE1[Frontend Instance 1<br/>Svelte App]
            FE2[Frontend Instance 2<br/>Svelte App]
            BE1[Backend Instance 1<br/>FastAPI]
            BE2[Backend Instance 2<br/>FastAPI]
        end
        
        subgraph "Data Tier"
            CHROMA[ChromaDB Cluster<br/>Vector Storage]
            FS[File System<br/>Shared Storage]
            REDIS[Redis Cache<br/>Session Storage]
        end
        
        subgraph "External Services"
            OR[OpenRouter API<br/>LLM Services]
            HF_EXT[Hugging Face<br/>Model Hub]
            MONITOR[Monitoring<br/>Logs & Metrics]
        end
    end

    LB --> FE1
    LB --> FE2
    LB --> BE1
    LB --> BE2
    
    BE1 --> CHROMA
    BE2 --> CHROMA
    BE1 --> FS
    BE2 --> FS
    BE1 --> REDIS
    BE2 --> REDIS
    
    BE1 --> OR
    BE2 --> OR
    BE1 --> HF_EXT
    BE2 --> HF_EXT
    
    BE1 --> MONITOR
    BE2 --> MONITOR

    %% Styling
    classDef lb fill:#ffcdd2
    classDef app fill:#c8e6c9
    classDef data fill:#bbdefb
    classDef external fill:#f8bbd9

    class LB lb
    class FE1,FE2,BE1,BE2 app
    class CHROMA,FS,REDIS data
    class OR,HF_EXT,MONITOR external
```

## Complete Workflow Flowchart (Top Down)

```mermaid
flowchart TD
    Start([User Opens Application]) --> Upload{Upload Design Image?}
    
    Upload -->|Yes| ValidateFile[Validate File Format & Size]
    Upload -->|No| Dashboard[View Analysis Dashboard]
    
    ValidateFile --> FileValid{File Valid?}
    FileValid -->|No| ErrorMsg[Show Error Message]
    ErrorMsg --> Upload
    
    FileValid -->|Yes| CreateAnalysisID[Generate Analysis ID]
    CreateAnalysisID --> SaveFile[Save Image to File System]
    SaveFile --> ParseContext[Parse Design Context]
    ParseContext --> InitializeStatus[Initialize Analysis Status]
    
    InitializeStatus --> StartBackground[Start Background Analysis]
    StartBackground --> ShowProgress[Show Progress to User]
    
    %% Background Analysis Flow
    StartBackground --> InitOrchestrator[Initialize LangGraph Orchestrator]
    InitOrchestrator --> CreateAgents[Create AI Agents with API Keys]
    CreateAgents --> RouteAnalysis{Route Analysis Type}
    
    RouteAnalysis -->|Visual Only| VisualFlow[Visual Analysis Flow]
    RouteAnalysis -->|UX Only| UXFlow[UX Analysis Flow]
    RouteAnalysis -->|Market Only| MarketFlow[Market Analysis Flow]
    RouteAnalysis -->|Multiple| ParallelFlow[Parallel Analysis Flow]
    
    %% Visual Analysis Flow
    VisualFlow --> VA_Start[Visual Agent: Start Analysis]
    VA_Start --> VA_ProcessImage[Process Image Features]
    VA_ProcessImage --> VA_QueryVector[Query Vector DB for Patterns]
    VA_QueryVector --> VA_LLM[Generate Analysis with LLM]
    VA_LLM --> VA_Complete[Visual Analysis Complete]
    
    %% UX Analysis Flow
    UXFlow --> UX_Start[UX Agent: Start Analysis]
    UX_Start --> UX_Usability[Analyze Usability Heuristics]
    UX_Usability --> UX_Navigation[Evaluate Navigation]
    UX_Navigation --> UX_Accessibility[Check Accessibility]
    UX_Accessibility --> UX_QueryVector[Query Vector DB for UX Patterns]
    UX_QueryVector --> UX_LLM[Generate UX Analysis with LLM]
    UX_LLM --> UX_Complete[UX Analysis Complete]
    
    %% Market Analysis Flow
    MarketFlow --> MR_Start[Market Agent: Start Analysis]
    MR_Start --> MR_Trends[Identify Design Trends]
    MR_Trends --> MR_Competitive[Competitive Analysis]
    MR_Competitive --> MR_QueryVector[Query Vector DB for Market Data]
    MR_QueryVector --> MR_LLM[Generate Market Analysis with LLM]
    MR_LLM --> MR_Complete[Market Analysis Complete]
    
    %% Parallel Flow
    ParallelFlow --> PA_Visual[Start Visual Analysis]
    ParallelFlow --> PA_UX[Start UX Analysis]
    ParallelFlow --> PA_Market[Start Market Analysis]
    
    PA_Visual --> VA_ProcessImage
    PA_UX --> UX_Usability
    PA_Market --> MR_Trends
    
    %% Convergence Points
    VA_Complete --> CheckComplete{All Analyses Complete?}
    UX_Complete --> CheckComplete
    MR_Complete --> CheckComplete
    
    CheckComplete -->|No| WaitForCompletion[Wait for Other Agents]
    WaitForCompletion --> CheckComplete
    
    CheckComplete -->|Yes| SynthesizeResults[Synthesize All Results]
    SynthesizeResults --> CalculateScores[Calculate Overall Scores]
    CalculateScores --> ExtractFindings[Extract Key Findings]
    ExtractFindings --> GenerateRecommendations[Generate Recommendations]
    GenerateRecommendations --> SaveResults[Save Results to File]
    
    SaveResults --> UpdateStatus[Update Analysis Status to Complete]
    UpdateStatus --> NotifyFrontend[Notify Frontend of Completion]
    
    %% Frontend Status Polling
    ShowProgress --> PollStatus[Poll Analysis Status]
    PollStatus --> StatusCheck{Analysis Complete?}
    StatusCheck -->|No| UpdateProgress[Update Progress Display]
    UpdateProgress --> PollStatus
    
    StatusCheck -->|Yes| FetchResults[Fetch Complete Results]
    FetchResults --> DisplayResults[Display Interactive Results]
    
    %% Results Display Flow
    DisplayResults --> ShowMetrics[Show Metrics & Scores]
    ShowMetrics --> ShowFindings[Show Detailed Findings]
    ShowFindings --> ShowRecommendations[Show Recommendations]
    ShowRecommendations --> ShowCharts[Show Interactive Charts]
    
    %% User Actions on Results
    ShowCharts --> UserAction{User Action?}
    UserAction -->|Download Report| DownloadFlow[Download Report Flow]
    UserAction -->|New Analysis| Upload
    UserAction -->|View Dashboard| Dashboard
    UserAction -->|Share Results| ShareFlow[Share Results Flow]
    
    %% Download Flow
    DownloadFlow --> SelectFormat{Select Download Format}
    SelectFormat -->|JSON| DownloadJSON[Download as JSON]
    SelectFormat -->|PDF| GeneratePDF[Generate PDF Report]
    SelectFormat -->|CSV| GenerateCSV[Generate CSV Export]
    SelectFormat -->|PNG| DownloadImage[Download Original Image]
    
    GeneratePDF --> DownloadPDF[Download PDF File]
    GenerateCSV --> DownloadCSVFile[Download CSV File]
    
    DownloadJSON --> DownloadComplete[Download Complete]
    DownloadPDF --> DownloadComplete
    DownloadCSVFile --> DownloadComplete
    DownloadImage --> DownloadComplete
    
    DownloadComplete --> UserAction
    
    %% Dashboard Flow
    Dashboard --> LoadHistory[Load Analysis History]
    LoadHistory --> FilterHistory{Apply Filters?}
    FilterHistory -->|Yes| ApplyFilters[Apply Search/Status Filters]
    FilterHistory -->|No| ShowHistory[Show Paginated History]
    ApplyFilters --> ShowHistory
    
    ShowHistory --> HistoryAction{User Action?}
    HistoryAction -->|View Analysis| LoadSpecificResult[Load Specific Result]
    HistoryAction -->|Delete Analysis| DeleteAnalysis[Delete Analysis & Files]
    HistoryAction -->|Download| DownloadFlow
    HistoryAction -->|New Analysis| Upload
    
    LoadSpecificResult --> DisplayResults
    DeleteAnalysis --> RefreshHistory[Refresh History List]
    RefreshHistory --> ShowHistory
    
    %% Error Handling
    VA_LLM -->|Error| VA_Error[Visual Analysis Failed]
    UX_LLM -->|Error| UX_Error[UX Analysis Failed]
    MR_LLM -->|Error| MR_Error[Market Analysis Failed]
    
    VA_Error --> LogError[Log Error Details]
    UX_Error --> LogError
    MR_Error --> LogError
    
    LogError --> PartialResults{Any Successful Results?}
    PartialResults -->|Yes| SynthesizeResults
    PartialResults -->|No| AnalysisError[Analysis Failed]
    
    AnalysisError --> UpdateStatusError[Update Status to Failed]
    UpdateStatusError --> ShowError[Show Error to User]
    ShowError --> Upload
    
    %% Share Flow
    ShareFlow --> GenerateShareLink[Generate Share Link]
    GenerateShareLink --> CopyToClipboard[Copy Link to Clipboard]
    CopyToClipboard --> ShareComplete[Share Complete]
    ShareComplete --> UserAction
    
    %% Styling
    classDef startEnd fill:#e8f5e8,stroke:#4caf50,stroke-width:2px
    classDef process fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    classDef decision fill:#fff3e0,stroke:#ff9800,stroke-width:2px
    classDef agent fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    classDef error fill:#ffebee,stroke:#f44336,stroke-width:2px
    classDef data fill:#e0f2f1,stroke:#009688,stroke-width:2px
    
    class Start,DownloadComplete,ShareComplete,AnalysisError startEnd
    class ValidateFile,CreateAnalysisID,SaveFile,ParseContext,InitializeStatus,StartBackground,ShowProgress,InitOrchestrator,CreateAgents,SynthesizeResults,CalculateScores,ExtractFindings,GenerateRecommendations,SaveResults,UpdateStatus,NotifyFrontend,FetchResults,DisplayResults,ShowMetrics,ShowFindings,ShowRecommendations,ShowCharts,LoadHistory,ApplyFilters,ShowHistory,LoadSpecificResult,DeleteAnalysis,RefreshHistory,GeneratePDF,GenerateCSV,DownloadJSON,DownloadImage,DownloadPDF,DownloadCSVFile,GenerateShareLink,CopyToClipboard process
    class Upload,FileValid,RouteAnalysis,CheckComplete,StatusCheck,UserAction,SelectFormat,FilterHistory,HistoryAction,PartialResults decision
    class VA_Start,VA_ProcessImage,VA_QueryVector,VA_LLM,VA_Complete,UX_Start,UX_Usability,UX_Navigation,UX_Accessibility,UX_QueryVector,UX_LLM,UX_Complete,MR_Start,MR_Trends,MR_Competitive,MR_QueryVector,MR_LLM,MR_Complete,PA_Visual,PA_UX,PA_Market agent
    class ErrorMsg,VA_Error,UX_Error,MR_Error,LogError,UpdateStatusError,ShowError error
    class Dashboard,PollStatus,UpdateProgress,WaitForCompletion,VisualFlow,UXFlow,MarketFlow,ParallelFlow,DownloadFlow,ShareFlow data
```
