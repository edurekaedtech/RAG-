# RAG App with Streamlit

A production-ready Retrieval-Augmented Generation (RAG) application built with Streamlit, FAISS, and OpenAI. This demo showcases a complete pipeline for document ingestion, semantic search, and context-aware LLM responses with specialized domain prompts.

## Features

 **Core Capabilities**
- **Document Ingestion**: Automatically processes and chunks documents from a specified folder
- **FAISS Vector Database**: Fast similarity search using Facebook's FAISS library
- **Semantic Retrieval**: Retrieves relevant document chunks using OpenAI embeddings
- **Domain Specialization**: Multiple system prompts for different assistant behaviors (e.g., Technical Expert, Friendly Assistant)
- **LLM Integration**: Seamless integration with OpenAI's GPT models (GPT-4, GPT-4 Turbo, GPT-4o-mini)

 **Optimization Controls**
- Configurable chunk size and overlap for document splitting
- Adjustable top-k retrieval parameters
- Model selection for cost vs. quality tradeoffs
- Prompt caching for improved performance
- Real-time parameter adjustments via sidebar controls

 **Security**
- Multi-source API key management (Streamlit Secrets, Environment Variables, Session State)
- Secure password input fields
- Support for Streamlit Cloud deployment

## Project Structure

```
rag-streamlit-demo/
├── app.py                 # Main Streamlit application
├── auth.py               # API key management utilities
├── requirements.txt      # Python dependencies
├── data/
│   ├── docs/            # Source documents (TXT, PDF)
│   │   ├── company_policies.txt
│   │   └── tech_docs.txt
│   └── faiss_db/        # FAISS index storage
│       ├── demo_docs_index.faiss
│       └── demo_docs_metadata.json
└── rag/                 # RAG pipeline modules
    ├── ingest.py        # Document ingestion & chunking
    ├── retrieve.py      # FAISS retrieval & embedding
    ├── prompts.py       # System prompts for different domains
    ├── utils.py         # Helper utilities
    └── __pycache__/
```

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key

### Setup

1. **Clone or download the project**
   ```bash
   cd rag-streamlit-demo
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Key**
   
   Choose one of the following options:
   
   - **Streamlit Cloud (Recommended)**
     Create `.streamlit/secrets.toml`:
     ```toml
     OPENAI_API_KEY = "sk-..."
     ```
   
   - **Local Environment**
     Create a `.env` file:
     ```env
     OPENAI_API_KEY=sk-...
     ```
   
   - **Runtime Input**
     Paste your API key directly in the sidebar when running the app

## Usage

### Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Ingesting Documents

1. Place your documents in the `data/docs/` folder
   - Supported formats: `.txt`, `.pdf`
2. Adjust **Storage** settings in the sidebar if using custom paths
3. Click the **"Ingest Documents"** button to process new documents
4. The FAISS index will be created and stored automatically

### Querying Documents

1. **Configure RAG Settings** in the sidebar:
   - **Assistant specialization**: Choose the domain (Technical Expert, Friendly Assistant, etc.)
   - **LLM model**: Select GPT model (default: gpt-4o-mini for cost efficiency)
   - **Chunk size**: 300-1500 chars (larger = more context, smaller = more precise)
   - **Chunk overlap**: 0-400 chars (prevents splitting important info)
   - **Top-k retrieval**: 1-10 chunks (higher = more recall, higher cost)

2. **Enter your question** in the main chat interface
3. The app will:
   - Embed your query using OpenAI's embedding model
   - Retrieve top-k relevant document chunks
   - Generate a response using the selected LLM and domain prompt
   - Display sources and retrieved context

## Configuration

### Sidebar Controls

| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| Assistant Specialization | - | Varies | System prompt domain |
| LLM Model | gpt-4o-mini, gpt-4-turbo, gpt-4 | gpt-4o-mini | Model selection |
| Chunk Size | 300-1500 | 900 | Tokens per document chunk |
| Chunk Overlap | 0-400 | 150 | Overlap between chunks |
| Top-k Retrieval | 1-10 | 4 | Number of chunks to retrieve |
| Docs Folder | - | data/docs | Document source path |
| FAISS Persist Dir | - | data/faiss_db | Index storage path |
| Collection Name | - | demo_docs | FAISS collection name |

## How It Works

### 1. **Document Ingestion** (`rag/ingest.py`)
- Reads documents from specified folder
- Splits into chunks with configurable overlap
- Generates OpenAI embeddings for each chunk
- Stores embeddings in FAISS index

### 2. **Semantic Retrieval** (`rag/retrieve.py`)
- Embeds user query using OpenAI API
- Performs similarity search on FAISS index
- Returns top-k most relevant chunks

### 3. **LLM Generation** (`app.py`)
- Constructs context from retrieved chunks
- Applies domain-specific system prompt
- Sends to OpenAI with conversation history
- Streams response to user

## API Keys & Security

The app uses a three-tier API key resolution strategy:

1. **Streamlit Secrets** (Cloud deployment) - Most secure
2. **Environment Variables** (Local setup) - Recommended for local dev
3. **Session State** (Runtime input) - Fallback for ad-hoc usage

Keys are **never persisted** to disk; they exist only during the session.

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | Latest | Web UI framework |
| openai | Latest | LLM & embeddings API |
| faiss-cpu | Latest | Vector similarity search |
| python-dotenv | Latest | Environment variable loading |
| pypdf | Latest | PDF document parsing |
| numpy | Latest | Array operations |

## Troubleshooting

### "Index or metadata not found"
**Solution**: Run the "Ingest Documents" button in the app to create the FAISS index.

###  "API key not found"
**Solution**: Ensure your API key is set via:
- `.streamlit/secrets.toml` (Cloud)
- `.env` file (Local)
- Sidebar input (Runtime)

###  "FAISS index version mismatch"
**Solution**: Delete `data/faiss_db/` and re-ingest documents.

###  "Slow retrieval or generation"
**Solution**: 
- Reduce `top_k` parameter
- Decrease `chunk_size` for faster embedding
- Use `gpt-4o-mini` model for lower latency

## Performance Tips

 **Optimize for Speed**
- Use `gpt-4o-mini` model (fastest & cheapest)
- Set `top_k = 2-3` for retrieval
- Reduce `chunk_size` to 300-500

 **Optimize for Quality**
- Use `gpt-4-turbo` or `gpt-4` model
- Set `top_k = 6-8` for more context
- Increase `chunk_size` to 1000-1500

 **Optimize for Cost**
- Use `gpt-4o-mini` model
- Enable prompt caching (default)
- Use smaller `chunk_size`
- Lower `top_k` retrieval

## Deployment

### Local Development
```bash
streamlit run app.py
```

### Streamlit Cloud
1. Push code to GitHub
2. Create app in [Streamlit Cloud](https://streamlit.io/cloud)
3. Set `OPENAI_API_KEY` in Secrets
4. Deploy

## Example Use Cases

- **Internal Knowledge Base**: Query company policies and documentation
- **Technical Support**: Specialized assistant for troubleshooting
- **Research Assistant**: Retrieve and synthesize academic papers
- **Customer Support**: Retrieve product documentation for responses
- **Content Creation**: Retrieve brand guidelines and reference materials

## Limitations & Future Enhancements

### Current Limitations
- Single-user session (Streamlit Community Cloud limitation)
- In-memory chat history (resets on app refresh)
- CPU-based FAISS only (no GPU acceleration)

## Contributing

Contributions welcome! Feel free to submit issues and pull requests.

## License

MIT License - See LICENSE file for details

## Support

- **OpenAI API Docs**: https://platform.openai.com/docs
- **Streamlit Docs**: https://docs.streamlit.io
- **FAISS Documentation**: https://github.com/facebookresearch/faiss

