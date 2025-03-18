# Llamaindex RAG Pipeline with ChromaDB Vector Store
# Imports required modules and configurations
import os
import chromadb
import llama_index

# Load environment variables from .env file
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from llama_index.core.evaluation import FaithfulnessEvaluator
from phoenix.otel import register
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

query = "What does attention mean for transformer models?"

load_dotenv()

# Initialize OpenAI API credentials from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
open_api_model_id = os.getenv("OPENAI_API_MODEL_ID")

# Initialize phoenix llamatrace
'''
PHOENIX_API_KEY = os.getenv("PHOENIX_API_KEY")
PHOENIX_LLAMA_TRACE_BASE_URL = os.getenv("PHOENIX_LLAMA_TRACE_BASE_URL")
PHOENIX_COLLECTOR_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")

os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = PHOENIX_COLLECTOR_ENDPOINT

tracer_provider = register(
    endpoint= f"{PHOENIX_LLAMA_TRACE_BASE_URL}/traces", set_global_tracer_provider=False
)

LlamaIndexInstrumentor().instrument(
    skip_dep_check=True, tracer_provider=tracer_provider
)
'''

# Create OpenAI LLM instance with specified model
llm = OpenAI(model=open_api_model_id, api_key=openai_api_key)

# Initialize ChromaDB persistent client and collection
db = chromadb.PersistentClient(
    path="./chroma_db"
)  # Persistent vector storage for faster retrieval between sessions
chroma_collection = db.get_or_create_collection(
    "alfred"
)  # Named collection for documents
vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection
)  # LlamaIndex adapter

# Configure the document processing pipeline
pipeline = IngestionPipeline(
    transformations=[
        # Text chunking: Split documents into 100-character chunks with no overlap
        # Optimal for dense retrieval while preventing information fragmentation
        SentenceSplitter(chunk_size=100, chunk_overlap=0),
        # Embedding generation: Convert text to 1536-dimension vectors using OpenAI's text-embedding-3-small
        OpenAIEmbedding(api_key=openai_api_key, model="text-embedding-3-small"),
    ],
    vector_store=vector_store,  # Store embeddings in ChromaDB for similarity search
)

# Process example document through the pipeline
reader = SimpleDirectoryReader(input_dir="./data/")
documents = reader.load_data()
nodes = pipeline.run(documents=documents)  # Execute chunking and embedding

# Initialize embedding model for query operations
embed_model = OpenAIEmbedding(api_key=openai_api_key)

# Create vector store index from existing ChromaDB collection
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

# Configure query engine with summarization mode
query_engine = index.as_query_engine(
    llm=llm,  # GPT-3.5-turbo for response generation
    response_mode="tree_summarize",  # Builds hierarchical summary from multiple leaf nodes
    # Other modes: "refine" (iterative improvement), "compact" (single-pass)
)

response = query_engine.query(query)

evaluator = FaithfulnessEvaluator(
    llm=llm
)  # Measures hallucination by checking response against source context

eval_result = evaluator.evaluate_response(response=response)
print(eval_result.passing)  # True means response stayed grounded in source material
