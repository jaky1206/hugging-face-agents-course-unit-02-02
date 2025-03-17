import os
import chromadb

from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool
import asyncio
from llama_index.core.agent.workflow import AgentWorkflow

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
open_api_model_id = os.getenv("OPENAI_API_MODEL_ID")

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


query_engine = index.as_query_engine(llm=llm, similarity_top_k=3)

query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="Attention",
    description="Attention in transformer",
    return_direct=False,
)
query_engine_agent = AgentWorkflow.from_tools_or_functions(
    [query_engine_tool],
    llm=llm,
    system_prompt="You are a helpful assistant that has access to a database containing informatrion about transformer models.",
)

# query = "What does attention mean for transformer models?"
# response = query_engine.query(query)
# print(response)

from llama_index.core.agent.workflow import (
    AgentWorkflow,
    ReActAgent,
)


# Define some tools
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b


# Create agent configs
# NOTE: we can use FunctionAgent or ReActAgent here.
# FunctionAgent works for LLMs with a function calling API.
# ReActAgent works for any LLM.
calculator_agent = ReActAgent(
    name="calculator",
    description="Performs basic arithmetic operations",
    system_prompt="You are a calculator assistant. Use your tools for any math operation.",
    tools=[add, subtract],
    llm=llm,
)

query_agent = ReActAgent(
    name="info_lookup",
    description="Looks up information about XYZ",
    system_prompt="Use your tool to query a RAG system to answer information about XYZ",
    tools=[query_engine_tool],
    llm=llm,
)

# Create and run the workflow
agent = AgentWorkflow(agents=[calculator_agent, query_agent], root_agent="calculator")

async def main():
    # Run the system
    response = await agent.run(user_msg="Can you add 5 and 3?")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
