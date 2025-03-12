import os
import chromadb

from dotenv import load_dotenv
from llama_index.core.tools import FunctionTool
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.tools.google import GmailToolSpec

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
open_api_model_id = os.getenv("OPENAI_API_MODEL_ID")

llm = OpenAI(model=open_api_model_id, api_key=openai_api_key)


def get_weather(location: str) -> str:
    """Useful for getting the weather for a given location."""
    print(f"Getting weather for {location}")
    return f"The weather in {location} is sunny"


functional_tool = FunctionTool.from_defaults(
    get_weather,
    name="my_weather_tool",
    description="Useful for getting the weather for a given location.",
)

# print(functional_tool.call("New York"))

embed_model = OpenAIEmbedding(api_key=openai_api_key)
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("alfred")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

query_engine = index.as_query_engine(llm=llm)
query_tool = QueryEngineTool.from_defaults(
    query_engine, name="Attention", description="Attention in transformer"
)

# print(query_tool.call("What is attention?"))

tool_spec = GmailToolSpec()
tool_spec_list = tool_spec.to_tool_list()

# Iterate through the list and print each tool's details
for tool in tool_spec_list:
    print(f"Tool Name: {tool.metadata.name}")
    print(f"Tool Description: {tool.metadata.description}")
    print("-" * 20)



