import os

from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
open_api_model_id = os.getenv("OPENAI_API_MODEL_ID")

llm = OpenAI(model=open_api_model_id, api_key=openai_api_key)

def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the resulting integer"""
    return a * b

import asyncio

agent = AgentWorkflow.from_tools_or_functions(
    [FunctionTool.from_defaults(multiply)],
    llm=llm
)

async def main():
    # stateless
    response = await agent.run("What is 2 times 2?")
    print(f"Response: {response}")

    # remembering state
    from llama_index.core.workflow import Context

    ctx = Context(agent)

    response = await agent.run("My name is Bob.", ctx=ctx)
    print(f"Response: {response}")
    response = await agent.run("What was my name again?", ctx=ctx)
    print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main())
