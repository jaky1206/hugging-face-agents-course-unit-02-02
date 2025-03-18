import os
import asyncio


from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_EY")
open_api_model_id = os.getenv("OPENAI_API_MODEL_ID")

llm = OpenAI(model=open_api_model_id, api_key=openai_api_key)


class MyWorkflow(Workflow):
    @step
    async def my_step(self, ev: StartEvent) -> StopEvent:
        # do something here
        return StopEvent(result="Hello, world!")


w = MyWorkflow(timeout=10, verbose=False)

async def main():
    result = await w.run()
    print(result)

if __name__ == "__main__":
    asyncio.run(main())

