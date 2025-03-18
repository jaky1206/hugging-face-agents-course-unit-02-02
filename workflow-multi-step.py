import os
import asyncio

from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import Event
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
open_api_model_id = os.getenv("OPENAI_API_MODEL_ID")

llm = OpenAI(model=open_api_model_id, api_key=openai_api_key)


class ProcessingEvent(Event):
    intermediate_result: str

class MultiStepWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent) -> ProcessingEvent:
        # Process initial data
        return ProcessingEvent(intermediate_result="Step 1 complete")

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        # Use the intermediate result
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)

w = MultiStepWorkflow(timeout=10, verbose=False)

async def main():
    result = await w.run()
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
