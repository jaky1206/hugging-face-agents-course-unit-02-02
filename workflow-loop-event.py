import os
import asyncio
import random

from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import Event
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step
from llama_index.utils.workflow import draw_all_possible_flows

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
open_api_model_id = os.getenv("OPENAI_API_MODEL_ID")

llm = OpenAI(model=open_api_model_id, api_key=openai_api_key)

class ProcessingEvent(Event):
    intermediate_result: str


class LoopEvent(Event):
    loop_output: str


class MultiStepWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent | LoopEvent) -> ProcessingEvent | LoopEvent:
        if random.randint(0, 1) == 0:
            print("Bad thing happened")
            return LoopEvent(loop_output="Back to step one.")
        else:
            print("Good thing happened")
            return ProcessingEvent(intermediate_result="First step complete.")

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        # Use the intermediate result
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)

w = MultiStepWorkflow(verbose=False)


#######################################
# Drawing Workflows
#######################################
draw_all_possible_flows(w, "flow.html")


async def main():
    result = await w.run()
    print(result)

if __name__ == "__main__":
    asyncio.run(main())

